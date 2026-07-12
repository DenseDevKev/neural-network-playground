// ── Training Web Worker ──
// Owns the engine instance, runs training off the main thread.
// Hybrid communication:
//   - Comlink RPC for commands (initialize, updateConfig, reset, step, etc.)
//   - MessagePort commands for high-frequency streamed snapshots during training

import * as Comlink from 'comlink';
import {
    Network,
    NonFiniteNumericalError,
    PRNG,
    buildGridInputs,
    generateDataset,
    generateDatasetV2,
    getDatasetContract,
    getActiveFeatures,
    transformPoint,
    transformDataset,
    countActiveFeatures,
    isLossCompatible,
    describeLossIncompatibility,
    detectWebGPU,
    WebGPUGridPredictor,
    exceedsGpuShape,
    flattenGridInputs,
} from '@nn-playground/engine';
import type {
    NetworkConfig,
    TrainingConfig,
    DataConfig,
    FeatureFlags,
    NetworkSnapshot,
    DataPoint,
    HistoryPoint,
    Metrics,
    PredictionTrace,
    ActivationHistogramResult,
    NetworkCheckpoint,
    BackpropExplanation,
    LossLandscapeProbe,
    LossLandscapeProbeOptions,
    CompiledExperimentConfig,
    CompiledTaskContract,
    MulticlassConfusionMatrixData,
} from '@nn-playground/engine';
import {
    GRID_SIZE,
    DEFAULT_DEMAND,
    isMainToWorkerCommand,
    normalizeVisualizationDemand,
    structuralEqual,
    normalizeAppConfig,
    parseWorkerEvidenceMessageV2,
    parseWorkerProtocolErrorMessageV2,
    parseCaptureRunArtifactRequestV2,
    validateExperimentRunRecordV2,
    compactEvenly,
    EXPERIMENT_MEMORY_MAX_EVALUATIONS,
    EXPERIMENT_MEMORY_MAX_TRENDS,
    WORKER_PROTOCOL_VERSION,
} from '@nn-playground/shared';
import type {
    PauseReason,
    VisualizationDemand,
    WorkerSnapshotMessage,
    WorkerStatusMessage,
    WorkerErrorMessage,
    WorkerSharedBuffersMessage,
    CheckpointTimeline,
    CheckpointSummary,
    ArenaScalarSnapshot,
    ArenaSide,
    ArenaModelSummary,
    DatasetRevision,
    EvaluationValues,
    ForcedEvaluationTriggerV2,
    ArtifactBasis,
    ArtifactProvenance,
    LiveTrainingSignal,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerEvidenceMessageV2,
    WorkerProtocolErrorCodeV2,
    WorkerProtocolErrorSourceV2,
    WorkerArtifactProvenanceV2,
    CaptureRunArtifactRequestV2,
    ExperimentRunRecordV2,
} from '@nn-playground/shared';
import type { FeatureSpec } from '@nn-playground/engine';
import {
    allocSharedSnapshotViews,
    canUseSharedBuffers,
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    publishSharedSnapshot,
    type SharedSnapshotViews,
} from './sharedSnapshot.ts';
import {
    DEFAULT_RUNTIME_STOP_CONDITIONS,
    createInitialStopConditionState,
    evaluateStopConditions,
    stopConditionsRequireCurrentEvaluation,
    type StopCondition,
    type StopConditionState,
} from './stopConditions.ts';
import {
    createMiniBatchScratch,
    fillMiniBatchScratch,
    getTrainingStepsForTick,
    normalizeTrainingSpeed,
    type MiniBatchScratch,
} from './trainingLoop.ts';
import { projectPreparedExperiment } from '../store/legacyProjection.ts';
import {
    EvaluationRuntime,
    TerminalDivergenceError,
} from './evaluationRuntime.ts';
import {
    RuntimeMetricHistory,
    type RuntimeMetricHistorySnapshot,
} from './runtimeMetricHistory.ts';
import {
    ExperimentRequestGate,
    ExperimentTransactionError,
    type ExperimentRequestGateDependencies,
} from './experimentTransaction.ts';

interface WorkerState {
    /** Canonical V2 preparation. Null means the worker is on the legacy path. */
    prepared: PreparedExperimentDocumentV2 | null;
    /** Exact compiled contract associated with `prepared`. */
    compiled: CompiledExperimentConfig | null;
    /** Versioned immutable dataset identity for scientific evidence. */
    datasetRevision: DatasetRevision | null;
    /** Owns live EMA and atomic paired evaluation publication. */
    evaluationRuntime: EvaluationRuntime | null;
    /** Worker-authoritative V2 metric histories. */
    metricHistory: RuntimeMetricHistory | null;
    /** Mutable epoch cell captured by the generation's evaluation callbacks. */
    v2EpochRef: { value: number } | null;
    network: Network | null;
    networkConfig: NetworkConfig | null;
    trainingConfig: TrainingConfig | null;
    dataConfig: DataConfig | null;
    features: FeatureFlags | null;
    activeFeatures: FeatureSpec[];
    trainInputs: number[][];
    trainTargets: number[][];
    testInputs: number[][];
    testTargets: number[][];
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    gridInputs: number[][];
    epoch: number;
    running: boolean;
    rafId: number | null;
    /** Shuffled index array — re-shuffled at the start of each epoch. */
    shuffledIndices: number[];
    /** Reusable mini-batch views, filled with sample references each step. */
    batchScratch: MiniBatchScratch | null;
    /** Separate PRNG for epoch shuffling, independent from network weights. */
    shufflePrng: PRNG | null;
    /** What visual data the UI currently needs. */
    demand: VisualizationDemand;
    /** Counter for test-eval frequency gating. */
    snapshotsSinceLastTestEval: number;
    /** Counter for train-eval frequency gating. Between full evals we report
     *  the EMA of per-step batch loss instead. */
    snapshotsSinceLastTrainEval: number;
    /** Counter for grid-rebuild frequency gating. */
    snapshotsSinceLastGrid: number;
    /** Counter for activation histogram frequency gating. */
    snapshotsSinceLastActivationHistogram: number;
    /** Cached last test metrics to reuse when skipping test evaluation. */
    lastTestMetrics: { loss: number; accuracy?: number; confusionMatrix?: { tp: number; tn: number; fp: number; fn: number } } | null;
    /** Cached last train metrics (full-dataset evaluation). Between
     *  intervals we overlay a running EMA of batch loss on top of the cached
     *  accuracy to give the UI a responsive-looking curve. */
    lastTrainMetrics: { loss: number; accuracy?: number } | null;
    /** Exponential moving average of per-step batch loss, used between full
     *  train-set evaluations. Initialized lazily to the first observed loss. */
    lossEma: number | null;
    /** EMA decay (0..1]. Lower = more responsive to recent batches. */
    lossEmaAlpha: number;
    /** True when the most recent snapshot reused cached test metrics instead of re-evaluating. */
    testMetricsStale: boolean;
    /** Runtime-only bounded checkpoint ring buffer. Heavy model state stays in the worker. */
    checkpoints: RuntimeCheckpoint[];
    nextCheckpointId: number;
    checkpointEvictedCount: number;
    restoredCheckpointId: number | null;
    /** Runtime-only stop-condition tracking. Reset on new worker runs/rebuilds. */
    stopConditionState: StopConditionState;
    /** Monotonic identity for confusion matrix payload freshness. */
    confusionMatrixVersion: number;
    /** Monotonic identity for bounded multiclass confusion matrix payload freshness. */
    multiclassConfusionMatrixVersion: number;
    /** Evaluation ID whose paired confusion bytes were most recently transported. */
    lastPackedConfusionEvaluationId: number | null;
    /** Monotonic identity for activation histogram payload freshness. */
    activationHistogramVersion: number;
    /** Clipped-gradient revision summarized by the current layer-statistics payload. */
    layerStatsGradientRevision: number | null;
    /** Monotonic identity for bounded multiclass boundary payload freshness. */
    multiclassBoundaryVersion: number;
    /** Pre-allocated buffers for grid predictions. */
    outputGridBuffer: Float32Array | null;
    neuronGridsBuffer: Float32Array | null;
    multiclassClassGridBuffer: Uint8Array | null;
    multiclassConfidenceGridBuffer: Float32Array | null;
    /** True only for the snapshot that freshly recomputed multiclass boundary data. */
    multiclassBoundaryFresh: boolean;
    /** True if the last computed grid is still fresh for this network state
     *  (i.e. the worker has trained since the grid was last recomputed). */
    gridStale: boolean;
    /** SharedArrayBuffer-backed snapshot transport. Allocated when the host
     *  is cross-origin isolated and torn down (re-allocated) on shape change.
     *  When non-null, heavy grid payloads are published through these views
     *  instead of being message-transferred. */
    sharedViews: SharedSnapshotViews | null;
    /** WebGPU grid predictor (AS-4). Allocated lazily on the first build
     *  after the user opts in to the GPU path AND the device is available
     *  AND the network shape fits the shader's compile-time caps. Disposed
     *  + reallocated on every shape change. */
    gpuPredictor: WebGPUGridPredictor | null;
    /** Whether the user has opted in to the WebGPU grid path. Set by the
     *  `setWebGpuEnabled` Comlink RPC. Defaults to false until the first
     *  initialize/updateConfig with the flag on. */
    gpuEnabled: boolean;
    /** Cached gridInputs as a flat Float32Array — the GPU path needs this
     *  shape and re-creating it per snapshot would be wasteful. */
    gridInputsFlat: Float32Array | null;
    /** Set true by `runGpuGridIfDue` after a successful GPU prediction.
     *  `computeSnapshot` consumes the flag to bypass the CPU recompute and
     *  hand the freshly-filled grid buffers straight to the snapshot. */
    gridFreshFromGpu: boolean;
    /** MessagePort for streaming snapshot delivery. */
    streamPort: MessagePort | null;
    /** Monotonically increasing run ID — incremented on init/reset/rebuild. */
    runId: number;
    /** Monotonically increasing snapshot ID within a run. */
    snapshotId: number;
    /** Steps per frame for the internal training loop. */
    stepsPerFrame: number;
    /** Timer ID for the internal training loop. */
    trainLoopTimer: ReturnType<typeof setTimeout> | null;
    /**
     * True when a snapshot has been posted to the main thread but has not yet
     * been acknowledged (applied to the frame buffer). Used for back-pressure:
     * while an ack is outstanding, the worker keeps training but skips further
     * postMessage calls so the transferable queue doesn't grow unbounded.
     */
    awaitingAck: boolean;
}

interface RuntimeCheckpoint {
    summary: CheckpointSummary;
    checkpoint: NetworkCheckpoint;
    epoch: number;
    shuffledIndices: number[];
    lossEma: number | null;
    lastTrainMetrics: WorkerState['lastTrainMetrics'];
    lastTestMetrics: WorkerState['lastTestMetrics'];
    snapshotsSinceLastTrainEval: number;
    snapshotsSinceLastTestEval: number;
    snapshotsSinceLastGrid: number;
    snapshotsSinceLastActivationHistogram: number;
}

interface ArenaModelInput {
    label?: string;
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
}

interface InitializeArenaRequest {
    modelA: ArenaModelInput;
    modelB: ArenaModelInput;
}

interface ArenaSlot {
    side: ArenaSide;
    label: string;
    network: Network;
    trainingConfig: TrainingConfig;
    dataConfig: DataConfig;
    trainInputs: number[][];
    trainTargets: number[][];
    testInputs: number[][];
    testTargets: number[][];
    epoch: number;
    shuffledIndices: number[];
    shufflePrng: PRNG;
    status: ArenaModelSummary['status'];
    pauseReason: PauseReason | null;
}

export type PredictionTraceSampleSource = 'train' | 'test' | 'custom';

export interface PredictionTraceRequest {
    source: PredictionTraceSampleSource;
    index?: number;
    x?: number;
    y?: number;
    label?: number;
}

export interface PredictionTraceResponse {
    runId: number;
    step: number;
    sample: {
        source: PredictionTraceSampleSource;
        index?: number;
        x: number;
        y: number;
        label?: number;
    };
    trace: PredictionTrace;
}

export interface BackpropExplanationResponse {
    runId: number;
    step: number;
    epoch: number;
    explanation: BackpropExplanation;
}

export interface SerializableLossLandscapeProbe {
    gridSize: number;
    sampleCount: number;
    radius: number;
    axisA: {
        parameter: LossLandscapeProbe['axisA']['parameter'];
        offsets: number[];
    };
    axisB: {
        parameter: LossLandscapeProbe['axisB']['parameter'];
        offsets: number[];
    };
    losses: number[];
    centerLoss: number;
    minLoss: number;
    maxLoss: number;
    best: LossLandscapeProbe['best'];
    summary: string;
}

export interface LossLandscapeProbeResponse {
    runId: number;
    step: number;
    epoch: number;
    probe: SerializableLossLandscapeProbe;
}

// Helper: reset back-pressure state. Called when the consumer on the other
// side is being torn down (stop, rebuild) so that a new run starts fresh
// instead of waiting for an ack that will never arrive.
function resetAck(): void {
    state.awaitingAck = false;
}

// Post the shared-buffers handshake. Called after SAB alloc (shape change)
// and after setStreamPort (so a late-arriving port still receives the
// handshake for buffers allocated earlier). No-op when SAB isn't active.
function postSharedBuffersHandshake(): void {
    if (!state.sharedViews || !state.streamPort || !state.network) return;
    const views = state.sharedViews;
    const handshake: WorkerSharedBuffersMessage = {
        type: 'sharedBuffers',
        runId: state.runId,
        control: views.controlSAB,
        outputGrid: views.outputGridSAB,
        neuronGrids: views.neuronGridsSAB,
        gridSize: views.gridSize,
        neuronGridLayout: {
            count: views.neuronCount,
            gridSize: views.gridSize,
        },
    };
    state.streamPort.postMessage(handshake);
}


// Post an error message to the main thread via the stream port. If the port
// isn't yet set (e.g. module-eval failure before setStreamPort), fall back to
// console so the error at least appears in devtools.
function postError(message: string): void {
    const errMsg: WorkerErrorMessage = {
        type: 'error',
        runId: state.runId,
        message,
    };
    if (state.streamPort) {
        state.streamPort.postMessage(errMsg);
    } else {
        console.error('[worker]', message);
    }
}

function postStatus(status: WorkerStatusMessage['status'], pauseReason?: PauseReason | null): void {
    if (!state.streamPort) return;
    const statusMsg: WorkerStatusMessage = {
        type: 'status',
        runId: state.runId,
        status,
    };
    if (pauseReason !== undefined) {
        statusMsg.pauseReason = pauseReason;
    }
    state.streamPort.postMessage(statusMsg);
}

/** Validate config compatibility — throws on incompatible loss/activation. */
function validateConfigs(network: NetworkConfig, training: TrainingConfig): void {
    if (!isLossCompatible(training.lossType, network.outputActivation)) {
        throw new Error(describeLossIncompatibility(training.lossType, network.outputActivation));
    }
}

const WORKER_MULTICLASS_OUTPUT_SIZE = 3;

function hasWorkerMulticlassContract(network: NetworkConfig, training: TrainingConfig): boolean {
    return (
        network.outputSize !== 1 ||
        network.outputActivation === 'softmax' ||
        training.lossType === 'categoricalCrossEntropy'
    );
}

function isApprovedWorkerMulticlassConfig(
    network: NetworkConfig,
    training: TrainingConfig,
    data: DataConfig,
): boolean {
    return (
        data.dataset === 'three-class-clusters' &&
        data.problemType === 'classification' &&
        network.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE &&
        network.outputActivation === 'softmax' &&
        training.lossType === 'categoricalCrossEntropy'
    );
}

function assertApprovedWorkerMulticlassConfig(
    network: NetworkConfig,
    training: TrainingConfig,
    data: DataConfig,
): void {
    if (!hasWorkerMulticlassContract(network, training)) return;
    if (isApprovedWorkerMulticlassConfig(network, training, data)) return;

    throw new Error(
        'Worker multiclass mode requires the approved three-class dataset, classification data, output size 3, softmax output activation, and categorical cross-entropy loss.',
    );
}

function assertScalarLiveArenaConfig(network: NetworkConfig, training: TrainingConfig): void {
    if (!hasWorkerMulticlassContract(network, training)) return;
    throw new Error('Multiclass live arena is not enabled; live arena is currently scalar-only.');
}

function normalizeWorkerConfig(
    networkConfig: NetworkConfig,
    trainingConfig: TrainingConfig,
    dataConfig: DataConfig,
    features: FeatureFlags,
): {
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
} {
    const isWorkerMulticlassRequest = hasWorkerMulticlassContract(networkConfig, trainingConfig);
    const result = normalizeAppConfig({
        network: networkConfig,
        training: trainingConfig,
        data: dataConfig,
        features,
        ui: { showTestData: false, discretizeOutput: false },
    }, { allowMulticlass: isWorkerMulticlassRequest });

    if (!result.config) {
        throw new Error(result.error ?? 'Invalid playground configuration.');
    }

    if (isWorkerMulticlassRequest) {
        assertApprovedWorkerMulticlassConfig(result.config.network, result.config.training, result.config.data);
    }

    const normalizedNetwork: NetworkConfig = result.config.network;
    const normalizedTraining: TrainingConfig = result.config.training;

    validateConfigs(normalizedNetwork, normalizedTraining);
    return {
        network: normalizedNetwork,
        training: normalizedTraining,
        data: result.config.data,
        features: result.config.features,
    };
}

// Structural equality has moved to @nn-playground/shared (`structuralEqual`).
// Keeping a local alias avoids touching every call site below.
const configsEqual = structuralEqual;

const WORKER_PERF_ENABLED = import.meta.env.DEV && import.meta.env.VITE_WORKER_PERF === '1';
const ACTIVATION_HISTOGRAM_BIN_COUNT = 12;
const ACTIVATION_HISTOGRAM_MAX_SAMPLES = 128;
const CHECKPOINT_MAX_COUNT = 8;
const CHECKPOINT_STEP_INTERVAL = 5;

function encodeTargetLabel(label: number, outputSize: number): number[] {
    if (outputSize === 1) return [label];
    if (!Number.isFinite(label) || !Number.isInteger(label) || label < 0 || label >= outputSize) {
        throw new RangeError(`Multiclass class index must be an integer between 0 and ${outputSize - 1}.`);
    }
    const target = Array.from({ length: outputSize }, () => 0);
    target[label] = 1;
    return target;
}

function encodeTargets(points: DataPoint[], outputSize: number): number[][] {
    return points.map((point) => encodeTargetLabel(point.label, outputSize));
}

function workerPerfMark(name: string): void {
    if (WORKER_PERF_ENABLED) performance.mark(name);
}

function workerPerfMeasure(name: string, startMark: string): void {
    if (!WORKER_PERF_ENABLED) return;
    performance.measure(name, startMark);
    performance.clearMarks(startMark);
    performance.clearMeasures(name);
}

const state: WorkerState = {
    prepared: null,
    compiled: null,
    datasetRevision: null,
    evaluationRuntime: null,
    metricHistory: null,
    v2EpochRef: null,
    network: null,
    networkConfig: null,
    trainingConfig: null,
    dataConfig: null,
    features: null,
    activeFeatures: [],
    trainInputs: [],
    trainTargets: [],
    testInputs: [],
    testTargets: [],
    trainPoints: [],
    testPoints: [],
    gridInputs: [],
    epoch: 0,
    running: false,
    rafId: null,
    shuffledIndices: [],
    batchScratch: null,
    shufflePrng: null,
    demand: { ...DEFAULT_DEMAND },
    snapshotsSinceLastTestEval: 0,
    snapshotsSinceLastTrainEval: 0,
    snapshotsSinceLastGrid: 0,
    snapshotsSinceLastActivationHistogram: 0,
    lastTestMetrics: null,
    lastTrainMetrics: null,
    lossEma: null,
    lossEmaAlpha: 0.1,
    testMetricsStale: false,
    checkpoints: [],
    nextCheckpointId: 1,
    checkpointEvictedCount: 0,
    restoredCheckpointId: null,
    stopConditionState: createInitialStopConditionState(),
    confusionMatrixVersion: 0,
    multiclassConfusionMatrixVersion: 0,
    lastPackedConfusionEvaluationId: null,
    activationHistogramVersion: 0,
    layerStatsGradientRevision: null,
    multiclassBoundaryVersion: 0,
    outputGridBuffer: null,
    neuronGridsBuffer: null,
    multiclassClassGridBuffer: null,
    multiclassConfidenceGridBuffer: null,
    multiclassBoundaryFresh: false,
    gridStale: true,
    sharedViews: null,
    gpuPredictor: null,
    gpuEnabled: false,
    gridInputsFlat: null,
    gridFreshFromGpu: false,
    streamPort: null,
    runId: 0,
    snapshotId: 0,
    stepsPerFrame: 5,
    trainLoopTimer: null,
    awaitingAck: false,
};

const arenaState: {
    slots: [ArenaSlot, ArenaSlot] | null;
    runId: number;
    snapshotId: number;
} = {
    slots: null,
    runId: 0,
    snapshotId: 0,
};

function cloneMetrics<T extends { loss: number; accuracy?: number; confusionMatrix?: { tp: number; tn: number; fp: number; fn: number } } | null>(metrics: T): T {
    if (metrics === null) return null as T;
    return {
        ...metrics,
        confusionMatrix: metrics.confusionMatrix
            ? { ...metrics.confusionMatrix }
            : undefined,
    } as T;
}

function buildCheckpointTimeline(): CheckpointTimeline {
    return {
        checkpoints: state.checkpoints.map((entry) => ({ ...entry.summary })),
        maxCheckpoints: CHECKPOINT_MAX_COUNT,
        evictedCount: state.checkpointEvictedCount,
        liveCheckpointId: state.checkpoints.at(-1)?.summary.id ?? null,
        restoredCheckpointId: state.restoredCheckpointId,
    };
}

function captureCheckpointFromSnapshot(snap: NetworkSnapshot): void {
    if (!state.network) return;
    const latest = state.checkpoints.at(-1);
    if (latest && latest.summary.step === snap.step) return;
    if (latest && snap.step - latest.summary.step < CHECKPOINT_STEP_INTERVAL) return;

    const id = state.nextCheckpointId++;
    const summary: CheckpointSummary = {
        id,
        step: snap.step,
        epoch: snap.epoch,
        trainLoss: snap.trainLoss,
        testLoss: snap.testLoss,
        trainAccuracy: snap.trainMetrics.accuracy,
        testAccuracy: snap.testMetrics.accuracy,
        label: `Step ${snap.step}`,
    };

    state.checkpoints.push({
        summary,
        checkpoint: state.network.createCheckpoint(),
        epoch: state.epoch,
        shuffledIndices: [...state.shuffledIndices],
        lossEma: state.lossEma,
        lastTrainMetrics: cloneMetrics(state.lastTrainMetrics),
        lastTestMetrics: cloneMetrics(state.lastTestMetrics),
        snapshotsSinceLastTrainEval: state.snapshotsSinceLastTrainEval,
        snapshotsSinceLastTestEval: state.snapshotsSinceLastTestEval,
        snapshotsSinceLastGrid: state.snapshotsSinceLastGrid,
        snapshotsSinceLastActivationHistogram: state.snapshotsSinceLastActivationHistogram,
    });

    while (state.checkpoints.length > CHECKPOINT_MAX_COUNT) {
        const evicted = state.checkpoints.shift();
        state.checkpointEvictedCount++;
        if (evicted && state.restoredCheckpointId === evicted.summary.id) {
            state.restoredCheckpointId = null;
        }
    }
}

function resetCheckpoints(): void {
    state.checkpoints = [];
    state.nextCheckpointId = 1;
    state.checkpointEvictedCount = 0;
    state.restoredCheckpointId = null;
}

// ── Scientific-trust V2 runtime ────────────────────────────────────────────

export interface WorkerExperimentResultV2 {
    readonly snapshot: NetworkSnapshot;
    readonly runId: number;
    readonly evidence: WorkerEvidenceMessageV2;
    readonly identities: PreparedExperimentDocumentV2['identities'];
    readonly artifacts?: WorkerArtifactProvenanceV2;
    readonly layerStatsGradientRevision?: number;
}

interface V2RuntimeBuild {
    readonly prepared: PreparedExperimentDocumentV2;
    readonly compiled: CompiledExperimentConfig;
    readonly projection: ReturnType<typeof projectPreparedExperiment>;
    readonly network: Network;
    readonly datasetRevision: DatasetRevision;
    readonly runtime: EvaluationRuntime;
    readonly history: RuntimeMetricHistory;
    readonly epochRef: { value: number };
    readonly initialEvidence: WorkerEvidenceMessageV2;
    readonly activeFeatures: FeatureSpec[];
    readonly trainInputs: number[][];
    readonly trainTargets: number[][];
    readonly testInputs: number[][];
    readonly testTargets: number[][];
    readonly trainPoints: DataPoint[];
    readonly testPoints: DataPoint[];
    readonly gridInputs: number[][];
    readonly gridInputsFlat: Float32Array;
    readonly shuffledIndices: number[];
    readonly shufflePrng: PRNG;
    readonly batchScratch: MiniBatchScratch;
    readonly outputGridBuffer: Float32Array;
    readonly neuronGridsBuffer: Float32Array;
    readonly multiclassClassGridBuffer: Uint8Array | null;
    readonly multiclassConfidenceGridBuffer: Float32Array | null;
    readonly sharedViews: SharedSnapshotViews | null;
}

let experimentRequestGate = new ExperimentRequestGate();
let v2MutationSequence = 0;
let v2MutationTail: Promise<void> = Promise.resolve();
let v2AllocationCount = 0;
let runtimeStopConditions: readonly StopCondition[] = DEFAULT_RUNTIME_STOP_CONDITIONS;
let detectWebGPUForRuntime: typeof detectWebGPU = detectWebGPU;
let createWebGPUGridPredictorForRuntime = (
    args: ConstructorParameters<typeof WebGPUGridPredictor>[0],
): WebGPUGridPredictor => new WebGPUGridPredictor(args);
export const MAX_MANUAL_V2_STEP_ITERATIONS = 10;

function enqueueV2Mutation<T>(operation: () => T | Promise<T>): Promise<T> {
    const result = v2MutationTail.then(operation, operation);
    v2MutationTail = result.then(() => undefined, () => undefined);
    return result;
}

/** Narrow diagnostic seam: forged/stale-boundary tests assert allocation never starts. */
export function getV2AllocationCountForTests(): number {
    return v2AllocationCount;
}

/** Test seam for exercising async preparation/action races. */
export function setV2PrepareForTests(
    prepare?: ExperimentRequestGateDependencies['prepare'],
): void {
    experimentRequestGate = new ExperimentRequestGate(
        prepare === undefined ? {} : { prepare },
    );
}

/** Narrow seam for proving comparison stop-condition behavior in the worker loop. */
export function setRuntimeStopConditionsForTests(
    conditions?: readonly StopCondition[],
): void {
    runtimeStopConditions = conditions === undefined
        ? DEFAULT_RUNTIME_STOP_CONDITIONS
        : [...conditions];
    state.stopConditionState = createInitialStopConditionState();
}

/** Narrow seam for deterministic async WebGPU readback race tests. */
export function setGpuPredictorForTests(
    predictor: Pick<
        WebGPUGridPredictor,
        | 'updateWeights'
        | 'predictGridInto'
        | 'predictGridWithNeuronsInto'
        | 'dispose'
    > | null,
): void {
    state.gpuPredictor = predictor as WebGPUGridPredictor | null;
}

/** Narrow seam for deterministic device-detection/toggle race tests. */
export function setGpuInitializationForTests(
    overrides?: {
        readonly detect?: typeof detectWebGPU;
        readonly create?: (
            args: ConstructorParameters<typeof WebGPUGridPredictor>[0],
        ) => WebGPUGridPredictor;
    },
): void {
    detectWebGPUForRuntime = overrides?.detect ?? detectWebGPU;
    createWebGPUGridPredictorForRuntime = overrides?.create
        ?? ((args) => new WebGPUGridPredictor(args));
}

/** Narrow seam for proving real engine overflow translation at worker boundaries. */
export function setV2OutputOverflowForTests(): void {
    const { network, compiled } = requireV2Runtime();
    const outputLayer = compiled.network.hiddenLayers.length;
    const fanIn = compiled.network.hiddenLayers.at(-1) ?? compiled.network.inputSize;
    for (let input = 0; input < fanIn; input++) {
        network.setWeight(outputLayer, 0, input, Number.MAX_VALUE);
    }
    network.setBias(outputLayer, 0, Number.MAX_VALUE);
}

function beginV2Mutation(): number {
    if (v2MutationSequence >= Number.MAX_SAFE_INTEGER) {
        throw new RangeError('V2 mutation sequence exhausted its safe integer range');
    }
    v2MutationSequence++;
    return v2MutationSequence;
}

function assertLegacyModelMutationAllowed(apiName: string): void {
    if (state.prepared !== null) {
        throw new Error(
            `Legacy ${apiName} is unavailable while a V2 experiment is active; use the V2 worker API.`,
        );
    }
}

function requireV2Runtime(): {
    prepared: PreparedExperimentDocumentV2;
    compiled: CompiledExperimentConfig;
    network: Network;
    runtime: EvaluationRuntime;
    history: RuntimeMetricHistory;
    epochRef: { value: number };
} {
    if (!state.prepared
        || !state.compiled
        || !state.network
        || !state.evaluationRuntime
        || !state.metricHistory
        || !state.v2EpochRef) {
        throw new Error('V2 experiment is not initialized');
    }
    return {
        prepared: state.prepared,
        compiled: state.compiled,
        network: state.network,
        runtime: state.evaluationRuntime,
        history: state.metricHistory,
        epochRef: state.v2EpochRef,
    };
}

function makeEvidenceV2(
    liveSignal?: LiveTrainingSignal,
    latestEvaluation?: PairedEvaluation,
): WorkerEvidenceMessageV2 {
    return parseWorkerEvidenceMessageV2({
        type: 'evidence',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        ...(liveSignal === undefined ? {} : { liveSignal }),
        ...(latestEvaluation === undefined ? {} : { latestEvaluation }),
    });
}

function postEvidenceV2(evidence: WorkerEvidenceMessageV2): void {
    try {
        state.streamPort?.postMessage(evidence);
    } catch (error) {
        postRuntimeErrorV2(error, 'runtime');
    }
}

function protocolErrorText(error: unknown, fallback: string): string {
    const text = error instanceof Error && error.message.length > 0
        ? error.message
        : fallback;
    return text.slice(0, 4_096);
}

function deliverProtocolErrorV2(message: ReturnType<typeof parseWorkerProtocolErrorMessageV2>): void {
    try {
        if (state.streamPort) state.streamPort.postMessage(message);
        else console.error('[worker:v2]', message.message);
    } catch {
        // Preserve the original worker failure when the transport itself is
        // broken; the caller still rejects through Comlink.
        console.error('[worker:v2]', message.message);
    }
}

function postTransactionErrorV2(error: unknown): void {
    const transaction = error instanceof ExperimentTransactionError ? error : null;
    const message = parseWorkerProtocolErrorMessageV2({
        type: 'worker-error',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: transaction?.requestId ?? null,
        // Initialization failures never belong to the still-current model
        // generation; attaching that ID would misattribute rejected config.
        generationId: null,
        code: transaction?.code ?? 'runtime-failure',
        path: transaction?.path ?? '$',
        message: protocolErrorText(error, 'Worker experiment transaction failed.'),
        source: transaction?.code === 'malformed-request'
            ? 'protocol'
            : transaction === null
                ? 'runtime'
                : 'preparation',
    });
    deliverProtocolErrorV2(message);
}

function postRuntimeErrorV2(
    error: unknown,
    source: WorkerProtocolErrorSourceV2,
    code: WorkerProtocolErrorCodeV2 = 'runtime-failure',
): void {
    const message = parseWorkerProtocolErrorMessageV2({
        type: 'worker-error',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: null,
        generationId: state.prepared && state.runId > 0 ? state.runId : null,
        code,
        path: error instanceof TerminalDivergenceError ? error.path : '$',
        message: protocolErrorText(error, 'Worker V2 runtime failure.'),
        source,
    });
    deliverProtocolErrorV2(message);
}

function postTerminalDivergenceV2(
    error: TerminalDivergenceError,
    source: WorkerProtocolErrorSourceV2,
    code: WorkerProtocolErrorCodeV2 = 'runtime-failure',
): void {
    stopInternalLoop();
    postRuntimeErrorV2(error, source, code);
    postStatus('paused', 'diverged');
}

function postTerminalDivergenceForPhase(
    error: TerminalDivergenceError,
    fallback: 'training' | 'evaluation',
): void {
    if (error.path.startsWith('$.snapshot') || error.path.startsWith('$.gpu')) {
        postTerminalDivergenceV2(error, 'artifact', 'artifact-failed');
        return;
    }
    postTerminalDivergenceV2(
        error,
        fallback,
        fallback === 'evaluation' ? 'evaluation-failed' : 'runtime-failure',
    );
}

function withEngineNumericalBoundary<T>(pathPrefix: string, operation: () => T): T {
    try {
        return operation();
    } catch (error) {
        if (error instanceof NonFiniteNumericalError) {
            const suffix = error.path.startsWith('[') ? error.path : `.${error.path}`;
            throw new TerminalDivergenceError(`${pathPrefix}${suffix}`, error.value);
        }
        throw error;
    }
}

function argmax(values: readonly number[]): number {
    let bestIndex = 0;
    for (let index = 1; index < values.length; index++) {
        if (values[index] > values[bestIndex]) bestIndex = index;
    }
    return bestIndex;
}

/** Task metrics only; predictive data loss is supplied by the compiled objective. */
function withTaskMetricsV2(
    network: Network,
    inputs: number[][],
    targets: number[][],
    task: CompiledTaskContract,
    dataLoss: number,
): EvaluationValues {
    if (task.kind === 'regression') return { dataLoss };

    if (task.kind === 'binary-classification') {
        let tp = 0;
        let tn = 0;
        let fp = 0;
        let fn = 0;
        for (let index = 0; index < inputs.length; index++) {
            const predicted = network.predict(inputs[index])[0] >= 0.5 ? 1 : 0;
            const actual = targets[index][0] >= 0.5 ? 1 : 0;
            if (predicted === 1 && actual === 1) tp++;
            else if (predicted === 0 && actual === 0) tn++;
            else if (predicted === 1) fp++;
            else fn++;
        }
        return {
            dataLoss,
            accuracy: (tp + tn) / inputs.length,
            confusionMatrix: { tp, tn, fp, fn },
        };
    }

    const counts: [number, number, number, number, number, number, number, number, number] = [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ];
    let correct = 0;
    for (let index = 0; index < inputs.length; index++) {
        const predicted = argmax(network.predict(inputs[index]));
        const actual = argmax(targets[index]);
        counts[actual * 3 + predicted]++;
        if (predicted === actual) correct++;
    }
    const confusionMatrix: MulticlassConfusionMatrixData = {
        classCount: 3,
        classLabels: [0, 1, 2],
        counts,
    };
    return {
        dataLoss,
        accuracy: correct / inputs.length,
        confusionMatrix,
    };
}

function stageV2Runtime(
    prepared: PreparedExperimentDocumentV2,
    generationId: number,
): V2RuntimeBuild {
    // This counter is intentionally inside the post-validation commit path.
    v2AllocationCount++;
    const compiled = prepared.compiled;
    const split = generateDatasetV2({ ...compiled.data });
    const projection = projectPreparedExperiment(prepared);
    const activeFeatures = getActiveFeatures(compiled.features);
    const trainInputs = transformDataset(split.train, activeFeatures);
    const trainTargets = encodeTargets(split.train, compiled.task.outputSize);
    const testInputs = transformDataset(split.test, activeFeatures);
    const testTargets = encodeTargets(split.test, compiled.task.outputSize);
    const gridInputs = buildGridInputs(GRID_SIZE, activeFeatures);
    const network = new Network(compiled.network, compiled.network.seed);
    const shuffledIndices = Array.from(
        { length: trainInputs.length },
        (_, index) => index,
    );
    const shufflePrng = new PRNG((compiled.data.seed + 1234) >>> 0);
    if (shuffledIndices.length > 0) shufflePrng.shuffle(shuffledIndices);
    const epochRef = { value: 0 };
    const datasetRevision: DatasetRevision = {
        generatorVersion: getDatasetContract(compiled.data.dataset).generatorVersion,
        datasetKey: prepared.identities.datasetKey,
        trainCount: trainInputs.length,
        testCount: testInputs.length,
    };

    let cachedTrainObjective: {
        generationId: number;
        revision: number;
        regularizationPenalty: number;
    } | null = null;
    const runtime = new EvaluationRuntime({
        generationId,
        dataset: datasetRevision,
        objectiveKey: prepared.identities.objectiveKey,
        emaAlpha: state.lossEmaAlpha,
        getCurrentModel: () => ({
            generationId,
            revision: network.getRevision(),
            step: network.getStep(),
            epoch: epochRef.value,
        }),
        evaluateTrain: (model) => withEngineNumericalBoundary('$.evaluation.train', () => {
            const objective = network.evaluateObjective(
                trainInputs,
                trainTargets,
                compiled.objective,
            );
            cachedTrainObjective = {
                generationId: model.generationId,
                revision: model.revision,
                regularizationPenalty: objective.regularizationPenalty,
            };
            return withTaskMetricsV2(
                network,
                trainInputs,
                trainTargets,
                compiled.task,
                objective.dataLoss,
            );
        }),
        evaluateTest: () => withEngineNumericalBoundary('$.evaluation.test', () => (
            withTaskMetricsV2(
                network,
                testInputs,
                testTargets,
                compiled.task,
                network.evaluateDataLoss(testInputs, testTargets, compiled.objective),
            )
        )),
        evaluateRegularizationPenalty: (model) => {
            if (cachedTrainObjective === null
                || cachedTrainObjective.generationId !== model.generationId
                || cachedTrainObjective.revision !== model.revision) {
                throw new Error('train objective penalty is not cached for this model revision');
            }
            const penalty = cachedTrainObjective.regularizationPenalty;
            cachedTrainObjective = null;
            return penalty;
        },
    });
    const history = new RuntimeMetricHistory({
        generationId,
        dataset: datasetRevision,
        objectiveKey: prepared.identities.objectiveKey,
    });
    const initialEvaluation = runtime.forceEvaluation('initial');
    history.appendEvaluation(initialEvaluation);
    const initialEvidence = makeEvidenceV2(undefined, initialEvaluation);

    const totalNeurons = network.getTotalNeuronCount();
    let sharedViews: SharedSnapshotViews | null = null;
    if (canUseSharedBuffers()) {
        try {
            sharedViews = allocSharedSnapshotViews(GRID_SIZE, totalNeurons);
        } catch (error) {
            console.warn('[worker] shared-snapshot alloc failed, falling back', error);
        }
    }

    return {
        prepared,
        compiled,
        projection,
        network,
        datasetRevision,
        runtime,
        history,
        epochRef,
        initialEvidence,
        activeFeatures,
        trainInputs,
        trainTargets,
        testInputs,
        testTargets,
        trainPoints: split.train,
        testPoints: split.test,
        gridInputs,
        gridInputsFlat: flattenGridInputs(gridInputs),
        shuffledIndices,
        shufflePrng,
        batchScratch: createMiniBatchScratch(compiled.training.batchSize),
        outputGridBuffer: new Float32Array(GRID_SIZE * GRID_SIZE),
        neuronGridsBuffer: new Float32Array(totalNeurons * GRID_SIZE * GRID_SIZE),
        multiclassClassGridBuffer: compiled.task.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
            ? new Uint8Array(GRID_SIZE * GRID_SIZE)
            : null,
        multiclassConfidenceGridBuffer: compiled.task.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
            ? new Float32Array(GRID_SIZE * GRID_SIZE)
            : null,
        sharedViews,
    };
}

function installStagedV2State(staged: V2RuntimeBuild, generationId: number): void {
    state.gpuPredictor = null;
    state.prepared = staged.prepared;
    state.compiled = staged.compiled;
    state.datasetRevision = staged.datasetRevision;
    state.evaluationRuntime = staged.runtime;
    state.metricHistory = staged.history;
    state.v2EpochRef = staged.epochRef;
    state.network = staged.network;
    state.networkConfig = { ...staged.projection.network };
    state.trainingConfig = { ...staged.projection.training };
    state.dataConfig = { ...staged.projection.data };
    state.features = { ...staged.projection.features };
    state.activeFeatures = staged.activeFeatures;
    state.trainInputs = staged.trainInputs;
    state.trainTargets = staged.trainTargets;
    state.testInputs = staged.testInputs;
    state.testTargets = staged.testTargets;
    state.trainPoints = staged.trainPoints;
    state.testPoints = staged.testPoints;
    state.gridInputs = staged.gridInputs;
    state.gridInputsFlat = staged.gridInputsFlat;
    state.shuffledIndices = staged.shuffledIndices;
    state.shufflePrng = staged.shufflePrng;
    state.batchScratch = staged.batchScratch;
    state.outputGridBuffer = staged.outputGridBuffer;
    state.neuronGridsBuffer = staged.neuronGridsBuffer;
    state.multiclassClassGridBuffer = staged.multiclassClassGridBuffer;
    state.multiclassConfidenceGridBuffer = staged.multiclassConfidenceGridBuffer;
    state.sharedViews = staged.sharedViews;
    state.epoch = 0;
    state.running = false;
    state.trainLoopTimer = null;
    state.runId = generationId;
    state.snapshotId = 0;
    state.snapshotsSinceLastTestEval = 0;
    state.snapshotsSinceLastTrainEval = 0;
    // A fresh strict generation must return its initially demanded boundary
    // artifacts immediately; subsequent snapshots resume normal cadence.
    state.snapshotsSinceLastGrid = state.demand.gridInterval;
    state.snapshotsSinceLastActivationHistogram = state.demand.needActivationHistograms
        ? state.demand.activationHistogramInterval
        : 0;
    state.lastTestMetrics = null;
    state.lastTrainMetrics = null;
    state.lossEma = null;
    state.testMetricsStale = false;
    state.multiclassBoundaryFresh = false;
    state.gridStale = true;
    state.gridFreshFromGpu = false;
    state.stopConditionState = createInitialStopConditionState();
    state.confusionMatrixVersion++;
    state.lastPackedConfusionEvaluationId = null;
    state.activationHistogramVersion++;
    state.layerStatsGradientRevision = null;
    resetCheckpoints();
    resetAck();
}

function commitV2Runtime(prepared: PreparedExperimentDocumentV2): WorkerExperimentResultV2 {
    const generationId = state.runId + 1;
    const staged = stageV2Runtime(prepared, generationId);

    // Build the compatibility snapshot against staged state before making the
    // generation observable. JavaScript cannot interleave another worker turn
    // during this synchronous section, and every prior field is restored on
    // failure, so rejected initialization leaves the active run untouched.
    const previousState: WorkerState = { ...state };
    let committedState: WorkerState;
    let snapshot: NetworkSnapshot;
    try {
        installStagedV2State(staged, generationId);
        snapshot = computeSnapshot();
        committedState = { ...state };
    } catch (error) {
        Object.assign(state, previousState);
        throw error;
    }

    Object.assign(state, previousState);
    stopInternalLoop();
    if (state.gpuPredictor) {
        try { state.gpuPredictor.dispose(); } catch { /* best-effort cleanup */ }
    }
    Object.assign(state, committedState);
    try {
        postSharedBuffersHandshake();
        postEvidenceV2(staged.initialEvidence);
    } catch (error) {
        // The committed generation is still available through this RPC. A
        // broken stream must not turn a successful atomic commit into a
        // rejected request with installed state.
        try { postRuntimeErrorV2(error, 'runtime'); } catch { /* stream unavailable */ }
    }

    return {
        snapshot,
        runId: generationId,
        evidence: staged.initialEvidence,
        identities: prepared.identities,
        ...directSnapshotArtifactBundle(snapshot),
    };
}

function buildArenaSlot(side: ArenaSide, input: ArenaModelInput): ArenaSlot {
    assertScalarLiveArenaConfig(input.network, input.training);
    const config = normalizeWorkerConfig(
        input.network,
        input.training,
        input.data,
        input.features,
    );
    const activeFeatures = getActiveFeatures(config.features);
    const split = generateDataset(
        config.data.dataset,
        config.data.numSamples,
        config.data.noise,
        config.data.trainTestRatio,
        config.data.seed,
    );
    const networkConfig = {
        ...config.network,
        inputSize: countActiveFeatures(config.features),
    };
    if (networkConfig.outputSize !== 1) {
        throw new Error('Multiclass live arena is not enabled; live arena is currently scalar-only.');
    }
    const network = new Network(networkConfig, networkConfig.seed);
    const trainInputs = transformDataset(split.train, activeFeatures);
    const trainTargets = encodeTargets(split.train, networkConfig.outputSize);
    const testInputs = transformDataset(split.test, activeFeatures);
    const testTargets = encodeTargets(split.test, networkConfig.outputSize);
    const shuffledIndices = Array.from({ length: trainInputs.length }, (_, index) => index);
    const shufflePrng = new PRNG((config.data.seed ?? 42) + (side === 'A' ? 2234 : 3234));
    if (shuffledIndices.length > 0) {
        shufflePrng.shuffle(shuffledIndices);
    }

    return {
        side,
        label: input.label ?? `Model ${side}`,
        network,
        trainingConfig: config.training,
        dataConfig: config.data,
        trainInputs,
        trainTargets,
        testInputs,
        testTargets,
        epoch: 0,
        shuffledIndices,
        shufflePrng,
        status: 'idle',
        pauseReason: null,
    };
}

function trainArenaSlot(slot: ArenaSlot): void {
    const n = slot.trainInputs.length;
    if (n === 0) return;

    const batchSize = slot.trainingConfig.batchSize;
    const stepBefore = slot.network.getStep();
    const numBatches = Math.ceil(n / batchSize);
    const batchSlot = stepBefore % numBatches;

    if (batchSlot === 0 && stepBefore > 0) {
        slot.shufflePrng.shuffle(slot.shuffledIndices);
    }

    const startIdx = batchSlot * batchSize;
    const endIdx = Math.min(startIdx + batchSize, n);
    if (startIdx === endIdx) return;

    const batchLoss = slot.network.trainBatchIndexed(
        slot.trainInputs,
        slot.trainTargets,
        slot.shuffledIndices,
        startIdx,
        endIdx,
        slot.trainingConfig,
    );
    if (!Number.isFinite(batchLoss)) {
        slot.status = 'paused';
        slot.pauseReason = 'diverged';
    }
    if (batchSlot === numBatches - 1) {
        slot.epoch++;
    }
}

function buildArenaSummary(slot: ArenaSlot): ArenaModelSummary {
    const lossType = slot.trainingConfig.lossType;
    const problemType = slot.dataConfig.problemType;
    const huberDelta = slot.trainingConfig.huberDelta;
    const trainMetrics = slot.network.evaluate(
        slot.trainInputs,
        slot.trainTargets,
        lossType,
        problemType,
        huberDelta,
    );
    const testMetrics = slot.network.evaluate(
        slot.testInputs,
        slot.testTargets,
        lossType,
        problemType,
        huberDelta,
    );

    return {
        side: slot.side,
        label: slot.label,
        status: slot.status,
        pauseReason: slot.pauseReason,
        step: slot.network.getStep(),
        epoch: slot.epoch,
        trainLoss: trainMetrics.loss,
        testLoss: testMetrics.loss,
        trainAccuracy: trainMetrics.accuracy,
        testAccuracy: testMetrics.accuracy,
    };
}

function buildArenaSnapshot(): ArenaScalarSnapshot {
    if (!arenaState.slots) {
        throw new Error('Arena is not initialized');
    }
    return {
        runId: arenaState.runId,
        snapshotId: arenaState.snapshotId++,
        summaries: arenaState.slots.map(buildArenaSummary),
    };
}

// Top-level error backstops — catch anything not handled by the per-function
// try/catch blocks (e.g. errors thrown during module evaluation or in callbacks
// we don't own). Both routes through postError so the main thread always sees
// the failure.
self.addEventListener('error', (event: ErrorEvent) => {
    postError(`Unhandled worker error: ${event.message ?? String(event)}`);
});

self.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    postError(`Unhandled worker rejection: ${reason}`);
});



function buildDataAndNetwork(): void {
    if (!state.dataConfig || !state.features || !state.networkConfig || !state.trainingConfig) return;

    // Entering through a legacy API intentionally leaves the strict V2 path.
    state.prepared = null;
    state.compiled = null;
    state.datasetRevision = null;
    state.evaluationRuntime = null;
    state.metricHistory = null;
    state.v2EpochRef = null;

    state.activeFeatures = getActiveFeatures(state.features);
    const inputSize = countActiveFeatures(state.features);

    // Generate data
    const split = generateDataset(
        state.dataConfig.dataset,
        state.dataConfig.numSamples,
        state.dataConfig.noise,
        state.dataConfig.trainTestRatio,
        state.dataConfig.seed,
    );

    // Create network
    const config: NetworkConfig = {
        ...state.networkConfig,
        inputSize,
    };
    state.networkConfig = config;

    state.trainPoints = split.train;
    state.testPoints = split.test;
    state.trainInputs = transformDataset(split.train, state.activeFeatures);
    state.trainTargets = encodeTargets(split.train, config.outputSize);
    state.testInputs = transformDataset(split.test, state.activeFeatures);
    state.testTargets = encodeTargets(split.test, config.outputSize);

    // Build grid inputs. Multiclass snapshots intentionally skip the scalar
    // boundary buffers until a dedicated visualization slice is approved.
    state.gridInputs = buildGridInputs(GRID_SIZE, state.activeFeatures);

    state.network = new Network(config, config.seed);

    // Allocate buffers
    state.outputGridBuffer = new Float32Array(GRID_SIZE * GRID_SIZE);
    state.multiclassClassGridBuffer = config.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
        ? new Uint8Array(GRID_SIZE * GRID_SIZE)
        : null;
    state.multiclassConfidenceGridBuffer = config.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
        ? new Float32Array(GRID_SIZE * GRID_SIZE)
        : null;
    state.multiclassBoundaryFresh = false;
    const totalNeurons = state.network.getTotalNeuronCount();
    state.neuronGridsBuffer = new Float32Array(totalNeurons * GRID_SIZE * GRID_SIZE);
    state.epoch = 0;

    // (Re-)allocate shared-memory transport. Always allocate fresh on shape
    // change: SAB sizes must match the new neuron count, and trying to
    // resize in place risks readers on the main thread indexing past the
    // end with a stale count.
    state.sharedViews = null;
    if (canUseSharedBuffers()) {
        try {
            state.sharedViews = allocSharedSnapshotViews(GRID_SIZE, totalNeurons);
        } catch (err) {
            // Allocation can legitimately fail on memory-constrained hosts or
            // if the SAB constructor is disabled at runtime. Fall back to the
            // postMessage path silently; logs only to worker console.
            console.warn('[worker] shared-snapshot alloc failed, falling back', err);
            state.sharedViews = null;
        }
    }

    // Cache the flattened grid inputs for the GPU path. Cheap to materialise
    // at build time (run once per shape) and avoids per-snapshot rebuilds.
    state.gridInputsFlat = flattenGridInputs(state.gridInputs);

    // Dispose any prior GPU predictor — its bind groups and buffers were
    // sized to the old shape and cannot be reused. The async (re-)alloc
    // happens in `ensureGpuPredictor()`, which the snapshot path awaits.
    if (state.gpuPredictor) {
        try { state.gpuPredictor.dispose(); } catch { /* ignore */ }
        state.gpuPredictor = null;
    }

    // Reset cadence gating + cached metrics + EMA.
    state.snapshotsSinceLastTestEval = 0;
    state.snapshotsSinceLastTrainEval = 0;
    state.snapshotsSinceLastGrid = 0;
    state.snapshotsSinceLastActivationHistogram = state.demand.needActivationHistograms
        ? state.demand.activationHistogramInterval
        : 0;
    state.lastTestMetrics = null;
    state.lastTrainMetrics = null;
    state.lossEma = null;
    state.gridStale = true;
    state.gridFreshFromGpu = false;
    state.testMetricsStale = false;
    state.stopConditionState = createInitialStopConditionState();
    state.confusionMatrixVersion++;
    state.activationHistogramVersion++;
    state.layerStatsGradientRevision = null;
    resetCheckpoints();

    // Increment run ID
    state.runId++;
    state.snapshotId = 0;

    // New run — drop any stale back-pressure gate.
    resetAck();

    // Hand the newly-allocated SABs to the main thread (if the stream port
    // is already connected). When the port arrives later, setStreamPort
    // will re-emit this handshake.
    postSharedBuffersHandshake();

    // Initialise shuffle state — seed is offset from data seed to stay independent.
    const n = state.trainInputs.length;
    state.shuffledIndices = Array.from({ length: n }, (_, i) => i);
    state.batchScratch = createMiniBatchScratch(state.trainingConfig!.batchSize);
    state.shufflePrng = new PRNG((state.dataConfig!.seed ?? 42) + 1234);
    // Shuffle once up front so the very first epoch is not in generator order
    // (important for datasets whose generators emit class-sorted samples).
    if (n > 0) {
        state.shufflePrng.shuffle(state.shuffledIndices);
    }
}

// ── GPU grid prediction (AS-4) ─────────────────────────────────────────────
export const MAX_WEBGPU_NEURON_READBACK_BYTES = 256 * 1024;

export type WebGpuGridReadbackMode = 'none' | 'outputOnly' | 'withNeurons' | 'cpu';

export function estimateWebGpuNeuronReadbackBytes(neuronCount: number, gridLen: number): number {
    return Math.max(0, neuronCount) * Math.max(0, gridLen) * Float32Array.BYTES_PER_ELEMENT;
}

export function selectWebGpuGridReadbackMode(args: {
    needDecisionBoundary: boolean;
    needNeuronGrids: boolean;
    gridLen: number;
    neuronCount: number;
}): WebGpuGridReadbackMode {
    if (!args.needDecisionBoundary && !args.needNeuronGrids) return 'none';
    if (!args.needNeuronGrids) return 'outputOnly';

    const neuronReadbackBytes = estimateWebGpuNeuronReadbackBytes(args.neuronCount, args.gridLen);
    return neuronReadbackBytes <= MAX_WEBGPU_NEURON_READBACK_BYTES ? 'withNeurons' : 'cpu';
}

function gpuPredictorSignature(): string | null {
    if (!state.network || !state.networkConfig || !state.gridInputsFlat) return null;
    const layerSizes = [
        state.networkConfig.inputSize,
        ...state.networkConfig.hiddenLayers,
        state.networkConfig.outputSize,
    ];
    return [
        state.runId,
        layerSizes.join(','),
        state.gridInputs.length,
        state.networkConfig.activation,
        state.networkConfig.outputActivation,
    ].join('|');
}

// Lazily allocate a WebGPUGridPredictor matching the current network shape.
// The first call after a build (or after the user toggles GPU on) does the
// async device + pipeline init; subsequent calls return the cached instance.
// Returns null when WebGPU is unavailable, the user hasn't opted in, or the
// network shape exceeds the shader's compile-time caps. Callers always
// fall back to the CPU predictor.
async function ensureGpuPredictor(): Promise<WebGPUGridPredictor | null> {
    if (!state.gpuEnabled) return null;
    if (state.gpuPredictor) return state.gpuPredictor;
    if (!state.network || !state.networkConfig || !state.gridInputsFlat) return null;

    const runId = state.runId;
    const signature = gpuPredictorSignature();
    if (signature === null) return null;

    const layerSizes = [
        state.networkConfig.inputSize,
        ...state.networkConfig.hiddenLayers,
        state.networkConfig.outputSize,
    ];
    if (exceedsGpuShape(layerSizes)) return null;

    const device = await detectWebGPUForRuntime();
    if (!state.gpuEnabled
        || state.runId !== runId
        || gpuPredictorSignature() !== signature) return null;
    if (!device) return null;

    try {
        const predictor = createWebGPUGridPredictorForRuntime({
            device,
            layerSizes,
            gridLen: state.gridInputs.length,
            hiddenActivation: state.networkConfig.activation,
            outputActivation: state.networkConfig.outputActivation,
        });
        // Grid inputs are constant per shape — upload once and never again
        // until the next shape change disposes this predictor.
        predictor.setGridInputs(state.gridInputsFlat);
        if (!state.gpuEnabled
            || state.runId !== runId
            || gpuPredictorSignature() !== signature) {
            predictor.dispose();
            return null;
        }
        state.gpuPredictor = predictor;
        return predictor;
    } catch (err) {
        console.warn('[worker] GPU predictor init failed, falling back to CPU', err);
        state.gpuPredictor = null;
        return null;
    }
}

/**
 * If the demand cadence says it's time to recompute the grid AND a GPU
 * predictor is available, run the GPU prediction into the existing
 * pre-allocated grid buffers and set `gridFreshFromGpu` so the upcoming
 * `computeSnapshot` call skips its CPU branch.
 *
 * On any failure the function silently returns; `computeSnapshot`'s CPU
 * branch then runs as if the GPU path didn't exist. This is the right
 * behaviour for an "accelerator" — never block the user when it breaks.
 */
async function runGpuGridIfDue(): Promise<void> {
    if (!state.network) return;
    if (state.networkConfig?.outputSize !== 1) return;
    const { demand } = state;
    const wantGrid = demand.needDecisionBoundary || demand.needNeuronGrids;
    const due =
        wantGrid &&
        state.gridStale &&
        state.snapshotsSinceLastGrid >= demand.gridInterval;
    if (!due) return;
    if (!state.outputGridBuffer || !state.neuronGridsBuffer) return;

    const readbackMode = selectWebGpuGridReadbackMode({
        needDecisionBoundary: demand.needDecisionBoundary,
        needNeuronGrids: demand.needNeuronGrids,
        gridLen: state.gridInputs.length,
        neuronCount: state.network.getTotalNeuronCount(),
    });
    if (readbackMode === 'none' || readbackMode === 'cpu') return;

    const requestedNetwork = state.network;
    const requestedRunId = state.runId;
    const requestedRevision = requestedNetwork.getRevision();
    const requestedMutationSequence = v2MutationSequence;
    const requestedDemand = state.demand;
    const requestedGpuEnabled = state.gpuEnabled;
    const requestedOutputBuffer = state.outputGridBuffer;
    const requestedNeuronBuffer = state.neuronGridsBuffer;
    const requestIsCurrent = (): boolean => (
        state.network === requestedNetwork
        && state.runId === requestedRunId
        && state.network.getRevision() === requestedRevision
        && v2MutationSequence === requestedMutationSequence
        && state.demand === requestedDemand
        && state.gpuEnabled === requestedGpuEnabled
        && state.outputGridBuffer === requestedOutputBuffer
        && state.neuronGridsBuffer === requestedNeuronBuffer
    );
    const forceCpuFallback = (): void => {
        // A generation replacement owns different network and buffer objects.
        // Its readiness must not be disturbed by completion of the discarded
        // predecessor. A revision-only race reuses the captured objects and
        // therefore must overwrite the stale readback on the CPU.
        if (state.network !== requestedNetwork
            || state.outputGridBuffer !== requestedOutputBuffer
            || state.neuronGridsBuffer !== requestedNeuronBuffer) {
            return;
        }
        state.gridFreshFromGpu = false;
        state.gridStale = true;
        state.snapshotsSinceLastGrid = Math.max(
            state.snapshotsSinceLastGrid,
            state.demand.gridInterval,
        );
    };

    const predictor = await ensureGpuPredictor();
    if (!requestIsCurrent()) {
        forceCpuFallback();
        return;
    }
    if (!predictor) return;

    const readbackIdentity = {
        runId: requestedRunId,
        revision: requestedRevision,
        mutationSequence: requestedMutationSequence,
        demand: requestedDemand,
        gpuEnabled: requestedGpuEnabled,
        readbackMode,
        network: requestedNetwork,
        predictor,
        outputBuffer: requestedOutputBuffer,
        neuronBuffer: requestedNeuronBuffer,
    };
    const readbackIsCurrent = (): boolean => (
        state.runId === readbackIdentity.runId
        && state.network === readbackIdentity.network
        && state.network.getRevision() === readbackIdentity.revision
        && v2MutationSequence === readbackIdentity.mutationSequence
        && state.demand === readbackIdentity.demand
        && state.gpuEnabled === readbackIdentity.gpuEnabled
        && selectWebGpuGridReadbackMode({
            needDecisionBoundary: state.demand.needDecisionBoundary,
            needNeuronGrids: state.demand.needNeuronGrids,
            gridLen: state.gridInputs.length,
            neuronCount: state.network.getTotalNeuronCount(),
        }) === readbackIdentity.readbackMode
        && state.gpuPredictor === readbackIdentity.predictor
        && state.outputGridBuffer === readbackIdentity.outputBuffer
        && state.neuronGridsBuffer === readbackIdentity.neuronBuffer
    );

    try {
        // Push the latest weights to the GPU. Float32 representability is a
        // scientific boundary, so typed failures become terminal divergence.
        const flat = readbackIdentity.network.getWeightsFlat();
        predictor.updateWeights(flat.buffer, readbackIdentity.network.getBiasesFlat());
        workerPerfMark('perf:worker:predictGridGpu:start');
        if (readbackMode === 'withNeurons') {
            await predictor.predictGridWithNeuronsInto(
                readbackIdentity.outputBuffer,
                readbackIdentity.neuronBuffer,
            );
        } else {
            await predictor.predictGridInto(readbackIdentity.outputBuffer);
        }
        workerPerfMeasure('perf:worker:predictGridGpu', 'perf:worker:predictGridGpu:start');
        if (!readbackIsCurrent()) {
            forceCpuFallback();
            return;
        }
        for (let index = 0; index < readbackIdentity.outputBuffer.length; index++) {
            const value = readbackIdentity.outputBuffer[index];
            if (!Number.isFinite(value)) {
                throw new NonFiniteNumericalError(`predictionGrid.output[${index}]`, value);
            }
        }
        if (readbackMode === 'withNeurons') {
            for (let index = 0; index < readbackIdentity.neuronBuffer.length; index++) {
                const value = readbackIdentity.neuronBuffer[index];
                if (!Number.isFinite(value)) {
                    throw new NonFiniteNumericalError(`predictionGrid.neurons[${index}]`, value);
                }
            }
        }
        state.gridFreshFromGpu = true;
    } catch (err) {
        if (!readbackIsCurrent()) {
            forceCpuFallback();
            return;
        }
        if (err instanceof NonFiniteNumericalError) {
            throw new TerminalDivergenceError(`$.gpu.${err.path}`, err.value);
        }
        console.warn('[worker] GPU grid prediction failed, falling back to CPU', err);
        forceCpuFallback();
    }
}

// ── Snapshot computation ──

function assertFiniteV2SnapshotScalars(snapshot: NetworkSnapshot): void {
    if (!state.prepared) return;
    const candidates: Array<readonly [string, number | undefined]> = [
        ['$.snapshot.trainLoss', snapshot.trainLoss],
        ['$.snapshot.testLoss', snapshot.testLoss],
        ['$.snapshot.trainMetrics.loss', snapshot.trainMetrics.loss],
        ['$.snapshot.testMetrics.loss', snapshot.testMetrics.loss],
        ['$.snapshot.trainMetrics.accuracy', snapshot.trainMetrics.accuracy],
        ['$.snapshot.testMetrics.accuracy', snapshot.testMetrics.accuracy],
    ];
    for (const [path, value] of candidates) {
        if (value !== undefined && !Number.isFinite(value)) {
            throw new TerminalDivergenceError(path, value);
        }
    }
}

/**
 * @param opts.lightweight — when true, skips the deep-copy of weights/biases
 * into the snapshot. The streaming path transfers flat buffers separately
 * (see packSnapshotMessage), so nested copies are pure waste there.
 */
function computeSnapshotUnchecked(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
    workerPerfMark('perf:worker:snapshot:start');
    if (!state.network || !state.trainingConfig || !state.dataConfig) {
        throw new Error('Not initialized');
    }

    const { demand } = state;
    const problemType = state.dataConfig.problemType;
    const lossType = state.trainingConfig.lossType;
    const huberDelta = state.trainingConfig.huberDelta;

    // ── Train metrics (gated by trainEvalInterval) ───────────────────────────
    // Between full dataset evaluations we overlay the running EMA of batch
    // loss on top of the last accuracy reading. This keeps the loss line
    // responsive without re-evaluating the entire training set every frame.
    const shouldRunTrainEval =
        state.snapshotsSinceLastTrainEval >= demand.trainEvalInterval ||
        state.lastTrainMetrics === null;

    let trainMetrics: Metrics;
    if (shouldRunTrainEval) {
        workerPerfMark('perf:worker:trainEval:start');
        trainMetrics = state.network.evaluate(
            state.trainInputs,
            state.trainTargets,
            lossType,
            problemType,
            huberDelta,
        );
        workerPerfMeasure('perf:worker:trainEval', 'perf:worker:trainEval:start');
        state.lastTrainMetrics = { loss: trainMetrics.loss, accuracy: trainMetrics.accuracy };
        // Re-align EMA to the just-measured true loss so the next cycle's
        // EMA readings start from a known-good point.
        state.lossEma = trainMetrics.loss;
        state.snapshotsSinceLastTrainEval = 0;
    } else {
        // Fall back to EMA-over-cached accuracy. EMA already updated in trainOneStep.
        const cachedTrain = state.lastTrainMetrics!;
        const emaLoss = state.lossEma ?? cachedTrain.loss;
        trainMetrics = {
            loss: emaLoss,
            accuracy: cachedTrain.accuracy,
        };
        state.snapshotsSinceLastTrainEval++;
    }

    // ── Test metrics (gated by testEvalInterval) ─────────────────────────────
    const shouldRunTestEval =
        state.snapshotsSinceLastTestEval >= demand.testEvalInterval ||
        state.lastTestMetrics === null;

    let testMetrics;
    if (shouldRunTestEval) {
        workerPerfMark('perf:worker:testEval:start');
        testMetrics = state.network.evaluate(
            state.testInputs,
            state.testTargets,
            lossType,
            problemType,
            huberDelta,
        );
        workerPerfMeasure('perf:worker:testEval', 'perf:worker:testEval:start');
        state.lastTestMetrics = {
            loss: testMetrics.loss,
            accuracy: testMetrics.accuracy,
            confusionMatrix: demand.needConfusionMatrix ? testMetrics.confusionMatrix : undefined,
        };
        if (demand.needConfusionMatrix && testMetrics.confusionMatrix) {
            state.confusionMatrixVersion++;
        }
        state.snapshotsSinceLastTestEval = 0;
        state.testMetricsStale = false;
    } else {
        testMetrics = state.lastTestMetrics!;
        state.snapshotsSinceLastTestEval++;
        state.testMetricsStale = true;
    }

    // ── Decision boundary grid (demand-gated AND cadence-gated) ──────────────
    // The grid is the single most expensive per-snapshot artifact. We only
    // recompute it on the first snapshot of a run, when the UI just turned
    // on a grid demand, when training has progressed since the last grid
    // was built, and when the cadence counter permits it.
    let outputGrid: number[] | Float32Array;
    let neuronGrids: number[][] | Float32Array | undefined;
    state.multiclassBoundaryFresh = false;

    const supportsScalarGrid = state.networkConfig?.outputSize === 1;
    const supportsMulticlassBoundary =
        state.networkConfig?.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE &&
        state.networkConfig.outputActivation === 'softmax';
    const wantGrid = supportsScalarGrid && (demand.needDecisionBoundary || demand.needNeuronGrids);
    const wantMulticlassBoundary = supportsMulticlassBoundary && demand.needDecisionBoundary;
    const shouldRebuildGrid =
        wantGrid &&
        state.gridStale &&
        state.snapshotsSinceLastGrid >= demand.gridInterval;
    const shouldRebuildMulticlassBoundary =
        wantMulticlassBoundary &&
        state.gridStale &&
        state.snapshotsSinceLastGrid >= demand.gridInterval;

    // GPU pre-fill (AS-4): runGpuGridIfDue ran earlier in trainTick and
    // already populated the grid buffers. Consume the flag here so the CPU
    // branches below stay short-circuited, and so a second call later in
    // the same frame doesn't double-count.
    if (state.gridFreshFromGpu && supportsScalarGrid && state.outputGridBuffer) {
        outputGrid = state.outputGridBuffer;
        if (demand.needNeuronGrids && state.neuronGridsBuffer) {
            neuronGrids = state.neuronGridsBuffer;
        }
        state.snapshotsSinceLastGrid = 0;
        state.gridStale = false;
        state.gridFreshFromGpu = false;
    } else if (state.gridFreshFromGpu && !supportsScalarGrid) {
        state.gridFreshFromGpu = false;
        outputGrid = [];
    } else if (shouldRebuildGrid && demand.needNeuronGrids && state.outputGridBuffer && state.neuronGridsBuffer) {
        workerPerfMark('perf:worker:predictGridNeurons:start');
        state.network.predictGridWithNeuronsInto(
            state.gridInputs,
            state.outputGridBuffer,
            state.neuronGridsBuffer,
        );
        workerPerfMeasure('perf:worker:predictGridNeurons', 'perf:worker:predictGridNeurons:start');
        outputGrid = state.outputGridBuffer;
        neuronGrids = state.neuronGridsBuffer;
        state.snapshotsSinceLastGrid = 0;
        state.gridStale = false;
    } else if (shouldRebuildGrid && demand.needDecisionBoundary && state.outputGridBuffer) {
        workerPerfMark('perf:worker:predictGrid:start');
        state.network.predictGridInto(state.gridInputs, state.outputGridBuffer);
        workerPerfMeasure('perf:worker:predictGrid', 'perf:worker:predictGrid:start');
        outputGrid = state.outputGridBuffer;
        state.snapshotsSinceLastGrid = 0;
        state.gridStale = false;
    } else if (
        shouldRebuildMulticlassBoundary &&
        state.multiclassClassGridBuffer &&
        state.multiclassConfidenceGridBuffer
    ) {
        workerPerfMark('perf:worker:predictMulticlassBoundary:start');
        state.network.predictMulticlassBoundaryInto(
            state.gridInputs,
            state.multiclassClassGridBuffer,
            state.multiclassConfidenceGridBuffer,
        );
        workerPerfMeasure(
            'perf:worker:predictMulticlassBoundary',
            'perf:worker:predictMulticlassBoundary:start',
        );
        outputGrid = [];
        state.snapshotsSinceLastGrid = 0;
        state.gridStale = false;
        state.multiclassBoundaryFresh = true;
    } else if (wantGrid && state.outputGridBuffer) {
        // Reuse the last computed grid(s) without recomputing. The main
        // thread retains the previous Float32Arrays in its frame buffer;
        // emitting undefined here causes packSnapshotMessage to skip the
        // buffer transfer entirely.
        outputGrid = [];
        state.snapshotsSinceLastGrid++;
    } else if (wantMulticlassBoundary) {
        outputGrid = [];
        state.snapshotsSinceLastGrid++;
    } else {
        outputGrid = [];
    }

    const snap = state.network.getSnapshot(
        state.network.getStep(),
        state.epoch,
        trainMetrics,
        testMetrics,
        outputGrid,
        GRID_SIZE,
        { includeParams: !opts.lightweight },
    );
    if (state.prepared) {
        delete snap.historyPoint;
    }
    // Direct RPC snapshots do not pass through WorkerSnapshotMessage.scalars,
    // so keep their cadence metadata alongside the metrics themselves.
    snap.testMetricsStale = state.testMetricsStale;
    // Streamed snapshots carry this through dedicated transferable fields.
    // Direct RPC snapshots need the same bounded payload so a paused manual
    // step can refresh the multiclass Boundary view.
    if (
        !opts.lightweight &&
        state.multiclassBoundaryFresh &&
        state.multiclassClassGridBuffer &&
        state.multiclassConfidenceGridBuffer
    ) {
        snap.multiclassBoundary = {
            classGrid: state.multiclassClassGridBuffer,
            confidenceGrid: state.multiclassConfidenceGridBuffer,
            gridSize: GRID_SIZE,
        };
    }

    if (neuronGrids) {
        snap.neuronGrids = neuronGrids;
    }

    if (demand.needLayerStats) {
        const populationCount = state.trainInputs.length;
        const sampleCount = Math.min(128, populationCount);
        const statistics = state.network.computeLayerStatistics(state.trainInputs, 128);
        if (statistics.sampleCount !== sampleCount
            || statistics.populationCount !== populationCount) {
            throw new Error(
                'Layer statistics sample basis does not match the training population.',
            );
        }
        if (statistics.revision !== state.network.getRevision()) {
            throw new Error('layer statistics revision must equal the current model revision');
        }
        snap.layerStats = statistics.layers;
        state.layerStatsGradientRevision = statistics.gradientRevision;
    } else {
        state.layerStatsGradientRevision = null;
    }

    const wantsActivationHistograms = demand.needActivationHistograms;
    const shouldComputeActivationHistograms =
        wantsActivationHistograms &&
        state.snapshotsSinceLastActivationHistogram >= demand.activationHistogramInterval;
    if (shouldComputeActivationHistograms) {
        snap.activationHistograms = state.network.computeActivationHistograms(
            state.trainInputs,
            {
                binCount: ACTIVATION_HISTOGRAM_BIN_COUNT,
                maxSamples: ACTIVATION_HISTOGRAM_MAX_SAMPLES,
            },
        );
        state.snapshotsSinceLastActivationHistogram = 0;
    } else if (wantsActivationHistograms) {
        state.snapshotsSinceLastActivationHistogram++;
    }

    if (state.prepared) {
        const pairedMatrix = state.demand.needConfusionMatrix
            ? state.evaluationRuntime?.latestEvaluation?.test.values.confusionMatrix
            : undefined;
        snap.testMetrics.confusionMatrix = undefined;
        snap.testMetrics.multiclassConfusionMatrix = undefined;
        if (pairedMatrix !== undefined) {
            if (isMulticlassConfusionMatrix(pairedMatrix)) {
                snap.testMetrics.multiclassConfusionMatrix = pairedMatrix;
            } else {
                snap.testMetrics.confusionMatrix = pairedMatrix;
            }
        }
    }

    // Legacy checkpoint payloads cannot restore strict V2 provenance. Keep
    // them disabled until the dedicated V2 checkpoint transaction lands.
    assertFiniteV2SnapshotScalars(snap);
    if (state.prepared) {
        // Prove the display transport can represent every parameter before
        // evidence/frame publication; Float64 values may overflow Float32.
        state.network.getWeightsFlat();
        state.network.getBiasesFlat();
    }
    if (!state.prepared) captureCheckpointFromSnapshot(snap);
    workerPerfMeasure('perf:worker:snapshot', 'perf:worker:snapshot:start');
    return snap;
}

function computeSnapshot(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
    if (!state.prepared) return computeSnapshotUnchecked(opts);
    return withEngineNumericalBoundary('$.snapshot.engine', () => computeSnapshotUnchecked(opts));
}

function artifactProvenanceAt(
    basis: ArtifactBasis,
    model: ArtifactProvenance['model'],
): ArtifactProvenance {
    if (!state.prepared || !state.datasetRevision) {
        throw new Error('strict artifact provenance requires an initialized V2 experiment');
    }
    return {
        model,
        dataset: state.datasetRevision,
        objectiveKey: state.prepared.identities.objectiveKey,
        basis,
    };
}

function currentArtifactProvenance(basis: ArtifactBasis): ArtifactProvenance {
    if (!state.network) throw new Error('strict artifact provenance requires a network');
    return artifactProvenanceAt(basis, {
        generationId: state.runId,
        revision: state.network.getRevision(),
        step: state.network.getStep(),
        epoch: state.epoch,
    });
}

function isMulticlassConfusionMatrix(
    matrix: NonNullable<PairedEvaluation['test']['values']['confusionMatrix']>,
): matrix is MulticlassConfusionMatrixData {
    return 'classCount' in matrix;
}

function directSnapshotArtifactBundle(
    snapshot: NetworkSnapshot,
): Pick<WorkerExperimentResultV2, 'artifacts' | 'layerStatsGradientRevision'> {
    if (!state.prepared || !state.datasetRevision) return {};
    const produced: {
        -readonly [Key in keyof WorkerArtifactProvenanceV2]: WorkerArtifactProvenanceV2[Key];
    } = {};
    if (snapshot.outputGrid.length > 0 || snapshot.multiclassBoundary !== undefined) {
        produced.decisionBoundary = currentArtifactProvenance({
            kind: 'prediction-grid',
            pointCount: snapshot.gridSize * snapshot.gridSize,
            domain: [-1, 1, -1, 1],
        });
    }
    if (snapshot.neuronGrids !== undefined && snapshot.neuronGrids.length > 0) {
        produced.neuronGrids = currentArtifactProvenance({
            kind: 'prediction-grid',
            pointCount: snapshot.gridSize * snapshot.gridSize,
            domain: [-1, 1, -1, 1],
        });
    }
    if (snapshot.layerStats !== undefined) {
        produced.activationStatistics = currentArtifactProvenance({
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, state.datasetRevision.trainCount),
            populationCount: state.datasetRevision.trainCount,
        });
    }
    if (snapshot.activationHistograms !== undefined) {
        produced.activationHistogram = currentArtifactProvenance({
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, state.datasetRevision.trainCount),
            populationCount: state.datasetRevision.trainCount,
        });
    }
    if (snapshot.testMetrics.confusionMatrix !== undefined
        || snapshot.testMetrics.multiclassConfusionMatrix !== undefined) {
        const evaluation = state.evaluationRuntime?.latestEvaluation;
        if (evaluation === undefined) {
            throw new Error('direct confusion artifact requires a paired evaluation');
        }
        produced.confusionMatrix = artifactProvenanceAt(
            evaluation.test.basis,
            evaluation.model,
        );
    }
    if (snapshot.layerStats !== undefined && state.layerStatsGradientRevision === null) {
        throw new Error('direct layer statistics require a clipped-gradient revision');
    }
    return {
        ...(Object.keys(produced).length === 0 ? {} : { artifacts: produced }),
        ...(snapshot.layerStats === undefined
            ? {}
            : { layerStatsGradientRevision: state.layerStatsGradientRevision! }),
    };
}

/**
 * Pack a NetworkSnapshot into a WorkerSnapshotMessage with Transferable Float32Arrays.
 * Returns { message, transferables }.
 */
function packSnapshotMessage(snap: NetworkSnapshot): { message: WorkerSnapshotMessage; transferables: Transferable[] } {
    workerPerfMark('perf:worker:snapshotPack:start');

    const transferables: Transferable[] = [];
    const gridSize = snap.gridSize;

    // Fast path: SharedArrayBuffer transport is active. Publish the grid
    // payloads into the shared buffers under a seqlock; leave the message
    // fields undefined and set `sharedSeq` so the main thread reads from
    // its installed views instead of expecting inline arrays. No
    // transferables / no per-frame reallocation on the worker side.
    //
    // Note: the Float32Array-or-array type union on snap.outputGrid comes
    // from the legacy postMessage path. When SAB is active we always fed
    // the Float32Array pre-alloc buffers into the network predictors, so
    // the `instanceof Float32Array` branches are the only ones that fire.
    const sharedViews = state.sharedViews;
    let sharedSeq: number | undefined;
    let outputGrid: Float32Array | undefined;
    let neuronGrids: Float32Array | undefined;
    let neuronGridLayout: { count: number; gridSize: number } | undefined;
    let activationHistogramBins: Float32Array | undefined;
    let activationHistogramLayout: WorkerSnapshotMessage['activationHistogramLayout'] | undefined;
    let activationHistogramVersion: number | undefined;
    let multiclassClassGrid: Uint8Array | undefined;
    let multiclassConfidenceGrid: Float32Array | undefined;
    let multiclassBoundaryLayout: WorkerSnapshotMessage['multiclassBoundaryLayout'] | undefined;
    let multiclassBoundaryVersion: number | undefined;
    let confusionMatrix: WorkerSnapshotMessage['confusionMatrix'] | undefined;
    let multiclassConfusionMatrix: WorkerSnapshotMessage['multiclassConfusionMatrix'] | undefined;
    let multiclassConfusionMatrixVersion: number | undefined;
    let confusionMatrixProvenance: ArtifactProvenance | undefined;
    let confusionMatrixEvaluationId: number | undefined;

    if (sharedViews) {
        let flags = 0;
        let outSrc: Float32Array | null = null;
        let neuSrc: Float32Array | null = null;
        if (snap.outputGrid instanceof Float32Array && snap.outputGrid.length > 0) {
            outSrc = snap.outputGrid;
            flags |= FLAG_OUTPUT_GRID;
        }
        if (snap.neuronGrids instanceof Float32Array && snap.neuronGrids.length > 0) {
            neuSrc = snap.neuronGrids;
            flags |= FLAG_NEURON_GRIDS;
            neuronGridLayout = {
                count: state.network!.getTotalNeuronCount(),
                gridSize,
            };
        }
        if (flags !== 0) {
            sharedSeq = publishSharedSnapshot(sharedViews, outSrc, neuSrc, flags);
        }
        // outputGrid / neuronGrids remain undefined on the outgoing message;
        // the main thread picks them up from the SAB views. Note that we
        // still emit neuronGridLayout when neurons are fresh, so the UI can
        // rebuild subarray slicing keyed to the current neuron count.
    } else {
        // Legacy postMessage-with-transferable path. Unchanged from before
        // AS-3 — keeps the codebase green on non-isolated hosts (GH Pages,
        // test runners).
        if (snap.outputGrid && snap.outputGrid.length > 0) {
            if (snap.outputGrid instanceof Float32Array) {
                outputGrid = snap.outputGrid;
                state.outputGridBuffer = null;
            } else {
                outputGrid = new Float32Array(snap.outputGrid);
            }
            transferables.push(outputGrid.buffer);
        }
        if (state.outputGridBuffer === null) {
            state.outputGridBuffer = new Float32Array(GRID_SIZE * GRID_SIZE);
        }
        if (snap.neuronGrids && snap.neuronGrids.length > 0) {
            if (snap.neuronGrids instanceof Float32Array) {
                neuronGrids = snap.neuronGrids;
                const totalNeurons = state.network!.getTotalNeuronCount();
                neuronGridLayout = { count: totalNeurons, gridSize };
                state.neuronGridsBuffer = null;
            } else {
                const count = snap.neuronGrids.length;
                const totalSize = count * gridSize * gridSize;
                neuronGrids = new Float32Array(totalSize);
                for (let n = 0; n < count; n++) {
                    neuronGrids.set(snap.neuronGrids[n], n * gridSize * gridSize);
                }
                neuronGridLayout = { count, gridSize };
            }
            transferables.push(neuronGrids.buffer);
        }
        if (state.neuronGridsBuffer === null && state.network) {
            const totalNeurons = state.network.getTotalNeuronCount();
            state.neuronGridsBuffer = new Float32Array(totalNeurons * GRID_SIZE * GRID_SIZE);
        }
    }

    // Pack weights
    const { buffer: weightsFlat, layerSizes } = state.network!.getWeightsFlat();
    transferables.push(weightsFlat.buffer);

    // Pack biases
    const biasesFlat = state.network!.getBiasesFlat();
    transferables.push(biasesFlat.buffer);

    if (snap.activationHistograms) {
        const histograms: ActivationHistogramResult = snap.activationHistograms;
        activationHistogramBins = histograms.bins;
        activationHistogramLayout = {
            binCount: histograms.layers[0]?.binCount ?? ACTIVATION_HISTOGRAM_BIN_COUNT,
            layers: histograms.layers,
        };
        state.activationHistogramVersion++;
        activationHistogramVersion = state.activationHistogramVersion;
        transferables.push(activationHistogramBins.buffer);
    }

    if (
        state.multiclassBoundaryFresh &&
        state.multiclassClassGridBuffer &&
        state.multiclassConfidenceGridBuffer
    ) {
        multiclassClassGrid = state.multiclassClassGridBuffer;
        multiclassConfidenceGrid = state.multiclassConfidenceGridBuffer;
        state.multiclassClassGridBuffer = null;
        state.multiclassConfidenceGridBuffer = null;
        state.multiclassBoundaryVersion++;
        multiclassBoundaryVersion = state.multiclassBoundaryVersion;
        multiclassBoundaryLayout = {
            gridSize,
            classCount: WORKER_MULTICLASS_OUTPUT_SIZE,
            classLabels: [0, 1, 2] as const,
        };
        transferables.push(multiclassClassGrid.buffer, multiclassConfidenceGrid.buffer);
    }
    if (state.multiclassClassGridBuffer === null && state.networkConfig?.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE) {
        state.multiclassClassGridBuffer = new Uint8Array(GRID_SIZE * GRID_SIZE);
    }
    if (state.multiclassConfidenceGridBuffer === null && state.networkConfig?.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE) {
        state.multiclassConfidenceGridBuffer = new Float32Array(GRID_SIZE * GRID_SIZE);
    }

    if (multiclassClassGrid && multiclassConfidenceGrid) {
        outputGrid = new Float32Array(0);
        neuronGrids = new Float32Array(0);
        neuronGridLayout = undefined;
        transferables.push(outputGrid.buffer, neuronGrids.buffer);
    }

    if (state.prepared) {
        const evaluation = state.evaluationRuntime?.latestEvaluation;
        const matrix = state.demand.needConfusionMatrix
            ? evaluation?.test.values.confusionMatrix
            : undefined;
        if (matrix !== undefined
            && evaluation !== undefined
            && evaluation.evaluationId !== state.lastPackedConfusionEvaluationId) {
            state.lastPackedConfusionEvaluationId = evaluation.evaluationId;
            confusionMatrixEvaluationId = evaluation.evaluationId;
            confusionMatrixProvenance = artifactProvenanceAt(
                evaluation.test.basis,
                evaluation.model,
            );
            if (isMulticlassConfusionMatrix(matrix)) {
                multiclassConfusionMatrix = matrix;
                state.multiclassConfusionMatrixVersion++;
                multiclassConfusionMatrixVersion = state.multiclassConfusionMatrixVersion;
            } else {
                confusionMatrix = matrix;
                state.confusionMatrixVersion++;
            }
        }
    } else {
        confusionMatrix = state.testMetricsStale
            ? undefined
            : snap.testMetrics?.confusionMatrix;
        if (
            !state.testMetricsStale &&
            state.demand.needConfusionMatrix &&
            state.networkConfig &&
            state.trainingConfig &&
            state.dataConfig &&
            isApprovedWorkerMulticlassConfig(
                state.networkConfig,
                state.trainingConfig,
                state.dataConfig,
            ) &&
            snap.testMetrics?.multiclassConfusionMatrix
        ) {
            multiclassConfusionMatrix = snap.testMetrics.multiclassConfusionMatrix;
            state.multiclassConfusionMatrixVersion++;
            multiclassConfusionMatrixVersion = state.multiclassConfusionMatrixVersion;
        }
    }

    const historyPoint: HistoryPoint | undefined = state.prepared
        ? undefined
        : {
            step: snap.step,
            trainLoss: snap.trainLoss,
            testLoss: snap.testLoss,
            trainAccuracy: snap.trainMetrics?.accuracy,
            testAccuracy: snap.testMetrics?.accuracy,
        };

    let artifacts: WorkerArtifactProvenanceV2 | undefined;
    if (state.prepared && state.datasetRevision) {
        const produced: WorkerArtifactProvenanceV2 = {
            ...((outputGrid !== undefined && outputGrid.length > 0)
                || multiclassClassGrid !== undefined
                || sharedSeq !== undefined
                ? {
                    decisionBoundary: currentArtifactProvenance({
                        kind: 'prediction-grid',
                        pointCount: gridSize * gridSize,
                        domain: [-1, 1, -1, 1],
                    }),
                }
                : {}),
            ...((neuronGrids !== undefined && neuronGrids.length > 0)
                || (sharedSeq !== undefined && neuronGridLayout !== undefined)
                ? {
                    neuronGrids: currentArtifactProvenance({
                        kind: 'prediction-grid',
                        pointCount: gridSize * gridSize,
                        domain: [-1, 1, -1, 1],
                    }),
                }
                : {}),
            ...(snap.layerStats !== undefined
                ? {
                    activationStatistics: currentArtifactProvenance({
                        kind: 'bounded-sample',
                        split: 'train',
                        sampleCount: Math.min(128, state.datasetRevision.trainCount),
                        populationCount: state.datasetRevision.trainCount,
                    }),
                }
                : {}),
            ...(activationHistogramBins !== undefined
                ? {
                    activationHistogram: currentArtifactProvenance({
                        kind: 'bounded-sample',
                        split: 'train',
                        sampleCount: Math.min(128, state.datasetRevision.trainCount),
                        populationCount: state.datasetRevision.trainCount,
                    }),
                }
                : {}),
            ...(confusionMatrixProvenance === undefined
                ? {}
                : { confusionMatrix: confusionMatrixProvenance }),
        };
        if (Object.keys(produced).length > 0) artifacts = produced;
    }

    const message: WorkerSnapshotMessage = {
        type: 'snapshot',
        protocolVersion: state.prepared ? WORKER_PROTOCOL_VERSION : undefined,
        runId: state.runId,
        snapshotId: ++state.snapshotId,
        model: state.prepared && state.network ? {
            generationId: state.runId,
            revision: state.network.getRevision(),
            step: state.network.getStep(),
            epoch: state.epoch,
        } : undefined,
        scalars: {
            step: snap.step,
            epoch: snap.epoch,
            trainLoss: snap.trainLoss,
            testLoss: snap.testLoss,
            trainAccuracy: snap.trainMetrics?.accuracy,
            testAccuracy: snap.testMetrics?.accuracy,
            gridSize,
            testMetricsStale: state.testMetricsStale,
        },
        outputGrid,
        neuronGrids,
        neuronGridLayout,
        weights: weightsFlat,
        biases: biasesFlat,
        weightLayout: { layerSizes },
        layerStats: snap.layerStats,
        layerStatsGradientRevision: snap.layerStats === undefined
            ? undefined
            : state.layerStatsGradientRevision ?? undefined,
        activationHistogramBins,
        activationHistogramLayout,
        activationHistogramVersion,
        multiclassClassGrid,
        multiclassConfidenceGrid,
        multiclassBoundaryLayout,
        multiclassBoundaryVersion,
        ...(historyPoint === undefined ? {} : { historyPoint }),
        artifacts,
        confusionMatrix,
        confusionMatrixEvaluationId,
        confusionMatrixVersion: state.confusionMatrixVersion,
        multiclassConfusionMatrix,
        multiclassConfusionMatrixVersion,
        checkpointTimeline: buildCheckpointTimeline(),
        sharedSeq,
    };

    workerPerfMeasure('perf:worker:snapshotPack', 'perf:worker:snapshotPack:start');
    return { message, transferables };
}

// ── Training ──

function trainOneStep(): void {
    if (!state.network || !state.trainingConfig) return;

    const bs = state.trainingConfig.batchSize;
    const n = state.trainInputs.length;
    if (n === 0) return;

    // Single source of truth for the step counter: the Network itself.
    const stepBefore = state.network.getStep();
    const numBatches = Math.ceil(n / bs);
    const batchSlot = stepBefore % numBatches;

    if (batchSlot === 0 && stepBefore > 0 && state.shufflePrng) {
        state.shufflePrng.shuffle(state.shuffledIndices);
    }

    const startIdx = batchSlot * bs;
    const endIdx = Math.min(startIdx + bs, n);
    if (!state.batchScratch) {
        state.batchScratch = createMiniBatchScratch(bs);
    }
    const batch = fillMiniBatchScratch(
        state.batchScratch,
        state.trainInputs,
        state.trainTargets,
        state.shuffledIndices,
        startIdx,
        endIdx,
    );

    if (batch.inputs.length > 0) {
        beginV2Mutation();
        const batchLoss = state.network.trainBatch(batch.inputs, batch.targets, state.trainingConfig);
        // Feed the running EMA of batch loss. This is what the UI line
        // actually follows between full-dataset evaluations.
        if (Number.isFinite(batchLoss)) {
            if (state.lossEma === null) state.lossEma = batchLoss;
            else state.lossEma = state.lossEmaAlpha * batchLoss + (1 - state.lossEmaAlpha) * state.lossEma;
        } else {
            // Propagate non-finite loss into the EMA so the outer loop's
            // divergence guard fires on the next snapshot.
            state.lossEma = batchLoss;
        }
        // The network has moved; any cached grid is out of date.
        state.gridStale = true;
    }

    if (batchSlot === numBatches - 1) {
        state.epoch++;
    }
}

function forceAndPublishEvaluationV2(
    trigger: ForcedEvaluationTriggerV2,
    publish: boolean = true,
): PairedEvaluation {
    const { runtime, history } = requireV2Runtime();
    beginV2Mutation();
    const evaluation = runtime.forceEvaluation(trigger);
    history.appendEvaluation(evaluation);
    if (publish) postEvidenceV2(makeEvidenceV2(undefined, evaluation));
    return evaluation;
}

/**
 * Advances one strict V2 batch. Every live signal is retained in worker
 * history, while only cadence pairs are posted immediately. The current live
 * signal is coalesced later with the produced visual frame.
 */
function trainOneStepV2(): {
    liveSignal: LiveTrainingSignal;
    cadenceEvaluation?: PairedEvaluation;
} | undefined {
    const { compiled, network, runtime, history, epochRef } = requireV2Runtime();
    const batchSize = compiled.training.batchSize;
    const sampleCount = state.trainInputs.length;
    if (sampleCount === 0) return undefined;

    const stepBefore = network.getStep();
    const batchCount = Math.ceil(sampleCount / batchSize);
    const batchSlot = stepBefore % batchCount;
    if (batchSlot === 0 && stepBefore > 0 && state.shufflePrng) {
        state.shufflePrng.shuffle(state.shuffledIndices);
    }
    const start = batchSlot * batchSize;
    const end = Math.min(start + batchSize, sampleCount);
    beginV2Mutation();
    const result = withEngineNumericalBoundary('$.training', () => (
        network.trainBatchIndexedV2(
            state.trainInputs,
            state.trainTargets,
            state.shuffledIndices,
            start,
            end,
            compiled.training,
        )
    ));

    if (batchSlot === batchCount - 1) {
        state.epoch++;
        epochRef.value = state.epoch;
    }
    const liveSignal = runtime.recordBatch({
        model: {
            generationId: state.runId,
            revision: result.revision,
            step: result.step,
            epoch: epochRef.value,
        },
        batchSize: result.sampleCount,
        dataLoss: result.objective.dataLoss,
    });
    history.appendLiveSignal(liveSignal);
    state.lossEma = liveSignal.dataLoss;
    state.gridStale = true;

    const cadenceEvaluation = runtime.takeCadenceEvaluation();
    if (cadenceEvaluation !== undefined) {
        history.appendEvaluation(cadenceEvaluation);
        // Cadence evidence is never coalesced into a latest-wins visual frame.
        postEvidenceV2(makeEvidenceV2(undefined, cadenceEvaluation));
    }
    return cadenceEvaluation === undefined
        ? { liveSignal }
        : { liveSignal, cadenceEvaluation };
}

// ── Internal training loop (worker-driven) ──

const TRAIN_TICK_INTERVAL_MS = 1000 / 60;

function scheduleNextTick(): void {
    if (state.trainLoopTimer !== null || !state.running) return;
    state.trainLoopTimer = setTimeout(() => {
        state.trainLoopTimer = null;
        if (!state.running) return;
        if (state.prepared) {
            // Timer-driven V2 batches share the same scientific mutation lane
            // as RPC capture/reset/step so an awaited capture stays exclusive.
            void enqueueV2Mutation(() => trainTick());
        } else {
            trainTick();
        }
    }, TRAIN_TICK_INTERVAL_MS);
}

// Snapshot pipeline split out from trainTick so the GPU grid pre-fill
// (AS-4) can be awaited without forcing the whole tick to be async. The
// back-pressure gate `state.awaitingAck` is set BEFORE we enter this
// async work, so subsequent ticks skip their snapshot block until the
// main thread acks.
interface SnapshotPipelineIdentity {
    readonly runId: number;
    readonly network: Network | null;
}

async function produceAndPostSnapshot(identity: SnapshotPipelineIdentity): Promise<void> {
    if (!state.streamPort) return;

    // GPU grid pre-fill (AS-4). When enabled + capable + due, this
    // populates state.outputGridBuffer / state.neuronGridsBuffer; the
    // synchronous computeSnapshot below detects the freshly-filled
    // buffers via state.gridFreshFromGpu and skips its CPU branch.
    await runGpuGridIfDue();
    // An awaited predecessor must never continue packing against a replacement
    // generation or interfere with its independent backpressure lifecycle.
    if (state.runId !== identity.runId || state.network !== identity.network) return;

    // Live evidence is transport-coalesced to one current signal per visual
    // frame. Worker history still contains every batch (see trainOneStepV2).
    const currentLive = state.evaluationRuntime?.latestLiveSignal;
    if (currentLive !== undefined) {
        postEvidenceV2(makeEvidenceV2(currentLive));
    }

    let stopEvaluation: ReturnType<typeof evaluateStopConditions> | null = null;
    if (state.running && state.prepared && state.network && state.evaluationRuntime) {
        const model = {
            generationId: state.runId,
            revision: state.network.getRevision(),
            step: state.network.getStep(),
            epoch: state.epoch,
        };
        let currentEvaluation = state.evaluationRuntime.latestEvaluation;
        const evaluationIsCurrent = currentEvaluation !== undefined
            && currentEvaluation.model.generationId === model.generationId
            && currentEvaluation.model.revision === model.revision
            && currentEvaluation.model.step === model.step
            && currentEvaluation.model.epoch === model.epoch;
        if (stopConditionsRequireCurrentEvaluation(runtimeStopConditions)
            && !evaluationIsCurrent) {
            try {
                currentEvaluation = forceAndPublishEvaluationV2('stop-condition');
            } catch (error) {
                if (error instanceof TerminalDivergenceError) {
                    postTerminalDivergenceV2(error, 'evaluation', 'evaluation-failed');
                    return;
                }
                throw error;
            }
        }
        stopEvaluation = evaluateStopConditions(
            runtimeStopConditions,
            {
                model,
                liveSignal: currentLive,
                currentEvaluation,
            },
            state.stopConditionState,
        );
        state.stopConditionState = stopEvaluation.nextState;
        if (stopEvaluation.terminalDivergence) {
            postTerminalDivergenceV2(
                new TerminalDivergenceError(
                    stopEvaluation.terminalDivergence.path,
                    stopEvaluation.terminalDivergence.value,
                ),
                'evaluation',
                'evaluation-failed',
            );
            return;
        }
    }

    const snap = computeSnapshot({ lightweight: true });
    const legacyDiverged = state.running
        && state.prepared === null
        && (!Number.isFinite(snap.trainLoss) || !Number.isFinite(snap.testLoss));

    const { message, transferables } = withEngineNumericalBoundary(
        '$.snapshot.parameters',
        () => packSnapshotMessage(snap),
    );
    state.streamPort.postMessage(message, transferables);
    const pauseReason = stopEvaluation?.pauseReason ?? (legacyDiverged ? 'diverged' : null);
    if (pauseReason) {
        stopInternalLoop();
        postStatus('paused', pauseReason);
    }
}

function trainTick(): void {
    if (!state.running) return;

    try {
        if (state.streamPort) {
            workerPerfMark('perf:worker:trainStep:start');

            // Keep the selected speed literal: each scheduled tick advances
            // by a bounded number of steps, then yields so pause/update
            // commands can be processed before more work starts.
            const burst = getTrainingStepsForTick(state.stepsPerFrame);
            for (let i = 0; i < burst && state.running; i++) {
                if (state.prepared) trainOneStepV2();
                else trainOneStep();
            }

            workerPerfMeasure('perf:worker:trainStep', 'perf:worker:trainStep:start');

            // Back-pressure: skip snapshot computation + posting while the main
            // thread hasn't yet applied the previous frame. Training still
            // progresses; the UI just coalesces to its render rate.
            if (!state.awaitingAck) {
                // Gate first so concurrent ticks don't fire while the GPU
                // pre-fill awaits the device. The ack will land after
                // postMessage in produceAndPostSnapshot() completes.
                state.awaitingAck = true;
                const pipelineIdentity: SnapshotPipelineIdentity = {
                    runId: state.runId,
                    network: state.network,
                };
                produceAndPostSnapshot(pipelineIdentity).catch((err) => {
                    if (state.runId !== pipelineIdentity.runId
                        || state.network !== pipelineIdentity.network) return;
                    state.awaitingAck = false;
                    stopInternalLoop();
                    if (state.prepared) {
                        if (err instanceof TerminalDivergenceError) {
                            postTerminalDivergenceForPhase(err, 'evaluation');
                        } else {
                            postRuntimeErrorV2(err, 'runtime');
                        }
                    }
                    else {
                        const msg = err instanceof Error ? err.message : String(err);
                        postError(`Training snapshot error: ${msg}`);
                    }
                });
            }
        }

        if (state.running) {
            scheduleNextTick();
        }
    } catch (err) {
        stopInternalLoop();
        if (state.prepared) {
            if (err instanceof TerminalDivergenceError) {
                postTerminalDivergenceForPhase(err, 'training');
            } else {
                postRuntimeErrorV2(err, 'training');
            }
        }
        else postError(`Training error: ${err instanceof Error ? err.message : String(err)}`);
    }
}

function startInternalLoop(): void {
    // Clear any stale gate from a previous run — no outstanding ack at start.
    resetAck();
    state.running = true;
    scheduleNextTick();
}

function stopInternalLoop(): void {
    state.running = false;
    if (state.trainLoopTimer !== null) {
        clearTimeout(state.trainLoopTimer);
        state.trainLoopTimer = null;
    }
    // Drop the gate — a paused loop must not block a later resume on an ack
    // for a snapshot we no longer care about.
    resetAck();
}

function applyDemand(demand: VisualizationDemand): void {
    const confusionDemandChanged = state.demand.needConfusionMatrix !== demand.needConfusionMatrix;
    state.demand = { ...demand };
    // Force the next snapshot to re-evaluate everything so the UI
    // immediately reflects the new demand mix, rather than waiting
    // up to one interval for the counters to roll over.
    state.snapshotsSinceLastTestEval = demand.testEvalInterval;
    state.snapshotsSinceLastTrainEval = demand.trainEvalInterval;
    state.snapshotsSinceLastGrid = demand.gridInterval;
    state.snapshotsSinceLastActivationHistogram = demand.activationHistogramInterval;
    state.gridStale = true;
    if (confusionDemandChanged) {
        state.lastTestMetrics = null;
        state.lastPackedConfusionEvaluationId = null;
        state.confusionMatrixVersion++;
    }
}

// ── MessageChannel command handler ──

function reportStreamCommandFailure(error: unknown): void {
    if (state.prepared) {
        if (error instanceof TerminalDivergenceError) {
            postTerminalDivergenceV2(error, 'evaluation', 'evaluation-failed');
        } else {
            postRuntimeErrorV2(error, 'runtime');
        }
    } else {
        postError(`Command handling error: ${error instanceof Error ? error.message : String(error)}`);
    }
}

function runOrQueueV2StreamMutation(operation: () => void): void {
    if (state.prepared) {
        void enqueueV2Mutation(operation).catch(reportStreamCommandFailure);
    } else {
        operation();
    }
}

function handleStreamCommand(cmd: unknown): void {
    try {
        if (!isMainToWorkerCommand(cmd)) {
            const error = new TypeError('Unknown worker stream command');
            if (state.prepared) postRuntimeErrorV2(error, 'protocol', 'malformed-request');
            else postError(error.message + ': ' + JSON.stringify(cmd));
            return;
        }
        switch (cmd.type) {
            case 'startTraining':
                runOrQueueV2StreamMutation(() => {
                    state.stepsPerFrame = normalizeTrainingSpeed(cmd.stepsPerFrame);
                    startInternalLoop();
                    postStatus('running');
                });
                break;

            case 'stopTraining':
            {
                runOrQueueV2StreamMutation(() => {
                    const wasRunning = state.running;
                    stopInternalLoop();
                    if (state.prepared && wasRunning) forceAndPublishEvaluationV2('pause');
                    postStatus('paused');
                });
                break;
            }

            case 'updateDemand':
                applyDemand(normalizeVisualizationDemand(cmd.demand)!);
                break;

            case 'updateSpeed':
                state.stepsPerFrame = normalizeTrainingSpeed(cmd.stepsPerFrame);
                break;

            case 'frameAck':
                // Main thread applied the previous snapshot — free the gate so
                // the next trainTick is allowed to post again.
                state.awaitingAck = false;
                break;
        }
    } catch (err) {
        reportStreamCommandFailure(err);
    }
}

function requireFiniteTraceValue(value: number | undefined, name: string): number {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
    }
    return value;
}

function resolveTraceSample(request: PredictionTraceRequest): {
    source: PredictionTraceSampleSource;
    index?: number;
    x: number;
    y: number;
    label?: number;
} {
    if (request.source === 'train' || request.source === 'test') {
        const points = request.source === 'train' ? state.trainPoints : state.testPoints;
        const index = request.index ?? 0;
        if (!Number.isInteger(index) || index < 0 || index >= points.length) {
            throw new RangeError(`${request.source} sample index is out of range`);
        }
        const point = points[index];
        return {
            source: request.source,
            index,
            x: point.x,
            y: point.y,
            label: point.label,
        };
    }

    if (request.source === 'custom') {
        const x = requireFiniteTraceValue(request.x, 'x');
        const y = requireFiniteTraceValue(request.y, 'y');
        const label = request.label === undefined
            ? undefined
            : requireFiniteTraceValue(request.label, 'label');
        return { source: 'custom', x, y, label };
    }

    throw new RangeError('trace source must be train, test, or custom');
}

function resolveBackpropPreviewBatch(): { inputs: number[][]; targets: number[][] } {
    if (!state.network || !state.trainingConfig) {
        throw new Error('Not initialized');
    }

    const n = state.trainInputs.length;
    if (n === 0) {
        throw new Error('No training samples are available for backprop preview');
    }

    const batchSize = state.trainingConfig.batchSize;
    const stepBefore = state.network.getStep();
    const numBatches = Math.ceil(n / batchSize);
    const batchSlot = stepBefore % numBatches;
    if (batchSlot === 0 && stepBefore > 0) {
        throw new Error('Backprop preview is unavailable at the epoch shuffle boundary; step once before previewing.');
    }

    const startIdx = batchSlot * batchSize;
    const endIdx = Math.min(startIdx + batchSize, n);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = startIdx; i < endIdx; i++) {
        const sampleIdx = state.shuffledIndices[i];
        inputs.push(state.trainInputs[sampleIdx]);
        targets.push(state.trainTargets[sampleIdx]);
    }

    return { inputs, targets };
}

function serializeLossLandscapeProbe(probe: LossLandscapeProbe): SerializableLossLandscapeProbe {
    return {
        gridSize: probe.gridSize,
        sampleCount: probe.sampleCount,
        radius: probe.radius,
        axisA: {
            parameter: { ...probe.axisA.parameter },
            offsets: [...probe.axisA.offsets],
        },
        axisB: {
            parameter: { ...probe.axisB.parameter },
            offsets: [...probe.axisB.offsets],
        },
        losses: Array.from(probe.losses),
        centerLoss: probe.centerLoss,
        minLoss: probe.minLoss,
        maxLoss: probe.maxLoss,
        best: { ...probe.best },
        summary: probe.summary,
    };
}

// ── Comlink API ──

async function initializeExperimentV2Now(
    request: unknown,
): Promise<WorkerExperimentResultV2> {
    try {
        let actionToken: number | null = null;
        let acceptedRequestId: number | null = null;
        const transaction = await experimentRequestGate.run(
            request,
            (prepared) => {
                if (actionToken === null || actionToken !== v2MutationSequence) {
                    throw new ExperimentTransactionError(
                        'stale-request',
                        '$.requestId',
                        `Request ${acceptedRequestId ?? 'unknown'} was superseded by a newer V2 action.`,
                        acceptedRequestId,
                    );
                }
                return commitV2Runtime(prepared);
            },
            (acceptedRequest) => {
                acceptedRequestId = acceptedRequest.requestId;
                actionToken = beginV2Mutation();
            },
        );
        return transaction.value;
    } catch (error) {
        postTransactionErrorV2(error);
        throw error;
    }
}

function stepExperimentV2Now(iterations: number): WorkerExperimentResultV2 {
    try {
        if (!Number.isSafeInteger(iterations)
            || iterations < 1
            || iterations > MAX_MANUAL_V2_STEP_ITERATIONS) {
            throw new RangeError(
                `V2 step iterations must be a safe integer from 1 to ${MAX_MANUAL_V2_STEP_ITERATIONS}`,
            );
        }
        requireV2Runtime();
        let latestLive: LiveTrainingSignal | undefined;
        for (let index = 0; index < iterations; index++) {
            latestLive = trainOneStepV2()?.liveSignal ?? latestLive;
        }
        const evaluation = forceAndPublishEvaluationV2('manual-step', false);
        const evidence = makeEvidenceV2(latestLive, evaluation);
        postEvidenceV2(evidence);
        const snapshot = computeSnapshot();
        return {
            snapshot,
            runId: state.runId,
            evidence,
            identities: state.prepared!.identities,
            ...directSnapshotArtifactBundle(snapshot),
        };
    } catch (error) {
        if (error instanceof TerminalDivergenceError) {
            postTerminalDivergenceForPhase(error, 'training');
        } else {
            postRuntimeErrorV2(error, 'training');
        }
        throw error;
    }
}

function resetExperimentV2Now(): WorkerExperimentResultV2 {
    try {
        const { prepared } = requireV2Runtime();
        beginV2Mutation();
        return commitV2Runtime(prepared);
    } catch (error) {
        postRuntimeErrorV2(error, 'runtime');
        throw error;
    }
}

function forceEvaluationV2Now(trigger: ForcedEvaluationTriggerV2): {
    readonly runId: number;
    readonly evidence: WorkerEvidenceMessageV2;
} {
    try {
        const allowed = new Set<ForcedEvaluationTriggerV2>([
            'manual-step',
            'pause',
            'checkpoint',
            'save',
            'stop-condition',
            'restore',
        ]);
        if (!allowed.has(trigger)) {
            throw new TypeError('Unsupported forced V2 evaluation trigger');
        }
        const evaluation = forceAndPublishEvaluationV2(trigger);
        return {
            runId: state.runId,
            evidence: makeEvidenceV2(undefined, evaluation),
        };
    } catch (error) {
        if (error instanceof TerminalDivergenceError) {
            postTerminalDivergenceV2(error, 'evaluation', 'evaluation-failed');
        } else {
            postRuntimeErrorV2(error, 'evaluation', 'evaluation-failed');
        }
        throw error;
    }
}

async function captureRunArtifactNow(
    metadata: CaptureRunArtifactRequestV2,
): Promise<ExperimentRunRecordV2> {
    try {
        const { prepared, history } = requireV2Runtime();
        const evaluation = forceAndPublishEvaluationV2('save');
        const capturedHistory = history.read();
        const candidate: ExperimentRunRecordV2 = {
            kind: 'nn-playground-run',
            schemaVersion: 2,
            ...metadata,
            recipe: prepared.document.recipe,
            recipeFingerprint: prepared.identities.recipeFingerprint,
            snapshot: {
                model: evaluation.model,
                evaluation,
                trendHistory: compactEvenly(
                    capturedHistory.trendHistory,
                    EXPERIMENT_MEMORY_MAX_TRENDS,
                ),
                evaluationHistory: compactEvenly(
                    capturedHistory.evaluationHistory,
                    EXPERIMENT_MEMORY_MAX_EVALUATIONS,
                ),
            },
        };
        const validated = await validateExperimentRunRecordV2(candidate);
        if (!validated.ok) {
            throw new TypeError(validated.issues.map(
                (entry) => `${entry.path}: ${entry.message}`,
            ).join('; '));
        }
        return validated.value;
    } catch (error) {
        postRuntimeErrorV2(error, 'persistence', 'capture-failed');
        throw error;
    }
}

export const workerApi = {
    /** Strict application boundary: validate, re-prepare, verify identities, then commit. */
    initializeExperimentV2(request: unknown): Promise<WorkerExperimentResultV2> {
        return enqueueV2Mutation(() => initializeExperimentV2Now(request));
    },

    /** Manual stepping always finishes with a current same-revision pair. */
    stepExperimentV2(iterations: number = 1): Promise<WorkerExperimentResultV2> {
        return enqueueV2Mutation(() => stepExperimentV2Now(iterations));
    },

    /** Rebuild the exact current prepared document as a fresh generation. */
    resetExperimentV2(): Promise<WorkerExperimentResultV2> {
        return enqueueV2Mutation(resetExperimentV2Now);
    },

    forceEvaluationV2(trigger: ForcedEvaluationTriggerV2): Promise<{
        readonly runId: number;
        readonly evidence: WorkerEvidenceMessageV2;
    }> {
        return enqueueV2Mutation(() => forceEvaluationV2Now(trigger));
    },

    /**
     * Capture recipe and scientific evidence in one worker-owned boundary.
     * Every mutable read completes before asynchronous fingerprint validation
     * yields, so a later command cannot be mixed into this saved artifact.
     */
    captureRunArtifact(request: unknown): Promise<ExperimentRunRecordV2> {
        let metadata: CaptureRunArtifactRequestV2;
        try {
            metadata = parseCaptureRunArtifactRequestV2(request);
        } catch (error) {
            postRuntimeErrorV2(error, 'persistence', 'capture-failed');
            return Promise.reject(error);
        }
        return enqueueV2Mutation(() => captureRunArtifactNow(metadata));
    },

    getMetricHistoryV2(): RuntimeMetricHistorySnapshot {
        return requireV2Runtime().history.read();
    },

    initialize(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
    ): { snapshot: NetworkSnapshot; runId: number } {
        const config = normalizeWorkerConfig(networkConfig, trainingConfig, dataConfig, features);
        beginV2Mutation();
        stopInternalLoop();
        state.networkConfig = { ...config.network };
        state.trainingConfig = { ...config.training };
        state.dataConfig = { ...config.data };
        state.features = { ...config.features };
        state.running = false;
        buildDataAndNetwork();
        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    updateConfig(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
        rebuild: boolean,
    ): { snapshot: NetworkSnapshot; runId: number } {
        assertLegacyModelMutationAllowed('updateConfig');
        const config = normalizeWorkerConfig(networkConfig, trainingConfig, dataConfig, features);
        beginV2Mutation();
        const needsRebuild = rebuild ||
            !configsEqual(state.networkConfig, config.network) ||
            !configsEqual(state.dataConfig, config.data) ||
            !configsEqual(state.features, config.features);

        state.networkConfig = { ...config.network };
        state.trainingConfig = { ...config.training };
        state.dataConfig = { ...config.data };
        state.features = { ...config.features };

        if (needsRebuild) {
            stopInternalLoop();
            state.running = false;
            buildDataAndNetwork();
        }

        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    step(iterations: number = 1): NetworkSnapshot {
        assertLegacyModelMutationAllowed('step');
        for (let i = 0; i < iterations; i++) {
            trainOneStep();
        }
        return computeSnapshot();
    },

    initializeArena(request: InitializeArenaRequest): ArenaScalarSnapshot {
        arenaState.slots = [
            buildArenaSlot('A', request.modelA),
            buildArenaSlot('B', request.modelB),
        ];
        arenaState.runId++;
        arenaState.snapshotId = 0;
        return buildArenaSnapshot();
    },

    stepArena(iterations: number = 1): ArenaScalarSnapshot {
        if (!Number.isInteger(iterations) || iterations < 1) {
            throw new RangeError('arena iterations must be a positive integer');
        }
        if (!arenaState.slots) {
            throw new Error('Arena is not initialized');
        }

        for (let i = 0; i < iterations; i++) {
            for (const slot of arenaState.slots) {
                if (slot.status !== 'paused') {
                    slot.status = 'running';
                    trainArenaSlot(slot);
                }
            }
        }

        for (const slot of arenaState.slots) {
            if (slot.status === 'running') {
                slot.status = 'paused';
                slot.pauseReason = null;
            }
        }

        return buildArenaSnapshot();
    },

    reset(): { snapshot: NetworkSnapshot; runId: number } {
        assertLegacyModelMutationAllowed('reset');
        beginV2Mutation();
        stopInternalLoop();
        buildDataAndNetwork();
        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    getCheckpointTimeline(): CheckpointTimeline {
        return buildCheckpointTimeline();
    },

    restoreCheckpoint(id: number): { snapshot: NetworkSnapshot; runId: number; timeline: CheckpointTimeline } {
        assertLegacyModelMutationAllowed('restoreCheckpoint');
        if (!Number.isInteger(id) || id <= 0) {
            throw new RangeError('checkpoint id must be a positive integer');
        }
        if (!state.network) {
            throw new Error('Not initialized');
        }

        const entry = state.checkpoints.find((candidate) => candidate.summary.id === id);
        if (!entry) {
            throw new RangeError('checkpoint not found');
        }

        beginV2Mutation();
        stopInternalLoop();
        state.network.restoreCheckpoint(entry.checkpoint);
        state.epoch = entry.epoch;
        state.shuffledIndices = [...entry.shuffledIndices];
        state.lossEma = entry.lossEma;
        state.lastTrainMetrics = cloneMetrics(entry.lastTrainMetrics);
        state.lastTestMetrics = cloneMetrics(entry.lastTestMetrics);
        state.snapshotsSinceLastTrainEval = entry.snapshotsSinceLastTrainEval;
        state.snapshotsSinceLastTestEval = entry.snapshotsSinceLastTestEval;
        state.snapshotsSinceLastGrid = entry.snapshotsSinceLastGrid;
        state.snapshotsSinceLastActivationHistogram = entry.snapshotsSinceLastActivationHistogram;
        state.restoredCheckpointId = id;
        state.gridStale = true;
        state.gridFreshFromGpu = false;
        state.confusionMatrixVersion++;
        state.activationHistogramVersion++;
        state.layerStatsGradientRevision = null;
        resetAck();

        const snapshot = computeSnapshot();
        return {
            snapshot,
            runId: state.runId,
            timeline: buildCheckpointTimeline(),
        };
    },

    getTrainPoints(): DataPoint[] {
        return state.trainPoints;
    },

    getTestPoints(): DataPoint[] {
        return state.testPoints;
    },

    getPredictionTrace(request: PredictionTraceRequest): PredictionTraceResponse {
        if (!state.network || !state.trainingConfig || !state.features) {
            throw new Error('Not initialized');
        }

        const sample = resolveTraceSample(request);
        const input = transformPoint(sample.x, sample.y, state.activeFeatures);
        const target = sample.label === undefined
            ? undefined
            : encodeTargetLabel(sample.label, state.networkConfig?.outputSize ?? 1);
        const trace = state.network.tracePrediction(
            input,
            target,
            target ? state.trainingConfig.lossType : undefined,
            state.trainingConfig.huberDelta,
        );

        return {
            runId: state.runId,
            step: state.network.getStep(),
            sample,
            trace,
        };
    },

    getBackpropExplanation(): BackpropExplanationResponse {
        if (!state.network || !state.trainingConfig) {
            throw new Error('Not initialized');
        }

        const batch = resolveBackpropPreviewBatch();
        return {
            runId: state.runId,
            step: state.network.getStep(),
            epoch: state.epoch,
            explanation: state.network.explainBackpropStep(
                batch.inputs,
                batch.targets,
                state.trainingConfig,
            ),
        };
    },

    getLossLandscapeProbe(options: LossLandscapeProbeOptions = {}): LossLandscapeProbeResponse {
        if (!state.network || !state.trainingConfig) {
            throw new Error('Not initialized');
        }

        const probe = state.network.probeLossLandscape(
            state.trainInputs,
            state.trainTargets,
            state.trainingConfig,
            options,
        );

        return {
            runId: state.runId,
            step: state.network.getStep(),
            epoch: state.epoch,
            probe: serializeLossLandscapeProbe(probe),
        };
    },

    /** Update what visual data the UI currently needs. */
    updateDemand(demand: VisualizationDemand): void {
        const normalized = normalizeVisualizationDemand(demand);
        if (!normalized) {
            throw new Error('Invalid visualization demand.');
        }
        applyDemand(normalized);
    },

    /**
     * Toggle the AS-4 WebGPU grid path. Disabling immediately disposes the
     * predictor (frees GPU memory); enabling lazily re-allocates on the
     * next snapshot. No-op when called with the current value.
     */
    setWebGpuEnabled(enabled: boolean): void {
        if (state.gpuEnabled === enabled) return;
        state.gpuEnabled = enabled;
        if (!enabled && state.gpuPredictor) {
            try { state.gpuPredictor.dispose(); } catch { /* ignore */ }
            state.gpuPredictor = null;
            // Force the next snapshot to recompute grids on the CPU so the
            // UI doesn't keep reading stale GPU output.
            state.gridStale = true;
            state.gridFreshFromGpu = false;
        }
    },

    /** Accept a MessagePort from the main thread for streaming. */
    setStreamPort(port: MessagePort): void {
        state.streamPort = port;
        port.addEventListener('message', (event: MessageEvent<unknown>) => {
            handleStreamCommand(event.data);
        });
        port.start();
        // If SABs were allocated before the port was connected (typical at
        // init), deliver the handshake now so the main thread can install
        // its views before the first snapshot arrives.
        postSharedBuffersHandshake();
    },
};

export type TrainingWorkerApi = typeof workerApi;

Comlink.expose(workerApi);
