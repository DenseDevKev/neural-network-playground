// ── Training Web Worker ──
// Owns the engine instance, runs training off the main thread.
// Hybrid communication:
//   - Comlink RPC for strict version-2 experiment commands
//   - MessagePort commands for high-frequency streamed snapshots during training

import * as Comlink from 'comlink';
import {
    Network,
    NonFiniteNumericalError,
    PRNG,
    buildGridInputs,
    generateDatasetV2,
    getDatasetContract,
    getActiveFeatures,
    transformPoint,
    transformDataset,
    detectWebGPU,
    WebGPUGridPredictor,
    exceedsGpuShape,
    flattenGridInputs,
} from '@nn-playground/engine';
import type {
    NetworkConfig,
    NetworkSnapshot,
    DataPoint,
    ActivationHistogramResult,
    LossLandscapeProbeOptions,
    CompiledExperimentConfig,
    CompiledTaskContract,
    MulticlassConfusionMatrixData,
    PredictionTraceV2,
    BackpropExplanationV2,
    ObjectiveLandscapeProbe,
} from '@nn-playground/engine';
import {
    GRID_SIZE,
    DEFAULT_DEMAND,
    isMainToWorkerCommand,
    normalizeVisualizationDemand,
    structuralEqual,
    parseWorkerEvidenceMessageV2,
    parseWorkerProtocolErrorMessageV2,
    parseCaptureRunArtifactRequestV2,
    parseMainToWorkerRequestV2,
    parseCheckpointTimelineV2,
    validateSessionCheckpointV2,
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
    WorkerSharedBuffersMessage,
    CheckpointTimeline,
    CheckpointSummary,
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
    SessionCheckpointV2,
    CaptureCheckpointRequestV2,
    RestoreCheckpointRequestV2,
    ModelRevision,
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
    getTrainingStepsForTick,
    normalizeTrainingSpeed,
} from './trainingLoop.ts';
import {
    EvaluationRuntime,
    TerminalDivergenceError,
    type PreparedCadenceEvaluation,
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
import { exposeWorkerApiAndAnnounceReady } from './workerReadiness.ts';

interface WorkerState {
    /** Canonical V2 preparation. Null means the worker is not initialized. */
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
    /** Mutable network cell keeps V2 evaluation callbacks valid across atomic restores. */
    v2NetworkRef: { value: Network } | null;
    network: Network | null;
    networkConfig: NetworkConfig | null;
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
    /** Shuffled index array — re-shuffled at the start of each epoch. */
    shuffledIndices: number[];
    /** Authoritative start offset of the next strict V2 batch. */
    batchStart: number;
    /** Separate PRNG for epoch shuffling, independent from network weights. */
    shufflePrng: PRNG | null;
    /** What visual data the UI currently needs. */
    demand: VisualizationDemand;
    /** Counter for grid-rebuild frequency gating. */
    snapshotsSinceLastGrid: number;
    /** Counter for activation histogram frequency gating. */
    snapshotsSinceLastActivationHistogram: number;
    /** Runtime-only bounded checkpoint ring buffer. Heavy model state stays in the worker. */
    checkpoints: V2RuntimeCheckpoint[];
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

interface V2RuntimeCheckpoint {
    kind: 'v2';
    summary: CheckpointSummary;
    checkpoint: SessionCheckpointV2;
}

export type PredictionTraceSampleSource = 'train' | 'test';

export interface PredictionTraceRequest {
    source: PredictionTraceSampleSource;
    index: number;
}

export interface PredictionTraceResponseV2 {
    readonly runId: number;
    readonly model: ModelRevision;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly sample: {
        readonly source: 'train' | 'test';
        readonly index: number;
        readonly x: number;
        readonly y: number;
        readonly label: number;
    };
    readonly trace: PredictionTraceV2;
}

export interface BackpropExplanationResponseV2 {
    readonly runId: number;
    readonly model: ModelRevision;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly basis: {
        readonly kind: 'next-mini-batch';
        readonly sampleCount: number;
        readonly populationCount: number;
    };
    readonly explanation: BackpropExplanationV2;
}

export interface SerializableObjectiveLandscapeProbe {
    readonly basis: 'training-objective';
    readonly gridSize: number;
    readonly sampleCount: number;
    readonly parameterPositionCount: number;
    readonly radius: number;
    readonly axisA: ObjectiveLandscapeProbe['axisA'];
    readonly axisB: ObjectiveLandscapeProbe['axisB'];
    readonly objectives: number[];
    readonly centerObjective: number;
    readonly minObjective: number;
    readonly maxObjective: number;
    readonly best: ObjectiveLandscapeProbe['best'];
    readonly summary: string;
}

export interface ObjectiveLandscapeResponseV2 {
    readonly runId: number;
    readonly model: ModelRevision;
    readonly provenance: ArtifactProvenance & {
        readonly basis: Extract<ArtifactBasis, { kind: 'parameter-grid' }>;
    };
    readonly probe: SerializableObjectiveLandscapeProbe;
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
        protocolVersion: WORKER_PROTOCOL_VERSION,
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


function postStatus(status: WorkerStatusMessage['status'], pauseReason?: PauseReason | null): void {
    if (!state.streamPort) return;
    const statusMsg: WorkerStatusMessage = {
        type: 'status',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        runId: state.runId,
        status,
    };
    if (pauseReason !== undefined) {
        statusMsg.pauseReason = pauseReason;
    }
    if (status === 'paused') {
        statusMsg.checkpointTimeline = buildCheckpointTimeline();
    }
    state.streamPort.postMessage(statusMsg);
}

const WORKER_MULTICLASS_OUTPUT_SIZE = 3;
const V2_LIVE_LOSS_EMA_ALPHA = 0.1;

const WORKER_PERF_ENABLED = import.meta.env.DEV && import.meta.env.VITE_WORKER_PERF === '1';
const ACTIVATION_HISTOGRAM_BIN_COUNT = 12;
const ACTIVATION_HISTOGRAM_MAX_SAMPLES = 128;
const CHECKPOINT_MAX_COUNT = 8;

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
    v2NetworkRef: null,
    network: null,
    networkConfig: null,
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
    shuffledIndices: [],
    batchStart: 0,
    shufflePrng: null,
    demand: { ...DEFAULT_DEMAND },
    snapshotsSinceLastGrid: 0,
    snapshotsSinceLastActivationHistogram: 0,
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

function buildCheckpointTimeline(): CheckpointTimeline {
    const timeline = {
        checkpoints: state.checkpoints.map((entry) => ({ ...entry.summary })),
        maxCheckpoints: CHECKPOINT_MAX_COUNT,
        evictedCount: state.checkpointEvictedCount,
        liveCheckpointId: state.checkpoints.at(-1)?.summary.id ?? null,
        restoredCheckpointId: state.restoredCheckpointId,
    };
    return parseCheckpointTimelineV2(timeline);
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
    readonly checkpointTimeline: CheckpointTimeline;
}

function checkpointValidationContext() {
    const { prepared, compiled, datasetRevision } = state;
    if (!prepared || !compiled || !datasetRevision) {
        throw new Error('V2 experiment is not initialized');
    }
    return {
        recipeFingerprint: prepared.identities.recipeFingerprint,
        objectiveKey: prepared.identities.objectiveKey,
        dataset: datasetRevision,
        layerSizes: [
            compiled.network.inputSize,
            ...compiled.network.hiddenLayers,
            compiled.network.outputSize,
        ],
        optimizer: compiled.training.optimizer,
    };
}

function prepareV2CheckpointEntry(evaluation: PairedEvaluation): V2RuntimeCheckpoint {
    const { prepared, compiled, datasetRevision, network } = state;
    if (!prepared || !compiled || !datasetRevision || !network) {
        throw new Error('V2 experiment is not initialized');
    }
    if (evaluation.model.generationId !== state.runId) {
        throw new RangeError('checkpoint model generation must equal the active generation');
    }
    const session = network.captureSessionState(compiled.training.optimizer);
    const validated = validateSessionCheckpointV2({
        kind: 'nn-playground-session-checkpoint',
        schemaVersion: 2,
        recipeFingerprint: prepared.identities.recipeFingerprint,
        objectiveKey: prepared.identities.objectiveKey,
        datasetKey: datasetRevision.datasetKey,
        model: evaluation.model,
        evaluation,
        network: session.network,
        optimizer: session.optimizer,
        cursor: {
            epoch: state.epoch,
            batchStart: state.batchStart,
            shuffledIndices: Uint32Array.from(state.shuffledIndices),
        },
        trajectoryGuarantee: 'parameters-and-optimizer-only',
    }, checkpointValidationContext());

    const id = state.nextCheckpointId;
    const summary: CheckpointSummary = {
        id,
        step: evaluation.model.step,
        epoch: evaluation.model.epoch,
        trainDataLoss: evaluation.train.values.dataLoss,
        testDataLoss: evaluation.test.values.dataLoss,
        ...(evaluation.train.values.accuracy === undefined
            ? {}
            : { trainAccuracy: evaluation.train.values.accuracy }),
        ...(evaluation.test.values.accuracy === undefined
            ? {}
            : { testAccuracy: evaluation.test.values.accuracy }),
        label: `Step ${evaluation.model.step}`,
    };
    const entry: V2RuntimeCheckpoint = { kind: 'v2', summary, checkpoint: validated };

    return entry;
}

function commitV2CheckpointEntry(entry: V2RuntimeCheckpoint): void {
    state.nextCheckpointId++;
    state.checkpoints.push(entry);
    while (state.checkpoints.length > CHECKPOINT_MAX_COUNT) {
        const evicted = state.checkpoints.shift();
        state.checkpointEvictedCount++;
        if (evicted && state.restoredCheckpointId === evicted.summary.id) {
            state.restoredCheckpointId = null;
        }
    }
}

function appendV2Checkpoint(evaluation: PairedEvaluation): V2RuntimeCheckpoint {
    const entry = prepareV2CheckpointEntry(evaluation);
    // The ring changes only after the entire envelope has validated and cloned.
    commitV2CheckpointEntry(entry);
    return entry;
}

function forceAndCaptureCheckpointV2(): PairedEvaluation {
    const { runtime, history } = requireV2Runtime();
    const preparedEvaluation = runtime.prepareCheckpointEvaluation();
    const entry = prepareV2CheckpointEntry(preparedEvaluation.evaluation);
    const evidence = makeEvidenceV2(undefined, preparedEvaluation.evaluation);

    beginV2Mutation();
    const evaluation = preparedEvaluation.commit();
    history.appendEvaluation(evaluation);
    commitV2CheckpointEntry(entry);
    postEvidenceV2(evidence);
    return evaluation;
}

function captureCheckpointFromCadenceV2(
    preparedCadence: PreparedCadenceEvaluation,
): PairedEvaluation {
    const { history } = requireV2Runtime();
    const checkpointEvaluation: PairedEvaluation = {
        ...preparedCadence.evaluation,
        trigger: 'checkpoint',
    };
    const entry = prepareV2CheckpointEntry(checkpointEvaluation);
    const cadenceEvidence = makeEvidenceV2(undefined, preparedCadence.evaluation);

    beginV2Mutation();
    const cadenceEvaluation = preparedCadence.commit();
    history.appendEvaluation(cadenceEvaluation);
    commitV2CheckpointEntry(entry);
    postEvidenceV2(cadenceEvidence);
    return cadenceEvaluation;
}

function resolvePendingCadenceBeforeTrainingV2(
    runtime: EvaluationRuntime,
    history: RuntimeMetricHistory,
): PairedEvaluation | undefined {
    const preparedCadence = runtime.prepareCadenceEvaluation();
    if (preparedCadence === undefined) return undefined;
    const pendingStep = preparedCadence.evaluation.model.step;
    if (pendingStep > 0 && pendingStep % runtime.policy.everySteps === 0) {
        return captureCheckpointFromCadenceV2(preparedCadence);
    }

    const evidence = makeEvidenceV2(undefined, preparedCadence.evaluation);
    beginV2Mutation();
    const cadenceEvaluation = preparedCadence.commit();
    history.appendEvaluation(cadenceEvaluation);
    postEvidenceV2(evidence);
    return cadenceEvaluation;
}

/** Narrow module-only seam for validating worker-private checkpoint transactions. */
export function getV2CheckpointForTests(id: number): SessionCheckpointV2 {
    const entry = state.checkpoints.find((candidate) => candidate.summary.id === id);
    if (!entry || entry.kind !== 'v2') throw new RangeError('V2 checkpoint not found');
    return structuredClone(entry.checkpoint);
}

/** Module-only checkpoint timeline seam for strict V2 runtime tests. */
export function getV2CheckpointTimelineForTests(): CheckpointTimeline {
    return buildCheckpointTimeline();
}

/** Narrow module-only corruption seam; checkpoint payload never enters the product RPC surface. */
export function replaceV2CheckpointForTests(id: number, checkpoint: unknown): void {
    const entry = state.checkpoints.find((candidate) => candidate.summary.id === id);
    if (!entry || entry.kind !== 'v2') throw new RangeError('V2 checkpoint not found');
    entry.checkpoint = checkpoint as SessionCheckpointV2;
}

interface V2RuntimeBuild {
    readonly prepared: PreparedExperimentDocumentV2;
    readonly compiled: CompiledExperimentConfig;
    readonly network: Network;
    readonly datasetRevision: DatasetRevision;
    readonly runtime: EvaluationRuntime;
    readonly history: RuntimeMetricHistory;
    readonly epochRef: { value: number };
    readonly networkRef: { value: Network };
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
    readonly batchStart: number;
    readonly shufflePrng: PRNG;
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
const clonePredictionTraceBoundary = typeof globalThis.structuredClone === 'function'
    ? globalThis.structuredClone.bind(globalThis)
    : null;

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
        || !state.v2EpochRef
        || !state.v2NetworkRef
        || state.v2NetworkRef.value !== state.network) {
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

interface EvaluationNetworkInputs {
    compiled: CompiledExperimentConfig;
    getNetwork: () => Network;
    trainInputs: number[][];
    trainTargets: number[][];
    testInputs: number[][];
    testTargets: number[][];
}

function createEvaluationComputationForNetwork(args: EvaluationNetworkInputs) {
    let cachedTrainObjective: {
        generationId: number;
        revision: number;
        regularizationPenalty: number;
    } | null = null;
    return {
        evaluateTrain: (model: PairedEvaluation['model']) => withEngineNumericalBoundary(
            '$.evaluation.train',
            () => {
                const network = args.getNetwork();
                const objective = network.evaluateObjective(
                    args.trainInputs,
                    args.trainTargets,
                    args.compiled.objective,
                );
                cachedTrainObjective = {
                    generationId: model.generationId,
                    revision: model.revision,
                    regularizationPenalty: objective.regularizationPenalty,
                };
                return withTaskMetricsV2(
                    network,
                    args.trainInputs,
                    args.trainTargets,
                    args.compiled.task,
                    objective.dataLoss,
                );
            },
        ),
        evaluateTest: () => withEngineNumericalBoundary('$.evaluation.test', () => {
            const network = args.getNetwork();
            return withTaskMetricsV2(
                network,
                args.testInputs,
                args.testTargets,
                args.compiled.task,
                network.evaluateDataLoss(
                    args.testInputs,
                    args.testTargets,
                    args.compiled.objective,
                ),
            );
        }),
        evaluateRegularizationPenalty: (model: PairedEvaluation['model']) => {
            if (cachedTrainObjective === null
                || cachedTrainObjective.generationId !== model.generationId
                || cachedTrainObjective.revision !== model.revision) {
                throw new Error('train objective penalty is not cached for this model revision');
            }
            const penalty = cachedTrainObjective.regularizationPenalty;
            cachedTrainObjective = null;
            return penalty;
        },
    };
}

function createEvaluationRuntimeForNetwork(args: EvaluationNetworkInputs & {
    generationId: number;
    objectiveKey: string;
    dataset: DatasetRevision;
    getCurrentModel: () => {
        generationId: number;
        revision: number;
        step: number;
        epoch: number;
    };
}): EvaluationRuntime {
    return new EvaluationRuntime({
        generationId: args.generationId,
        dataset: args.dataset,
        objectiveKey: args.objectiveKey,
        emaAlpha: V2_LIVE_LOSS_EMA_ALPHA,
        getCurrentModel: args.getCurrentModel,
        ...createEvaluationComputationForNetwork(args),
    });
}

function stageV2Runtime(
    prepared: PreparedExperimentDocumentV2,
    generationId: number,
): V2RuntimeBuild {
    // This counter is intentionally inside the post-validation commit path.
    v2AllocationCount++;
    const compiled = prepared.compiled;
    const split = generateDatasetV2({ ...compiled.data });
    const activeFeatures = getActiveFeatures(compiled.features);
    const trainInputs = transformDataset(split.train, activeFeatures);
    const trainTargets = encodeTargets(split.train, compiled.task.outputSize);
    const testInputs = transformDataset(split.test, activeFeatures);
    const testTargets = encodeTargets(split.test, compiled.task.outputSize);
    const gridInputs = buildGridInputs(GRID_SIZE, activeFeatures);
    const network = new Network(compiled.network, compiled.network.seed);
    const networkRef = { value: network };
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

    const runtime = createEvaluationRuntimeForNetwork({
        generationId,
        dataset: datasetRevision,
        objectiveKey: prepared.identities.objectiveKey,
        compiled,
        getNetwork: () => networkRef.value,
        trainInputs,
        trainTargets,
        testInputs,
        testTargets,
        getCurrentModel: () => ({
            generationId,
            revision: networkRef.value.getRevision(),
            step: networkRef.value.getStep(),
            epoch: epochRef.value,
        }),
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
        network,
        datasetRevision,
        runtime,
        history,
        epochRef,
        networkRef,
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
        batchStart: 0,
        shufflePrng,
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
    state.v2NetworkRef = staged.networkRef;
    state.network = staged.network;
    state.networkConfig = { ...staged.compiled.network };
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
    state.batchStart = staged.batchStart;
    state.shufflePrng = staged.shufflePrng;
    state.outputGridBuffer = staged.outputGridBuffer;
    state.neuronGridsBuffer = staged.neuronGridsBuffer;
    state.multiclassClassGridBuffer = staged.multiclassClassGridBuffer;
    state.multiclassConfidenceGridBuffer = staged.multiclassConfidenceGridBuffer;
    state.sharedViews = staged.sharedViews;
    state.epoch = 0;
    state.batchStart = 0;
    state.running = false;
    clearTrainLoopTimer();
    state.runId = generationId;
    state.snapshotId = 0;
    // A fresh strict generation must return its initially demanded boundary
    // artifacts immediately; subsequent snapshots resume normal cadence.
    state.snapshotsSinceLastGrid = state.demand.gridInterval;
    state.snapshotsSinceLastActivationHistogram = state.demand.needActivationHistograms
        ? state.demand.activationHistogramInterval
        : 0;
    state.multiclassBoundaryFresh = false;
    state.gridStale = true;
    state.gridFreshFromGpu = false;
    state.stopConditionState = createInitialStopConditionState();
    state.confusionMatrixVersion++;
    state.lastPackedConfusionEvaluationId = null;
    state.activationHistogramVersion++;
    state.layerStatsGradientRevision = null;
    resetCheckpoints();
    const initialEvaluation = staged.initialEvidence.latestEvaluation;
    if (initialEvaluation === undefined) {
        throw new Error('initial V2 evidence must include a paired evaluation');
    }
    appendV2Checkpoint(initialEvaluation);
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
        checkpointTimeline: buildCheckpointTimeline(),
        ...directSnapshotArtifactBundle(snapshot),
    };
}

// Top-level error backstops keep every worker failure on the structured V2
// error channel, including failures that happen before initialization.
self.addEventListener('error', (event: ErrorEvent) => {
    postRuntimeErrorV2(
        new Error(`Unhandled worker error: ${event.message ?? String(event)}`),
        'runtime',
    );
});

self.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    postRuntimeErrorV2(new Error(`Unhandled worker rejection: ${reason}`), 'runtime');
});



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

/**
 * @param opts.lightweight — when true, skips the deep-copy of weights/biases
 * into the snapshot. The streaming path transfers flat buffers separately
 * (see packSnapshotMessage), so nested copies are pure waste there.
 */
function computeSnapshotUnchecked(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
    workerPerfMark('perf:worker:snapshot:start');
    const { network, runtime } = requireV2Runtime();
    const evaluation = runtime.latestEvaluation;
    if (evaluation === undefined) throw new Error('V2 snapshot requires a paired evaluation');

    const { demand } = state;
    const trainMetrics = {
        loss: evaluation.train.values.dataLoss,
        ...(evaluation.train.values.accuracy === undefined
            ? {}
            : { accuracy: evaluation.train.values.accuracy }),
    };
    const testMetrics = {
        loss: evaluation.test.values.dataLoss,
        ...(evaluation.test.values.accuracy === undefined
            ? {}
            : { accuracy: evaluation.test.values.accuracy }),
    };

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
    const wantMulticlassBoundary = supportsMulticlassBoundary && (demand.needDecisionBoundary || demand.needNeuronGrids);
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
        network.predictGridWithNeuronsInto(
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
        network.predictGridInto(state.gridInputs, state.outputGridBuffer);
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
        if (demand.needNeuronGrids && state.outputGridBuffer && state.neuronGridsBuffer) {
            network.predictGridWithNeuronsInto(state.gridInputs, state.outputGridBuffer, state.neuronGridsBuffer);
            neuronGrids = state.neuronGridsBuffer;
        }
        if (demand.needDecisionBoundary) network.predictMulticlassBoundaryInto(
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
        state.multiclassBoundaryFresh = demand.needDecisionBoundary;
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

    const snap = network.getSnapshot(
        network.getStep(),
        state.epoch,
        trainMetrics,
        testMetrics,
        outputGrid,
        GRID_SIZE,
        { includeParams: !opts.lightweight },
    );
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
        const statistics = network.computeLayerStatistics(state.trainInputs, 128);
        if (statistics.sampleCount !== sampleCount
            || statistics.populationCount !== populationCount) {
            throw new Error(
                'Layer statistics sample basis does not match the training population.',
            );
        }
        if (statistics.revision !== network.getRevision()) {
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
        snap.activationHistograms = network.computeActivationHistograms(
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

    const pairedMatrix = demand.needConfusionMatrix
        ? evaluation.test.values.confusionMatrix
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

    // Prove the display transport can represent every parameter before
    // evidence/frame publication; Float64 values may overflow Float32.
    network.getWeightsFlat();
    network.getBiasesFlat();
    workerPerfMeasure('perf:worker:snapshot', 'perf:worker:snapshot:start');
    return snap;
}

function computeSnapshot(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
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
    requireV2Runtime();
    if (!state.datasetRevision) {
        throw new Error('strict direct snapshot requires a dataset revision');
    }
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
    // from the transferable postMessage path. When SAB is active we always fed
    // the Float32Array pre-alloc buffers into the network predictors, so
    // the `instanceof Float32Array` branches are the only ones that fire.
    // The existing shared protocol carries a scalar output plane. Multiclass
    // neuron maps travel inline alongside their separate winning-class field.
    const sharedViews = state.networkConfig?.outputSize === 1 ? state.sharedViews : null;
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
        // Transferable postMessage path keeps non-isolated hosts (GH Pages
        // and test runners) fully functional without SharedArrayBuffer.
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
        transferables.push(outputGrid.buffer);
    }

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

    let artifacts: WorkerArtifactProvenanceV2 | undefined;
    if (state.datasetRevision) {
        const produced: WorkerArtifactProvenanceV2 = {
            ...((outputGrid !== undefined && outputGrid.length > 0)
                || multiclassClassGrid !== undefined
                || (sharedSeq !== undefined && snap.outputGrid.length > 0)
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

    const messageBase = {
        type: 'snapshot',
        runId: state.runId,
        snapshotId: ++state.snapshotId,
        scalars: {
            step: snap.step,
            epoch: snap.epoch,
            gridSize,
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
        artifacts,
        confusionMatrix,
        confusionMatrixEvaluationId,
        confusionMatrixVersion: state.confusionMatrixVersion,
        multiclassConfusionMatrix,
        multiclassConfusionMatrixVersion,
        sharedSeq,
    } as const;
    if (!state.network) throw new Error('strict snapshot requires an active network');
    const message: WorkerSnapshotMessage = {
        ...messageBase,
        protocolVersion: WORKER_PROTOCOL_VERSION,
        model: {
            generationId: state.runId,
            revision: state.network.getRevision(),
            step: state.network.getStep(),
            epoch: state.epoch,
        },
        recipeFingerprint: state.prepared!.identities.recipeFingerprint,
        checkpointTimeline: buildCheckpointTimeline(),
    };

    workerPerfMeasure('perf:worker:snapshotPack', 'perf:worker:snapshotPack:start');
    return { message, transferables };
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
    let cadenceEvaluation = resolvePendingCadenceBeforeTrainingV2(runtime, history);
    const batchSize = compiled.training.batchSize;
    const sampleCount = state.trainInputs.length;
    if (sampleCount === 0) return undefined;

    if (state.batchStart === sampleCount) {
        // The boundary sentinel already records the completed epoch. Future
        // shuffle PRNG state is intentionally not checkpointed, so normalize
        // only when the next batch is actually requested.
        state.batchStart = 0;
        state.shufflePrng?.shuffle(state.shuffledIndices);
    }
    if (state.batchStart < 0 || state.batchStart >= sampleCount) {
        throw new RangeError('V2 batch cursor is outside the training population');
    }
    const start = state.batchStart;
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

    state.batchStart = end;
    if (state.batchStart === sampleCount) {
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
        dataLoss: result.postUpdateDataLoss,
    });
    history.appendLiveSignal(liveSignal);
    state.gridStale = true;

    const checkpointDue = result.step > 0
        && result.step % runtime.policy.everySteps === 0;
    const preparedCadence = runtime.prepareCadenceEvaluation();
    if (preparedCadence !== undefined) {
        if (checkpointDue) {
            cadenceEvaluation = captureCheckpointFromCadenceV2(preparedCadence);
        } else {
            cadenceEvaluation = preparedCadence.commit();
            history.appendEvaluation(cadenceEvaluation);
            // Cadence evidence is never coalesced into a latest-wins visual frame.
            postEvidenceV2(makeEvidenceV2(undefined, cadenceEvaluation));
        }
    }
    return cadenceEvaluation === undefined
        ? { liveSignal }
        : { liveSignal, cadenceEvaluation };
}

// ── Internal training loop (worker-driven) ──

const TRAIN_TICK_INTERVAL_MS = 1000 / 60;

/**
 * Cancel a pending training tick. Generation swaps and restores must cancel
 * (not merely drop) the outstanding handle, otherwise the orphaned callback
 * fires later and clobbers the next generation's timer cell.
 */
function clearTrainLoopTimer(): void {
    if (state.trainLoopTimer !== null) {
        clearTimeout(state.trainLoopTimer);
        state.trainLoopTimer = null;
    }
}

function scheduleNextTick(): void {
    if (state.trainLoopTimer !== null || !state.running) return;
    state.trainLoopTimer = setTimeout(() => {
        state.trainLoopTimer = null;
        if (!state.running) return;
        // Timer-driven V2 batches share the same scientific mutation lane as
        // RPC capture/reset/step so an awaited capture stays exclusive.
        void enqueueV2Mutation(() => trainTick());
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
    if (state.running && state.network && state.evaluationRuntime) {
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
    const { message, transferables } = withEngineNumericalBoundary(
        '$.snapshot.parameters',
        () => packSnapshotMessage(snap),
    );
    state.streamPort.postMessage(message, transferables);
    const pauseReason = stopEvaluation?.pauseReason ?? null;
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
                trainOneStepV2();
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
                    if (err instanceof TerminalDivergenceError) {
                        postTerminalDivergenceForPhase(err, 'evaluation');
                    } else {
                        postRuntimeErrorV2(err, 'runtime');
                    }
                });
            }
        }

        if (state.running) {
            scheduleNextTick();
        }
    } catch (err) {
        stopInternalLoop();
        if (err instanceof TerminalDivergenceError) {
            postTerminalDivergenceForPhase(err, 'training');
        } else {
            postRuntimeErrorV2(err, 'training');
        }
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
    clearTrainLoopTimer();
    // Drop the gate — a paused loop must not block a later resume on an ack
    // for a snapshot we no longer care about.
    resetAck();
}

function applyDemand(demand: VisualizationDemand): void {
    const confusionDemandChanged = state.demand.needConfusionMatrix !== demand.needConfusionMatrix;
    const newlyVisible = Object.keys(demand).some((key) =>
        key.startsWith('need') && demand[key as keyof VisualizationDemand] === true
        && state.demand[key as keyof VisualizationDemand] !== true);
    state.demand = { ...demand };
    // Force the next snapshot to refresh its requested visualization payloads.
    state.snapshotsSinceLastGrid = demand.gridInterval;
    state.snapshotsSinceLastActivationHistogram = demand.activationHistogramInterval;
    state.gridStale = true;
    if (confusionDemandChanged) {
        state.lastPackedConfusionEvaluationId = null;
        state.confusionMatrixVersion++;
    }
    // A paused view has no training tick to publish its newly requested grids.
    // Use the existing ordered lane and snapshot transport without creating an
    // evaluation, checkpoint, parameter update, or status transition.
    const requestedDemand = state.demand;
    const requestedRun = state.runId;
    if (newlyVisible && !state.running && state.prepared && state.streamPort) {
        void enqueueV2Mutation(() => {
            if (state.running || state.demand !== requestedDemand
                || state.runId !== requestedRun || !state.streamPort) return;
            const { message, transferables } = packSnapshotMessage(computeSnapshot({ lightweight: true }));
            state.streamPort.postMessage(message, transferables);
            // The next model update should still get a fresh visual frame.
            state.snapshotsSinceLastGrid = demand.gridInterval;
            state.snapshotsSinceLastActivationHistogram = demand.activationHistogramInterval;
        }).catch(reportStreamCommandFailure);
    }
}

// ── MessageChannel command handler ──

function reportStreamCommandFailure(error: unknown): void {
    if (error instanceof TerminalDivergenceError) {
        postTerminalDivergenceV2(error, 'evaluation', 'evaluation-failed');
    } else {
        postRuntimeErrorV2(error, 'runtime');
    }
}

function runOrQueueV2StreamMutation(operation: () => void): void {
    void enqueueV2Mutation(operation).catch(reportStreamCommandFailure);
}

function handleStreamCommand(cmd: unknown): void {
    try {
        if (!isMainToWorkerCommand(cmd)) {
            const error = new TypeError('Unknown worker stream command');
            postRuntimeErrorV2(error, 'protocol', 'malformed-request');
            return;
        }
        switch (cmd.type) {
            case 'startTraining':
                runOrQueueV2StreamMutation(() => {
                    requireV2Runtime();
                    state.stepsPerFrame = normalizeTrainingSpeed(cmd.stepsPerFrame);
                    startInternalLoop();
                    postStatus('running');
                });
                break;

            case 'stopTraining':
            {
                runOrQueueV2StreamMutation(() => {
                    requireV2Runtime();
                    const wasRunning = state.running;
                    stopInternalLoop();
                    if (wasRunning) forceAndPublishEvaluationV2('pause');
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

function parsePredictionTraceRequest(request: unknown): PredictionTraceRequest {
    if (typeof request !== 'object' || request === null || Array.isArray(request)) {
        throw new TypeError('prediction trace request must be a plain record');
    }
    const prototype = Object.getPrototypeOf(request);
    if (prototype !== Object.prototype && prototype !== null) {
        throw new TypeError('prediction trace request must have a plain record prototype');
    }
    const expectedKeys = new Set(['source', 'index']);
    const ownKeys = Reflect.ownKeys(request);
    if (ownKeys.length !== expectedKeys.size
        || ownKeys.some((key) => typeof key !== 'string' || !expectedKeys.has(key))) {
        throw new TypeError('prediction trace request must contain exactly index and source');
    }
    for (const key of expectedKeys) {
        const descriptor = Object.getOwnPropertyDescriptor(request, key);
        if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
            throw new TypeError(`prediction trace request ${key} must be an enumerable data property`);
        }
    }

    if (clonePredictionTraceBoundary === null) {
        throw new TypeError('prediction trace requests cannot be authenticated in this environment');
    }
    let snapshot: unknown;
    try {
        snapshot = clonePredictionTraceBoundary(request);
    } catch {
        throw new TypeError('prediction trace request must be an authentic cloneable plain record');
    }
    if (typeof snapshot !== 'object' || snapshot === null || Array.isArray(snapshot)) {
        throw new TypeError('prediction trace request snapshot must be a plain record');
    }
    const snapshotPrototype = Object.getPrototypeOf(snapshot);
    if (snapshotPrototype !== Object.prototype && snapshotPrototype !== null) {
        throw new TypeError('prediction trace request snapshot must have a plain record prototype');
    }
    const source = (snapshot as Record<string, unknown>)['source'];
    if (source !== 'train' && source !== 'test') {
        throw new TypeError('prediction trace request source must be train or test');
    }
    const index = (snapshot as Record<string, unknown>)['index'];
    if (!Number.isSafeInteger(index) || (index as number) < 0) {
        throw new RangeError('prediction trace request index must be a non-negative safe integer');
    }
    return { source, index: index as number };
}

function resolveTraceSample(request: PredictionTraceRequest): {
    source: PredictionTraceSampleSource;
    index: number;
    x: number;
    y: number;
    label: number;
} {
    const points = request.source === 'train' ? state.trainPoints : state.testPoints;
    const { index } = request;
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

function resolveBackpropPreviewBatch(): { inputs: number[][]; targets: number[][] } {
    const { compiled } = requireV2Runtime();

    const n = state.trainInputs.length;
    if (n === 0) {
        throw new Error('No training samples are available for backprop preview');
    }

    const batchSize = compiled.training.batchSize;
    if (state.batchStart === n) {
        throw new Error('Backprop preview is unavailable at the epoch shuffle boundary; step once before previewing.');
    }
    if (state.batchStart < 0 || state.batchStart >= n) {
        throw new RangeError('Backprop preview batch cursor is outside the training population');
    }
    const startIdx = state.batchStart;
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

function currentV2Model(): ModelRevision {
    const { network, epochRef } = requireV2Runtime();
    return {
        generationId: state.runId,
        revision: network.getRevision(),
        step: network.getStep(),
        epoch: epochRef.value,
    };
}

function serializeObjectiveLandscapeProbe(
    probe: ObjectiveLandscapeProbe,
): SerializableObjectiveLandscapeProbe {
    return {
        basis: probe.basis,
        gridSize: probe.gridSize,
        sampleCount: probe.sampleCount,
        parameterPositionCount: probe.parameterPositionCount,
        radius: probe.radius,
        axisA: {
            parameter: { ...probe.axisA.parameter },
            offsets: [...probe.axisA.offsets],
        },
        axisB: {
            parameter: { ...probe.axisB.parameter },
            offsets: [...probe.axisB.offsets],
        },
        objectives: Array.from(probe.objectives),
        centerObjective: probe.centerObjective,
        minObjective: probe.minObjective,
        maxObjective: probe.maxObjective,
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
            checkpointTimeline: buildCheckpointTimeline(),
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

function captureCheckpointV2Now(): WorkerExperimentResultV2 {
    try {
        requireV2Runtime();
        const evaluation = forceAndCaptureCheckpointV2();
        const evidence = makeEvidenceV2(undefined, evaluation);
        const snapshot = computeSnapshot();
        return {
            snapshot,
            runId: state.runId,
            evidence,
            identities: state.prepared!.identities,
            checkpointTimeline: buildCheckpointTimeline(),
            ...directSnapshotArtifactBundle(snapshot),
        };
    } catch (error) {
        postRuntimeErrorV2(error, 'checkpoint', 'checkpoint-failed');
        throw error;
    }
}

function prepareRestoreCandidate(
    checkpoint: SessionCheckpointV2,
    checkpointId: number,
    prospectiveRevision: number,
    runtime: EvaluationRuntime,
): {
    readonly network: Network;
    readonly snapshot: NetworkSnapshot;
    readonly evidence: WorkerEvidenceMessageV2;
    readonly checkpointTimeline: CheckpointTimeline;
    readonly nextState: WorkerState;
    readonly commitEvaluation: () => PairedEvaluation;
    readonly artifacts: ReturnType<typeof directSnapshotArtifactBundle>;
} {
    const { prepared, compiled, datasetRevision, v2EpochRef, v2NetworkRef } = state;
    if (!prepared || !compiled || !datasetRevision || !v2EpochRef || !v2NetworkRef) {
        throw new Error('V2 experiment is not initialized');
    }
    const candidate = new Network(compiled.network, compiled.network.seed);
    candidate.restoreSessionState(
        { network: checkpoint.network, optimizer: checkpoint.optimizer },
        compiled.training.optimizer,
        checkpoint.model.step,
        prospectiveRevision,
    );
    const readCandidateModel = () => ({
        generationId: state.runId,
        revision: candidate.getRevision(),
        step: candidate.getStep(),
        epoch: checkpoint.cursor.epoch,
    });
    const restoredModel = readCandidateModel();
    const computation = createEvaluationComputationForNetwork({
        compiled,
        getNetwork: () => candidate,
        trainInputs: state.trainInputs,
        trainTargets: state.trainTargets,
        testInputs: state.testInputs,
        testTargets: state.testTargets,
    });
    const preparedEvaluation = runtime.prepareRestoreEvaluation({
        model: restoredModel,
        getCurrentModel: readCandidateModel,
        ...computation,
    });
    const capturedScientificValues = {
        dataset: checkpoint.evaluation.dataset,
        objectiveKey: checkpoint.evaluation.objectiveKey,
        train: checkpoint.evaluation.train,
        test: checkpoint.evaluation.test,
        objective: checkpoint.evaluation.objective,
    };
    const recomputedScientificValues = {
        dataset: preparedEvaluation.evaluation.dataset,
        objectiveKey: preparedEvaluation.evaluation.objectiveKey,
        train: preparedEvaluation.evaluation.train,
        test: preparedEvaluation.evaluation.test,
        objective: preparedEvaluation.evaluation.objective,
    };
    if (!structuralEqual(capturedScientificValues, recomputedScientificValues)) {
        throw new RangeError(
            'checkpoint evaluation does not match its restored parameters',
        );
    }
    const evidence = makeEvidenceV2(undefined, preparedEvaluation.evaluation);
    const replacementHistory = new RuntimeMetricHistory({
        generationId: state.runId,
        dataset: datasetRevision,
        objectiveKey: prepared.identities.objectiveKey,
    });
    replacementHistory.appendEvaluation(preparedEvaluation.evaluation);
    const checkpointTimeline = parseCheckpointTimelineV2({
        ...buildCheckpointTimeline(),
        restoredCheckpointId: checkpointId,
    });

    // A temporary runtime lets computeSnapshot exercise the exact demanded
    // artifact path without touching the active EvaluationRuntime.
    const previewRuntime = createEvaluationRuntimeForNetwork({
        generationId: state.runId,
        objectiveKey: prepared.identities.objectiveKey,
        dataset: datasetRevision,
        compiled,
        getNetwork: () => candidate,
        trainInputs: state.trainInputs,
        trainTargets: state.trainTargets,
        testInputs: state.testInputs,
        testTargets: state.testTargets,
        getCurrentModel: readCandidateModel,
    });
    previewRuntime.forceEvaluation('checkpoint');

    const previousState: WorkerState = { ...state };
    let snapshot: NetworkSnapshot;
    let artifacts: ReturnType<typeof directSnapshotArtifactBundle>;
    let nextState: WorkerState;
    try {
        state.network = candidate;
        state.evaluationRuntime = previewRuntime;
        state.metricHistory = replacementHistory;
        state.v2EpochRef = { value: checkpoint.cursor.epoch };
        state.v2NetworkRef = { value: candidate };
        state.epoch = checkpoint.cursor.epoch;
        state.batchStart = checkpoint.cursor.batchStart;
        state.shuffledIndices = Array.from(checkpoint.cursor.shuffledIndices);
        state.running = false;
        clearTrainLoopTimer();
        state.gpuPredictor = null;
        state.sharedViews = null;
        state.restoredCheckpointId = checkpointId;
        resetRestoreDerivedState(candidate);
        snapshot = computeSnapshot();
        artifacts = directSnapshotArtifactBundle(snapshot);
        nextState = {
            ...state,
            evaluationRuntime: runtime,
            metricHistory: replacementHistory,
            v2EpochRef,
            v2NetworkRef,
            sharedViews: previousState.sharedViews,
            gpuPredictor: null,
        };
    } finally {
        Object.assign(state, previousState);
    }

    return {
        network: candidate,
        snapshot,
        evidence,
        checkpointTimeline,
        nextState,
        commitEvaluation: preparedEvaluation.commit,
        artifacts,
    };
}

function resetRestoreDerivedState(network: Network): void {
    state.snapshotsSinceLastGrid = state.demand.gridInterval;
    state.snapshotsSinceLastActivationHistogram = state.demand.needActivationHistograms
        ? state.demand.activationHistogramInterval
        : 0;
    state.stopConditionState = createInitialStopConditionState();
    state.gridStale = true;
    state.gridFreshFromGpu = false;
    state.multiclassBoundaryFresh = false;
    state.lastPackedConfusionEvaluationId = null;
    state.layerStatsGradientRevision = null;
    state.confusionMatrixVersion++;
    state.multiclassConfusionMatrixVersion++;
    state.multiclassBoundaryVersion++;
    state.activationHistogramVersion++;
    state.outputGridBuffer = new Float32Array(GRID_SIZE * GRID_SIZE);
    state.neuronGridsBuffer = new Float32Array(
        network.getTotalNeuronCount() * GRID_SIZE * GRID_SIZE,
    );
    state.multiclassClassGridBuffer = network.config.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
        ? new Uint8Array(GRID_SIZE * GRID_SIZE)
        : null;
    state.multiclassConfidenceGridBuffer = network.config.outputSize === WORKER_MULTICLASS_OUTPUT_SIZE
        ? new Float32Array(GRID_SIZE * GRID_SIZE)
        : null;
    resetAck();
}

function restoreCheckpointV2Now(request: RestoreCheckpointRequestV2): WorkerExperimentResultV2 {
    try {
        const { network, runtime, epochRef } = requireV2Runtime();
        const entry = state.checkpoints.find(
            (candidate) => candidate.summary.id === request.checkpointId,
        );
        if (!entry || entry.kind !== 'v2') throw new RangeError('V2 checkpoint not found');
        const checkpoint = validateSessionCheckpointV2(
            entry.checkpoint,
            checkpointValidationContext(),
        );
        if (checkpoint.model.generationId !== state.runId) {
            throw new RangeError('checkpoint generation does not match the active generation');
        }

        const previousRevision = network.getRevision();
        const prospectiveRevision = previousRevision + 1;
        const prepared = prepareRestoreCandidate(
            checkpoint,
            request.checkpointId,
            prospectiveRevision,
            runtime,
        );

        // All scientific computation and allocation happened above. Commit
        // the monotonic pair, then swap complete references synchronously.
        stopInternalLoop();
        beginV2Mutation();
        if (state.gpuPredictor) {
            try { state.gpuPredictor.dispose(); } catch { /* best-effort invalidation */ }
        }
        prepared.commitEvaluation();
        epochRef.value = checkpoint.cursor.epoch;
        state.v2NetworkRef!.value = prepared.network;
        Object.assign(state, prepared.nextState);
        state.epoch = checkpoint.cursor.epoch;
        state.batchStart = checkpoint.cursor.batchStart;
        postEvidenceV2(prepared.evidence);
        return {
            snapshot: prepared.snapshot,
            runId: state.runId,
            evidence: prepared.evidence,
            identities: state.prepared!.identities,
            checkpointTimeline: prepared.checkpointTimeline,
            ...prepared.artifacts,
        };
    } catch (error) {
        postRuntimeErrorV2(error, 'checkpoint', 'checkpoint-failed');
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

    captureCheckpointV2(request: unknown): Promise<WorkerExperimentResultV2> {
        let parsed: CaptureCheckpointRequestV2;
        try {
            const candidate = parseMainToWorkerRequestV2(request);
            if (candidate.type !== 'capture-checkpoint') {
                throw new TypeError('capture checkpoint request has the wrong discriminator');
            }
            parsed = candidate;
        } catch (error) {
            postRuntimeErrorV2(error, 'checkpoint', 'checkpoint-failed');
            return Promise.reject(error);
        }
        void parsed;
        return enqueueV2Mutation(captureCheckpointV2Now);
    },

    restoreCheckpointV2(request: unknown): Promise<WorkerExperimentResultV2> {
        let parsed: RestoreCheckpointRequestV2;
        try {
            const candidate = parseMainToWorkerRequestV2(request);
            if (candidate.type !== 'restore-checkpoint') {
                throw new TypeError('restore checkpoint request has the wrong discriminator');
            }
            parsed = candidate;
        } catch (error) {
            postRuntimeErrorV2(error, 'checkpoint', 'checkpoint-failed');
            return Promise.reject(error);
        }
        return enqueueV2Mutation(() => restoreCheckpointV2Now(parsed));
    },

    getMetricHistoryV2(): RuntimeMetricHistorySnapshot {
        return requireV2Runtime().history.read();
    },

    getTrainPointsV2(): DataPoint[] {
        requireV2Runtime();
        return state.trainPoints;
    },

    getTestPointsV2(): DataPoint[] {
        requireV2Runtime();
        return state.testPoints;
    },

    getPredictionTraceV2(request: unknown): Promise<PredictionTraceResponseV2> {
        let parsed: PredictionTraceRequest;
        try {
            parsed = parsePredictionTraceRequest(request);
        } catch (error) {
            return Promise.reject(error);
        }
        return enqueueV2Mutation(() => {
            const { network, compiled, runtime } = requireV2Runtime();
            const sample = resolveTraceSample(parsed);
            const input = transformPoint(sample.x, sample.y, state.activeFeatures);
            const target = encodeTargetLabel(sample.label, compiled.network.outputSize);
            return {
                runId: state.runId,
                model: currentV2Model(),
                dataset: runtime.dataset,
                objectiveKey: runtime.objectiveKey,
                sample: {
                    source: sample.source,
                    index: sample.index,
                    x: sample.x,
                    y: sample.y,
                    label: sample.label,
                },
                trace: network.tracePredictionV2(input, target, compiled.objective),
            };
        });
    },

    getBackpropExplanationV2(): Promise<BackpropExplanationResponseV2> {
        return enqueueV2Mutation(() => {
            const { network, compiled, runtime } = requireV2Runtime();
            const batch = resolveBackpropPreviewBatch();
            return {
                runId: state.runId,
                model: currentV2Model(),
                dataset: runtime.dataset,
                objectiveKey: runtime.objectiveKey,
                basis: {
                    kind: 'next-mini-batch',
                    sampleCount: batch.inputs.length,
                    populationCount: state.trainInputs.length,
                },
                explanation: network.explainBackpropStepV2(
                    batch.inputs,
                    batch.targets,
                    compiled.training,
                ),
            };
        });
    },

    getObjectiveLandscapeV2(
        options: LossLandscapeProbeOptions = {},
    ): Promise<ObjectiveLandscapeResponseV2> {
        return enqueueV2Mutation(() => {
            const { network, compiled, runtime } = requireV2Runtime();
            const model = currentV2Model();
            const probe = network.probeObjectiveLandscape(
                state.trainInputs,
                state.trainTargets,
                compiled.training,
                options,
            );
            return {
                runId: state.runId,
                model,
                provenance: {
                    model,
                    dataset: runtime.dataset,
                    objectiveKey: runtime.objectiveKey,
                    basis: {
                        kind: 'parameter-grid',
                        sampleCount: probe.sampleCount,
                        parameterPositions: probe.parameterPositionCount,
                    },
                },
                probe: serializeObjectiveLandscapeProbe(probe),
            };
        });
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

exposeWorkerApiAndAnnounceReady(
    () => Comlink.expose(workerApi),
    self as unknown as { postMessage(message: unknown): void },
);
