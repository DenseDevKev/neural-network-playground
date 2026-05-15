// ── Training Web Worker ──
// Owns the engine instance, runs training off the main thread.
// Hybrid communication:
//   - Comlink RPC for commands (initialize, updateConfig, reset, step, etc.)
//   - MessagePort commands for high-frequency streamed snapshots during training

import * as Comlink from 'comlink';
import {
    Network,
    PRNG,
    buildGridInputs,
    generateDataset,
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
} from '@nn-playground/engine';
import {
    GRID_SIZE,
    DEFAULT_DEMAND,
    isMainToWorkerCommand,
    normalizeVisualizationDemand,
    structuralEqual,
    normalizeAppConfig,
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
    type StopConditionState,
} from './stopConditions.ts';
import {
    createMiniBatchScratch,
    fillMiniBatchScratch,
    getTrainingStepsForTick,
    normalizeTrainingSpeed,
    type MiniBatchScratch,
} from './trainingLoop.ts';

interface WorkerState {
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
    /** Monotonic identity for activation histogram payload freshness. */
    activationHistogramVersion: number;
    /** Pre-allocated buffers for grid predictions. */
    outputGridBuffer: Float32Array | null;
    neuronGridsBuffer: Float32Array | null;
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
        'Worker multiclass mode requires classification data, output size 3, softmax output activation, and categorical cross-entropy loss.',
    );
}

function normalizeScalarSurrogateForSharedValidation(
    networkConfig: NetworkConfig,
    trainingConfig: TrainingConfig,
): { network: NetworkConfig; training: TrainingConfig } {
    return {
        network: {
            ...networkConfig,
            outputSize: 1,
            outputActivation: 'sigmoid',
        },
        training: {
            ...trainingConfig,
            lossType: 'crossEntropy',
        },
    };
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
    const validationConfig = isWorkerMulticlassRequest
        ? normalizeScalarSurrogateForSharedValidation(networkConfig, trainingConfig)
        : { network: networkConfig, training: trainingConfig };

    const result = normalizeAppConfig({
        network: validationConfig.network,
        training: validationConfig.training,
        data: dataConfig,
        features,
        ui: { showTestData: false, discretizeOutput: false },
    });

    if (!result.config) {
        throw new Error(result.error ?? 'Invalid playground configuration.');
    }

    if (isWorkerMulticlassRequest) {
        assertApprovedWorkerMulticlassConfig(networkConfig, trainingConfig, result.config.data);
    }

    const normalizedNetwork: NetworkConfig = isWorkerMulticlassRequest
        ? {
            ...result.config.network,
            outputSize: WORKER_MULTICLASS_OUTPUT_SIZE,
            outputActivation: 'softmax',
        }
        : result.config.network;
    const normalizedTraining: TrainingConfig = isWorkerMulticlassRequest
        ? {
            ...result.config.training,
            lossType: 'categoricalCrossEntropy',
        }
        : result.config.training;

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
    activationHistogramVersion: 0,
    outputGridBuffer: null,
    neuronGridsBuffer: null,
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

function buildArenaSlot(side: ArenaSide, input: ArenaModelInput): ArenaSlot {
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

    const device = await detectWebGPU();
    if (state.runId !== runId || gpuPredictorSignature() !== signature) return null;
    if (!device) return null;

    try {
        const predictor = new WebGPUGridPredictor({
            device,
            layerSizes,
            gridLen: state.gridInputs.length,
            hiddenActivation: state.networkConfig.activation,
            outputActivation: state.networkConfig.outputActivation,
        });
        // Grid inputs are constant per shape — upload once and never again
        // until the next shape change disposes this predictor.
        predictor.setGridInputs(state.gridInputsFlat);
        if (state.runId !== runId || gpuPredictorSignature() !== signature) {
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

    const predictor = await ensureGpuPredictor();
    if (!predictor) return;

    // Push the latest weights to the GPU. The flat accessors allocate a
    // fresh Float32Array per call — we accept that small cost in exchange
    // for not having to reach into Network's private packed buffers.
    const flat = state.network.getWeightsFlat();
    predictor.updateWeights(flat.buffer, state.network.getBiasesFlat());

    try {
        workerPerfMark('perf:worker:predictGridGpu:start');
        if (readbackMode === 'withNeurons') {
            await predictor.predictGridWithNeuronsInto(
                state.outputGridBuffer,
                state.neuronGridsBuffer,
            );
        } else {
            await predictor.predictGridInto(state.outputGridBuffer);
        }
        workerPerfMeasure('perf:worker:predictGridGpu', 'perf:worker:predictGridGpu:start');
        state.gridFreshFromGpu = true;
    } catch (err) {
        console.warn('[worker] GPU grid prediction failed, falling back to CPU', err);
        // Leave gridFreshFromGpu false; computeSnapshot will run the CPU
        // branch this frame.
    }
}

// ── Snapshot computation ──

/**
 * @param opts.lightweight — when true, skips the deep-copy of weights/biases
 * into the snapshot. The streaming path transfers flat buffers separately
 * (see packSnapshotMessage), so nested copies are pure waste there.
 */
function computeSnapshot(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
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

    const supportsScalarGrid = state.networkConfig?.outputSize === 1;
    const wantGrid = supportsScalarGrid && (demand.needDecisionBoundary || demand.needNeuronGrids);
    const shouldRebuildGrid =
        wantGrid &&
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
    } else if (wantGrid && state.outputGridBuffer) {
        // Reuse the last computed grid(s) without recomputing. The main
        // thread retains the previous Float32Arrays in its frame buffer;
        // emitting undefined here causes packSnapshotMessage to skip the
        // buffer transfer entirely.
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

    if (neuronGrids) {
        snap.neuronGrids = neuronGrids;
    }

    if (demand.needLayerStats) {
        snap.layerStats = state.network.getLayerStats();
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

    captureCheckpointFromSnapshot(snap);
    workerPerfMeasure('perf:worker:snapshot', 'perf:worker:snapshot:start');
    return snap;
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

    if (state.networkConfig?.outputSize !== 1) {
        outputGrid = new Float32Array(0);
        neuronGrids = new Float32Array(0);
        neuronGridLayout = undefined;
        transferables.push(outputGrid.buffer, neuronGrids.buffer);
    }

    // History point
    const historyPoint: HistoryPoint = {
        step: snap.step,
        trainLoss: snap.trainLoss,
        testLoss: snap.testLoss,
        trainAccuracy: snap.trainMetrics?.accuracy,
        testAccuracy: snap.testMetrics?.accuracy,
    };

    const message: WorkerSnapshotMessage = {
        type: 'snapshot',
        runId: state.runId,
        snapshotId: ++state.snapshotId,
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
        activationHistogramBins,
        activationHistogramLayout,
        activationHistogramVersion,
        historyPoint,
        confusionMatrix: state.testMetricsStale ? undefined : snap.testMetrics?.confusionMatrix,
        confusionMatrixVersion: state.confusionMatrixVersion,
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

// ── Internal training loop (worker-driven) ──

const TRAIN_TICK_INTERVAL_MS = 1000 / 60;

function scheduleNextTick(): void {
    if (state.trainLoopTimer !== null || !state.running) return;
    state.trainLoopTimer = setTimeout(() => {
        state.trainLoopTimer = null;
        if (state.running) trainTick();
    }, TRAIN_TICK_INTERVAL_MS);
}

// Snapshot pipeline split out from trainTick so the GPU grid pre-fill
// (AS-4) can be awaited without forcing the whole tick to be async. The
// back-pressure gate `state.awaitingAck` is set BEFORE we enter this
// async work, so subsequent ticks skip their snapshot block until the
// main thread acks.
async function produceAndPostSnapshot(): Promise<void> {
    if (!state.streamPort) return;

    // GPU grid pre-fill (AS-4). When enabled + capable + due, this
    // populates state.outputGridBuffer / state.neuronGridsBuffer; the
    // synchronous computeSnapshot below detects the freshly-filled
    // buffers via state.gridFreshFromGpu and skips its CPU branch.
    await runGpuGridIfDue();

    const snap = computeSnapshot({ lightweight: true });
    const stopEvaluation = state.running
        ? evaluateStopConditions(
            DEFAULT_RUNTIME_STOP_CONDITIONS,
            {
                step: snap.step,
                trainLoss: snap.trainLoss,
                testLoss: snap.testLoss,
                trainAccuracy: snap.trainMetrics.accuracy,
                testAccuracy: snap.testMetrics.accuracy,
            },
            state.stopConditionState,
        )
        : null;
    if (stopEvaluation) {
        state.stopConditionState = stopEvaluation.nextState;
    }

    const { message, transferables } = packSnapshotMessage(snap);
    state.streamPort.postMessage(message, transferables);
    if (stopEvaluation?.pauseReason) {
        stopInternalLoop();
        postStatus('paused', stopEvaluation.pauseReason);
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
                trainOneStep();
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
                produceAndPostSnapshot().catch((err) => {
                    state.awaitingAck = false;
                    stopInternalLoop();
                    const msg = err instanceof Error ? err.message : String(err);
                    postError(`Training snapshot error: ${msg}`);
                });
            }
        }

        if (state.running) {
            scheduleNextTick();
        }
    } catch (err) {
        stopInternalLoop();
        postError(`Training error: ${err instanceof Error ? err.message : String(err)}`);
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
        state.confusionMatrixVersion++;
    }
}

// ── MessageChannel command handler ──

function handleStreamCommand(cmd: unknown): void {
    try {
        if (!isMainToWorkerCommand(cmd)) {
            postError('Unknown command: ' + JSON.stringify(cmd));
            return;
        }
        switch (cmd.type) {
            case 'startTraining':
                state.stepsPerFrame = normalizeTrainingSpeed(cmd.stepsPerFrame);
                startInternalLoop();
                postStatus('running');
                break;

            case 'stopTraining':
                stopInternalLoop();
                postStatus('paused');
                break;

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
        postError(`Command handling error: ${err instanceof Error ? err.message : String(err)}`);
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

export const workerApi = {
    initialize(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
    ): { snapshot: NetworkSnapshot; runId: number } {
        const config = normalizeWorkerConfig(networkConfig, trainingConfig, dataConfig, features);
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
        const config = normalizeWorkerConfig(networkConfig, trainingConfig, dataConfig, features);
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
        stopInternalLoop();
        buildDataAndNetwork();
        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    getCheckpointTimeline(): CheckpointTimeline {
        return buildCheckpointTimeline();
    },

    restoreCheckpoint(id: number): { snapshot: NetworkSnapshot; runId: number; timeline: CheckpointTimeline } {
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
