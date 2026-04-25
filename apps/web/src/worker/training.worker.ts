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
    VisualizationDemand,
    WorkerSnapshotMessage,
    WorkerStatusMessage,
    WorkerErrorMessage,
    WorkerSharedBuffersMessage,
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
import { getTrainingStepsForTick, normalizeTrainingSpeed } from './trainingLoop.ts';

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
    /** Monotonic identity for confusion matrix payload freshness. */
    confusionMatrixVersion: number;
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

/** Validate config compatibility — throws on incompatible loss/activation. */
function validateConfigs(network: NetworkConfig, training: TrainingConfig): void {
    if (!isLossCompatible(training.lossType, network.outputActivation)) {
        throw new Error(describeLossIncompatibility(training.lossType, network.outputActivation));
    }
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
    const result = normalizeAppConfig({
        network: networkConfig,
        training: trainingConfig,
        data: dataConfig,
        features,
        ui: { showTestData: false, discretizeOutput: false },
    });

    if (!result.config) {
        throw new Error(result.error ?? 'Invalid playground configuration.');
    }

    validateConfigs(result.config.network, result.config.training);
    return {
        network: result.config.network,
        training: result.config.training,
        data: result.config.data,
        features: result.config.features,
    };
}

// Structural equality has moved to @nn-playground/shared (`structuralEqual`).
// Keeping a local alias avoids touching every call site below.
const configsEqual = structuralEqual;

const WORKER_PERF_ENABLED = import.meta.env.DEV && import.meta.env.VITE_WORKER_PERF === '1';

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
    shufflePrng: null,
    demand: { ...DEFAULT_DEMAND },
    snapshotsSinceLastTestEval: 0,
    snapshotsSinceLastTrainEval: 0,
    snapshotsSinceLastGrid: 0,
    lastTestMetrics: null,
    lastTrainMetrics: null,
    lossEma: null,
    lossEmaAlpha: 0.1,
    testMetricsStale: false,
    confusionMatrixVersion: 0,
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

    state.trainPoints = split.train;
    state.testPoints = split.test;
    state.trainInputs = transformDataset(split.train, state.activeFeatures);
    state.trainTargets = split.train.map((p) => [p.label]);
    state.testInputs = transformDataset(split.test, state.activeFeatures);
    state.testTargets = split.test.map((p) => [p.label]);

    // Build grid inputs
    state.gridInputs = buildGridInputs(GRID_SIZE, state.activeFeatures);

    // Create network
    const config: NetworkConfig = {
        ...state.networkConfig,
        inputSize,
    };
    state.networkConfig = config;
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
    state.lastTestMetrics = null;
    state.lastTrainMetrics = null;
    state.lossEma = null;
    state.gridStale = true;
    state.testMetricsStale = false;
    state.confusionMatrixVersion++;

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

    const wantGrid = demand.needDecisionBoundary || demand.needNeuronGrids;
    const shouldRebuildGrid =
        wantGrid &&
        state.gridStale &&
        state.snapshotsSinceLastGrid >= demand.gridInterval;

    // GPU pre-fill (AS-4): runGpuGridIfDue ran earlier in trainTick and
    // already populated the grid buffers. Consume the flag here so the CPU
    // branches below stay short-circuited, and so a second call later in
    // the same frame doesn't double-count.
    if (state.gridFreshFromGpu && state.outputGridBuffer) {
        outputGrid = state.outputGridBuffer;
        if (demand.needNeuronGrids && state.neuronGridsBuffer) {
            neuronGrids = state.neuronGridsBuffer;
        }
        state.snapshotsSinceLastGrid = 0;
        state.gridStale = false;
        state.gridFreshFromGpu = false;
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
        historyPoint,
        confusionMatrix: state.testMetricsStale ? undefined : snap.testMetrics?.confusionMatrix,
        confusionMatrixVersion: state.confusionMatrixVersion,
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

    if (startIdx < endIdx) {
        const batchLoss = state.network.trainBatchIndexed(
            state.trainInputs,
            state.trainTargets,
            state.shuffledIndices,
            startIdx,
            endIdx,
            state.trainingConfig,
        );
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

    // NaN / divergence guard — stop the loop and notify the UI. Training
    // with NaN weights is a dead end; keep state readable for debugging.
    if (!Number.isFinite(snap.trainLoss)) {
        stopInternalLoop();
        postError(
            'Training diverged (non-finite loss). Try a smaller learning rate, ' +
            'enable gradient clipping, or reset the network.',
        );
        return;
    }

    const { message, transferables } = packSnapshotMessage(snap);
    state.streamPort.postMessage(message, transferables);
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
                if (state.streamPort) {
                    const statusMsg: WorkerStatusMessage = {
                        type: 'status',
                        runId: state.runId,
                        status: 'running',
                    };
                    state.streamPort.postMessage(statusMsg);
                }
                break;

            case 'stopTraining':
                stopInternalLoop();
                if (state.streamPort) {
                    const statusMsg: WorkerStatusMessage = {
                        type: 'status',
                        runId: state.runId,
                        status: 'paused',
                    };
                    state.streamPort.postMessage(statusMsg);
                }
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

    reset(): { snapshot: NetworkSnapshot; runId: number } {
        stopInternalLoop();
        buildDataAndNetwork();
        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    getTrainPoints(): DataPoint[] {
        return state.trainPoints;
    },

    getTestPoints(): DataPoint[] {
        return state.testPoints;
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
