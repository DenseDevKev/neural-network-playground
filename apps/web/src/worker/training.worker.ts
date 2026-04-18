// ── Training Web Worker ──
// Owns the engine instance, runs training off the main thread.
// Hybrid communication:
//   - Comlink RPC for commands (initialize, updateConfig, reset, step, etc.)
//   - MessageChannel for high-frequency streamed snapshots during training

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
} from '@nn-playground/engine';
import type {
    NetworkConfig,
    TrainingConfig,
    DataConfig,
    FeatureFlags,
    NetworkSnapshot,
    DataPoint,
    HistoryPoint,
} from '@nn-playground/engine';
import { GRID_SIZE, DEFAULT_DEMAND } from '@nn-playground/shared';
import type {
    VisualizationDemand,
    WorkerSnapshotMessage,
    WorkerStatusMessage,
    WorkerErrorMessage,
    MainToWorkerCommand,
} from '@nn-playground/shared';
import type { FeatureSpec } from '@nn-playground/engine';

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
    /** Cached last test metrics to reuse when skipping test evaluation. */
    lastTestMetrics: { loss: number; accuracy?: number; confusionMatrix?: { tp: number; tn: number; fp: number; fn: number } } | null;
    /** True when the most recent snapshot reused cached test metrics instead of re-evaluating. */
    testMetricsStale: boolean;
    /** Pre-allocated buffers for grid predictions. */
    outputGridBuffer: Float32Array | null;
    neuronGridsBuffer: Float32Array | null;
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
    /** Training history for getHistory() API. */
    history: HistoryPoint[];
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

/**
 * Shallow structural equality for our config objects. Assumes values are
 * primitives or small arrays of primitives (e.g. NetworkConfig.hiddenLayers).
 * Faster and more robust than JSON.stringify comparison — which is sensitive
 * to key ordering and mishandles `undefined`.
 */
function configsEqual(a: unknown, b: unknown): boolean {
    if (a === b) return true;
    if (a == null || b == null) return a === b;
    if (typeof a !== 'object' || typeof b !== 'object') return false;
    if (Array.isArray(a) || Array.isArray(b)) {
        if (!Array.isArray(a) || !Array.isArray(b)) return false;
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const aKeys = Object.keys(ao);
    const bKeys = Object.keys(bo);
    if (aKeys.length !== bKeys.length) return false;
    for (const k of aKeys) {
        if (!configsEqual(ao[k], bo[k])) return false;
    }
    return true;
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
    lastTestMetrics: null,
    testMetricsStale: false,
    outputGridBuffer: null,
    neuronGridsBuffer: null,
    streamPort: null,
    runId: 0,
    snapshotId: 0,
    stepsPerFrame: 5,
    trainLoopTimer: null,
    history: [],
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

    // Reset test-eval gating
    state.snapshotsSinceLastTestEval = 0;
    state.lastTestMetrics = null;

    // Increment run ID
    state.runId++;
    state.snapshotId = 0;

    // New run — drop any stale back-pressure gate.
    resetAck();

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

// ── Snapshot computation ──

/**
 * @param opts.lightweight — when true, skips the deep-copy of weights/biases
 * into the snapshot. The streaming path transfers flat buffers separately
 * (see packSnapshotMessage), so nested copies are pure waste there.
 */
function computeSnapshot(opts: { lightweight?: boolean } = {}): NetworkSnapshot {
    performance.mark('perf:worker:snapshot:start');
    if (!state.network || !state.trainingConfig || !state.dataConfig) {
        throw new Error('Not initialized');
    }

    const { demand } = state;
    const problemType = state.dataConfig.problemType;
    const lossType = state.trainingConfig.lossType;

    // ── Train metrics (always computed — cheap, uses cached forward pass) ──
    const huberDelta = state.trainingConfig.huberDelta;
    const trainMetrics = state.network.evaluate(
        state.trainInputs,
        state.trainTargets,
        lossType,
        problemType,
        huberDelta,
    );

    // ── Test metrics (gated by testEvalInterval) ──
    const shouldRunTestEval =
        state.snapshotsSinceLastTestEval >= demand.testEvalInterval ||
        state.lastTestMetrics === null;

    let testMetrics;
    if (shouldRunTestEval) {
        testMetrics = state.network.evaluate(
            state.testInputs,
            state.testTargets,
            lossType,
            problemType,
            huberDelta,
        );
        state.lastTestMetrics = {
            loss: testMetrics.loss,
            accuracy: testMetrics.accuracy,
            confusionMatrix: demand.needConfusionMatrix ? testMetrics.confusionMatrix : undefined,
        };
        state.snapshotsSinceLastTestEval = 0;
        state.testMetricsStale = false;
    } else {
        testMetrics = state.lastTestMetrics!;
        state.snapshotsSinceLastTestEval++;
        state.testMetricsStale = true;
    }

    // ── Decision boundary grid (demand-gated) ──
    let outputGrid: number[] | Float32Array;
    let neuronGrids: number[][] | Float32Array | undefined;

    if (demand.needNeuronGrids && state.outputGridBuffer && state.neuronGridsBuffer) {
        performance.mark('perf:worker:predictGridNeurons:start');
        state.network.predictGridWithNeuronsInto(
            state.gridInputs,
            state.outputGridBuffer,
            state.neuronGridsBuffer,
        );
        performance.measure('perf:worker:predictGridNeurons', 'perf:worker:predictGridNeurons:start');
        outputGrid = state.outputGridBuffer;
        neuronGrids = state.neuronGridsBuffer;
    } else if (demand.needDecisionBoundary && state.outputGridBuffer) {
        performance.mark('perf:worker:predictGrid:start');
        state.network.predictGridInto(state.gridInputs, state.outputGridBuffer);
        performance.measure('perf:worker:predictGrid', 'perf:worker:predictGrid:start');
        outputGrid = state.outputGridBuffer;
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

    performance.measure('perf:worker:snapshot', 'perf:worker:snapshot:start');
    return snap;
}

/**
 * Maximum number of history points retained in memory. When exceeded, every
 * other point is dropped (a simple in-place compaction) before pushing the
 * new one — giving a log-spaced density profile over very long runs.
 */
const MAX_HISTORY = 2048;

function pushHistory(point: HistoryPoint | undefined): void {
    if (!point) return;
    if (state.history.length >= MAX_HISTORY) {
        const kept: HistoryPoint[] = [];
        for (let i = 0; i < state.history.length; i += 2) {
            kept.push(state.history[i]);
        }
        state.history = kept;
    }
    state.history.push(point);
}

/**
 * Pack a NetworkSnapshot into a WorkerSnapshotMessage with Transferable Float32Arrays.
 * Returns { message, transferables }.
 */
function packSnapshotMessage(snap: NetworkSnapshot): { message: WorkerSnapshotMessage; transferables: Transferable[] } {
    performance.mark('perf:worker:snapshotPack:start');

    const transferables: Transferable[] = [];
    const gridSize = snap.gridSize;

    // Pack outputGrid
    let outputGrid: Float32Array | undefined;
    if (snap.outputGrid && snap.outputGrid.length > 0) {
        if (snap.outputGrid instanceof Float32Array) {
            outputGrid = snap.outputGrid;
            // Since we transfer the buffer, we must null out the reference in state
            // and re-allocate it for the next snapshot.
            state.outputGridBuffer = null;
        } else {
            outputGrid = new Float32Array(snap.outputGrid);
        }
        transferables.push(outputGrid.buffer);
    }

    // Re-allocate outputGridBuffer if it was transferred
    if (state.outputGridBuffer === null) {
        state.outputGridBuffer = new Float32Array(GRID_SIZE * GRID_SIZE);
    }

    // Pack neuronGrids (concatenated into one flat array)
    let neuronGrids: Float32Array | undefined;
    let neuronGridLayout: { count: number; gridSize: number } | undefined;
    if (snap.neuronGrids && snap.neuronGrids.length > 0) {
        if (snap.neuronGrids instanceof Float32Array) {
            neuronGrids = snap.neuronGrids;
            const totalNeurons = state.network!.getTotalNeuronCount();
            neuronGridLayout = { count: totalNeurons, gridSize };
            // Transfer and null out
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

    // Re-allocate neuronGridsBuffer if it was transferred
    if (state.neuronGridsBuffer === null && state.network) {
        const totalNeurons = state.network.getTotalNeuronCount();
        state.neuronGridsBuffer = new Float32Array(totalNeurons * GRID_SIZE * GRID_SIZE);
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
        confusionMatrix: snap.testMetrics?.confusionMatrix,
    };

    performance.measure('perf:worker:snapshotPack', 'perf:worker:snapshotPack:start');
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
    const batchIndices = state.shuffledIndices.slice(startIdx, endIdx);
    const batchInputs = batchIndices.map((i) => state.trainInputs[i]);
    const batchTargets = batchIndices.map((i) => state.trainTargets[i]);

    if (batchInputs.length > 0) {
        state.network.trainBatch(batchInputs, batchTargets, state.trainingConfig);
    }

    if (batchSlot === numBatches - 1) {
        state.epoch++;
    }
}

// ── Internal training loop (worker-driven) ──

function trainTick(): void {
    if (!state.running) return;

    try {
        if (state.streamPort) {
            performance.mark('perf:worker:trainStep:start');
            for (let i = 0; i < state.stepsPerFrame; i++) {
                trainOneStep();
            }
            performance.measure('perf:worker:trainStep', 'perf:worker:trainStep:start');

            // Back-pressure: skip snapshot computation + posting while the main
            // thread hasn't yet applied the previous frame. Training still
            // progresses; the UI just coalesces to its render rate.
            if (!state.awaitingAck) {
                // Compute snapshot and stream it to the main thread. Lightweight
                // mode skips the nested weight/bias deep-copy — we transfer flat
                // buffers.
                const snap = computeSnapshot({ lightweight: true });

                // NaN / divergence guard — stop the loop and notify the UI.
                // Training with NaN weights is a dead end; keep state readable
                // for debugging.
                if (!Number.isFinite(snap.trainLoss)) {
                    stopInternalLoop();
                    postError(
                        'Training diverged (non-finite loss). Try a smaller learning rate, ' +
                        'enable gradient clipping, or reset the network.',
                    );
                    return;
                }

                // Only training paths accumulate history; passive snapshot RPCs do not.
                pushHistory(snap.historyPoint);

                const { message, transferables } = packSnapshotMessage(snap);
                state.awaitingAck = true;
                state.streamPort.postMessage(message, transferables);
            }
        }

        // Schedule next tick (yield to allow message processing)
        if (state.running) {
            state.trainLoopTimer = setTimeout(trainTick, 0);
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
    state.trainLoopTimer = setTimeout(trainTick, 0);
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

// ── MessageChannel command handler ──

function handleStreamCommand(cmd: MainToWorkerCommand): void {
    try {
        switch (cmd.type) {
            case 'startTraining':
                state.stepsPerFrame = cmd.stepsPerFrame;
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
                state.demand = { ...cmd.demand };
                state.snapshotsSinceLastTestEval = cmd.demand.testEvalInterval;
                break;

            case 'updateSpeed':
                state.stepsPerFrame = cmd.stepsPerFrame;
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

const workerApi = {
    initialize(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
    ): { snapshot: NetworkSnapshot; runId: number } {
        validateConfigs(networkConfig, trainingConfig);
        stopInternalLoop();
        state.networkConfig = { ...networkConfig };
        state.trainingConfig = { ...trainingConfig };
        state.dataConfig = { ...dataConfig };
        state.features = { ...features };
        state.running = false;
        state.history = [];
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
        validateConfigs(networkConfig, trainingConfig);
        const needsRebuild = rebuild ||
            !configsEqual(state.networkConfig, networkConfig) ||
            !configsEqual(state.dataConfig, dataConfig) ||
            !configsEqual(state.features, features);

        state.networkConfig = { ...networkConfig };
        state.trainingConfig = { ...trainingConfig };
        state.dataConfig = { ...dataConfig };
        state.features = { ...features };

        if (needsRebuild) {
            stopInternalLoop();
            state.running = false;
            state.history = [];
            buildDataAndNetwork();
        }

        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    step(iterations: number = 1): NetworkSnapshot {
        for (let i = 0; i < iterations; i++) {
            trainOneStep();
        }
        const snap = computeSnapshot();
        pushHistory(snap.historyPoint);
        return snap;
    },

    reset(): { snapshot: NetworkSnapshot; runId: number } {
        stopInternalLoop();
        state.history = [];
        buildDataAndNetwork();
        return { snapshot: computeSnapshot(), runId: state.runId };
    },

    getSnapshot(): NetworkSnapshot {
        return computeSnapshot();
    },

    getHistory(): HistoryPoint[] {
        return state.history;
    },

    getTrainPoints(): DataPoint[] {
        return state.trainPoints;
    },

    getTestPoints(): DataPoint[] {
        return state.testPoints;
    },

    setRunning(running: boolean): void {
        state.running = running;
    },

    isRunning(): boolean {
        return state.running;
    },

    /** Update what visual data the UI currently needs. */
    updateDemand(demand: VisualizationDemand): void {
        state.demand = { ...demand };
        state.snapshotsSinceLastTestEval = demand.testEvalInterval;
    },

    /** Accept a MessagePort from the main thread for streaming. */
    setStreamPort(port: MessagePort): void {
        state.streamPort = port;
        port.addEventListener('message', (event: MessageEvent<MainToWorkerCommand>) => {
            handleStreamCommand(event.data);
        });
        port.start();
    },

    // Legacy: Run continuous training — called in a loop from main thread
    trainAndSnapshot(stepsPerFrame: number): NetworkSnapshot {
        performance.mark('perf:worker:trainStep:start');
        for (let i = 0; i < stepsPerFrame; i++) {
            trainOneStep();
        }
        performance.measure('perf:worker:trainStep', 'perf:worker:trainStep:start');
        const snap = computeSnapshot();
        pushHistory(snap.historyPoint);
        return snap;
    },
};

export type TrainingWorkerApi = typeof workerApi;

Comlink.expose(workerApi);
