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
    step: number;
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
    step: 0,
    epoch: 0,
    running: false,
    rafId: null,
    shuffledIndices: [],
    shufflePrng: null,
    demand: { ...DEFAULT_DEMAND },
    snapshotsSinceLastTestEval: 0,
    lastTestMetrics: null,
    outputGridBuffer: null,
    neuronGridsBuffer: null,
    streamPort: null,
    runId: 0,
    snapshotId: 0,
    stepsPerFrame: 5,
    trainLoopTimer: null,
    history: [],
};



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
    state.step = 0;

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

    // Initialise shuffle state — seed is offset from data seed to stay independent.
    const n = state.trainInputs.length;
    state.shuffledIndices = Array.from({ length: n }, (_, i) => i);
    state.shufflePrng = new PRNG((state.dataConfig!.seed ?? 42) + 1234);
}

// ── Snapshot computation ──

function computeSnapshot(): NetworkSnapshot {
    performance.mark('perf:worker:snapshot:start');
    if (!state.network || !state.trainingConfig || !state.dataConfig) {
        throw new Error('Not initialized');
    }

    const { demand } = state;
    const problemType = state.dataConfig.problemType;
    const lossType = state.trainingConfig.lossType;

    // ── Train metrics (always computed — cheap, uses cached forward pass) ──
    const trainMetrics = state.network.evaluate(
        state.trainInputs,
        state.trainTargets,
        lossType,
        problemType,
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
        );
        state.lastTestMetrics = {
            loss: testMetrics.loss,
            accuracy: testMetrics.accuracy,
            confusionMatrix: demand.needConfusionMatrix ? testMetrics.confusionMatrix : undefined,
        };
        state.snapshotsSinceLastTestEval = 0;
    } else {
        testMetrics = state.lastTestMetrics!;
        state.snapshotsSinceLastTestEval++;
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
        state.step,
        state.epoch,
        trainMetrics,
        testMetrics,
        outputGrid,
        GRID_SIZE,
    );

    if (neuronGrids) {
        snap.neuronGrids = neuronGrids;
    }

    if (demand.needLayerStats) {
        snap.layerStats = state.network.getLayerStats();
    }

    // Add history point to worker history
    if (snap.historyPoint) {
        state.history.push(snap.historyPoint);
    }

    performance.measure('perf:worker:snapshot', 'perf:worker:snapshot:start');
    return snap;
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

    const numBatches = Math.ceil(n / bs);
    const batchSlot = state.step % numBatches;

    if (batchSlot === 0 && state.step > 0 && state.shufflePrng) {
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

    state.step++;
    if (batchSlot === numBatches - 1) {
        state.epoch++;
    }
}

// ── Internal training loop (worker-driven) ──

function trainTick(): void {
    if (!state.running) return;

    if (state.streamPort) {
        performance.mark('perf:worker:trainStep:start');
        for (let i = 0; i < state.stepsPerFrame; i++) {
            trainOneStep();
        }
        performance.measure('perf:worker:trainStep', 'perf:worker:trainStep:start');

        // Compute snapshot and stream it to the main thread
        const snap = computeSnapshot();
        const { message, transferables } = packSnapshotMessage(snap);
        state.streamPort.postMessage(message, transferables);
    }

    // Schedule next tick (yield to allow message processing)
    if (state.running) {
        state.trainLoopTimer = setTimeout(trainTick, 0);
    }
}

function startInternalLoop(): void {
    state.running = true;
    state.trainLoopTimer = setTimeout(trainTick, 0);
}

function stopInternalLoop(): void {
    state.running = false;
    if (state.trainLoopTimer !== null) {
        clearTimeout(state.trainLoopTimer);
        state.trainLoopTimer = null;
    }
}

// ── MessageChannel command handler ──

function handleStreamCommand(cmd: MainToWorkerCommand): void {
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
        const needsRebuild = rebuild ||
            JSON.stringify(state.networkConfig) !== JSON.stringify(networkConfig) ||
            JSON.stringify(state.dataConfig) !== JSON.stringify(dataConfig) ||
            JSON.stringify(state.features) !== JSON.stringify(features);

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
        return computeSnapshot();
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
        return computeSnapshot();
    },
};

export type TrainingWorkerApi = typeof workerApi;

Comlink.expose(workerApi);
