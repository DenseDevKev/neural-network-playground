// ── Training Web Worker ──
// Owns the engine instance, runs training off the main thread.
// Communicates via Comlink RPC.

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
} from '@nn-playground/engine';
import { GRID_SIZE } from '@nn-playground/shared';
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
    state.epoch = 0;

    // Initialise shuffle state — seed is offset from data seed to stay independent.
    const n = state.trainInputs.length;
    state.shuffledIndices = Array.from({ length: n }, (_, i) => i);
    state.shufflePrng = new PRNG((state.dataConfig!.seed ?? 42) + 1234);
}

function computeSnapshot(): NetworkSnapshot {
    if (!state.network || !state.trainingConfig || !state.dataConfig) {
        throw new Error('Not initialized');
    }

    const problemType = state.dataConfig.problemType;
    const lossType = state.trainingConfig.lossType;

    const trainMetrics = state.network.evaluate(
        state.trainInputs,
        state.trainTargets,
        lossType,
        problemType,
    );

    const testMetrics = state.network.evaluate(
        state.testInputs,
        state.testTargets,
        lossType,
        problemType,
    );

    const { outputGrid, neuronGrids } = state.network.predictGridWithNeurons(state.gridInputs);

    const snap = state.network.getSnapshot(
        state.step,
        state.epoch,
        trainMetrics,
        testMetrics,
        outputGrid,
        GRID_SIZE,
    );
    snap.neuronGrids = neuronGrids;
    snap.layerStats = state.network.getLayerStats();
    return snap;
}

function trainOneStep(): void {
    if (!state.network || !state.trainingConfig) return;

    const bs = state.trainingConfig.batchSize;
    const n = state.trainInputs.length;
    if (n === 0) return;

    const numBatches = Math.ceil(n / bs);
    const batchSlot = state.step % numBatches;

    // Re-shuffle at the start of each epoch (except epoch 0, which uses the initial order).
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

const workerApi = {
    initialize(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
    ): NetworkSnapshot {
        state.networkConfig = { ...networkConfig };
        state.trainingConfig = { ...trainingConfig };
        state.dataConfig = { ...dataConfig };
        state.features = { ...features };
        state.running = false;
        buildDataAndNetwork();
        return computeSnapshot();
    },

    updateConfig(
        networkConfig: NetworkConfig,
        trainingConfig: TrainingConfig,
        dataConfig: DataConfig,
        features: FeatureFlags,
        rebuild: boolean,
    ): NetworkSnapshot {
        const needsRebuild = rebuild ||
            JSON.stringify(state.networkConfig) !== JSON.stringify(networkConfig) ||
            JSON.stringify(state.dataConfig) !== JSON.stringify(dataConfig) ||
            JSON.stringify(state.features) !== JSON.stringify(features);

        state.networkConfig = { ...networkConfig };
        state.trainingConfig = { ...trainingConfig };
        state.dataConfig = { ...dataConfig };
        state.features = { ...features };

        if (needsRebuild) {
            state.running = false;
            buildDataAndNetwork();
        }

        return computeSnapshot();
    },

    step(iterations: number = 1): NetworkSnapshot {
        for (let i = 0; i < iterations; i++) {
            trainOneStep();
        }
        return computeSnapshot();
    },

    reset(): NetworkSnapshot {
        buildDataAndNetwork();
        return computeSnapshot();
    },

    getSnapshot(): NetworkSnapshot {
        return computeSnapshot();
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

    // Run continuous training — called in a loop from main thread
    trainAndSnapshot(stepsPerFrame: number): NetworkSnapshot {
        for (let i = 0; i < stepsPerFrame; i++) {
            trainOneStep();
        }
        return computeSnapshot();
    },
};

export type TrainingWorkerApi = typeof workerApi;

Comlink.expose(workerApi);
