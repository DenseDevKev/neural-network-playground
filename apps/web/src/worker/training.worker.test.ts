import { describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

vi.mock('comlink', () => ({
    expose: vi.fn(),
}));

import { workerApi } from './training.worker.ts';

function containsTypedArray(value: unknown): boolean {
    if (ArrayBuffer.isView(value)) return true;
    if (value instanceof ArrayBuffer || value instanceof SharedArrayBuffer) return true;
    if (value === null || typeof value !== 'object') return false;
    return Object.values(value).some((child) => containsTypedArray(child));
}

describe('training worker loss landscape probe RPC', () => {
    it('rejects before worker initialization', async () => {
        const { workerApi: freshWorkerApi } = await import('./training.worker.ts?loss-landscape-before-init');

        expect(() => freshWorkerApi.getLossLandscapeProbe()).toThrow('Not initialized');
    });

    it('returns bounded serializable loss-grid metadata for the current worker model', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK, hiddenLayers: [3, 2] },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 930, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getLossLandscapeProbe({
            gridSize: 7,
            maxSamples: 64,
            radius: 0.1,
        });

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.epoch).toBe(0);
        expect(response.probe.gridSize).toBe(7);
        expect(response.probe.sampleCount).toBeLessThanOrEqual(64);
        expect(response.probe.losses).toHaveLength(49);
        expect(response.probe.losses.every((loss) => Number.isFinite(loss))).toBe(true);
        expect(Number.isFinite(response.probe.centerLoss)).toBe(true);
        expect(Number.isFinite(response.probe.minLoss)).toBe(true);
        expect(Number.isFinite(response.probe.maxLoss)).toBe(true);
        expect(response.probe.axisA.offsets).toHaveLength(7);
        expect(response.probe.axisB.offsets).toHaveLength(7);
        expect(containsTypedArray(response)).toBe(false);
        expect(JSON.stringify(response)).not.toMatch(/inputs|targets|weights|biases|checkpoint/i);
    });

    it('is deterministic and does not advance the next training step', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 931, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const first = workerApi.getLossLandscapeProbe();
        const second = workerApi.getLossLandscapeProbe();
        const afterProbeStep = workerApi.step(1);

        expect(second).toEqual(first);
        expect(afterProbeStep.step).toBe(first.step + 1);
    });

    it('leaves the next real training step equivalent to a control run without probing', () => {
        const network = { ...DEFAULT_NETWORK, hiddenLayers: [3] };
        const training = { ...DEFAULT_TRAINING, batchSize: 5 };
        const data = { ...DEFAULT_DATA, seed: 932, numSamples: 20 };
        const features = { ...DEFAULT_FEATURES };

        workerApi.initialize(network, training, data, features);
        workerApi.getLossLandscapeProbe();
        const probeThenStep = workerApi.step(1);

        workerApi.initialize(network, training, data, features);
        const controlStep = workerApi.step(1);

        expect(probeThenStep.step).toBe(controlStep.step);
        expect(probeThenStep.trainLoss).toBeCloseTo(controlStep.trainLoss, 12);
        expect(probeThenStep.testLoss).toBeCloseTo(controlStep.testLoss, 12);
        expect(probeThenStep.weights).toEqual(controlStep.weights);
        expect(probeThenStep.biases).toEqual(controlStep.biases);
    });

    it('propagates engine bounds for invalid probe options', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 933, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.getLossLandscapeProbe({ gridSize: 8 })).toThrow(RangeError);
        expect(() => workerApi.getLossLandscapeProbe({ maxSamples: 65 })).toThrow(RangeError);
        expect(() => workerApi.getLossLandscapeProbe({ radius: 1.01 })).toThrow(RangeError);
    });
});

describe('training worker backprop explanation RPC', () => {
    it('rejects before worker initialization', async () => {
        const { workerApi: freshWorkerApi } = await import('./training.worker.ts?backprop-before-init');

        expect(() => freshWorkerApi.getBackpropExplanation()).toThrow('Not initialized');
    });

    it('returns finite bounded scalar summaries for the current worker model', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK, hiddenLayers: [3, 2] },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 920, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getBackpropExplanation();

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.epoch).toBe(0);
        expect(response.explanation.batchSize).toBeGreaterThan(0);
        expect(response.explanation.layers).toHaveLength(3);
        expect(Number.isFinite(response.explanation.loss)).toBe(true);
        expect(Number.isFinite(response.explanation.globalGradientNorm)).toBe(true);
        expect(response.explanation.layers.every((layer) => Number.isFinite(layer.meanAbsUpdate))).toBe(true);
        expect(containsTypedArray(response)).toBe(false);
        expect(JSON.stringify(response)).not.toMatch(/inputs|targets|weights|biases/i);
    });

    it('is deterministic and does not advance the next training step', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 921, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const first = workerApi.getBackpropExplanation();
        const second = workerApi.getBackpropExplanation();
        const afterPreviewStep = workerApi.step(1);

        expect(second).toEqual(first);
        expect(afterPreviewStep.step).toBe(first.step + 1);
    });

    it('leaves the next real training step equivalent to a control run without preview', () => {
        const network = { ...DEFAULT_NETWORK, hiddenLayers: [3] };
        const training = { ...DEFAULT_TRAINING, batchSize: 5 };
        const data = { ...DEFAULT_DATA, seed: 922, numSamples: 20 };
        const features = { ...DEFAULT_FEATURES };

        workerApi.initialize(network, training, data, features);
        workerApi.getBackpropExplanation();
        const previewThenStep = workerApi.step(1);

        workerApi.initialize(network, training, data, features);
        const controlStep = workerApi.step(1);

        expect(previewThenStep.step).toBe(controlStep.step);
        expect(previewThenStep.trainLoss).toBeCloseTo(controlStep.trainLoss, 12);
        expect(previewThenStep.testLoss).toBeCloseTo(controlStep.testLoss, 12);
        expect(previewThenStep.weights).toEqual(controlStep.weights);
        expect(previewThenStep.biases).toEqual(controlStep.biases);
    });

    it('refuses the epoch reshuffle boundary instead of mutating shuffle state during preview', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, batchSize: 5 },
            { ...DEFAULT_DATA, seed: 923, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.step(2);

        expect(() => workerApi.getBackpropExplanation()).toThrow(/epoch shuffle boundary/i);
        expect(workerApi.step(1).step).toBe(3);
    });
});

describe('training worker prediction trace RPC', () => {
    it('returns an on-demand trace for a training sample', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 123, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getPredictionTrace({ source: 'train', index: 0 });

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.sample.source).toBe('train');
        expect(response.sample.index).toBe(0);
        expect(response.trace.target).toHaveLength(1);
        expect(response.trace.input.length).toBeGreaterThan(0);
        expect(response.trace.output).toHaveLength(1);
        expect(response.trace.layers.length).toBeGreaterThan(0);
    });

    it('rejects out-of-range sample indexes', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 456, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.getPredictionTrace({ source: 'test', index: 999 })).toThrow(RangeError);
    });
});

describe('training worker multiclass target encoding', () => {
    const multiclassNetwork = {
        ...DEFAULT_NETWORK,
        outputSize: 3,
        outputActivation: 'softmax' as const,
    };
    const multiclassTraining = {
        ...DEFAULT_TRAINING,
        lossType: 'categoricalCrossEntropy' as const,
    };

    it('encodes classification labels as bounded one-hot targets for worker training and traces', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...DEFAULT_DATA, seed: 934, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const snapshot = workerApi.step(1);
        const trace = workerApi.getPredictionTrace({ source: 'train', index: 0 });

        expect(Number.isFinite(snapshot.trainLoss)).toBe(true);
        expect(Number.isFinite(snapshot.testLoss)).toBe(true);
        expect(trace.trace.output).toHaveLength(3);
        expect(trace.trace.target).toHaveLength(3);
        expect(trace.trace.target.reduce((sum, value) => sum + value, 0)).toBe(1);
        expect(trace.trace.target[trace.sample.label ?? -1]).toBe(1);
    });

    it('does not source hidden three-class samples through public worker dataset generation', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...DEFAULT_DATA, seed: 940, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        );

        const labels = [
            ...workerApi.getTrainPoints(),
            ...workerApi.getTestPoints(),
        ].map((point) => point.label);

        expect(new Set(labels)).toEqual(new Set([0, 1]));
        expect(labels).not.toContain(2);
    });

    it('encodes custom multiclass trace labels and rejects out-of-range classes', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...DEFAULT_DATA, seed: 935, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const trace = workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 2,
        });

        expect(trace.trace.target).toEqual([0, 0, 1]);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: -1,
        })).toThrow(/class index/i);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 1.5,
        })).toThrow(/class index/i);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 3,
        })).toThrow(/class index/i);
    });

    it('keeps one-shot inspection RPCs finite and non-mutating with multiclass targets', () => {
        workerApi.initialize(
            multiclassNetwork,
            { ...multiclassTraining, batchSize: 4 },
            { ...DEFAULT_DATA, seed: 939, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const backprop = workerApi.getBackpropExplanation();
        const landscape = workerApi.getLossLandscapeProbe({ gridSize: 5, maxSamples: 16, radius: 0.05 });
        const afterPreviewStep = workerApi.step(1);

        expect(Number.isFinite(backprop.explanation.loss)).toBe(true);
        expect(backprop.explanation.layers.length).toBeGreaterThan(0);
        expect(containsTypedArray(backprop)).toBe(false);
        expect(Number.isFinite(landscape.probe.centerLoss)).toBe(true);
        expect(landscape.probe.losses).toHaveLength(25);
        expect(containsTypedArray(landscape)).toBe(false);
        expect(afterPreviewStep.step).toBe(1);
    });

    it('requires the approved softmax plus categorical cross-entropy worker pairing', () => {
        expect(() => workerApi.initialize(
            {
                ...DEFAULT_NETWORK,
                outputSize: 3,
                outputActivation: 'sigmoid',
            },
            multiclassTraining,
            { ...DEFAULT_DATA, seed: 936, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        )).toThrow(/softmax.*categorical cross-entropy/i);
    });

    it.each([
        [
            'two outputs',
            { outputSize: 2, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { problemType: 'classification' as const },
        ],
        [
            'four outputs',
            { outputSize: 4, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { problemType: 'classification' as const },
        ],
        [
            'softmax with scalar loss',
            { outputSize: 3, outputActivation: 'softmax' as const },
            { lossType: 'crossEntropy' as const },
            { problemType: 'classification' as const },
        ],
        [
            'categorical loss without softmax',
            { outputSize: 3, outputActivation: 'sigmoid' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { problemType: 'classification' as const },
        ],
        [
            'regression data',
            { outputSize: 3, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { problemType: 'regression' as const },
        ],
    ])('rejects unsupported multiclass worker config: %s', (_label, networkOverrides, trainingOverrides, dataOverrides) => {
        expect(() => workerApi.initialize(
            {
                ...DEFAULT_NETWORK,
                ...networkOverrides,
            },
            {
                ...DEFAULT_TRAINING,
                ...trainingOverrides,
            },
            {
                ...DEFAULT_DATA,
                ...dataOverrides,
                seed: 936,
                numSamples: 24,
            },
            { ...DEFAULT_FEATURES },
        )).toThrow(/output size 3, softmax output activation, and categorical cross-entropy loss/i);
    });

    it('does not emit scalar decision-boundary grids or binary confusion matrices for multiclass snapshots', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...DEFAULT_DATA, seed: 938, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: true,
            needConfusionMatrix: true,
        });

        const snapshot = workerApi.step(1);

        expect(snapshot.outputGrid).toHaveLength(0);
        expect(snapshot.neuronGrids).toBeUndefined();
        expect(snapshot.testMetrics.confusionMatrix).toBeUndefined();
    });

    it('keeps live arena runtime guarded to scalar models', () => {
        expect(() => workerApi.initializeArena({
            modelA: {
                label: 'Multiclass A',
                network: multiclassNetwork,
                training: multiclassTraining,
                data: { ...DEFAULT_DATA, seed: 937, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Scalar B',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 937, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
        })).toThrow(/multiclass live arena/i);
    });
});

describe('training worker activation histogram demand', () => {
    it('omits activation histograms until explicitly requested', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 789, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
            needActivationHistograms: false,
        });
        expect(workerApi.step(1).activationHistograms).toBeUndefined();

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
            needActivationHistograms: true,
            activationHistogramInterval: 2,
        });
        const snapshot = workerApi.step(1);

        expect(snapshot.activationHistograms).toBeDefined();
        expect(snapshot.activationHistograms?.bins).toBeInstanceOf(Float32Array);
        expect(snapshot.activationHistograms?.bins.length).toBeGreaterThan(0);
        expect(snapshot.activationHistograms?.layers.length).toBeGreaterThan(0);
        expect(workerApi.step(1).activationHistograms).toBeUndefined();
    });

    it('does not compute activation histograms for layer-stat demand alone', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 890, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
            needActivationHistograms: false,
        });

        expect(workerApi.step(1).activationHistograms).toBeUndefined();
    });

    it('computes activation histograms on the first rebuild snapshot while requested', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 891, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needActivationHistograms: true,
            activationHistogramInterval: 5,
        });

        const result = workerApi.updateConfig(
            {
                ...DEFAULT_NETWORK,
                hiddenLayers: [3],
            },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 891, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            true,
        );

        expect(result.snapshot.activationHistograms).toBeDefined();
        expect(result.snapshot.activationHistograms?.layers).toHaveLength(2);
    });
});

describe('training worker lifecycle and demand cadence', () => {
    it('initializes and steps two scalar-only arena model slots sequentially', () => {
        const arena = workerApi.initializeArena({
            modelA: {
                label: 'Capacity 2',
                network: { ...DEFAULT_NETWORK, hiddenLayers: [2] },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 910, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Capacity 4',
                network: { ...DEFAULT_NETWORK, hiddenLayers: [4] },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 910, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
        });

        expect(arena.summaries.map((summary) => summary.side)).toEqual(['A', 'B']);
        expect(arena.summaries.map((summary) => summary.label)).toEqual(['Capacity 2', 'Capacity 4']);
        expect(arena.summaries.every((summary) => summary.step === 0)).toBe(true);

        const stepped = workerApi.stepArena(3);

        expect(stepped.snapshotId).toBeGreaterThan(arena.snapshotId);
        expect(stepped.summaries.map((summary) => summary.step)).toEqual([3, 3]);
        expect(stepped.summaries.every((summary) => Number.isFinite(summary.trainLoss))).toBe(true);
        expect(stepped.summaries.every((summary) => Number.isFinite(summary.testLoss))).toBe(true);
    });

    it('keeps live arena stepping isolated from the existing single-model worker run', () => {
        const single = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 911, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.initializeArena({
            modelA: {
                label: 'Arena A',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 912, numSamples: 20 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Arena B',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 913, numSamples: 20 },
                features: { ...DEFAULT_FEATURES },
            },
        });
        workerApi.stepArena(2);

        const singleAfterArena = workerApi.step(1);

        expect(singleAfterArena.step).toBe(1);
        expect(workerApi.getCheckpointTimeline().checkpoints[0].step).toBe(single.snapshot.step);
    });

    it('captures bounded checkpoint metadata and restores an earlier checkpoint', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, optimizer: 'adam' },
            { ...DEFAULT_DATA, seed: 906, numSamples: 30 },
            { ...DEFAULT_FEATURES },
        );

        let timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints).toHaveLength(1);
        expect(timeline.checkpoints[0].step).toBe(init.snapshot.step);

        workerApi.step(6);
        timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints.map((checkpoint) => checkpoint.step)).toContain(6);

        const firstCheckpointId = timeline.checkpoints[0].id;
        const restored = workerApi.restoreCheckpoint(firstCheckpointId);
        expect(restored.snapshot.step).toBe(0);
        expect(restored.timeline.restoredCheckpointId).toBe(firstCheckpointId);
        expect(workerApi.step(1).step).toBe(1);
    });

    it('evicts old checkpoints when the runtime timeline reaches its bound', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 907, numSamples: 30 },
            { ...DEFAULT_FEATURES },
        );

        for (let i = 0; i < 12; i++) {
            workerApi.step(5);
        }

        const timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints.length).toBeLessThanOrEqual(timeline.maxCheckpoints);
        expect(timeline.evictedCount).toBeGreaterThan(0);
        expect(timeline.checkpoints[0].step).toBeGreaterThan(0);
    });

    it('preserves the run on training-only config updates and rebuilds for shape changes', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const trainingOnly = workerApi.updateConfig(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, learningRate: DEFAULT_TRAINING.learningRate / 2 },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            false,
        );
        expect(trainingOnly.runId).toBe(init.runId);

        const rebuilt = workerApi.updateConfig(
            { ...DEFAULT_NETWORK, hiddenLayers: [2] },
            { ...DEFAULT_TRAINING, learningRate: DEFAULT_TRAINING.learningRate / 2 },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            false,
        );
        expect(rebuilt.runId).toBeGreaterThan(init.runId);
        expect(rebuilt.snapshot.step).toBe(0);
    });

    it('reset creates a fresh run after manual training steps', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 902, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(workerApi.step(3).step).toBe(3);

        const reset = workerApi.reset();
        expect(reset.runId).toBeGreaterThan(init.runId);
        expect(reset.snapshot.step).toBe(0);
        expect(reset.snapshot.epoch).toBe(0);
    });

    it('honors decision-boundary cadence without recomputing every snapshot', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 903, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: false,
            gridInterval: 2,
        });

        const fresh = workerApi.step(1);
        const skipped = workerApi.step(1);

        expect(fresh.outputGrid).toBeInstanceOf(Float32Array);
        expect(fresh.outputGrid).toHaveLength(fresh.gridSize * fresh.gridSize);
        expect(skipped.outputGrid).toHaveLength(0);
        expect(skipped.neuronGrids).toBeUndefined();
    });

    it('emits layer stats only while layer-stat demand is enabled', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 904, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
        });
        expect(workerApi.step(1).layerStats).toBeUndefined();

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
        });
        expect(workerApi.step(1).layerStats?.length).toBeGreaterThan(0);
    });

    it('rejects malformed demand updates without clearing the previous demand', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 905, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
        });

        expect(() => workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: 'yes',
        } as unknown as typeof DEFAULT_DEMAND)).toThrow('Invalid visualization demand.');

        expect(workerApi.step(1).layerStats?.length).toBeGreaterThan(0);
    });
});
