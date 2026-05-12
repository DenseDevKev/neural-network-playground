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
