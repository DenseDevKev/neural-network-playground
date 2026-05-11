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
