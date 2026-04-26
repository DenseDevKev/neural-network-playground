import { describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_DATA,
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
