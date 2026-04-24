import { afterEach, describe, expect, it, vi } from 'vitest';
import { Network } from '@nn-playground/engine';
import { DEFAULT_DATA, DEFAULT_FEATURES, DEFAULT_NETWORK, DEFAULT_TRAINING } from '@nn-playground/shared';
import { getTrainingStepsForTick, normalizeTrainingSpeed } from './trainingLoop.ts';
import {
    MAX_WEBGPU_NEURON_READBACK_BYTES,
    estimateWebGpuNeuronReadbackBytes,
    selectWebGpuGridReadbackMode,
    workerApi,
} from './training.worker.ts';

describe('training loop speed bounds', () => {
    it('maps the selected speed to a fixed number of steps per tick', () => {
        expect(getTrainingStepsForTick(1)).toBe(1);
        expect(getTrainingStepsForTick(5)).toBe(5);
        expect(getTrainingStepsForTick(50)).toBe(50);
    });

    it('clamps malformed speed values before the worker loop uses them', () => {
        expect(normalizeTrainingSpeed(0)).toBe(1);
        expect(normalizeTrainingSpeed(Number.POSITIVE_INFINITY)).toBe(1);
        expect(normalizeTrainingSpeed(5000)).toBe(100);
    });
});

describe('WebGPU grid readback policy', () => {
    it('uses GPU output-only mode when only the decision boundary is needed', () => {
        expect(selectWebGpuGridReadbackMode({
            needDecisionBoundary: true,
            needNeuronGrids: false,
            gridLen: 40 * 40,
            neuronCount: 10_000,
        })).toBe('outputOnly');
    });

    it('allows WebGPU neuron grids when the estimated readback is small', () => {
        const gridLen = 40 * 40;
        const neuronCount = Math.floor(MAX_WEBGPU_NEURON_READBACK_BYTES / (gridLen * 4));

        expect(estimateWebGpuNeuronReadbackBytes(neuronCount, gridLen)).toBeLessThanOrEqual(
            MAX_WEBGPU_NEURON_READBACK_BYTES,
        );
        expect(selectWebGpuGridReadbackMode({
            needDecisionBoundary: true,
            needNeuronGrids: true,
            gridLen,
            neuronCount,
        })).toBe('withNeurons');
    });

    it('falls back to CPU when neuron grid readback would be large', () => {
        const gridLen = 40 * 40;
        const neuronCount = Math.floor(MAX_WEBGPU_NEURON_READBACK_BYTES / (gridLen * 4)) + 1;

        expect(estimateWebGpuNeuronReadbackBytes(neuronCount, gridLen)).toBeGreaterThan(
            MAX_WEBGPU_NEURON_READBACK_BYTES,
        );
        expect(selectWebGpuGridReadbackMode({
            needDecisionBoundary: true,
            needNeuronGrids: true,
            gridLen,
            neuronCount,
        })).toBe('cpu');
    });
});

describe('worker training batches', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('trains mini-batches through the indexed engine API without materialized batch arrays', () => {
        const trainBatchSpy = vi.spyOn(Network.prototype, 'trainBatch');
        const trainBatchIndexedSpy = vi.spyOn(Network.prototype, 'trainBatchIndexed');
        const training = { ...DEFAULT_TRAINING, batchSize: 2 };

        workerApi.initialize(
            { ...DEFAULT_NETWORK, inputSize: 2 },
            training,
            { ...DEFAULT_DATA, numSamples: 8, trainTestRatio: 0.5 },
            { ...DEFAULT_FEATURES },
        );
        trainBatchSpy.mockClear();
        trainBatchIndexedSpy.mockClear();

        const snapshot = workerApi.step(1);

        expect(snapshot.step).toBe(1);
        expect(trainBatchSpy).not.toHaveBeenCalled();
        expect(trainBatchIndexedSpy).toHaveBeenCalledTimes(1);
        const [inputs, targets, indices, start, end, config] = trainBatchIndexedSpy.mock.calls[0];
        expect(inputs).toHaveLength(4);
        expect(targets).toHaveLength(4);
        expect(indices).toHaveLength(4);
        expect(start).toBe(0);
        expect(end).toBe(2);
        expect(config).toEqual(training);
    });
});
