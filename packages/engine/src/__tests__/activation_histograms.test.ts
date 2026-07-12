import { describe, expect, it } from 'vitest';
import { Network } from '../network.js';
import { NonFiniteNumericalError } from '../numericalError.js';
import type { NetworkConfig } from '../types.js';

function makeConfig(overrides: Partial<NetworkConfig> = {}): NetworkConfig {
    return {
        inputSize: 1,
        hiddenLayers: [1],
        outputSize: 1,
        activation: 'linear',
        outputActivation: 'linear',
        weightInit: 'zeros',
        seed: 7,
        ...overrides,
    };
}

function captureNonFinite(action: () => unknown): NonFiniteNumericalError {
    try {
        action();
    } catch (error) {
        expect(error).toBeInstanceOf(NonFiniteNumericalError);
        return error as NonFiniteNumericalError;
    }
    throw new Error('expected a NonFiniteNumericalError');
}

describe('Network activation histograms', () => {
    it('computes deterministic bounded bins without exposing raw activations', () => {
        const net = new Network(makeConfig());
        net.setWeight(0, 0, 0, 1);
        net.setWeight(1, 0, 0, 1);

        const result = net.computeActivationHistograms(
            [[-1], [0], [1]],
            { binCount: 2, maxSamples: 16 },
        );

        expect(result.layers).toHaveLength(2);
        expect(result.bins).toBeInstanceOf(Float32Array);
        expect(result.bins).toHaveLength(4);
        expect(Array.from(result.bins)).toEqual([1, 2, 1, 2]);
        expect(result.layers[0]).toMatchObject({
            layerIndex: 0,
            binCount: 2,
            binStart: -1,
            binWidth: 1,
            minActivation: -1,
            maxActivation: 1,
            totalCount: 3,
            nearZeroCount: 1,
            saturatedCount: 0,
        });
    });

    it('counts bounded activation saturation for sigmoid layers', () => {
        const net = new Network(makeConfig({
            hiddenLayers: [],
            outputActivation: 'sigmoid',
        }));
        net.setWeight(0, 0, 0, 10);

        const result = net.computeActivationHistograms(
            [[-1], [0], [1]],
            { binCount: 4, saturationThreshold: 0.95 },
        );

        expect(result.layers).toHaveLength(1);
        expect(result.layers[0].totalCount).toBe(3);
        expect(result.layers[0].nearZeroCount).toBe(1);
        expect(result.layers[0].saturatedCount).toBe(2);
        expect(result.bins.reduce((sum, count) => sum + count, 0)).toBe(3);
    });

    it('caps sample count deterministically for bounded worker payloads', () => {
        const net = new Network(makeConfig());
        net.setWeight(0, 0, 0, 1);
        net.setWeight(1, 0, 0, 1);

        const result = net.computeActivationHistograms(
            [[-2], [-1], [0], [1], [2]],
            { binCount: 4, maxSamples: 2 },
        );

        expect(result.layers[0].totalCount).toBe(2);
        expect(result.layers[1].totalCount).toBe(2);
        expect(result.bins.reduce((sum, count) => sum + count, 0)).toBe(4);
    });

    it('reports the exact activation that becomes non-finite', () => {
        const net = new Network(makeConfig({ hiddenLayers: [] }));
        net.setWeight(0, 0, 0, Number.MAX_VALUE);

        const error = captureNonFinite(() => net.computeActivationHistograms([
            [Number.MAX_VALUE],
        ]));

        expect(error).toMatchObject({
            path: 'activationHistograms.activations[0][0][0]',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('reports overflow in derived histogram metadata', () => {
        const net = new Network(makeConfig({ hiddenLayers: [] }));
        net.setWeight(0, 0, 0, 1);

        const error = captureNonFinite(() => net.computeActivationHistograms(
            [[-Number.MAX_VALUE], [Number.MAX_VALUE]],
        ));

        expect(error).toMatchObject({
            path: 'activationHistograms.layers[0].range',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('reports non-finite derived bin positions before indexing the bins array', () => {
        const net = new Network(makeConfig({ hiddenLayers: [] }));
        net.setWeight(0, 0, 0, 1);

        const error = captureNonFinite(() => net.computeActivationHistograms(
            [[0], [Number.MIN_VALUE]],
        ));

        expect(error.path).toBe('activationHistograms.binning[0][0][0].position');
        expect(Number.isNaN(error.value)).toBe(true);
    });
});
