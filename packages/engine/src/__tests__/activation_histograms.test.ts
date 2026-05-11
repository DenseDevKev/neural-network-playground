import { describe, expect, it } from 'vitest';
import { Network } from '../network.js';
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
});
