import { describe, expect, it } from 'vitest';
import {
    extractNeuronGrid,
    flattenBiases,
    flattenNeuronGrids,
    flattenWeights,
    layerBiasOffset,
    layerWeightOffset,
    readBias,
    readWeight,
    unflattenBiases,
    unflattenWeights,
} from './frameBufferLayout.ts';

describe('frameBufferLayout', () => {
    it('flattens and unflattens weights without changing layout order', () => {
        const weights = [
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [
                [0.5, 0.6],
            ],
        ];

        const { buffer, layerSizes } = flattenWeights(weights);

        expect(layerSizes).toEqual([2, 2, 1]);
        expect(Array.from(buffer)).toEqual(expect.arrayContaining([
            expect.closeTo(0.1),
            expect.closeTo(0.2),
            expect.closeTo(0.3),
            expect.closeTo(0.4),
            expect.closeTo(0.5),
            expect.closeTo(0.6),
        ]));
        expect(unflattenWeights(buffer, layerSizes)).toEqual([
            [
                [expect.closeTo(0.1), expect.closeTo(0.2)],
                [expect.closeTo(0.3), expect.closeTo(0.4)],
            ],
            [
                [expect.closeTo(0.5), expect.closeTo(0.6)],
            ],
        ]);
    });

    it('flattens and unflattens biases without changing layer order', () => {
        const biases = [
            [0.1, 0.2, 0.3],
            [0.4],
        ];

        const buffer = flattenBiases(biases);

        expect(Array.from(buffer)).toEqual([
            expect.closeTo(0.1),
            expect.closeTo(0.2),
            expect.closeTo(0.3),
            expect.closeTo(0.4),
        ]);
        expect(unflattenBiases(buffer, [2, 3, 1])).toEqual([
            [expect.closeTo(0.1), expect.closeTo(0.2), expect.closeTo(0.3)],
            [expect.closeTo(0.4)],
        ]);
    });

    it('flattens neuron grids and extracts subarray views by neuron index', () => {
        const { buffer, layout } = flattenNeuronGrids([
            new Float32Array([0, 1, 2, 3]),
            new Float32Array([4, 5, 6, 7]),
            new Float32Array([8, 9, 10, 11]),
        ], 2);

        expect(layout).toEqual({ count: 3, gridSize: 2 });
        expect(Array.from(extractNeuronGrid(buffer, 0, 4))).toEqual([0, 1, 2, 3]);
        expect(Array.from(extractNeuronGrid(buffer, 1, 4))).toEqual([4, 5, 6, 7]);
        expect(Array.from(extractNeuronGrid(buffer, 2, 4))).toEqual([8, 9, 10, 11]);
        expect(extractNeuronGrid(buffer, 1, 4).buffer).toBe(buffer.buffer);
    });

    it('computes flat weight and bias offsets used by graph renderers', () => {
        const layerSizes = [2, 3, 1];
        const weights = new Float32Array([
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
            0.7, 0.8, 0.9,
        ]);
        const biases = new Float32Array([0.01, 0.02, 0.03, 0.04]);

        expect(layerWeightOffset(layerSizes, 0)).toBe(0);
        expect(layerWeightOffset(layerSizes, 1)).toBe(6);
        expect(readWeight(weights, layerSizes, 0, 2, 1)).toBeCloseTo(0.6);
        expect(readWeight(weights, layerSizes, 1, 0, 2)).toBeCloseTo(0.9);

        expect(layerBiasOffset(layerSizes, 0)).toBe(0);
        expect(layerBiasOffset(layerSizes, 1)).toBe(3);
        expect(readBias(biases, layerSizes, 0, 2)).toBeCloseTo(0.03);
        expect(readBias(biases, layerSizes, 1, 0)).toBeCloseTo(0.04);
    });
});
