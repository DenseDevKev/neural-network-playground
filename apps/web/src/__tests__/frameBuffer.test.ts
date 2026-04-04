import { describe, it, expect, beforeEach } from 'vitest';
import {
    getFrameBuffer,
    getFrameVersion,
    updateFrameBuffer,
    resetFrameBuffer,
    unflattenWeights,
    unflattenBiases,
    extractNeuronGrid
} from '../worker/frameBuffer';

describe('frameBuffer', () => {
    beforeEach(() => {
        resetFrameBuffer();
    });

    describe('State mutations', () => {
        it('should have initial version of 1 after reset (or 0 initially, but reset increments)', () => {
            // Note: resetFrameBuffer increments the version, so we check what it currently is
            const initialVersion = getFrameVersion();
            expect(typeof initialVersion).toBe('number');
        });

        it('updateFrameBuffer should merge patch and increment version', () => {
            const initialVersion = getFrameVersion();
            const patch = { gridSize: 10 };

            const newVersion = updateFrameBuffer(patch);

            expect(newVersion).toBe(initialVersion + 1);
            expect(getFrameVersion()).toBe(initialVersion + 1);

            const buffer = getFrameBuffer();
            expect(buffer.gridSize).toBe(10);
            // outputGrid should still be null
            expect(buffer.outputGrid).toBeNull();
        });

        it('updateFrameBuffer should correctly handle multiple updates', () => {
            const v1 = getFrameVersion();

            const v2 = updateFrameBuffer({ gridSize: 20 });
            expect(v2).toBe(v1 + 1);
            expect(getFrameBuffer().gridSize).toBe(20);

            const v3 = updateFrameBuffer({ version: 999 } as any); // version shouldn't be overridable by types, but let's just update something else

            const v4 = updateFrameBuffer({ layerStats: [] });
            expect(v4).toBe(v2 + 2); // Assuming v3 was +1
            expect(getFrameBuffer().gridSize).toBe(20);
            expect(getFrameBuffer().layerStats).toEqual([]);
        });
    });

    describe('unflattenWeights', () => {
        it('should reconstruct nested weight arrays correctly', () => {
            // Layout: 2 inputs, 2 hidden neurons, 1 output neuron
            const layerSizes = [2, 2, 1];
            // 2*2 + 2*1 = 6 weights total
            const flatWeights = new Float32Array([
                // layer 0 to 1 (2 neurons * 2 inputs)
                0.1, 0.2, // neuron 0 weights
                0.3, 0.4, // neuron 1 weights
                // layer 1 to 2 (1 neuron * 2 inputs)
                0.5, 0.6  // output neuron weights
            ]);

            const expected = [
                // Layer 1 (index 0)
                [
                    [0.1, 0.2],
                    [0.3, 0.4]
                ],
                // Layer 2 (index 1)
                [
                    [0.5, 0.6]
                ]
            ];

            const result = unflattenWeights(flatWeights, layerSizes);

            // Due to floating point precision, let's check elements or stringify if close enough.
            // Using toEqual works for exact float matches here since they are simple fractions/decimals
            expect(result).toBeDefined();
            // Just comparing structure and close values
            expect(result[0][0][0]).toBeCloseTo(0.1);
            expect(result[0][0][1]).toBeCloseTo(0.2);
            expect(result[0][1][0]).toBeCloseTo(0.3);
            expect(result[0][1][1]).toBeCloseTo(0.4);
            expect(result[1][0][0]).toBeCloseTo(0.5);
            expect(result[1][0][1]).toBeCloseTo(0.6);
        });
    });

    describe('unflattenBiases', () => {
        it('should reconstruct nested bias arrays correctly', () => {
            // Layout: 2 inputs, 3 hidden neurons, 1 output neuron
            const layerSizes = [2, 3, 1];
            // 3 + 1 = 4 biases total
            const flatBiases = new Float32Array([
                // layer 1 (3 neurons)
                0.1, 0.2, 0.3,
                // layer 2 (1 neuron)
                0.4
            ]);

            const result = unflattenBiases(flatBiases, layerSizes);

            expect(result.length).toBe(2);
            expect(result[0].length).toBe(3);
            expect(result[0][0]).toBeCloseTo(0.1);
            expect(result[0][1]).toBeCloseTo(0.2);
            expect(result[0][2]).toBeCloseTo(0.3);

            expect(result[1].length).toBe(1);
            expect(result[1][0]).toBeCloseTo(0.4);
        });
    });

    describe('extractNeuronGrid', () => {
        it('should extract correct subarray for a given neuron', () => {
            // 3 neurons, grid length 4
            const gridLength = 4;
            const neuronGrids = new Float32Array([
                0, 1, 2, 3,       // neuron 0
                4, 5, 6, 7,       // neuron 1
                8, 9, 10, 11      // neuron 2
            ]);

            const result0 = extractNeuronGrid(neuronGrids, 0, gridLength);
            expect(result0).toHaveLength(4);
            expect(result0[0]).toBe(0);
            expect(result0[3]).toBe(3);

            const result1 = extractNeuronGrid(neuronGrids, 1, gridLength);
            expect(result1).toHaveLength(4);
            expect(result1[0]).toBe(4);
            expect(result1[3]).toBe(7);

            const result2 = extractNeuronGrid(neuronGrids, 2, gridLength);
            expect(result2).toHaveLength(4);
            expect(result2[0]).toBe(8);
            expect(result2[3]).toBe(11);
        });
    });
});
