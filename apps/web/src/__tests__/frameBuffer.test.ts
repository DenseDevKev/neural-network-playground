import { describe, it, expect, beforeEach } from 'vitest';
import {
    getFrameBuffer,
    getFrameVersions,
    updateFrameBuffer,
    resetFrameBuffer,
} from '../worker/frameBuffer';
import {
    unflattenWeights,
    unflattenBiases,
    extractNeuronGrid
} from '../worker/frameBufferLayout';

describe('frameBuffer', () => {
    beforeEach(() => {
        resetFrameBuffer();
    });

    describe('State mutations', () => {
        it('should expose numeric versions after reset', () => {
            const versions = getFrameVersions();
            expect(typeof versions.frameVersion).toBe('number');
            expect(typeof versions.outputGridVersion).toBe('number');
            expect(typeof versions.neuronGridsVersion).toBe('number');
            expect(typeof versions.paramsVersion).toBe('number');
            expect(typeof versions.layerStatsVersion).toBe('number');
            expect(typeof versions.confusionMatrixVersion).toBe('number');
        });

        it('updateFrameBuffer({}) should not bump any version', () => {
            const initialVersions = getFrameVersions();

            const newVersion = updateFrameBuffer({});

            expect(newVersion).toBe(initialVersions.frameVersion);
            expect(getFrameVersions()).toEqual(initialVersions);
        });

        it('output grid patch should bump only output grid and broad frame versions', () => {
            const initialVersions = getFrameVersions();
            const outputGrid = new Float32Array([0, 0.25, 0.75, 1]);

            const newVersion = updateFrameBuffer({ outputGrid, gridSize: 2 });

            expect(newVersion).toBe(initialVersions.frameVersion + 1);
            expect(getFrameVersions()).toEqual({
                ...initialVersions,
                frameVersion: initialVersions.frameVersion + 1,
                outputGridVersion: initialVersions.outputGridVersion + 1,
            });
            const buffer = getFrameBuffer();
            expect(buffer.outputGrid).toBe(outputGrid);
            expect(buffer.gridSize).toBe(2);
        });

        it('weights/params patch should bump only params and broad frame versions', () => {
            const initialVersions = getFrameVersions();
            const weights = new Float32Array([0.1, 0.2]);
            const biases = new Float32Array([0.3]);

            const newVersion = updateFrameBuffer({
                weights,
                biases,
                weightLayout: { layerSizes: [2, 1] },
            });

            expect(newVersion).toBe(initialVersions.frameVersion + 1);
            expect(getFrameVersions()).toEqual({
                ...initialVersions,
                frameVersion: initialVersions.frameVersion + 1,
                paramsVersion: initialVersions.paramsVersion + 1,
            });
            expect(getFrameBuffer().weights).toBe(weights);
            expect(getFrameBuffer().biases).toBe(biases);
        });

        it('resetFrameBuffer should bump all versions', () => {
            const initialVersions = getFrameVersions();

            resetFrameBuffer();

            expect(getFrameVersions()).toEqual({
                frameVersion: initialVersions.frameVersion + 1,
                outputGridVersion: initialVersions.outputGridVersion + 1,
                neuronGridsVersion: initialVersions.neuronGridsVersion + 1,
                paramsVersion: initialVersions.paramsVersion + 1,
                layerStatsVersion: initialVersions.layerStatsVersion + 1,
                confusionMatrixVersion: initialVersions.confusionMatrixVersion + 1,
            });
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
