import { describe, expect, it } from 'vitest';
import { Network, buildGridInputs } from '../network.js';
import type { NetworkConfig } from '../types.js';
import { getActiveFeatures } from '../features.js';

const config: NetworkConfig = {
    inputSize: 2,
    hiddenLayers: [16, 16, 16],
    outputSize: 1,
    activation: 'tanh',
    outputActivation: 'sigmoid',
    weightInit: 'xavier',
    seed: 42,
};

const GRID_SIZE = 100; // Larger grid for better measurement
const features = { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false };
const activeFeatures = getActiveFeatures(features);
const gridInputs = buildGridInputs(GRID_SIZE, activeFeatures);

function expectFiniteArray(values: ArrayLike<number>): void {
    expect(values.length).toBeGreaterThan(0);
    for (let i = 0; i < values.length; i++) {
        expect(Number.isFinite(values[i])).toBe(true);
    }
}

describe('Grid Prediction Performance Benchmark', () => {
    it('compares predictGrid vs predictGridInto', { timeout: 60_000 }, () => {
        const net = new Network(config);
        const target = new Float32Array(gridInputs.length);

        // Warm up
        for (let i = 0; i < 5; i++) {
            net.predictGrid(gridInputs);
            net.predictGridInto(gridInputs, target);
        }

        const iterations = 100;

        const startOld = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.predictGrid(gridInputs);
        }
        const endOld = performance.now();

        const startNew = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.predictGridInto(gridInputs, target);
        }
        const endNew = performance.now();
        const oldTotal = endOld - startOld;
        const newTotal = endNew - startNew;
        const reference = net.predictGrid(gridInputs);
        net.predictGridInto(gridInputs, target);

        console.log(`predictGrid: ${oldTotal.toFixed(4)}ms total for ${iterations} iterations`);
        console.log(`predictGridInto: ${newTotal.toFixed(4)}ms total for ${iterations} iterations`);

        expect(gridInputs).toHaveLength(GRID_SIZE * GRID_SIZE);
        expect(gridInputs[0]).toHaveLength(activeFeatures.length);
        expect(reference).toHaveLength(gridInputs.length);
        expect(target).toHaveLength(gridInputs.length);
        expect(Number.isFinite(oldTotal)).toBe(true);
        expect(Number.isFinite(newTotal)).toBe(true);
        expect(oldTotal).toBeGreaterThanOrEqual(0);
        expect(newTotal).toBeGreaterThanOrEqual(0);
        expectFiniteArray(reference);
        expectFiniteArray(target);
        for (let i = 0; i < reference.length; i++) {
            expect(target[i]).toBeCloseTo(reference[i], 6);
        }
    });

    it('compares predictGridWithNeurons vs predictGridWithNeuronsInto', { timeout: 60_000 }, () => {
        const net = new Network(config);
        const outputTarget = new Float32Array(gridInputs.length);
        const totalNeurons = net.getTotalNeuronCount();
        const neuronTarget = new Float32Array(totalNeurons * gridInputs.length);

        // Warm up
        for (let i = 0; i < 5; i++) {
            net.predictGridWithNeurons(gridInputs);
            net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);
        }

        const iterations = 50;

        const startOld = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.predictGridWithNeurons(gridInputs);
        }
        const endOld = performance.now();

        const startNew = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);
        }
        const endNew = performance.now();
        const oldTotal = endOld - startOld;
        const newTotal = endNew - startNew;
        const reference = net.predictGridWithNeurons(gridInputs);
        net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);

        console.log(`predictGridWithNeurons: ${oldTotal.toFixed(4)}ms total for ${iterations} iterations`);
        console.log(`predictGridWithNeuronsInto: ${newTotal.toFixed(4)}ms total for ${iterations} iterations`);

        expect(Number.isFinite(oldTotal)).toBe(true);
        expect(Number.isFinite(newTotal)).toBe(true);
        expect(oldTotal).toBeGreaterThanOrEqual(0);
        expect(newTotal).toBeGreaterThanOrEqual(0);
        expect(reference.outputGrid).toHaveLength(gridInputs.length);
        expect(reference.neuronGrids).toHaveLength(totalNeurons);
        expect(outputTarget).toHaveLength(gridInputs.length);
        expect(neuronTarget).toHaveLength(totalNeurons * gridInputs.length);
        expectFiniteArray(reference.outputGrid);
        expectFiniteArray(outputTarget);
        expectFiniteArray(neuronTarget);
        for (let neuronIdx = 0; neuronIdx < reference.neuronGrids.length; neuronIdx++) {
            const neuronGrid = reference.neuronGrids[neuronIdx];
            expect(neuronGrid).toHaveLength(gridInputs.length);
            expectFiniteArray(neuronGrid);
            for (let gridIdx = 0; gridIdx < gridInputs.length; gridIdx++) {
                expect(neuronTarget[neuronIdx * gridInputs.length + gridIdx]).toBeCloseTo(neuronGrid[gridIdx], 6);
            }
        }
        for (let i = 0; i < reference.outputGrid.length; i++) {
            expect(outputTarget[i]).toBeCloseTo(reference.outputGrid[i], 6);
        }
    });
});
