import { describe, it } from 'vitest';
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

describe('Grid Prediction Performance Benchmark', () => {
    it('compares predictGrid vs predictGridInto', () => {
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

        console.log(`predictGrid: ${(endOld - startOld).toFixed(4)}ms total for ${iterations} iterations`);
        console.log(`predictGridInto: ${(endNew - startNew).toFixed(4)}ms total for ${iterations} iterations`);
    });

    it('compares predictGridWithNeurons vs predictGridWithNeuronsInto', () => {
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

        console.log(`predictGridWithNeurons: ${(endOld - startOld).toFixed(4)}ms total for ${iterations} iterations`);
        console.log(`predictGridWithNeuronsInto: ${(endNew - startNew).toFixed(4)}ms total for ${iterations} iterations`);
    });
});
