import console from 'node:console';
import { describe, expect, it } from 'vitest';
import { Network, buildGridInputs } from '../network.js';
import type { NetworkConfig } from '../types.js';
import { getActiveFeatures } from '../features.js';
import {
    assertPerformanceBudgets,
    measureMedianMsPerIteration,
    type PerformanceBudgetResult,
} from './performanceStatistics.js';

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
const BASELINE_120_PERCENT_MAX_MS = {
    predictGrid: 11.2504104,
    predictGridInto: 11.1464268,
    predictGridWithNeurons: 14.37312,
    predictGridWithNeuronsInto: 12.2388,
} as const;
const WARMUP_ITERATIONS = 5;
const MEASURED_ROUNDS = 7;
const GRID_ITERATIONS_PER_ROUND = 20;
const NEURON_GRID_ITERATIONS_PER_ROUND = 10;

function expectFiniteArray(values: ArrayLike<number>): void {
    expect(values.length).toBeGreaterThan(0);
    for (let i = 0; i < values.length; i++) {
        expect(Number.isFinite(values[i])).toBe(true);
    }
}

describe('Grid Prediction Performance Benchmark', () => {
    it('measures and checks all grid prediction paths', { timeout: 60_000 }, () => {
        const net = new Network(config);
        const outputTarget = new Float32Array(gridInputs.length);
        const totalNeurons = net.getTotalNeuronCount();
        const neuronTarget = new Float32Array(totalNeurons * gridInputs.length);
        const predictionTarget = new Float32Array(gridInputs.length);

        const results: Array<PerformanceBudgetResult & { iterationsPerRound: number }> = [
            {
                name: 'predictGrid',
                measured: measureMedianMsPerIteration(
                    () => { net.predictGrid(gridInputs); },
                    {
                        warmupIterations: WARMUP_ITERATIONS,
                        measuredRounds: MEASURED_ROUNDS,
                        iterationsPerRound: GRID_ITERATIONS_PER_ROUND,
                    },
                ),
                limit: BASELINE_120_PERCENT_MAX_MS.predictGrid,
                iterationsPerRound: GRID_ITERATIONS_PER_ROUND,
            },
            {
                name: 'predictGridInto',
                measured: measureMedianMsPerIteration(
                    () => { net.predictGridInto(gridInputs, predictionTarget); },
                    {
                        warmupIterations: WARMUP_ITERATIONS,
                        measuredRounds: MEASURED_ROUNDS,
                        iterationsPerRound: GRID_ITERATIONS_PER_ROUND,
                    },
                ),
                limit: BASELINE_120_PERCENT_MAX_MS.predictGridInto,
                iterationsPerRound: GRID_ITERATIONS_PER_ROUND,
            },
            {
                name: 'predictGridWithNeurons',
                measured: measureMedianMsPerIteration(
                    () => { net.predictGridWithNeurons(gridInputs); },
                    {
                        warmupIterations: WARMUP_ITERATIONS,
                        measuredRounds: MEASURED_ROUNDS,
                        iterationsPerRound: NEURON_GRID_ITERATIONS_PER_ROUND,
                    },
                ),
                limit: BASELINE_120_PERCENT_MAX_MS.predictGridWithNeurons,
                iterationsPerRound: NEURON_GRID_ITERATIONS_PER_ROUND,
            },
            {
                name: 'predictGridWithNeuronsInto',
                measured: measureMedianMsPerIteration(
                    () => {
                        net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);
                    },
                    {
                        warmupIterations: WARMUP_ITERATIONS,
                        measuredRounds: MEASURED_ROUNDS,
                        iterationsPerRound: NEURON_GRID_ITERATIONS_PER_ROUND,
                    },
                ),
                limit: BASELINE_120_PERCENT_MAX_MS.predictGridWithNeuronsInto,
                iterationsPerRound: NEURON_GRID_ITERATIONS_PER_ROUND,
            },
        ];

        for (const result of results) {
            const medianRoundTotal = result.measured * result.iterationsPerRound;
            console.log(
                `${result.name}: ${result.measured.toFixed(4)}ms median per iteration; `
                + `${medianRoundTotal.toFixed(4)}ms median round `
                + `(${MEASURED_ROUNDS} rounds x ${result.iterationsPerRound} iterations)`,
            );
        }

        const reference = net.predictGrid(gridInputs);
        net.predictGridInto(gridInputs, predictionTarget);
        const neuronReference = net.predictGridWithNeurons(gridInputs);
        net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);

        expect(gridInputs).toHaveLength(GRID_SIZE * GRID_SIZE);
        expect(gridInputs[0]).toHaveLength(activeFeatures.length);
        expect(reference).toHaveLength(gridInputs.length);
        expect(predictionTarget).toHaveLength(gridInputs.length);
        expectFiniteArray(reference);
        expectFiniteArray(predictionTarget);
        for (let i = 0; i < reference.length; i++) {
            expect(predictionTarget[i]).toBeCloseTo(reference[i], 6);
        }
        expect(neuronReference.outputGrid).toHaveLength(gridInputs.length);
        expect(neuronReference.neuronGrids).toHaveLength(totalNeurons);
        expect(outputTarget).toHaveLength(gridInputs.length);
        expect(neuronTarget).toHaveLength(totalNeurons * gridInputs.length);
        expectFiniteArray(neuronReference.outputGrid);
        expectFiniteArray(outputTarget);
        expectFiniteArray(neuronTarget);
        for (let neuronIdx = 0; neuronIdx < neuronReference.neuronGrids.length; neuronIdx++) {
            const neuronGrid = neuronReference.neuronGrids[neuronIdx];
            expect(neuronGrid).toHaveLength(gridInputs.length);
            expectFiniteArray(neuronGrid);
            for (let gridIdx = 0; gridIdx < gridInputs.length; gridIdx++) {
                expect(neuronTarget[neuronIdx * gridInputs.length + gridIdx]).toBeCloseTo(neuronGrid[gridIdx], 6);
            }
        }
        for (let i = 0; i < neuronReference.outputGrid.length; i++) {
            expect(outputTarget[i]).toBeCloseTo(neuronReference.outputGrid[i], 6);
        }

        assertPerformanceBudgets(results);
    });
});
