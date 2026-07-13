import { describe, expect, it } from 'vitest';
import {
    compileObjective,
    Network,
    type CompiledTrainingContractV2,
    type NetworkConfig,
} from '../index.js';

function makeLinearNetwork(overrides: Partial<NetworkConfig> = {}): Network {
    return new Network({
        inputSize: 1,
        hiddenLayers: [],
        outputSize: 1,
        activation: 'linear',
        outputActivation: 'linear',
        weightInit: 'zeros',
        seed: 1,
        ...overrides,
    });
}

describe('deterministic layer statistics', () => {
    it('aggregates population moments across every selected record and neuron', () => {
        const network = makeLinearNetwork({ inputSize: 2, hiddenLayers: [2] });
        network.setWeight(0, 0, 0, 1);
        network.setWeight(0, 0, 1, 2);
        network.setWeight(0, 1, 0, -1);
        network.setWeight(0, 1, 1, 1);
        network.setBias(0, 0, 0);
        network.setBias(0, 1, 1);
        network.setWeight(1, 0, 0, 2);
        network.setWeight(1, 0, 1, -1);
        network.setBias(1, 0, 0.5);

        const result = network.computeLayerStatistics([[1, 0], [0, 1]]);

        expect(result.revision).toBe(network.getRevision());
        expect(result.gradientRevision).toBe(0);
        expect(result.sampleCount).toBe(2);
        expect(result.populationCount).toBe(2);
        expect(result.layers).toHaveLength(2);
        expect(result.layers[0]).toMatchObject({
            meanActivation: 1.25,
            meanAbsWeight: 1.25,
            meanAbsGradient: 0,
        });
        expect(result.layers[0].activationStd).toBeCloseTo(Math.sqrt(0.6875), 12);
        expect(result.layers[1]).toEqual({
            meanActivation: 2.5,
            activationStd: 0,
            meanAbsWeight: 1.5,
            meanAbsGradient: 0,
        });
    });

    it('preserves tightly clustered population variance at high activation magnitudes', () => {
        const network = makeLinearNetwork();
        network.setWeight(0, 0, 0, 1);

        const result = network.computeLayerStatistics([[1e16], [1e16 + 2]]);
        const layer = result.layers[0];

        expect(Number.isFinite(layer.meanActivation)).toBe(true);
        expect(layer.meanActivation).toBeGreaterThanOrEqual(1e16);
        expect(layer.meanActivation).toBeLessThanOrEqual(1e16 + 2);
        expect(layer.activationStd).toBe(1);
    });

    it('keeps representable population spread finite near Number.MAX_VALUE', () => {
        const network = makeLinearNetwork();
        network.setWeight(0, 0, 0, 1);
        const high = 1e308;
        const low = high - 2e292;
        const expectedStd = Math.abs(high - low) / 2;

        const result = network.computeLayerStatistics([[high], [low]]);
        const layer = result.layers[0];

        expect(Number.isFinite(expectedStd)).toBe(true);
        expect(expectedStd).toBeGreaterThan(0);
        expect(Number.isFinite(layer.meanActivation)).toBe(true);
        expect(layer.meanActivation).toBeGreaterThanOrEqual(low);
        expect(layer.meanActivation).toBeLessThanOrEqual(high);
        expect(Number.isFinite(layer.activationStd)).toBe(true);
        expect(layer.activationStd).toBeGreaterThan(0);
        expect(layer.activationStd / expectedStd).toBeCloseTo(1, 12);
    });

    it('selects a deterministic prefix capped at 128 and ignores unrelated forwards', () => {
        const network = makeLinearNetwork();
        network.setWeight(0, 0, 0, 1);
        const inputs = Array.from({ length: 200 }, (_, value) => [value]);

        const defaultResult = network.computeLayerStatistics(inputs);
        const explicitResult = network.computeLayerStatistics(inputs, 5);
        network.forward([999]);
        network.predictGrid([[500], [-500]]);
        const repeatedResult = network.computeLayerStatistics(inputs);

        expect(defaultResult.sampleCount).toBe(128);
        expect(defaultResult.populationCount).toBe(200);
        expect(defaultResult.layers[0].meanActivation).toBe(63.5);
        expect(defaultResult.layers[0].activationStd).toBeCloseTo(Math.sqrt(1365.25), 12);
        expect(explicitResult.sampleCount).toBe(5);
        expect(explicitResult.layers[0].meanActivation).toBe(2);
        expect(explicitResult.layers[0].activationStd).toBeCloseTo(Math.sqrt(2), 12);
        expect(repeatedResult).toEqual(defaultResult);
    });

    it('reports the most recently applied clipped weight gradient and its revision', () => {
        const network = makeLinearNetwork();
        network.setBias(0, 0, 1);
        const training: CompiledTrainingContractV2 = {
            learningRate: 0.1,
            batchSize: 1,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: {
                kind: 'global-norm',
                maximumNorm: 0.1,
                scope: 'total-objective-gradient',
            },
            objective: compileObjective({
                dataLoss: { kind: 'mean-squared-error' },
                penalty: { kind: 'none' },
                reduction: 'mean-per-sample',
            }, network.config),
        };
        const update = network.trainBatchV2([[1]], [[0]], training);
        const clippedGradient = network.getRecentGradientSnapshot().weightGradients[0][0][0];
        network.setBias(0, 0, network.getBias(0, 0) + 1);
        network.forward([100]);
        network.backward([1000], 'mse');

        const result = network.computeLayerStatistics([[0], [1]]);

        expect(result.revision).toBe(network.getRevision());
        expect(result.gradientRevision).toBe(update.revision);
        expect(result.gradientRevision).not.toBe(result.revision);
        expect(result.layers[0].meanAbsGradient).toBeCloseTo(Math.abs(clippedGradient), 12);
        expect(result.layers[0].meanAbsGradient).not.toBe(
            Math.abs(network.getWeightGrads()[0][0][0]),
        );
    });

    it('rejects empty, non-finite, wrong-shape, and invalid sample bounds', () => {
        const network = makeLinearNetwork();

        expect(() => network.computeLayerStatistics([])).toThrow(RangeError);
        expect(() => network.computeLayerStatistics([[1, 2]])).toThrow(RangeError);
        expect(() => network.computeLayerStatistics([[Number.NaN]])).toThrow(RangeError);
        expect(() => network.computeLayerStatistics([[1]], 0)).toThrow(RangeError);
        expect(() => network.computeLayerStatistics([[1]], 1.5)).toThrow(RangeError);
    });
});
