import { describe, expect, it } from 'vitest';
import {
    compileObjective,
    Network,
    type CompiledTrainingContractV2,
    type NetworkConfig,
    type ObjectiveSpecV2,
    type OptimizerSpecV2,
    type TrainingConfig,
} from '../index.js';

const regressionObjective: ObjectiveSpecV2 = {
    dataLoss: { kind: 'mean-squared-error' },
    penalty: { kind: 'none' },
    reduction: 'mean-per-sample',
};

function makeNetwork(overrides: Partial<NetworkConfig> = {}): Network {
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

function compileTraining(
    network: Network,
    objectiveSpec: ObjectiveSpecV2,
    options: {
        learningRate?: number;
        optimizer?: OptimizerSpecV2;
        gradientClipping?: CompiledTrainingContractV2['gradientClipping'];
    } = {},
): CompiledTrainingContractV2 {
    return {
        learningRate: options.learningRate ?? 1,
        batchSize: 1,
        schedule: { kind: 'constant' },
        optimizer: options.optimizer ?? { kind: 'sgd' },
        gradientClipping: options.gradientClipping ?? { kind: 'none' },
        objective: compileObjective(objectiveSpec, network.config),
    };
}

function snapshotNorm(snapshot: {
    weightGradients: number[][][];
    biasGradients: number[][];
}): number {
    let squaredNorm = 0;
    for (const layer of snapshot.weightGradients) {
        for (const neuron of layer) {
            for (const gradient of neuron) squaredNorm += gradient * gradient;
        }
    }
    for (const layer of snapshot.biasGradients) {
        for (const gradient of layer) squaredNorm += gradient * gradient;
    }
    return Math.sqrt(squaredNorm);
}

describe('Network V2 objective-gradient updates', () => {
    it('adds the L2 gradient before clipping the complete objective gradient', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        });
        network.setWeight(0, 0, 0, 100);
        const objective = compileObjective({
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'l2', coefficient: 1, applyTo: 'weights' },
            reduction: 'mean-per-sample',
        }, network.config);
        const training: CompiledTrainingContractV2 = {
            learningRate: 1,
            batchSize: 1,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: {
                kind: 'global-norm',
                maximumNorm: 0.01,
                scope: 'total-objective-gradient',
            },
            objective,
        };

        const result = network.trainBatchV2([[0]], [[0]], training);

        expect(result.objective).toEqual({
            dataLoss: 0,
            regularizationPenalty: 5000,
            totalObjective: 5000,
        });
        expect(result.gradients).toEqual({
            dataGradientNorm: 0,
            penaltyGradientNorm: 100,
            totalGradientNorm: 100,
            clippedGradientNorm: 0.01,
            clipScale: 0.0001,
        });
        expect(network.getWeight(0, 0, 0)).toBeCloseTo(99.99, 12);
    });

    it.each([
        ['SGD', { kind: 'sgd' }],
        ['momentum', { kind: 'sgd-momentum', momentum: 0.9 }],
        ['Adam', { kind: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }],
    ] as const)('captures the exact clipped weight and bias gradients for %s', (_name, optimizer) => {
        const network = makeNetwork();
        network.setBias(0, 0, 1);
        const training = compileTraining(network, regressionObjective, {
            optimizer,
            gradientClipping: {
                kind: 'global-norm',
                maximumNorm: 0.01,
                scope: 'total-objective-gradient',
            },
        });

        const result = network.trainBatchV2([[1]], [[0]], training);
        const snapshot = network.getRecentGradientSnapshot();

        expect(snapshot.revision).toBe(result.revision);
        expect(snapshot.weightGradients[0][0][0]).not.toBe(0);
        expect(snapshot.biasGradients[0][0]).not.toBe(0);
        expect(snapshotNorm(snapshot)).toBeCloseTo(0.01, 12);

        snapshot.weightGradients[0][0][0] = 999;
        snapshot.biasGradients[0][0] = 999;
        expect(snapshotNorm(network.getRecentGradientSnapshot())).toBeCloseTo(0.01, 12);
    });

    it('reduces accumulated gradients by sample count rather than output width', () => {
        const network = makeNetwork({ outputSize: 3, outputActivation: 'softmax' });
        const training = compileTraining(network, {
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        });

        const result = network.trainBatchV2([[1]], [[1, 0, 0]], training);

        expect(result.sampleCount).toBe(1);
        expect(result.objective.dataLoss).toBeCloseTo(Math.log(3), 12);
        expect(network.getWeight(0, 0, 0)).toBeCloseTo(2 / 3, 12);
        expect(network.getWeight(0, 1, 0)).toBeCloseTo(-1 / 3, 12);
        expect(network.getWeight(0, 2, 0)).toBeCloseTo(-1 / 3, 12);
        expect(network.getBias(0, 0)).toBeCloseTo(2 / 3, 12);
        expect(network.getBias(0, 1)).toBeCloseTo(-1 / 3, 12);
        expect(network.getBias(0, 2)).toBeCloseTo(-1 / 3, 12);
    });

    it('uses the compiled binary logits loss and delta at extreme logits', () => {
        const network = makeNetwork({ outputActivation: 'sigmoid' });
        network.setBias(0, 0, 1000);
        const training = compileTraining(network, {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        });

        const result = network.trainBatchV2([[0]], [[0]], training);

        expect(result.objective.dataLoss).toBe(1000);
        expect(Number.isFinite(result.objective.totalObjective)).toBe(true);
        expect(network.getRecentGradientSnapshot().biasGradients[0][0]).toBe(1);
        expect(network.getBias(0, 0)).toBe(999);
    });

    it('uses the compiled categorical logits loss and delta at extreme logits', () => {
        const network = makeNetwork({ outputSize: 3, outputActivation: 'softmax' });
        network.setBias(0, 0, 1000);
        network.setBias(0, 1, 0);
        network.setBias(0, 2, -1000);
        const training = compileTraining(network, {
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        });

        const result = network.trainBatchV2([[0]], [[0, 1, 0]], training);

        expect(result.objective.dataLoss).toBe(1000);
        expect(Number.isFinite(result.objective.totalObjective)).toBe(true);
        expect(network.getRecentGradientSnapshot().biasGradients[0]).toEqual([1, -1, 0]);
        expect(network.getBiases()[0]).toEqual([999, 1, -1000]);
    });

    it('advances model revision for setters and updates while step advances only for updates', () => {
        const network = makeNetwork();
        const training = compileTraining(network, regressionObjective, { learningRate: 0.1 });

        expect(network.getRevision()).toBe(0);
        expect(network.getStep()).toBe(0);

        network.setWeight(0, 0, 0, 0.25);
        expect(network.getRevision()).toBe(1);
        expect(network.getStep()).toBe(0);
        network.setBias(0, 0, -0.5);
        expect(network.getRevision()).toBe(2);
        expect(network.getStep()).toBe(0);

        const first = network.trainBatchV2([[1]], [[0]], training);
        expect(first).toMatchObject({ revision: 3, step: 1, sampleCount: 1 });
        expect(network.getRevision()).toBe(3);
        expect(network.getStep()).toBe(1);

        const second = network.trainBatchIndexedV2(
            [[1], [2]],
            [[0], [1]],
            new Uint32Array([1, 0]),
            0,
            1,
            training,
        );
        expect(second).toMatchObject({ revision: 4, step: 2, sampleCount: 1 });
        expect(network.getRecentGradientSnapshot().revision).toBe(4);

        network.setBias(0, 0, network.getBias(0, 0) + 1);
        expect(network.getRevision()).toBe(5);
        expect(network.getStep()).toBe(2);
        expect(network.getRecentGradientSnapshot().revision).toBe(4);
    });

    it('clears partial accumulators when compiled objective validation aborts a batch', () => {
        const failedThenValid = makeNetwork({ outputActivation: 'sigmoid' });
        const clean = makeNetwork({ outputActivation: 'sigmoid' });
        const objective: ObjectiveSpecV2 = {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        };
        const failedTraining = compileTraining(failedThenValid, objective, { learningRate: 0.1 });
        const cleanTraining = compileTraining(clean, objective, { learningRate: 0.1 });

        expect(() => failedThenValid.trainBatchV2(
            [[1], [1]],
            [[1], [0.5]],
            failedTraining,
        )).toThrow(RangeError);
        expect(failedThenValid.getStep()).toBe(0);
        expect(failedThenValid.getRevision()).toBe(0);

        failedThenValid.trainBatchV2([[1]], [[0]], failedTraining);
        clean.trainBatchV2([[1]], [[0]], cleanTraining);

        expect(failedThenValid.getWeights()).toEqual(clean.getWeights());
        expect(failedThenValid.getBiases()).toEqual(clean.getBiases());
    });

    it('preserves the legacy adapter while adding its L2 gradient exactly once', () => {
        const network = makeNetwork();
        network.setWeight(0, 0, 0, 10);
        const training: TrainingConfig = {
            learningRate: 0.1,
            batchSize: 1,
            lossType: 'mse',
            optimizer: 'sgd',
            momentum: 0,
            regularization: 'l2',
            regularizationRate: 0.5,
            gradientClip: null,
        };

        network.applyGradients(training, 1);

        expect(network.getWeight(0, 0, 0)).toBeCloseTo(9.5, 12);
        expect(network.getRecentGradientSnapshot().weightGradients[0][0][0]).toBe(5);
    });
});
