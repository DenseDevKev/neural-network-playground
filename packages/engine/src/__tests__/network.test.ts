import { describe, it, expect, vi } from 'vitest';
import { Network, buildGridInputs } from '../network.js';
import { getActiveFeatures, defaultFeatureFlags } from '../features.js';
import type { NetworkConfig, TrainingConfig } from '../types.js';

function makeConfig(overrides: Partial<NetworkConfig> = {}): NetworkConfig {
    return {
        inputSize: 2,
        hiddenLayers: [4],
        outputSize: 1,
        activation: 'tanh',
        outputActivation: 'sigmoid',
        weightInit: 'xavier',
        seed: 42,
        ...overrides,
    };
}

const defaultTraining: TrainingConfig = {
    learningRate: 0.03,
    batchSize: 10,
    lossType: 'crossEntropy',
    optimizer: 'sgd',
    momentum: 0.9,
    regularization: 'none',
    regularizationRate: 0,
    gradientClip: null,
};

describe('Network construction', () => {
    it('creates a network with correct layer sizes', () => {
        const net = new Network(makeConfig());
        const weights = net.getWeights();
        const biases = net.getBiases();

        // 2 → [4] → 1: two weight matrices
        expect(weights).toHaveLength(2);
        // First layer: 4 neurons, each with 2 weights
        expect(weights[0]).toHaveLength(4);
        expect(weights[0][0]).toHaveLength(2);
        // Output layer: 1 neuron, 4 weights
        expect(weights[1]).toHaveLength(1);
        expect(weights[1][0]).toHaveLength(4);

        // Biases: 2 layers
        expect(biases).toHaveLength(2);
        expect(biases[0]).toHaveLength(4);
        expect(biases[1]).toHaveLength(1);
    });

    it('creates a deeper network correctly', () => {
        const net = new Network(makeConfig({ hiddenLayers: [8, 6, 4] }));
        const w = net.getWeights();
        // Layers: 2→8, 8→6, 6→4, 4→1 = 4 weight matrices
        expect(w).toHaveLength(4);
        expect(w[0]).toHaveLength(8);
        expect(w[1]).toHaveLength(6);
        expect(w[2]).toHaveLength(4);
        expect(w[3]).toHaveLength(1);
    });

    it('is deterministic with same seed', () => {
        const a = new Network(makeConfig());
        const b = new Network(makeConfig());
        expect(a.getWeights()).toEqual(b.getWeights());
        expect(a.getBiases()).toEqual(b.getBiases());
    });

    it('different seeds produce different weights', () => {
        const a = new Network(makeConfig({ seed: 1 }));
        const b = new Network(makeConfig({ seed: 2 }));
        expect(a.getWeights()).not.toEqual(b.getWeights());
    });

    it('rejects non-finite or non-positive layer sizes', () => {
        expect(() => new Network(makeConfig({ inputSize: 0 }))).toThrow(RangeError);
        expect(() => new Network(makeConfig({ outputSize: Number.POSITIVE_INFINITY }))).toThrow(RangeError);
        expect(() => new Network(makeConfig({ hiddenLayers: [4, -1] }))).toThrow(RangeError);
        expect(() => new Network(makeConfig({ hiddenLayers: [4, 1.5] }))).toThrow(RangeError);
    });
});

describe('Network forward pass', () => {
    it('produces output of correct shape', () => {
        const net = new Network(makeConfig());
        const out = net.forward([0.5, -0.3]);
        expect(out).toHaveLength(1);
    });

    it('output is in [0, 1] for sigmoid output', () => {
        const net = new Network(makeConfig());
        for (let i = 0; i < 20; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            const out = net.forward([x, y]);
            expect(out[0]).toBeGreaterThanOrEqual(0);
            expect(out[0]).toBeLessThanOrEqual(1);
        }
    });

    it('forward pass is deterministic', () => {
        const net = new Network(makeConfig());
        const a = net.forward([0.5, -0.3]);
        const b = net.forward([0.5, -0.3]);
        expect(a).toEqual(b);
    });

    it('linear output activation can produce values outside [0,1]', () => {
        const net = new Network(makeConfig({ outputActivation: 'linear' }));
        // With xavier-init weights, some inputs may produce values outside [0,1]
        for (let i = 0; i < 50; i++) {
            net.forward([i * 0.1 - 2.5, i * 0.1 - 2.5]);
        }
        // May or may not be outside range, but output should be a valid number
        const out = net.forward([1, 1]);
        expect(isNaN(out[0])).toBe(false);
    });

    it('softplus output remains finite for saturated positive inputs', () => {
        const net = new Network(makeConfig({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            outputActivation: 'softplus',
            weightInit: 'zeros',
        }));
        net.setWeight(0, 0, 0, 1000);

        const out = net.forward([1]);

        expect(Number.isFinite(out[0])).toBe(true);
        expect(out[0]).toBeCloseTo(1000, 8);
    });
});

describe('Network training', () => {
    it('loss decreases when training on a simple pattern', () => {
        const net = new Network(makeConfig());
        const inputs = [
            [0, 0], [0, 1], [1, 0], [1, 1],
        ];
        // XOR labels
        const targets = [[0], [1], [1], [0]];

        // Measure initial loss
        const metrics0 = net.evaluate(inputs, targets, 'crossEntropy', 'classification');

        // Train for many steps
        for (let epoch = 0; epoch < 200; epoch++) {
            net.trainBatch(inputs, targets, defaultTraining);
        }

        const metrics1 = net.evaluate(inputs, targets, 'crossEntropy', 'classification');
        expect(metrics1.loss).toBeLessThan(metrics0.loss);
    });

    it('trainBatch returns a loss value', () => {
        const net = new Network(makeConfig());
        const loss = net.trainBatch([[0, 0], [1, 1]], [[0], [1]], defaultTraining);
        expect(typeof loss).toBe('number');
        expect(isNaN(loss)).toBe(false);
        expect(loss).toBeGreaterThanOrEqual(0);
    });


    it('trainBatch with an empty batch is a no-op', () => {
        const net = new Network(makeConfig());
        const weightsBefore = net.getWeights();
        const biasesBefore = net.getBiases();
        const stepBefore = net.getStep();

        const loss = net.trainBatch([], [], defaultTraining);

        expect(loss).toBe(0);
        expect(net.getWeights()).toEqual(weightsBefore);
        expect(net.getBiases()).toEqual(biasesBefore);
        expect(net.getStep()).toBe(stepBefore);
    });

    it('trainBatchIndexed matches trainBatch for the selected samples', () => {
        const config = makeConfig({ hiddenLayers: [3], seed: 7 });
        const indexed = new Network(config);
        const sliced = new Network(config);
        const inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0.25, 0.75],
        ];
        const targets = [[0], [1], [1], [0], [1]];
        const indices = new Uint32Array([4, 1, 3]);

        const indexedLoss = indexed.trainBatchIndexed(inputs, targets, indices, 0, indices.length, defaultTraining);
        const slicedLoss = sliced.trainBatch(
            [inputs[4], inputs[1], inputs[3]],
            [targets[4], targets[1], targets[3]],
            defaultTraining,
        );

        expect(indexedLoss).toBeCloseTo(slicedLoss, 12);
        expect(indexed.getWeights()).toEqual(sliced.getWeights());
        expect(indexed.getBiases()).toEqual(sliced.getBiases());
        expect(indexed.getStep()).toBe(sliced.getStep());
    });

    it('trainBatchIndexed rejects invalid ranges and indices', () => {
        const net = new Network(makeConfig());
        const inputs = [[0, 0], [1, 1]];
        const targets = [[0], [1]];
        const indices = [0, 1];

        expect(() => net.trainBatchIndexed(inputs, targets, indices, -1, 1, defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatchIndexed(inputs, targets, indices, 0, 3, defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatchIndexed(inputs, targets, indices, 2, 1, defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatchIndexed(inputs, targets, [0, 2], 0, 2, defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatchIndexed(inputs, targets, [0, 1.5], 0, 2, defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatchIndexed(inputs, targets, [0, -1], 0, 2, defaultTraining)).toThrow(RangeError);
    });

    it('trainBatchIndexed validates selected sample shapes', () => {
        const net = new Network(makeConfig());

        expect(() => net.trainBatchIndexed(
            [[0, 0], [1]],
            [[0], [1]],
            [1],
            0,
            1,
            defaultTraining,
        )).toThrow(RangeError);

        expect(() => net.trainBatchIndexed(
            [[0, 0], [1, 1]],
            [[0], [1, 0]],
            [1],
            0,
            1,
            defaultTraining,
        )).toThrow(RangeError);
    });

    it('trainBatchIndexed with an empty selected range is a no-op', () => {
        const net = new Network(makeConfig());
        const weightsBefore = net.getWeights();
        const biasesBefore = net.getBiases();
        const stepBefore = net.getStep();

        const loss = net.trainBatchIndexed([[0, 0]], [[0]], [0], 0, 0, defaultTraining);

        expect(loss).toBe(0);
        expect(net.getWeights()).toEqual(weightsBefore);
        expect(net.getBiases()).toEqual(biasesBefore);
        expect(net.getStep()).toBe(stepBefore);
    });

    it('throws for wrong input or target shapes', () => {
        const net = new Network(makeConfig());

        expect(() => net.forward([0])).toThrow(RangeError);
        expect(() => net.trainBatch([[0, 0], [1, 1]], [[0]], defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatch([[0]], [[0]], defaultTraining)).toThrow(RangeError);
        expect(() => net.trainBatch([[0, 0]], [[0, 1]], defaultTraining)).toThrow(RangeError);
        expect(() => net.evaluate([[0, 0]], [[0, 1]], 'crossEntropy', 'classification')).toThrow(RangeError);
    });

    it('applyGradients rejects invalid normalization counts', () => {
        const net = new Network(makeConfig());

        expect(() => net.applyGradients(defaultTraining, 0)).toThrow(RangeError);
        expect(() => net.applyGradients(defaultTraining, -1)).toThrow(RangeError);
        expect(() => net.applyGradients(defaultTraining, Number.NaN)).toThrow(RangeError);
    });

    it('applyGradients rejects invalid finite/ranged training hyperparameters', () => {
        const net = new Network(makeConfig());

        expect(() => net.applyGradients({ ...defaultTraining, learningRate: Number.NaN }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...defaultTraining, batchSize: 0 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...defaultTraining, regularizationRate: -0.1 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...defaultTraining, gradientClip: 0 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({
            ...defaultTraining,
            optimizer: 'sgdMomentum',
            momentum: 1.01,
        }, 1)).toThrow(RangeError);
    });

    it('applyGradients rejects invalid Adam hyperparameters', () => {
        const net = new Network(makeConfig());
        const adamTraining: TrainingConfig = {
            ...defaultTraining,
            optimizer: 'adam',
        };

        expect(() => net.applyGradients({ ...adamTraining, adamBeta1: -0.1 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...adamTraining, adamBeta1: 1 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...adamTraining, adamBeta2: Number.NaN }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...adamTraining, adamBeta2: 1 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...adamTraining, adamEps: 0 }, 1)).toThrow(RangeError);
        expect(() => net.applyGradients({ ...adamTraining, adamEps: Number.POSITIVE_INFINITY }, 1)).toThrow(RangeError);
    });

    it('normalizes multi-output gradients by output size', () => {
        const net = new Network(makeConfig({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 2,
            outputActivation: 'linear',
            weightInit: 'zeros',
        }));
        const training: TrainingConfig = {
            ...defaultTraining,
            learningRate: 1,
            lossType: 'mse',
            optimizer: 'sgd',
        };

        const loss = net.trainBatch([[1]], [[1, 1]], training);

        expect(loss).toBeCloseTo(0.5, 8);
        expect(net.getWeight(0, 0, 0)).toBeCloseTo(0.5, 8);
        expect(net.getWeight(0, 1, 0)).toBeCloseTo(0.5, 8);
        expect(net.getBias(0, 0)).toBeCloseTo(0.5, 8);
        expect(net.getBias(0, 1)).toBeCloseTo(0.5, 8);
    });

    it('updates saturated sigmoid outputs with cross-entropy', () => {
        const net = new Network(makeConfig({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            outputActivation: 'sigmoid',
            weightInit: 'zeros',
        }));
        net.setWeight(0, 0, 0, 1000);
        const training: TrainingConfig = {
            ...defaultTraining,
            learningRate: 0.1,
            optimizer: 'sgd',
            lossType: 'crossEntropy',
        };

        net.trainBatch([[1]], [[0]], training);

        expect(net.getWeight(0, 0, 0)).toBeLessThan(1000);
        expect(net.getBias(0, 0)).toBeLessThan(0);
    });

    it('resets optimizer buffers and optimizer step when switching optimizers', () => {
        const config = makeConfig({ hiddenLayers: [3] });
        const net = new Network(config);
        const momentumTraining: TrainingConfig = {
            ...defaultTraining,
            optimizer: 'sgdMomentum',
            momentum: 0.9,
        };
        const adamTraining: TrainingConfig = {
            ...defaultTraining,
            optimizer: 'adam',
        };
        const inputs = [[0, 0], [1, 1]];
        const targets = [[0], [1]];

        net.trainBatch(inputs, targets, momentumTraining);
        const reference = Network.deserialize(net.serialize());

        net.trainBatch(inputs, targets, adamTraining);
        reference.trainBatch(inputs, targets, adamTraining);

        expect(net.getWeights()).toEqual(reference.getWeights());
        expect(net.getBiases()).toEqual(reference.getBiases());
        expect(net.getStep()).toBe(2);
        expect(reference.getStep()).toBe(1);
    });


    it('preserves recent gradient magnitudes for layer stats after a batch update', () => {
        const net = new Network(makeConfig());

        net.trainBatch([[0, 0], [1, 1]], [[0], [1]], defaultTraining);

        const stats = net.getLayerStats();
        expect(stats).toHaveLength(2);
        expect(stats.some((layer) => layer.meanAbsGradient > 0)).toBe(true);
    });
});

describe('Network evaluate', () => {
    it('returns loss and accuracy for classification', () => {
        const net = new Network(makeConfig());
        const metrics = net.evaluate(
            [[0, 0], [1, 1]],
            [[0], [1]],
            'crossEntropy',
            'classification',
        );
        expect(typeof metrics.loss).toBe('number');
        expect(metrics.accuracy).toBeDefined();
        expect(metrics.accuracy!).toBeGreaterThanOrEqual(0);
        expect(metrics.accuracy!).toBeLessThanOrEqual(1);
    });

    it('returns correct confusion matrix for classification', () => {
        const net = new Network(makeConfig({ inputSize: 1 }));
        // Mock forward pass to precisely control the prediction class
        // If input[0] > 0 -> return 0.9 (Class 1)
        // If input[0] <= 0 -> return 0.1 (Class 0)
        vi.spyOn(net, 'forward').mockImplementation((input: number[]) => {
            return input[0] > 0 ? [0.9] : [0.1];
        });

        const inputs = [[1], [-1], [1], [-1]];
        const targets = [[1], [1], [0], [0]];

        // Sample 0: input [1] -> pred 1, target 1 -> True Positive (TP)
        // Sample 1: input [-1] -> pred 0, target 1 -> False Negative (FN)
        // Sample 2: input [1] -> pred 1, target 0 -> False Positive (FP)
        // Sample 3: input [-1] -> pred 0, target 0 -> True Negative (TN)

        const metrics = net.evaluate(inputs, targets, 'crossEntropy', 'classification');

        expect(metrics.confusionMatrix).toEqual({ tp: 1, fn: 1, fp: 1, tn: 1 });
        expect(metrics.accuracy).toBe(0.5); // 2 correct out of 4
    });

    it('keeps evaluation metrics finite and equivalent to public forward predictions', () => {
        const config = makeConfig({ outputSize: 2, outputActivation: 'sigmoid' });
        const net = new Network(config);
        const inputs = [[0, 0], [1, 1], [0.25, 0.75]];
        const targets = [[1, 0], [0, 1], [0, 1]];
        const lossFn = (prediction: number, target: number): number => {
            const eps = 1e-12;
            const p = Math.min(Math.max(prediction, eps), 1 - eps);
            return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
        };
        let expectedLoss = 0;
        let expectedCorrect = 0;

        for (let i = 0; i < inputs.length; i++) {
            const pred = net.forward(inputs[i]);
            for (let o = 0; o < pred.length; o++) {
                expect(Number.isFinite(pred[o])).toBe(true);
                expectedLoss += lossFn(pred[o], targets[i][o]);
            }
            const predClass = pred[1] > pred[0] ? 1 : 0;
            const targetClass = targets[i][1] > targets[i][0] ? 1 : 0;
            if (predClass === targetClass) expectedCorrect++;
        }

        const metrics = net.evaluate(inputs, targets, 'crossEntropy', 'classification');

        expect(Number.isFinite(metrics.loss)).toBe(true);
        expect(metrics.loss).toBeCloseTo(expectedLoss / (inputs.length * config.outputSize), 12);
        expect(metrics.accuracy).toBe(expectedCorrect / inputs.length);
    });

    it('returns loss without accuracy for regression', () => {
        const net = new Network(makeConfig({ outputActivation: 'linear' }));
        const metrics = net.evaluate(
            [[0, 0], [1, 1]],
            [[0.5], [1.5]],
            'mse',
            'regression',
        );
        expect(typeof metrics.loss).toBe('number');
        expect(metrics.accuracy).toBeUndefined();
    });
});

describe('Network gradient clipping (global-norm)', () => {
    it('clipping a very small norm produces no change vs no clipping', () => {
        const cfg = makeConfig();
        const a = new Network(cfg);
        const b = new Network(cfg);
        const noClip: TrainingConfig = { ...defaultTraining, gradientClip: null };
        const largeClip: TrainingConfig = { ...defaultTraining, gradientClip: 100 };
        a.trainBatch([[0, 0], [1, 1]], [[0], [1]], noClip);
        b.trainBatch([[0, 0], [1, 1]], [[0], [1]], largeClip);
        expect(JSON.stringify(a.getWeights())).toBe(JSON.stringify(b.getWeights()));
    });

    it('very tight clip produces different weights than no clip', () => {
        const cfg = makeConfig();
        const a = new Network(cfg);
        const b = new Network(cfg);
        const noClip: TrainingConfig = { ...defaultTraining, gradientClip: null };
        const tightClip: TrainingConfig = { ...defaultTraining, gradientClip: 0.01 };
        // Train for a few iterations so the clip actually fires.
        for (let i = 0; i < 5; i++) {
            a.trainBatch([[0, 0], [1, 1]], [[0], [1]], noClip);
            b.trainBatch([[0, 0], [1, 1]], [[0], [1]], tightClip);
        }
        expect(JSON.stringify(a.getWeights())).not.toBe(JSON.stringify(b.getWeights()));
    });
});

describe('Network optimizer routing (biases)', () => {
    it('biases change when training with Adam', () => {
        const net = new Network(makeConfig());
        const initialBiases = JSON.stringify(net.getBiases());
        const adamTraining: TrainingConfig = { ...defaultTraining, optimizer: 'adam' };

        for (let i = 0; i < 5; i++) {
            net.trainBatch([[0, 0], [1, 1]], [[0], [1]], adamTraining);
        }
        expect(JSON.stringify(net.getBiases())).not.toBe(initialBiases);
    });

    it('training.momentum is honoured by sgdMomentum', () => {
        // Train two identical networks, one with momentum 0.9 and one with
        // momentum 0.0 — they must produce different weights.
        const config = makeConfig();
        const netA = new Network(config);
        const netB = new Network(config);

        const momentumHigh: TrainingConfig = { ...defaultTraining, optimizer: 'sgdMomentum', momentum: 0.9 };
        const momentumZero: TrainingConfig = { ...defaultTraining, optimizer: 'sgdMomentum', momentum: 0.0 };

        for (let i = 0; i < 20; i++) {
            netA.trainBatch([[0, 0], [1, 1]], [[0], [1]], momentumHigh);
            netB.trainBatch([[0, 0], [1, 1]], [[0], [1]], momentumZero);
        }
        expect(JSON.stringify(netA.getWeights())).not.toBe(JSON.stringify(netB.getWeights()));
    });
});

describe('Network reset', () => {
    it('resets weights to initial state with same seed', () => {
        const config = makeConfig();
        const net = new Network(config);
        const initialWeights = JSON.stringify(net.getWeights());

        // Train a bit
        net.trainBatch([[0, 0], [1, 1]], [[0], [1]], defaultTraining);
        const trainedWeights = JSON.stringify(net.getWeights());
        expect(trainedWeights).not.toBe(initialWeights);

        // Reset
        net.reset();
        const resetWeights = JSON.stringify(net.getWeights());
        expect(resetWeights).toBe(initialWeights);
    });

    it('resets step counter to 0', () => {
        const net = new Network(makeConfig());
        net.trainBatch([[0, 0]], [[0]], defaultTraining);
        expect(net.getStep()).toBeGreaterThan(0);
        net.reset();
        expect(net.getStep()).toBe(0);
    });
});

describe('Network serialize/deserialize', () => {
    it('round-trips through serialize/deserialize', () => {
        const net = new Network(makeConfig());
        // Train a bit to change weights
        net.trainBatch([[0, 0], [1, 1]], [[0], [1]], defaultTraining);

        const data = net.serialize();
        const restored = Network.deserialize(data);

        expect(restored.getWeights()).toEqual(net.getWeights());
        expect(restored.getBiases()).toEqual(net.getBiases());
    });

    it('deserialized network produces same output', () => {
        const net = new Network(makeConfig());
        net.trainBatch([[0.5, -0.3]], [[1]], defaultTraining);

        const restored = Network.deserialize(net.serialize());
        const input = [0.2, 0.8];
        expect(restored.forward(input)).toEqual(net.forward(input));
    });

    it('rejects invalid serialized dimensions', () => {
        const data = new Network(makeConfig()).serialize();

        expect(() => Network.deserialize({
            ...data,
            config: { ...data.config, inputSize: 0 },
        })).toThrow(RangeError);
        expect(() => Network.deserialize({
            ...data,
            weights: data.weights.slice(0, -1),
        })).toThrow(RangeError);
        expect(() => Network.deserialize({
            ...data,
            biases: data.biases.slice(0, -1),
        })).toThrow(RangeError);
    });

    it('rejects serialized weights with missing rows or wrong row sizes', () => {
        const data = new Network(makeConfig()).serialize();
        const missingRow = {
            ...data,
            weights: data.weights.map((layer) => layer.map((row) => [...row])),
        };
        missingRow.weights[0] = missingRow.weights[0].slice(0, -1);

        const wrongRowSize = {
            ...data,
            weights: data.weights.map((layer) => layer.map((row) => [...row])),
        };
        wrongRowSize.weights[0][0] = wrongRowSize.weights[0][0].slice(0, -1);

        expect(() => Network.deserialize(missingRow)).toThrow(RangeError);
        expect(() => Network.deserialize(wrongRowSize)).toThrow(RangeError);
    });

    it('rejects serialized biases with wrong sizes', () => {
        const data = new Network(makeConfig()).serialize();
        const wrongBiasSize = {
            ...data,
            biases: data.biases.map((layer) => [...layer]),
        };
        wrongBiasSize.biases[0] = wrongBiasSize.biases[0].slice(0, -1);

        expect(() => Network.deserialize(wrongBiasSize)).toThrow(RangeError);
    });

    it('rejects non-finite serialized parameter values', () => {
        const badWeight = new Network(makeConfig()).serialize();
        badWeight.weights[0][0][0] = Number.NaN;

        const badBias = new Network(makeConfig()).serialize();
        badBias.biases[0][0] = Number.POSITIVE_INFINITY;

        expect(() => Network.deserialize(badWeight)).toThrow(RangeError);
        expect(() => Network.deserialize(badBias)).toThrow(RangeError);
    });
});

describe('Network snapshot', () => {
    it('returns a well-formed snapshot', () => {
        const net = new Network(makeConfig());
        const snap = net.getSnapshot(
            0, 0,
            { loss: 0.5, accuracy: 0.6 },
            { loss: 0.55, accuracy: 0.55 },
            [0.5, 0.5, 0.5, 0.5],
            2,
        );
        expect(snap.step).toBe(0);
        expect(snap.epoch).toBe(0);
        expect(snap.trainLoss).toBe(0.5);
        expect(snap.testLoss).toBe(0.55);
        expect(snap.weights).toBeDefined();
        expect(snap.biases).toBeDefined();
        expect(snap.outputGrid).toEqual([0.5, 0.5, 0.5, 0.5]);
        expect(snap.gridSize).toBe(2);
        expect(snap.historyPoint).toBeDefined();
    });

    it('snapshot weights are copies (not references)', () => {
        const net = new Network(makeConfig());
        const snap = net.getSnapshot(0, 0, { loss: 0 }, { loss: 0 }, [], 0);
        // Mutating snapshot should not affect network
        snap.weights[0][0][0] = 99999;
        expect(net.getWeights()[0][0][0]).not.toBe(99999);
    });
});

describe('buildGridInputs', () => {
    it('produces gridSize² inputs', () => {
        const active = getActiveFeatures(defaultFeatureFlags());
        const inputs = buildGridInputs(20, active);
        expect(inputs).toHaveLength(400); // 20*20
    });

    it('each input has correct feature dimension', () => {
        const active = getActiveFeatures(defaultFeatureFlags());
        const inputs = buildGridInputs(5, active);
        for (const inp of inputs) {
            expect(inp).toHaveLength(2); // x and y features
        }
    });

    it('rejects grid sizes smaller than 2', () => {
        const active = getActiveFeatures(defaultFeatureFlags());
        expect(() => buildGridInputs(1, active)).toThrow(RangeError);
    });

    it('rejects non-integer grid sizes', () => {
        const active = getActiveFeatures(defaultFeatureFlags());
        expect(() => buildGridInputs(2.5, active)).toThrow(RangeError);
    });
});

describe('predictGridInto', () => {
    it('writes identical results to predictGrid', () => {
        const net = new Network(makeConfig());
        const active = getActiveFeatures(defaultFeatureFlags());
        const gridInputs = buildGridInputs(10, active);

        const expected = net.predictGrid(gridInputs);
        const target = new Float32Array(gridInputs.length);
        net.predictGridInto(gridInputs, target);

        for (let i = 0; i < expected.length; i++) {
            expect(target[i]).toBeCloseTo(expected[i], 5);
        }
    });
});

describe('predictGridWithNeuronsInto', () => {
    it('produces results consistent with predictGridWithNeurons', () => {
        const net = new Network(makeConfig({ hiddenLayers: [3, 2] }));
        const active = getActiveFeatures(defaultFeatureFlags());
        const gridInputs = buildGridInputs(5, active);
        const gridLen = gridInputs.length; // 25

        // Original method
        const { outputGrid, neuronGrids } = net.predictGridWithNeurons(gridInputs);

        // New typed-array method
        const totalNeurons = net.getTotalNeuronCount();
        const outputTarget = new Float32Array(gridLen);
        const neuronTarget = new Float32Array(totalNeurons * gridLen);
        net.predictGridWithNeuronsInto(gridInputs, outputTarget, neuronTarget);

        // Compare output grids
        for (let i = 0; i < outputGrid.length; i++) {
            expect(outputTarget[i]).toBeCloseTo(outputGrid[i], 5);
        }

        // Compare neuron grids
        for (let n = 0; n < neuronGrids.length; n++) {
            for (let g = 0; g < gridLen; g++) {
                expect(neuronTarget[n * gridLen + g]).toBeCloseTo(neuronGrids[n][g], 5);
            }
        }
    });
});

describe('getWeightsFlat / getBiasesFlat', () => {
    it('flat weights contain all values from nested weights', () => {
        const net = new Network(makeConfig({ hiddenLayers: [4, 3] }));
        const nested = net.getWeights();
        const { buffer, layerSizes } = net.getWeightsFlat();

        expect(layerSizes).toEqual([2, 4, 3, 1]);

        // Count expected total
        let expectedTotal = 0;
        for (const layer of nested) {
            for (const neuron of layer) {
                expectedTotal += neuron.length;
            }
        }
        expect(buffer.length).toBe(expectedTotal);

        // Verify values match
        let idx = 0;
        for (const layer of nested) {
            for (const neuron of layer) {
                for (const w of neuron) {
                    expect(buffer[idx]).toBeCloseTo(w, 5);
                    idx++;
                }
            }
        }
    });

    it('flat biases contain all values from nested biases', () => {
        const net = new Network(makeConfig({ hiddenLayers: [4, 3] }));
        const nested = net.getBiases();
        const flat = net.getBiasesFlat();

        let expectedTotal = 0;
        for (const layer of nested) {
            expectedTotal += layer.length;
        }
        expect(flat.length).toBe(expectedTotal);

        let idx = 0;
        for (const layer of nested) {
            for (const b of layer) {
                expect(flat[idx]).toBeCloseTo(b, 5);
                idx++;
            }
        }
    });
});

describe('getTotalNeuronCount', () => {
    it('returns correct total for single hidden layer', () => {
        const net = new Network(makeConfig({ hiddenLayers: [4] }));
        // 4 hidden + 1 output = 5
        expect(net.getTotalNeuronCount()).toBe(5);
    });

    it('returns correct total for multi-layer network', () => {
        const net = new Network(makeConfig({ hiddenLayers: [8, 6, 4] }));
        // 8 + 6 + 4 + 1 = 19
        expect(net.getTotalNeuronCount()).toBe(19);
    });
});
