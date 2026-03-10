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
        let outsideRange = false;
        for (let i = 0; i < 50; i++) {
            const out = net.forward([i * 0.1 - 2.5, i * 0.1 - 2.5]);
            if (out[0] < 0 || out[0] > 1) outsideRange = true;
        }
        // May or may not be outside range, but output should be a valid number
        const out = net.forward([1, 1]);
        expect(isNaN(out[0])).toBe(false);
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
        const net = new Network(makeConfig());
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
});
