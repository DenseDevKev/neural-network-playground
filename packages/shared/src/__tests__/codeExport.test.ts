import { describe, expect, it } from 'vitest';
import {
    generatePseudocode,
    generateNumPy,
    generateTFJS,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    DEFAULT_FEATURES,
} from '../index.js';
import type { NetworkSnapshot, NetworkConfig } from '@nn-playground/engine';

const mockConfig: NetworkConfig = {
    ...DEFAULT_NETWORK,
    inputSize: 2,
    hiddenLayers: [2], // 2 -> 2 -> 1
};

const mockSnapshot: NetworkSnapshot = {
    step: 100,
    epoch: 1,
    weights: [
        [[0.1, 0.2], [0.3, 0.4]], // Layer 1: 2 neurons, 2 inputs
        [[0.5, 0.6]]             // Layer 2: 1 neuron, 2 inputs
    ],
    biases: [
        [0.1, 0.2],
        [0.3]
    ],
    trainLoss: 0.1,
    testLoss: 0.12,
    trainMetrics: { loss: 0.1, accuracy: 0.9 },
    testMetrics: { loss: 0.12, accuracy: 0.88 },
    outputGrid: [],
    gridSize: 0,
    historyPoint: { step: 100, trainLoss: 0.1, testLoss: 0.12 }
};

describe('codeExport', () => {
    describe('generatePseudocode', () => {
        it('generates pseudocode without snapshot', () => {
            const output = generatePseudocode(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, null);
            expect(output).toContain('Neural Network — Pseudocode');
            expect(output).toContain('Architecture: 2 → 2 → 1');
            expect(output).toContain('INPUT features = [x, y]');
            expect(output).not.toContain('Trained weights');
        });

        it('generates pseudocode with snapshot', () => {
            const output = generatePseudocode(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, mockSnapshot);
            expect(output).toContain('Trained weights (step 100)');
            expect(output).toContain('neuron 0: bias=0.1000  weights=[0.1000, 0.2000]');
        });
    });

    describe('generateNumPy', () => {
        it('generates NumPy code without snapshot', () => {
            const output = generateNumPy(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, null);
            expect(output).toContain('import numpy as np');
            expect(output).toContain('def tanh(x):');
            expect(output).toContain('Train the model first');
        });

        it('generates NumPy code with snapshot', () => {
            const output = generateNumPy(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, mockSnapshot);
            expect(output).toContain('W1 = np.array([');
            expect(output).toContain('[0.100000, 0.200000]');
            expect(output).toContain('def predict(x):');
        });
    });

    describe('generateTFJS', () => {
        it('generates TFJS code without snapshot', () => {
            const output = generateTFJS(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, null);
            expect(output).toContain("import * as tf from '@tensorflow/tfjs'");
            expect(output).toContain('model.add(tf.layers.dense');
            expect(output).toContain("loss: 'binaryCrossentropy'");
            expect(output).not.toContain('Load trained weights');
        });

        it('generates TFJS code with snapshot', () => {
            const output = generateTFJS(mockConfig, DEFAULT_TRAINING, DEFAULT_FEATURES, mockSnapshot);
            expect(output).toContain('Load trained weights');
            expect(output).toContain('Layer 1: 2×2, bias: 2');
        });
    });
});
