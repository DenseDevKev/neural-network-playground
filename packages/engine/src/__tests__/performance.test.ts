import { describe, it } from 'vitest';
import { Network } from '../network.js';
import type { NetworkConfig, TrainingConfig } from '../types.js';

const largeConfig: NetworkConfig = {
    inputSize: 100,
    hiddenLayers: [512, 512, 512],
    outputSize: 10,
    activation: 'tanh',
    outputActivation: 'sigmoid',
    weightInit: 'xavier',
    seed: 42,
};

const trainingConfig: TrainingConfig = {
    learningRate: 0.03,
    batchSize: 32,
    lossType: 'crossEntropy',
    optimizer: 'adam',
    momentum: 0.9,
    regularization: 'l2',
    regularizationRate: 0.01,
    gradientClip: 1.0,
};

describe('Network performance benchmark', () => {
    it('measures applyGradients execution time', { timeout: 60_000 }, () => {
        const net = new Network(largeConfig);
        const batchSize = 32;

        // Warm up
        for (let i = 0; i < 5; i++) {
            net.applyGradients(trainingConfig, batchSize);
        }

        const iterations = 50;
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.applyGradients(trainingConfig, batchSize);
        }
        const end = performance.now();
        const averageTime = (end - start) / iterations;

        console.log(`Average applyGradients time (Adam, L2, Clip): ${averageTime.toFixed(4)}ms`);
    });

    it('measures applyGradients execution time (SGD)', { timeout: 60_000 }, () => {
        const net = new Network(largeConfig);
        const batchSize = 32;
        const sgdConfig = { ...trainingConfig, optimizer: 'sgd' as const, regularization: 'none' as const, gradientClip: null };

        // Warm up
        for (let i = 0; i < 5; i++) {
            net.applyGradients(sgdConfig, batchSize);
        }

        const iterations = 50;
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            net.applyGradients(sgdConfig, batchSize);
        }
        const end = performance.now();
        const averageTime = (end - start) / iterations;

        console.log(`Average applyGradients time (SGD): ${averageTime.toFixed(4)}ms`);
    });
});
