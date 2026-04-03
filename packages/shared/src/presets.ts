// ── Curated presets ──
import type { Preset } from './types.js';

export const PRESETS: Preset[] = [
    {
        id: 'single-neuron',
        title: 'Single Neuron Linear Separator',
        description: 'A single neuron can only learn a linear boundary.',
        learningGoal: 'Understand that a single neuron computes a weighted sum and can only separate linearly.',
        difficulty: 'beginner',
        config: {
            data: { dataset: 'gauss', problemType: 'classification', noise: 0, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [], outputSize: 1, activation: 'tanh', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
    {
        id: 'xor-hidden',
        title: 'XOR Needs Hidden Layers',
        description: 'XOR is not linearly separable — you need at least one hidden layer.',
        learningGoal: 'See that hidden layers enable non-linear decision boundaries.',
        difficulty: 'beginner',
        config: {
            data: { dataset: 'xor', problemType: 'classification', noise: 0, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [4, 4], outputSize: 1, activation: 'tanh', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
    {
        id: 'circle-one-layer',
        title: 'Circle with One Hidden Layer',
        description: 'A simple circle dataset with a single hidden layer.',
        learningGoal: 'One hidden layer with enough neurons can learn a circular boundary.',
        difficulty: 'intermediate',
        config: {
            data: { dataset: 'circle', problemType: 'classification', noise: 0, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [6], outputSize: 1, activation: 'relu', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
    {
        id: 'spiral-deep',
        title: 'Spiral with Deeper Network',
        description: 'The spiral dataset requires a deeper network to learn.',
        learningGoal: 'Deeper networks can learn more complex decision boundaries.',
        difficulty: 'advanced',
        config: {
            data: { dataset: 'spiral', problemType: 'classification', noise: 0, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [8, 8, 4], outputSize: 1, activation: 'tanh', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            training: { learningRate: 0.03, batchSize: 10, lossType: 'crossEntropy', optimizer: 'sgd', momentum: 0.9, regularization: 'none', regularizationRate: 0, gradientClip: null },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
    {
        id: 'regression-plane',
        title: 'Regression with No Hidden Layer',
        description: 'A simple linear regression on a plane surface.',
        learningGoal: 'Linear regression can perfectly fit a plane with no hidden layers.',
        difficulty: 'beginner',
        config: {
            data: { dataset: 'reg-plane', problemType: 'regression', noise: 5, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [], outputSize: 1, activation: 'tanh', outputActivation: 'linear', weightInit: 'xavier', seed: 42 },
            training: { learningRate: 0.01, batchSize: 10, lossType: 'mse', optimizer: 'sgd', momentum: 0.9, regularization: 'none', regularizationRate: 0, gradientClip: null },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
    {
        id: 'feature-engineering',
        title: 'Feature Engineering Helps',
        description: 'Adding x² and y² features makes the circle trivially separable.',
        learningGoal: 'Good features can simplify a problem — even without hidden layers.',
        difficulty: 'intermediate',
        config: {
            data: { dataset: 'circle', problemType: 'classification', noise: 0, trainTestRatio: 0.5, numSamples: 300, seed: 42 },
            network: { inputSize: 4, hiddenLayers: [], outputSize: 1, activation: 'tanh', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            features: { x: true, y: true, xSquared: true, ySquared: true, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
        },
    },
];
