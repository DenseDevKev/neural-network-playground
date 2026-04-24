// ── Shared constants ──
import type { FeatureFlags, NetworkConfig, TrainingConfig, DataConfig } from '@nn-playground/engine';

export const DEFAULT_SEED = 42;
export const DEFAULT_NUM_SAMPLES = 300;
export const GRID_SIZE = 40; // heatmap resolution (40×40 = 1600 predictions)

export const MAX_HIDDEN_LAYERS = 6;
export const MAX_NEURONS_PER_LAYER = 32;
export const MIN_NEURONS_PER_LAYER = 1;

export const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
export const BATCH_SIZES = [1, 2, 4, 8, 10, 16, 32, 64];
export const REGULARIZATION_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
export const MOMENTUM_VALUES = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
export const GRADIENT_CLIP_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5];
export const ADAM_BETA1_VALUES = [0.7, 0.8, 0.85, 0.9, 0.95];
export const ADAM_BETA2_VALUES = [0.9, 0.95, 0.98, 0.99, 0.999];
export const HUBER_DELTA_VALUES = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 5];
export const LR_SCHEDULE_GAMMA_VALUES = [0.1, 0.25, 0.5, 0.6, 0.75, 0.9];


export const DEFAULT_FEATURES: FeatureFlags = {
    x: true, y: true,
    xSquared: false, ySquared: false,
    xy: false,
    sinX: false, sinY: false,
    cosX: false, cosY: false,
};

export const DEFAULT_NETWORK: Omit<NetworkConfig, 'inputSize'> = {
    hiddenLayers: [4, 4],
    outputSize: 1,
    activation: 'tanh',
    outputActivation: 'sigmoid',
    weightInit: 'xavier',
    seed: DEFAULT_SEED,
};

export const DEFAULT_TRAINING: TrainingConfig = {
    learningRate: 0.03,
    batchSize: 10,
    lossType: 'crossEntropy',
    optimizer: 'sgd',
    momentum: 0.9,
    regularization: 'none',
    regularizationRate: 0,
    gradientClip: null,
};

export const DEFAULT_DATA: DataConfig = {
    dataset: 'circle',
    problemType: 'classification',
    trainTestRatio: 0.5,
    noise: 0,
    numSamples: DEFAULT_NUM_SAMPLES,
    seed: DEFAULT_SEED,
};
