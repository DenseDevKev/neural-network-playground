// ── Domain types for the neural network engine ──
// Framework-agnostic — no React, no DOM dependencies.

export type ProblemType = 'classification' | 'regression';

export type ActivationType =
    | 'relu'
    | 'tanh'
    | 'sigmoid'
    | 'linear'
    | 'leakyRelu'
    | 'elu'
    | 'swish'
    | 'softplus';

export type LossType = 'mse' | 'crossEntropy' | 'huber';

export type OptimizerType = 'sgd' | 'sgdMomentum' | 'adam';

export type WeightInitType = 'xavier' | 'he' | 'uniform' | 'zeros';

export type RegularizationType = 'none' | 'l1' | 'l2';

export type DatasetType =
    | 'circle'
    | 'xor'
    | 'gauss'
    | 'spiral'
    | 'moons'
    | 'checkerboard'
    | 'rings'
    | 'heart'
    | 'reg-plane'
    | 'reg-gauss';

/** Configuration for the network architecture. */
export interface NetworkConfig {
    inputSize: number;
    hiddenLayers: number[]; // neurons per hidden layer
    outputSize: number;
    activation: ActivationType;
    outputActivation: ActivationType;
    weightInit: WeightInitType;
    seed: number;
}

/** Configuration for the training process. */
export interface TrainingConfig {
    learningRate: number;
    batchSize: number;
    lossType: LossType;
    optimizer: OptimizerType;
    /** SGD-with-momentum coefficient; ignored for other optimizers. */
    momentum: number;
    regularization: RegularizationType;
    regularizationRate: number;
    /** Global-norm gradient clip threshold; null disables clipping. */
    gradientClip: number | null;
    /** Adam first-moment decay. Default 0.9 when omitted. */
    adamBeta1?: number;
    /** Adam second-moment decay. Default 0.999 when omitted. */
    adamBeta2?: number;
    /** Adam numerical-stability epsilon. Default 1e-8 when omitted. */
    adamEps?: number;
    /** Huber loss transition point. Default 1.0 when omitted. */
    huberDelta?: number;
    /** Optional learning-rate schedule; omit for constant LR. */
    lrSchedule?: import('./schedules.js').LRSchedule;
}

/** Configuration for data generation. */
export interface DataConfig {
    dataset: DatasetType;
    problemType: ProblemType;
    trainTestRatio: number; // 0–1, fraction used for training
    noise: number;
    numSamples: number;
    seed: number;
}

/** Feature toggle flags — which input features are active. */
export interface FeatureFlags {
    x: boolean;
    y: boolean;
    xSquared: boolean;
    ySquared: boolean;
    xy: boolean;
    sinX: boolean;
    sinY: boolean;
    cosX: boolean;
    cosY: boolean;
}

/** Per-sample data record. */
export interface DataPoint {
    x: number;
    y: number;
    label: number;
}

/** Dataset split result. */
export interface DataSplit {
    train: DataPoint[];
    test: DataPoint[];
}

/** Confusion Matrix for classification tasks. */
export interface ConfusionMatrixData {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
}

/** Metrics for a single evaluation pass. */
export interface Metrics {
    loss: number;
    accuracy?: number;
    confusionMatrix?: ConfusionMatrixData;
}

/** Per-layer statistics for inspection. */
export interface LayerStats {
    meanActivation: number;
    activationStd: number;
    meanAbsWeight: number;
    meanAbsGradient: number;
}

/** A single training history entry. */
export interface HistoryPoint {
    step: number;
    trainLoss: number;
    testLoss: number;
    trainAccuracy?: number;
    testAccuracy?: number;
}

/** Per-layer activations captured for one inspected prediction. */
export interface PredictionTraceLayer {
    layerIndex: number;
    preActivations: number[];
    activations: number[];
}

/** Pure-copy explanation of a single forward prediction. */
export interface PredictionTrace {
    input: number[];
    target?: number[];
    output: number[];
    prediction: number | number[];
    lossContribution?: number;
    layers: PredictionTraceLayer[];
}

/** Serializable network state for save/restore. */
export interface SerializedNetwork {
    config: NetworkConfig;
    weights: number[][][];
    biases: number[][];
}

/**
 * Full snapshot returned from the worker to the main thread.
 * Contains everything the UI needs to render.
 */
export interface NetworkSnapshot {
    step: number;
    epoch: number;

    weights: number[][][];
    biases: number[][];

    trainLoss: number;
    testLoss: number;
    trainMetrics: Metrics;
    testMetrics: Metrics;

    /** Flattened prediction grid for decision boundary heatmap. */
    outputGrid: ArrayLike<number>;
    gridSize: number; // width/height of the square grid

    /** Per-neuron heatmap grids (optional, on-demand). */
    neuronGrids?: number[][] | Float32Array[] | Float32Array;

    /** Per-layer statistics for inspection panel. */
    layerStats?: LayerStats[];

    historyPoint: HistoryPoint;
}
