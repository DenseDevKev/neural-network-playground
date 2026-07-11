// ── Domain types for the neural network engine ──
// Framework-agnostic — no React, no DOM dependencies.

export type ProblemType = 'classification' | 'regression';

export type ScalarActivationType =
    | 'relu'
    | 'tanh'
    | 'sigmoid'
    | 'linear'
    | 'leakyRelu'
    | 'elu'
    | 'swish'
    | 'softplus';

export type ActivationType = ScalarActivationType | 'softmax';

export type ScalarLossType = 'mse' | 'crossEntropy' | 'huber';

export type LossType = ScalarLossType | 'categoricalCrossEntropy';

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
    | 'three-class-clusters'
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

/** Row-major actual-class by predicted-class counts for the bounded 3-class path. */
export type MulticlassConfusionMatrixCounts = readonly [
    number, number, number,
    number, number, number,
    number, number, number,
];

/** Bounded 3x3 confusion matrix for the approved multiclass foundation. */
export interface MulticlassConfusionMatrixData {
    classCount: 3;
    classLabels: readonly [0, 1, 2];
    counts: MulticlassConfusionMatrixCounts;
}

/** Bounded 3-class decision-boundary grid for direct worker snapshots. */
export interface MulticlassBoundaryData {
    classGrid: Uint8Array;
    confidenceGrid: Float32Array;
    gridSize: number;
}

/** Metrics for a single evaluation pass. */
export interface Metrics {
    loss: number;
    accuracy?: number;
    confusionMatrix?: ConfusionMatrixData;
    multiclassConfusionMatrix?: MulticlassConfusionMatrixData;
}

/** Per-layer statistics for inspection. */
export interface LayerStats {
    meanActivation: number;
    activationStd: number;
    meanAbsWeight: number;
    meanAbsGradient: number;
}

/** Runtime-only options for compact activation histogram inspection. */
export interface ActivationHistogramOptions {
    /** Number of fixed-width bins per layer. Defaults to 12. */
    binCount?: number;
    /** Maximum samples to scan. Defaults to 128. */
    maxSamples?: number;
    /** Values with absolute magnitude at or below this count as near-zero. */
    zeroThreshold?: number;
    /** Bounded activations at or beyond this threshold count as saturated. */
    saturationThreshold?: number;
}

/** Per-layer metadata for compact activation histogram bins. */
export interface ActivationHistogramLayer {
    layerIndex: number;
    binCount: number;
    binStart: number;
    binWidth: number;
    minActivation: number;
    maxActivation: number;
    totalCount: number;
    nearZeroCount: number;
    saturatedCount: number;
}

/** Bounded activation histogram payload. `bins` is flattened by layer. */
export interface ActivationHistogramResult {
    layers: ActivationHistogramLayer[];
    bins: Float32Array;
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

export type BackpropExplanationStatus = 'tiny' | 'healthy' | 'large' | 'clipped';

/** Bounded scalar summary for one layer in a dry-run backprop explanation. */
export interface BackpropExplanationLayer {
    layerIndex: number;
    meanAbsErrorSignal: number;
    maxAbsErrorSignal: number;
    meanAbsGradient: number;
    maxAbsGradient: number;
    meanAbsUpdate: number;
    maxAbsUpdate: number;
    meanActivation: number;
    activationStd: number;
    status: BackpropExplanationStatus;
    note: string;
}

/** Pure dry-run preview of the next mini-batch update. */
export interface BackpropExplanation {
    batchSize: number;
    loss: number;
    learningRate: number;
    globalGradientNorm: number;
    globalClipScale: number;
    clipped: boolean;
    layers: BackpropExplanationLayer[];
    summary: string;
}

/** Runtime-only options for a bounded local loss-landscape probe. */
export interface LossLandscapeProbeOptions {
    /** Odd grid size for the 2D probe. Defaults to 7 and is capped at 7. */
    gridSize?: number;
    /** Maximum samples to evaluate. Defaults to 64 and is capped at 64. */
    maxSamples?: number;
    /** Symmetric scalar weight perturbation radius. Defaults to 0.1 and is capped at 1.0. */
    radius?: number;
}

/** Public coordinate metadata for one scalar parameter axis in the probe. */
export interface LossLandscapeParameter {
    kind: 'weight';
    layerIndex: number;
    neuronIndex: number;
    inputIndex: number;
    label: string;
}

/** One of the two deterministic axes used by a local loss-landscape probe. */
export interface LossLandscapeProbeAxis {
    parameter: LossLandscapeParameter;
    offsets: number[];
}

/** Best cell found in the bounded 2D loss grid. */
export interface LossLandscapeBestCell {
    row: number;
    col: number;
    loss: number;
    offsetA: number;
    offsetB: number;
}

/** Bounded, engine-local dry-run result for a tiny 2D loss slice. */
export interface LossLandscapeProbe {
    gridSize: number;
    sampleCount: number;
    radius: number;
    axisA: LossLandscapeProbeAxis;
    axisB: LossLandscapeProbeAxis;
    losses: Float32Array;
    centerLoss: number;
    minLoss: number;
    maxLoss: number;
    best: LossLandscapeBestCell;
    summary: string;
}

/** Serializable network state for save/restore. */
export interface SerializedNetwork {
    config: NetworkConfig;
    weights: number[][][];
    biases: number[][];
}

/** Runtime-only checkpoint state for pausable training timelines. */
export interface NetworkCheckpoint {
    config: NetworkConfig;
    currentStep: number;
    optimizerStep: number;
    activeOptimizer: OptimizerType | null;
    hasMomentumState: boolean;
    hasAdamState: boolean;
    weights: Float64Array[];
    biases: Float64Array[];
    mWeights: Float64Array[];
    mBiases: Float64Array[];
    vWeights: Float64Array[];
    vBiases: Float64Array[];
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
    /** True when test metrics were reused from a previous full evaluation. */
    testMetricsStale?: boolean;

    /** Flattened prediction grid for decision boundary heatmap. */
    outputGrid: ArrayLike<number>;
    gridSize: number; // width/height of the square grid

    /** Per-neuron heatmap grids (optional, on-demand). */
    neuronGrids?: number[][] | Float32Array[] | Float32Array;

    /** Bounded multiclass boundary payload when the worker computed one for this snapshot. */
    multiclassBoundary?: MulticlassBoundaryData;

    /** Per-layer statistics for inspection panel. */
    layerStats?: LayerStats[];

    /** Compact, demand-gated activation histograms for inspection. */
    activationHistograms?: ActivationHistogramResult;

    historyPoint: HistoryPoint;
}
