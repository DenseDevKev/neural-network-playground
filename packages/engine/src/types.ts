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

export type DataLossSpecV2 =
    | { readonly kind: 'binary-cross-entropy-with-logits' }
    | { readonly kind: 'categorical-cross-entropy-with-logits' }
    | { readonly kind: 'mean-squared-error' }
    | { readonly kind: 'huber'; readonly delta: number };

export type PenaltySpecV2 =
    | { readonly kind: 'none' }
    | {
        readonly kind: 'l1' | 'l2';
        readonly coefficient: number;
        readonly applyTo: 'weights';
    };

export interface ObjectiveSpecV2 {
    readonly dataLoss: DataLossSpecV2;
    readonly penalty: PenaltySpecV2;
    readonly reduction: 'mean-per-sample';
}

export type GradientClipSpecV2 =
    | { readonly kind: 'none' }
    | {
        readonly kind: 'global-norm';
        readonly maximumNorm: number;
        readonly scope: 'total-objective-gradient';
    };

export type OptimizerSpecV2 =
    | { kind: 'sgd' }
    | { kind: 'sgd-momentum'; momentum: number }
    | { kind: 'adam'; beta1: number; beta2: number; epsilon: number };

export type LearningRateScheduleV2 =
    | { kind: 'constant' }
    | { kind: 'step'; interval: number; gamma: number }
    | { kind: 'cosine'; totalSteps: number; minimumRate: number };

export interface ObjectiveBreakdown {
    dataLoss: number;
    regularizationPenalty: number;
    totalObjective: number;
}

export interface GradientDiagnostics {
    dataGradientNorm: number;
    penaltyGradientNorm: number;
    totalGradientNorm: number;
    clippedGradientNorm: number;
    clipScale: number;
}

export interface BatchTrainingResult {
    revision: number;
    step: number;
    sampleCount: number;
    objective: ObjectiveBreakdown;
    gradients: GradientDiagnostics;
}

/** Exact complete gradient supplied to the optimizer for the most recent update. */
export interface RecentGradientSnapshot {
    revision: number;
    weightGradients: number[][][];
    biasGradients: number[][];
}

export interface ClipResult {
    totalGradientNorm: number;
    clippedGradientNorm: number;
    clipScale: number;
}

/** Mutable numeric storage accepted by objective kernels without framework coupling. */
export interface MutableNumericArray extends ArrayLike<number> {
    [index: number]: number;
}

/** Network fields needed to compile and validate an objective. */
export interface ObjectiveNetworkConfig {
    outputSize: number;
    outputActivation: ActivationType;
}

export interface CompiledObjective {
    readonly spec: ObjectiveSpecV2;
    evaluateDataSample(
        logits: ArrayLike<number>,
        outputs: ArrayLike<number>,
        target: ArrayLike<number>,
    ): number;
    seedOutputDeltaInto(
        logits: ArrayLike<number>,
        outputs: ArrayLike<number>,
        target: ArrayLike<number>,
        destination: MutableNumericArray,
    ): void;
    regularizationPenalty(weights: readonly ArrayLike<number>[]): number;
    addPenaltyGradientInto(
        weights: readonly ArrayLike<number>[],
        weightGradients: readonly MutableNumericArray[],
    ): number;
}

export type TaskKind =
    | 'binary-classification'
    | 'multiclass-classification'
    | 'regression';

export type BinaryDatasetId =
    | 'circle'
    | 'xor'
    | 'gauss'
    | 'spiral'
    | 'moons'
    | 'checkerboard'
    | 'rings'
    | 'heart';

export type MulticlassDatasetId = 'three-class-clusters';

export type RegressionDatasetId =
    | 'reg-plane'
    | 'reg-gauss';

export type DatasetId = BinaryDatasetId | MulticlassDatasetId | RegressionDatasetId;

/** Compatibility alias for the pre-version-2 positional generator API. */
export type DatasetType = DatasetId;

export interface DatasetContract {
    id: DatasetId;
    taskKind: TaskKind;
    inputDomain: {
        x: readonly [number, number];
        y: readonly [number, number];
    };
    targetDomain:
        | { kind: 'binary'; values: readonly [0, 1] }
        | { kind: 'classes'; classCount: 3 }
        | {
            kind: 'continuous';
            boundsForNoise: (noise: number) => readonly [number, number];
        };
    noise: {
        minimum: number;
        maximum: number;
        meaning: 'coordinate-perturbation' | 'target-perturbation';
    };
    generatorVersion: number;
}

/** Strict, validated request accepted by the version-2 dataset generator. */
export interface DatasetGenerationRequest {
    dataset: DatasetId;
    sampleCount: number;
    noise: number;
    seed: number;
    trainFraction?: number;
}

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

export type FeatureId = keyof FeatureFlags;

export interface CompilableExperimentRecipe {
    data: { sampleCount: number; trainFraction: number; noise: number; seed: number };
    inputs: { featureIds: readonly FeatureId[] };
    model: {
        hiddenLayers: readonly number[];
        hiddenActivation: ScalarActivationType;
        initialization: WeightInitType;
        seed: number;
    };
    training: {
        batchSize: number;
        learningRate: number;
        schedule: LearningRateScheduleV2;
        optimizer: OptimizerSpecV2;
        gradientClipping: GradientClipSpecV2;
    };
    task:
        | { kind: 'binary-classification'; dataset: BinaryDatasetId }
        | { kind: 'multiclass-classification'; dataset: 'three-class-clusters' }
        | { kind: 'regression'; dataset: RegressionDatasetId };
    objective: ObjectiveSpecV2;
}

export interface CompiledTrainingContractV2 {
    learningRate: number;
    batchSize: number;
    schedule: LearningRateScheduleV2;
    optimizer: OptimizerSpecV2;
    gradientClipping: GradientClipSpecV2;
    objective: CompiledObjective;
}

export interface CompiledDataConfig {
    dataset: DatasetId;
    sampleCount: number;
    trainFraction: number;
    noise: number;
    seed: number;
}

export type CompiledTaskContract =
    | {
        kind: 'binary-classification';
        dataset: BinaryDatasetId;
        outputSize: 1;
        outputActivation: 'sigmoid';
        target: { kind: 'scalar'; values: readonly [0, 1] };
    }
    | {
        kind: 'multiclass-classification';
        dataset: 'three-class-clusters';
        outputSize: 3;
        outputActivation: 'softmax';
        target: { kind: 'one-hot'; length: 3 };
    }
    | {
        kind: 'regression';
        dataset: RegressionDatasetId;
        outputSize: 1;
        outputActivation: 'linear';
        target: { kind: 'scalar'; finite: true };
    };

export interface CompiledExperimentConfig {
    network: NetworkConfig;
    training: CompiledTrainingContractV2;
    data: CompiledDataConfig;
    features: FeatureFlags;
    objective: CompiledObjective;
    task: CompiledTaskContract;
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
