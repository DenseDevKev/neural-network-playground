// ── Engine barrel export ──
export { Network, buildGridInputs } from './network.js';
export { normalizeUint32Seed, PRNG } from './prng.js';
export { getActivation, softmax, ACTIVATION_LABELS } from './activations.js';
export type { ActivationFn } from './activations.js';
export {
    getLoss,
    batchLoss,
    categoricalCrossEntropy,
    categoricalCrossEntropyLogitGradient,
    LOSS_LABELS,
    isLossCompatible,
    describeLossIncompatibility,
} from './losses.js';
export {
    applyGradientTransformInto,
    binaryCrossEntropyLogitDelta,
    binaryCrossEntropyWithLogits,
    buildObjectiveBreakdown,
    categoricalCrossEntropyLogitDelta,
    categoricalCrossEntropyWithLogits,
    compileObjective,
    computeGradientTransform,
    gradientNorm,
    huberLoss,
    huberLossDelta,
    l1Penalty,
    l1PenaltyGradient,
    l2Penalty,
    l2PenaltyGradient,
    meanSquaredError,
    meanSquaredErrorDelta,
} from './objective.js';
export { getOptimizer, createOptimizerState } from './optimizers.js';
export type { OptimizerHyperparams } from './optimizers.js';
export { computeLearningRate, sanitizeLRSchedule, validateLRSchedule } from './schedules.js';
export type { LRSchedule, LRScheduleType } from './schedules.js';
export {
    DatasetGenerationError,
    generateDataset,
    generateDatasetV2,
    getDefaultProblemType,
} from './datasets.js';
export type { DatasetGenerationErrorCode } from './datasets.js';
export {
    BINARY_DATASET_IDS,
    DATASET_IDS,
    getDatasetContract,
    MULTICLASS_DATASET_IDS,
    REGRESSION_DATASET_IDS,
} from './datasetContracts.js';
export {
    ALL_FEATURES,
    getActiveFeatures,
    transformPoint,
    transformDataset,
    countActiveFeatures,
    defaultFeatureFlags,
} from './features.js';
export type { FeatureSpec } from './features.js';
export { initWeights, initBiases } from './initialization.js';
export * from './types.js';

// AS-4 — WebGPU grid predictor (capability-detected fallback to CPU).
export {
    WebGPUGridPredictor,
    flattenGridInputs,
    exceedsGpuShape,
    MAX_GPU_WIDTH,
    MAX_GPU_LAYERS,
} from './webgpu/predictGridGPU.js';
export { detectWebGPU, resetWebGPUDetectionCache } from './webgpu/detect.js';
