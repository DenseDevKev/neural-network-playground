// ── Engine barrel export ──
export { Network, buildGridInputs } from './network.js';
export { PRNG } from './prng.js';
export { getActivation, ACTIVATION_LABELS } from './activations.js';
export type { ActivationFn } from './activations.js';
export {
    getLoss,
    batchLoss,
    LOSS_LABELS,
    isLossCompatible,
    describeLossIncompatibility,
} from './losses.js';
export { getOptimizer, createOptimizerState } from './optimizers.js';
export type { OptimizerHyperparams } from './optimizers.js';
export { computeLearningRate } from './schedules.js';
export type { LRSchedule, LRScheduleType } from './schedules.js';
export { generateDataset, getDefaultProblemType } from './datasets.js';
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
