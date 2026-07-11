import type {
    LRSchedule,
    NetworkConfig,
    TrainingConfig,
} from '@nn-playground/engine';
import type {
    AppConfig,
    PreparedExperimentDocumentV2,
} from '@nn-playground/shared';
import { DEFAULT_TRAINING } from '@nn-playground/shared';

function deepFreeze<T>(value: T, seen = new WeakSet<object>()): T {
    if (typeof value !== 'object' || value === null || seen.has(value)) return value;
    seen.add(value);
    for (const child of Object.values(value)) deepFreeze(child, seen);
    return Object.freeze(value);
}

function projectSchedule(
    prepared: PreparedExperimentDocumentV2,
): LRSchedule | undefined {
    const schedule = prepared.document.recipe.training.schedule;
    switch (schedule.kind) {
        case 'constant':
            return undefined;
        case 'step':
            return {
                type: 'step',
                stepSize: schedule.interval,
                gamma: schedule.gamma,
            };
        case 'cosine':
            return {
                type: 'cosine',
                totalSteps: schedule.totalSteps,
                minLr: schedule.minimumRate,
            };
    }
}

function projectNetwork(prepared: PreparedExperimentDocumentV2): NetworkConfig {
    const network = prepared.compiled.network;
    return {
        inputSize: network.inputSize,
        hiddenLayers: [...network.hiddenLayers],
        outputSize: prepared.compiled.task.outputSize,
        activation: network.activation,
        outputActivation: prepared.compiled.task.outputActivation,
        weightInit: network.weightInit,
        seed: network.seed,
    };
}

function projectTraining(prepared: PreparedExperimentDocumentV2): TrainingConfig {
    const recipe = prepared.document.recipe;
    const optimizer = recipe.training.optimizer;
    const clipping = recipe.training.gradientClipping;
    const penalty = recipe.objective.penalty;
    const dataLoss = recipe.objective.dataLoss;
    const lossType = dataLoss.kind === 'binary-cross-entropy-with-logits'
        ? 'crossEntropy'
        : dataLoss.kind === 'categorical-cross-entropy-with-logits'
            ? 'categoricalCrossEntropy'
            : dataLoss.kind === 'mean-squared-error'
                ? 'mse'
                : 'huber';
    const optimizerType = optimizer.kind === 'sgd-momentum'
        ? 'sgdMomentum'
        : optimizer.kind;
    const schedule = projectSchedule(prepared);

    return {
        learningRate: recipe.training.learningRate,
        batchSize: recipe.training.batchSize,
        lossType,
        optimizer: optimizerType,
        momentum: optimizer.kind === 'sgd-momentum'
            ? optimizer.momentum
            : DEFAULT_TRAINING.momentum,
        regularization: penalty.kind,
        regularizationRate: penalty.kind === 'none' ? 0 : penalty.coefficient,
        gradientClip: clipping.kind === 'none' ? null : clipping.maximumNorm,
        ...(optimizer.kind === 'adam' ? {
            adamBeta1: optimizer.beta1,
            adamBeta2: optimizer.beta2,
            adamEps: optimizer.epsilon,
        } : {}),
        ...(dataLoss.kind === 'huber' ? { huberDelta: dataLoss.delta } : {}),
        ...(schedule ? { lrSchedule: schedule } : {}),
    };
}

/**
 * Temporary one-way adapter for pre-V2 UI and worker consumers.
 * The returned values are detached display/runtime projections; they are never
 * accepted as input for reconstructing a version-2 experiment document.
 */
export function projectPreparedExperiment(
    prepared: PreparedExperimentDocumentV2,
): AppConfig {
    const recipe = prepared.document.recipe;
    return deepFreeze({
        network: projectNetwork(prepared),
        training: projectTraining(prepared),
        data: {
            dataset: recipe.task.dataset,
            problemType: recipe.task.kind === 'regression' ? 'regression' : 'classification',
            trainTestRatio: recipe.data.trainFraction,
            noise: recipe.data.noise,
            numSamples: recipe.data.sampleCount,
            seed: recipe.data.seed,
        },
        features: { ...prepared.compiled.features },
        ui: {
            showTestData: prepared.document.view.showTestData,
            discretizeOutput: prepared.document.view.discretizeOutput,
        },
    });
}
