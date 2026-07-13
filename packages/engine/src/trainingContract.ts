import { compileObjective } from './objective.js';
import type {
    CompilableExperimentRecipe,
    CompiledDataConfig,
    CompiledExperimentConfig,
    CompiledTaskContract,
    CompiledTrainingContractV2,
    FeatureFlags,
    FeatureId,
    GradientClipSpecV2,
    LearningRateScheduleV2,
    OptimizerSpecV2,
} from './types.js';

export type {
    CompilableExperimentRecipe,
    CompiledDataConfig,
    CompiledExperimentConfig,
    CompiledTaskContract,
    CompiledTrainingContractV2,
    LearningRateScheduleV2,
    OptimizerSpecV2,
} from './types.js';

function compileFeatureFlags(featureIds: readonly FeatureId[]): FeatureFlags {
    const flags: FeatureFlags = {
        x: false,
        y: false,
        xSquared: false,
        ySquared: false,
        xy: false,
        sinX: false,
        sinY: false,
        cosX: false,
        cosY: false,
    };
    for (const featureId of featureIds) flags[featureId] = true;
    return flags;
}

function copySchedule(schedule: LearningRateScheduleV2): LearningRateScheduleV2 {
    switch (schedule.kind) {
        case 'constant':
            return { kind: 'constant' };
        case 'step':
            return { kind: 'step', interval: schedule.interval, gamma: schedule.gamma };
        case 'cosine':
            return {
                kind: 'cosine',
                totalSteps: schedule.totalSteps,
                minimumRate: schedule.minimumRate,
            };
    }
}

function copyOptimizer(optimizer: OptimizerSpecV2): OptimizerSpecV2 {
    switch (optimizer.kind) {
        case 'sgd':
            return { kind: 'sgd' };
        case 'sgd-momentum':
            return { kind: 'sgd-momentum', momentum: optimizer.momentum };
        case 'adam':
            return {
                kind: 'adam',
                beta1: optimizer.beta1,
                beta2: optimizer.beta2,
                epsilon: optimizer.epsilon,
            };
    }
}

function copyGradientClipping(spec: GradientClipSpecV2): GradientClipSpecV2 {
    if (spec.kind === 'none') return { kind: 'none' };
    return {
        kind: 'global-norm',
        maximumNorm: spec.maximumNorm,
        scope: spec.scope,
    };
}

function compileTask(
    task: CompilableExperimentRecipe['task'],
): CompiledTaskContract {
    switch (task.kind) {
        case 'binary-classification':
            return {
                kind: task.kind,
                dataset: task.dataset,
                outputSize: 1,
                outputActivation: 'sigmoid',
                target: { kind: 'scalar', values: [0, 1] },
            };
        case 'multiclass-classification':
            return {
                kind: task.kind,
                dataset: task.dataset,
                outputSize: 3,
                outputActivation: 'softmax',
                target: { kind: 'one-hot', length: 3 },
            };
        case 'regression':
            return {
                kind: task.kind,
                dataset: task.dataset,
                outputSize: 1,
                outputActivation: 'linear',
                target: { kind: 'scalar', finite: true },
            };
    }
}

function assertTaskObjectiveCompatibility(recipe: CompilableExperimentRecipe): void {
    const dataLossKind = recipe.objective.dataLoss.kind;
    const compatible = recipe.task.kind === 'binary-classification'
        ? dataLossKind === 'binary-cross-entropy-with-logits'
        : recipe.task.kind === 'multiclass-classification'
            ? dataLossKind === 'categorical-cross-entropy-with-logits'
            : dataLossKind === 'mean-squared-error' || dataLossKind === 'huber';
    if (!compatible) {
        throw new RangeError(
            `objective ${dataLossKind} is incompatible with task ${recipe.task.kind}`,
        );
    }
}

/** Compile a validated recipe into engine-owned, allocation-free runtime contracts. */
export function compileExperimentRecipe(
    recipe: CompilableExperimentRecipe,
): CompiledExperimentConfig {
    assertTaskObjectiveCompatibility(recipe);
    const task = compileTask(recipe.task);
    const network = {
        inputSize: recipe.inputs.featureIds.length,
        hiddenLayers: [...recipe.model.hiddenLayers],
        outputSize: task.outputSize,
        activation: recipe.model.hiddenActivation,
        outputActivation: task.outputActivation,
        weightInit: recipe.model.initialization,
        seed: recipe.model.seed,
    };
    const objective = compileObjective(recipe.objective, network);
    const data: CompiledDataConfig = {
        dataset: recipe.task.dataset,
        sampleCount: recipe.data.sampleCount,
        trainFraction: recipe.data.trainFraction,
        noise: recipe.data.noise,
        seed: recipe.data.seed,
    };
    const training: CompiledTrainingContractV2 = {
        learningRate: recipe.training.learningRate,
        batchSize: recipe.training.batchSize,
        schedule: copySchedule(recipe.training.schedule),
        optimizer: copyOptimizer(recipe.training.optimizer),
        gradientClipping: copyGradientClipping(recipe.training.gradientClipping),
        objective,
    };

    return {
        network,
        training,
        data,
        features: compileFeatureFlags(recipe.inputs.featureIds),
        objective,
        task,
    };
}
