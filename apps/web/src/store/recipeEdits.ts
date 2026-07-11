import {
    ALL_FEATURES,
    getDatasetContract,
    type BinaryDatasetId,
    type DataLossSpecV2,
    type DatasetId,
    type FeatureId,
    type GradientClipSpecV2,
    type LearningRateScheduleV2,
    type MulticlassDatasetId,
    type OptimizerSpecV2,
    type PenaltySpecV2,
    type RegressionDatasetId,
    type ScalarActivationType,
    type WeightInitType,
} from '@nn-playground/engine';
import type { StandardExperimentRecipeV2 } from '@nn-playground/shared';

type DeepReadonlyValue<T> = T extends (...args: never[]) => unknown
    ? T
    : T extends readonly (infer Element)[]
        ? readonly DeepReadonlyValue<Element>[]
        : T extends object
            ? { readonly [Key in keyof T]: DeepReadonlyValue<T[Key]> }
            : T;

export type ReadonlyExperimentRecipeV2 = DeepReadonlyValue<StandardExperimentRecipeV2>;

export type RecipeEditIssue =
    | {
        kind: 'batch-exceeds-training-population';
        attemptedEdit: 'batch-size' | 'sample-count' | 'train-fraction' | 'dataset';
        batchSize: number;
        trainCount: number;
    }
    | { kind: 'seed-overflow'; target: 'data'; seed: number }
    | { kind: 'unknown-feature'; featureId: string }
    | { kind: 'duplicate-feature'; featureId: string }
    | { kind: 'empty-feature-set' }
    | { kind: 'last-feature-removal'; featureId: string }
    | { kind: 'hidden-layer-index-out-of-range'; layerIndex: number; layerCount: number }
    | {
        kind: 'incompatible-task-edit';
        editor: 'regression-data-loss';
        taskKind: StandardExperimentRecipeV2['task']['kind'];
    };

export type RecipeEditResult =
    | { ok: true; recipe: StandardExperimentRecipeV2 }
    | { ok: false; issue: RecipeEditIssue };

export type RegressionDataLossEdit = Extract<
    DataLossSpecV2,
    { readonly kind: 'mean-squared-error' } | { readonly kind: 'huber' }
>;

const MAX_UINT32 = 4_294_967_295;
const FEATURE_IDS = new Set<string>(ALL_FEATURES.map((feature) => feature.id));

function accepted(recipe: StandardExperimentRecipeV2): RecipeEditResult {
    return { ok: true, recipe };
}

function rejected(issue: RecipeEditIssue): RecipeEditResult {
    return { ok: false, issue };
}

function isSchemaValidSampleCount(sampleCount: number): boolean {
    return Number.isInteger(sampleCount) && sampleCount >= 2 && sampleCount <= 1_000;
}

function isSchemaValidTrainFraction(trainFraction: number): boolean {
    return Number.isFinite(trainFraction) && trainFraction >= 0.1 && trainFraction <= 0.9;
}

function isSchemaValidBatchSize(batchSize: number): boolean {
    return Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= 512;
}

function rejectBatchConflictForValues(
    batchSize: number,
    sampleCount: number,
    trainFraction: number,
    attemptedEdit: 'batch-size' | 'sample-count' | 'train-fraction' | 'dataset',
): RecipeEditResult | undefined {
    if (!isSchemaValidBatchSize(batchSize)
        || !isSchemaValidSampleCount(sampleCount)
        || !isSchemaValidTrainFraction(trainFraction)) {
        return undefined;
    }
    const trainCount = Math.floor(sampleCount * trainFraction);
    if (batchSize <= trainCount) return undefined;
    return rejected({
        kind: 'batch-exceeds-training-population',
        attemptedEdit,
        batchSize,
        trainCount,
    });
}

function rejectBatchConflict(
    recipe: ReadonlyExperimentRecipeV2,
    attemptedEdit: 'dataset',
): RecipeEditResult | undefined {
    return rejectBatchConflictForValues(
        recipe.training.batchSize,
        recipe.data.sampleCount,
        recipe.data.trainFraction,
        attemptedEdit,
    );
}

function isBinaryDataset(dataset: DatasetId): dataset is BinaryDatasetId {
    return getDatasetContract(dataset).taskKind === 'binary-classification';
}

function isMulticlassDataset(dataset: DatasetId): dataset is MulticlassDatasetId {
    return getDatasetContract(dataset).taskKind === 'multiclass-classification';
}

function isRegressionDataset(dataset: DatasetId): dataset is RegressionDatasetId {
    return getDatasetContract(dataset).taskKind === 'regression';
}

export function switchDataset(
    recipe: ReadonlyExperimentRecipeV2,
    dataset: DatasetId,
): RecipeEditResult {
    const batchConflict = rejectBatchConflict(recipe, 'dataset');
    if (batchConflict) return batchConflict;

    const contract = getDatasetContract(dataset);
    if (contract.taskKind === 'binary-classification' && isBinaryDataset(dataset)) {
        return accepted({
            ...recipe,
            task: { kind: 'binary-classification', dataset },
            objective: {
                dataLoss: { kind: 'binary-cross-entropy-with-logits' },
                penalty: recipe.objective.penalty,
                reduction: 'mean-per-sample',
            },
        });
    }
    if (contract.taskKind === 'multiclass-classification' && isMulticlassDataset(dataset)) {
        return accepted({
            ...recipe,
            task: { kind: 'multiclass-classification', dataset },
            objective: {
                dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
                penalty: recipe.objective.penalty,
                reduction: 'mean-per-sample',
            },
        });
    }
    if (contract.taskKind === 'regression' && isRegressionDataset(dataset)) {
        return accepted({
            ...recipe,
            task: { kind: 'regression', dataset },
            objective: {
                dataLoss: { kind: 'mean-squared-error' },
                penalty: recipe.objective.penalty,
                reduction: 'mean-per-sample',
            },
        });
    }

    throw new Error(`Dataset contract mismatch for ${dataset}`);
}

export function setSampleCount(
    recipe: ReadonlyExperimentRecipeV2,
    sampleCount: number,
): RecipeEditResult {
    const batchConflict = rejectBatchConflictForValues(
        recipe.training.batchSize,
        sampleCount,
        recipe.data.trainFraction,
        'sample-count',
    );
    if (batchConflict) return batchConflict;
    return accepted({ ...recipe, data: { ...recipe.data, sampleCount } });
}

export function setTrainFraction(
    recipe: ReadonlyExperimentRecipeV2,
    trainFraction: number,
): RecipeEditResult {
    const batchConflict = rejectBatchConflictForValues(
        recipe.training.batchSize,
        recipe.data.sampleCount,
        trainFraction,
        'train-fraction',
    );
    if (batchConflict) return batchConflict;
    return accepted({ ...recipe, data: { ...recipe.data, trainFraction } });
}

export function setNoise(
    recipe: ReadonlyExperimentRecipeV2,
    noise: number,
): RecipeEditResult {
    return accepted({ ...recipe, data: { ...recipe.data, noise } });
}

export function setDataSeed(
    recipe: ReadonlyExperimentRecipeV2,
    seed: number,
): RecipeEditResult {
    return accepted({ ...recipe, data: { ...recipe.data, seed } });
}

export function reshuffleDataSeed(recipe: ReadonlyExperimentRecipeV2): RecipeEditResult {
    if (recipe.data.seed === MAX_UINT32) {
        return rejected({
            kind: 'seed-overflow',
            target: 'data',
            seed: recipe.data.seed,
        });
    }
    return setDataSeed(recipe, recipe.data.seed + 1);
}

function isFeatureId(featureId: string): featureId is FeatureId {
    return FEATURE_IDS.has(featureId);
}

export function setFeatureIds(
    recipe: ReadonlyExperimentRecipeV2,
    featureIds: readonly string[],
): RecipeEditResult {
    if (featureIds.length === 0) {
        return rejected({ kind: 'empty-feature-set' });
    }

    const selected = new Set<FeatureId>();
    for (const featureId of featureIds) {
        if (!isFeatureId(featureId)) {
            return rejected({ kind: 'unknown-feature', featureId });
        }
        if (selected.has(featureId)) {
            return rejected({ kind: 'duplicate-feature', featureId });
        }
        selected.add(featureId);
    }

    const canonicalFeatureIds = ALL_FEATURES
        .filter((feature) => selected.has(feature.id))
        .map((feature) => feature.id);
    return accepted({
        ...recipe,
        inputs: { featureIds: canonicalFeatureIds },
    });
}

export function toggleFeature(
    recipe: ReadonlyExperimentRecipeV2,
    featureId: string,
): RecipeEditResult {
    if (!isFeatureId(featureId)) {
        return rejected({ kind: 'unknown-feature', featureId });
    }
    const isSelected = recipe.inputs.featureIds.includes(featureId);
    if (isSelected && recipe.inputs.featureIds.length === 1) {
        return rejected({ kind: 'last-feature-removal', featureId });
    }
    const featureIds = isSelected
        ? recipe.inputs.featureIds.filter((selected) => selected !== featureId)
        : [...recipe.inputs.featureIds, featureId];
    return setFeatureIds(recipe, featureIds);
}

export function setHiddenLayers(
    recipe: ReadonlyExperimentRecipeV2,
    hiddenLayers: readonly number[],
): RecipeEditResult {
    return accepted({
        ...recipe,
        model: { ...recipe.model, hiddenLayers: [...hiddenLayers] },
    });
}

export function setHiddenLayerWidth(
    recipe: ReadonlyExperimentRecipeV2,
    layerIndex: number,
    width: number,
): RecipeEditResult {
    if (!Number.isInteger(layerIndex)
        || layerIndex < 0
        || layerIndex >= recipe.model.hiddenLayers.length) {
        return rejected({
            kind: 'hidden-layer-index-out-of-range',
            layerIndex,
            layerCount: recipe.model.hiddenLayers.length,
        });
    }
    const hiddenLayers = [...recipe.model.hiddenLayers];
    hiddenLayers[layerIndex] = width;
    return setHiddenLayers(recipe, hiddenLayers);
}

export function setHiddenActivation(
    recipe: ReadonlyExperimentRecipeV2,
    hiddenActivation: ScalarActivationType,
): RecipeEditResult {
    return accepted({
        ...recipe,
        model: { ...recipe.model, hiddenActivation },
    });
}

export function setInitialization(
    recipe: ReadonlyExperimentRecipeV2,
    initialization: WeightInitType,
): RecipeEditResult {
    return accepted({
        ...recipe,
        model: { ...recipe.model, initialization },
    });
}

export function setModelSeed(
    recipe: ReadonlyExperimentRecipeV2,
    seed: number,
): RecipeEditResult {
    return accepted({ ...recipe, model: { ...recipe.model, seed } });
}

export function setLearningRate(
    recipe: ReadonlyExperimentRecipeV2,
    learningRate: number,
): RecipeEditResult {
    return accepted({
        ...recipe,
        training: { ...recipe.training, learningRate },
    });
}

export function setBatchSize(
    recipe: ReadonlyExperimentRecipeV2,
    batchSize: number,
): RecipeEditResult {
    const batchConflict = rejectBatchConflictForValues(
        batchSize,
        recipe.data.sampleCount,
        recipe.data.trainFraction,
        'batch-size',
    );
    if (batchConflict) return batchConflict;
    return accepted({
        ...recipe,
        training: { ...recipe.training, batchSize },
    });
}

function exactSchedule(
    schedule: DeepReadonlyValue<LearningRateScheduleV2>,
): LearningRateScheduleV2 {
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

export function setSchedule(
    recipe: ReadonlyExperimentRecipeV2,
    schedule: DeepReadonlyValue<LearningRateScheduleV2>,
): RecipeEditResult {
    return accepted({
        ...recipe,
        training: { ...recipe.training, schedule: exactSchedule(schedule) },
    });
}

function exactOptimizer(
    optimizer: DeepReadonlyValue<OptimizerSpecV2>,
): OptimizerSpecV2 {
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

export function setOptimizer(
    recipe: ReadonlyExperimentRecipeV2,
    optimizer: DeepReadonlyValue<OptimizerSpecV2>,
): RecipeEditResult {
    return accepted({
        ...recipe,
        training: { ...recipe.training, optimizer: exactOptimizer(optimizer) },
    });
}

function exactGradientClipping(
    clipping: DeepReadonlyValue<GradientClipSpecV2>,
): GradientClipSpecV2 {
    switch (clipping.kind) {
        case 'none':
            return { kind: 'none' };
        case 'global-norm':
            return {
                kind: 'global-norm',
                maximumNorm: clipping.maximumNorm,
                scope: clipping.scope,
            };
    }
}

export function setGradientClipping(
    recipe: ReadonlyExperimentRecipeV2,
    clipping: DeepReadonlyValue<GradientClipSpecV2>,
): RecipeEditResult {
    return accepted({
        ...recipe,
        training: {
            ...recipe.training,
            gradientClipping: exactGradientClipping(clipping),
        },
    });
}

function exactRegressionDataLoss(
    dataLoss: DeepReadonlyValue<RegressionDataLossEdit>,
): RegressionDataLossEdit {
    switch (dataLoss.kind) {
        case 'mean-squared-error':
            return { kind: 'mean-squared-error' };
        case 'huber':
            return { kind: 'huber', delta: dataLoss.delta };
    }
}

export function setRegressionDataLoss(
    recipe: ReadonlyExperimentRecipeV2,
    dataLoss: DeepReadonlyValue<RegressionDataLossEdit>,
): RecipeEditResult {
    if (recipe.task.kind !== 'regression') {
        return rejected({
            kind: 'incompatible-task-edit',
            editor: 'regression-data-loss',
            taskKind: recipe.task.kind,
        });
    }
    return accepted({
        data: recipe.data,
        inputs: recipe.inputs,
        model: recipe.model,
        training: recipe.training,
        task: { kind: 'regression', dataset: recipe.task.dataset },
        objective: {
            dataLoss: exactRegressionDataLoss(dataLoss),
            penalty: recipe.objective.penalty,
            reduction: 'mean-per-sample',
        },
    });
}

function exactPenalty(penalty: DeepReadonlyValue<PenaltySpecV2>): PenaltySpecV2 {
    switch (penalty.kind) {
        case 'none':
            return { kind: 'none' };
        case 'l1':
            return {
                kind: 'l1',
                coefficient: penalty.coefficient,
                applyTo: penalty.applyTo,
            };
        case 'l2':
            return {
                kind: 'l2',
                coefficient: penalty.coefficient,
                applyTo: penalty.applyTo,
            };
    }
}

export function setPenalty(
    recipe: ReadonlyExperimentRecipeV2,
    penalty: DeepReadonlyValue<PenaltySpecV2>,
): RecipeEditResult {
    const nextPenalty = exactPenalty(penalty);
    switch (recipe.task.kind) {
        case 'binary-classification':
            return accepted({
                data: recipe.data,
                inputs: recipe.inputs,
                model: recipe.model,
                training: recipe.training,
                task: {
                    kind: 'binary-classification',
                    dataset: recipe.task.dataset,
                },
                objective: {
                    dataLoss: { kind: 'binary-cross-entropy-with-logits' },
                    penalty: nextPenalty,
                    reduction: 'mean-per-sample',
                },
            });
        case 'multiclass-classification':
            return accepted({
                data: recipe.data,
                inputs: recipe.inputs,
                model: recipe.model,
                training: recipe.training,
                task: {
                    kind: 'multiclass-classification',
                    dataset: recipe.task.dataset,
                },
                objective: {
                    dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
                    penalty: nextPenalty,
                    reduction: 'mean-per-sample',
                },
            });
        case 'regression':
            switch (recipe.objective.dataLoss.kind) {
                case 'mean-squared-error':
                case 'huber':
                    return accepted({
                        data: recipe.data,
                        inputs: recipe.inputs,
                        model: recipe.model,
                        training: recipe.training,
                        task: { kind: 'regression', dataset: recipe.task.dataset },
                        objective: {
                            dataLoss: exactRegressionDataLoss(recipe.objective.dataLoss),
                            penalty: nextPenalty,
                            reduction: 'mean-per-sample',
                        },
                    });
                default:
                    throw new Error('Regression recipe has a classification data loss');
            }
    }
}
