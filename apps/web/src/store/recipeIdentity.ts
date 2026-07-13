import type {
    PreparedExperimentDocumentV2,
    RecipeFingerprint,
    ValidatedStandardExperimentRecipeV2,
} from '@nn-playground/shared';
import type { FeatureId } from '@nn-playground/engine';

export type RecipeDriftGroup = 'data' | 'features' | 'network' | 'training';

export interface RecipeDriftItem {
    group: RecipeDriftGroup;
    groupLabel: string;
    field: string;
    label: string;
    snapshotValue: string;
    currentValue: string;
}

export interface RecipeDriftState {
    hasTrainedRecipe: boolean;
    hasDrift: boolean;
    headline: string;
    resolution: string;
    items: RecipeDriftItem[];
    visibleItems: RecipeDriftItem[];
    remainingCount: number;
    groupLabels: string[];
}

export interface RecipeSummary {
    dataset: string;
    architecture: string;
    training: string;
    lossAndBatch: string;
    features: string;
    featureCount: string;
}

export interface RecipeIdentityComparison {
    trainedRecipeFingerprint: RecipeFingerprint | null;
    currentRecipeFingerprint: RecipeFingerprint | null;
}

const GROUP_LABELS: Record<RecipeDriftGroup, string> = {
    data: 'Dataset',
    features: 'Features',
    network: 'Network',
    training: 'Training',
};

const FEATURE_LABELS: Record<FeatureId, string> = {
    x: 'x',
    y: 'y',
    xSquared: 'x^2',
    ySquared: 'y^2',
    xy: 'x*y',
    sinX: 'sin(x)',
    sinY: 'sin(y)',
    cosX: 'cos(x)',
    cosY: 'cos(y)',
};

const DATASET_LABELS: Record<string, string> = {
    circle: 'Circle',
    xor: 'XOR',
    gauss: 'Gaussian',
    spiral: 'Spiral',
    moons: 'Moons',
    checkerboard: 'Checkerboard',
    rings: 'Rings',
    heart: 'Heart',
    'three-class-clusters': 'Three-class clusters',
    'reg-plane': 'Regression plane',
    'reg-gauss': 'Regression gaussian',
};

const LOSS_LABELS: Record<ValidatedStandardExperimentRecipeV2['objective']['dataLoss']['kind'], string> = {
    'binary-cross-entropy-with-logits': 'binary cross entropy',
    'categorical-cross-entropy-with-logits': 'categorical cross entropy',
    'mean-squared-error': 'mean squared error',
    huber: 'Huber',
};

function titleCase(value: string): string {
    if (!value) return value;
    return `${value[0].toUpperCase()}${value.slice(1)}`;
}

function formatNumber(value: number): string {
    return Number.isInteger(value) ? String(value) : Number(value.toPrecision(5)).toString();
}

function formatValue(value: unknown): string {
    if (Array.isArray(value)) return value.length > 0 ? value.map(formatValue).join(' x ') : 'none';
    if (value === null || value === undefined) return 'none';
    if (typeof value === 'number') return formatNumber(value);
    if (typeof value === 'boolean') return value ? 'on' : 'off';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

function activeFeatureLabel(featureIds: readonly FeatureId[]): string {
    return featureIds.length > 0
        ? featureIds.map((featureId) => FEATURE_LABELS[featureId]).join(', ')
        : 'none';
}

function taskLabel(recipe: ValidatedStandardExperimentRecipeV2): string {
    return recipe.task.kind === 'regression' ? 'regression' : 'classification';
}

function outputSize(recipe: ValidatedStandardExperimentRecipeV2): number {
    return recipe.task.kind === 'multiclass-classification' ? 3 : 1;
}

function optimizerLabel(recipe: ValidatedStandardExperimentRecipeV2): string {
    switch (recipe.training.optimizer.kind) {
        case 'sgd':
            return 'SGD';
        case 'sgd-momentum':
            return 'SGD + momentum';
        case 'adam':
            return 'Adam';
    }
}

function addItem(
    items: RecipeDriftItem[],
    group: RecipeDriftGroup,
    field: string,
    label: string,
    snapshotValue: unknown,
    currentValue: unknown,
) {
    const formattedSnapshot = formatValue(snapshotValue);
    const formattedCurrent = formatValue(currentValue);
    if (formattedSnapshot === formattedCurrent) return;
    items.push({
        group,
        groupLabel: GROUP_LABELS[group],
        field,
        label,
        snapshotValue: formattedSnapshot,
        currentValue: formattedCurrent,
    });
}

/** Local recipe selection is exact canonical recipe equality, never fragments. */
export function isSameCanonicalRecipe(
    left: PreparedExperimentDocumentV2 | null,
    right: PreparedExperimentDocumentV2 | null,
): boolean {
    return left !== null
        && right !== null
        && left.identities.canonicalRecipeKey === right.identities.canonicalRecipeKey;
}

export function summarizeRecipe(recipe: ValidatedStandardExperimentRecipeV2): RecipeSummary {
    const hidden = formatValue(recipe.model.hiddenLayers);
    const datasetName = DATASET_LABELS[recipe.task.dataset] ?? recipe.task.dataset;
    const loss = LOSS_LABELS[recipe.objective.dataLoss.kind];

    return {
        dataset: `${datasetName} ${taskLabel(recipe)}`,
        architecture: `${recipe.inputs.featureIds.length} -> ${hidden} -> ${outputSize(recipe)}, ${recipe.model.hiddenActivation}`,
        training: `${optimizerLabel(recipe)}, lr ${formatValue(recipe.training.learningRate)}`,
        lossAndBatch: `${loss}, batch ${formatValue(recipe.training.batchSize)}`,
        features: activeFeatureLabel(recipe.inputs.featureIds),
        featureCount: `${recipe.inputs.featureIds.length} features`,
    };
}

export function getRecipeDrift(
    trainedRecipe: ValidatedStandardExperimentRecipeV2 | null,
    currentRecipe: ValidatedStandardExperimentRecipeV2 | null,
    visibleLimit = 3,
    identity?: RecipeIdentityComparison,
): RecipeDriftState {
    if (!trainedRecipe || (identity && identity.trainedRecipeFingerprint === null)) {
        return {
            hasTrainedRecipe: false,
            hasDrift: false,
            headline: 'No trained snapshot yet.',
            resolution: 'Run or step the experiment to produce a trained snapshot for this recipe.',
            items: [],
            visibleItems: [],
            remainingCount: 0,
            groupLabels: [],
        };
    }

    const items: RecipeDriftItem[] = [];
    if (currentRecipe) {
        addItem(items, 'data', 'dataset', 'Dataset', trainedRecipe.task.dataset, currentRecipe.task.dataset);
        addItem(items, 'data', 'problemType', 'Problem type', taskLabel(trainedRecipe), taskLabel(currentRecipe));
        addItem(items, 'data', 'sampleCount', 'Sample count', trainedRecipe.data.sampleCount, currentRecipe.data.sampleCount);
        addItem(items, 'data', 'noise', 'Noise', trainedRecipe.data.noise, currentRecipe.data.noise);
        addItem(items, 'data', 'trainFraction', 'Train split', trainedRecipe.data.trainFraction, currentRecipe.data.trainFraction);
        addItem(items, 'data', 'seed', 'Data seed', trainedRecipe.data.seed, currentRecipe.data.seed);

        addItem(
            items,
            'features',
            'featureIds',
            'Active features',
            activeFeatureLabel(trainedRecipe.inputs.featureIds),
            activeFeatureLabel(currentRecipe.inputs.featureIds),
        );

        addItem(items, 'network', 'inputSize', 'Input size', trainedRecipe.inputs.featureIds.length, currentRecipe.inputs.featureIds.length);
        addItem(items, 'network', 'hiddenLayers', 'Hidden layers', trainedRecipe.model.hiddenLayers, currentRecipe.model.hiddenLayers);
        addItem(items, 'network', 'outputSize', 'Output size', outputSize(trainedRecipe), outputSize(currentRecipe));
        addItem(items, 'network', 'hiddenActivation', 'Hidden activation', trainedRecipe.model.hiddenActivation, currentRecipe.model.hiddenActivation);
        addItem(items, 'network', 'initialization', 'Weight init', trainedRecipe.model.initialization, currentRecipe.model.initialization);
        addItem(items, 'network', 'seed', 'Weight seed', trainedRecipe.model.seed, currentRecipe.model.seed);

        addItem(items, 'training', 'learningRate', 'Learning rate', trainedRecipe.training.learningRate, currentRecipe.training.learningRate);
        addItem(items, 'training', 'batchSize', 'Batch size', trainedRecipe.training.batchSize, currentRecipe.training.batchSize);
        addItem(items, 'training', 'dataLoss', 'Loss', trainedRecipe.objective.dataLoss, currentRecipe.objective.dataLoss);
        addItem(items, 'training', 'optimizer', 'Optimizer', trainedRecipe.training.optimizer, currentRecipe.training.optimizer);
        addItem(items, 'training', 'penalty', 'Regularization', trainedRecipe.objective.penalty, currentRecipe.objective.penalty);
        addItem(items, 'training', 'gradientClipping', 'Gradient clipping', trainedRecipe.training.gradientClipping, currentRecipe.training.gradientClipping);
        addItem(items, 'training', 'schedule', 'Learning-rate schedule', trainedRecipe.training.schedule, currentRecipe.training.schedule);
    }

    const hasDrift = identity
        ? identity.currentRecipeFingerprint === null
            || identity.trainedRecipeFingerprint !== identity.currentRecipeFingerprint
        : currentRecipe === null || items.length > 0;
    const visibleIdentityItems = hasDrift ? items : [];
    const groupLabels = [...new Set(visibleIdentityItems.map((item) => item.groupLabel))];

    return {
        hasTrainedRecipe: true,
        hasDrift,
        headline: hasDrift
            ? 'Current recipe differs from trained snapshot.'
            : 'Current recipe matches the trained snapshot.',
        resolution: hasDrift
            ? 'Run or reset training to produce evidence for the current recipe.'
            : 'Evidence is aligned with the current recipe.',
        items: visibleIdentityItems,
        visibleItems: visibleIdentityItems.slice(0, visibleLimit),
        remainingCount: Math.max(0, visibleIdentityItems.length - visibleLimit),
        groupLabels,
    };
}

export function getRecipeGroupLabel(group: RecipeDriftGroup): string {
    return GROUP_LABELS[group] ?? titleCase(group);
}
