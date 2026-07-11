import type {
    AppConfig,
    PreparedExperimentDocumentV2,
    RecipeFingerprint,
} from '@nn-playground/shared';
import type { FeatureFlags } from '@nn-playground/engine';

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

const FEATURE_LABELS: Record<keyof FeatureFlags, string> = {
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

const OPTIMIZER_LABELS: Record<string, string> = {
    sgd: 'SGD',
    sgdMomentum: 'SGD + momentum',
    adam: 'Adam',
};

const LOSS_LABELS: Record<string, string> = {
    mse: 'MSE',
    crossEntropy: 'cross entropy',
    categoricalCrossEntropy: 'categorical cross entropy',
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

function activeFeatureLabel(features: FeatureFlags): string {
    const enabled = (Object.keys(FEATURE_LABELS) as (keyof FeatureFlags)[])
        .filter((key) => features[key])
        .map((key) => FEATURE_LABELS[key]);
    return enabled.length > 0 ? enabled.join(', ') : 'none';
}

function activeFeatureCount(features: FeatureFlags): number {
    return (Object.keys(FEATURE_LABELS) as (keyof FeatureFlags)[])
        .filter((key) => features[key])
        .length;
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

export function summarizeRecipe(config: AppConfig): RecipeSummary {
    const hidden = formatValue(config.network.hiddenLayers);
    const datasetName = DATASET_LABELS[config.data.dataset] ?? config.data.dataset;
    const optimizer = OPTIMIZER_LABELS[config.training.optimizer] ?? config.training.optimizer;
    const loss = LOSS_LABELS[config.training.lossType] ?? config.training.lossType;

    return {
        dataset: `${datasetName} ${config.data.problemType}`,
        architecture: `${config.network.inputSize} -> ${hidden} -> ${config.network.outputSize}, ${config.network.activation}`,
        training: `${optimizer}, lr ${formatValue(config.training.learningRate)}`,
        lossAndBatch: `${loss}, batch ${formatValue(config.training.batchSize)}`,
        features: activeFeatureLabel(config.features),
        featureCount: `${activeFeatureCount(config.features)} features`,
    };
}

export function getRecipeDrift(
    trainedConfig: AppConfig | null,
    currentConfig: AppConfig,
    visibleLimit = 3,
    identity?: RecipeIdentityComparison,
): RecipeDriftState {
    if (!trainedConfig || (identity && identity.trainedRecipeFingerprint === null)) {
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

    addItem(items, 'data', 'dataset', 'Dataset', trainedConfig.data.dataset, currentConfig.data.dataset);
    addItem(items, 'data', 'problemType', 'Problem type', trainedConfig.data.problemType, currentConfig.data.problemType);
    addItem(items, 'data', 'numSamples', 'Sample count', trainedConfig.data.numSamples, currentConfig.data.numSamples);
    addItem(items, 'data', 'noise', 'Noise', trainedConfig.data.noise, currentConfig.data.noise);
    addItem(items, 'data', 'trainTestRatio', 'Train split', trainedConfig.data.trainTestRatio, currentConfig.data.trainTestRatio);
    addItem(items, 'data', 'seed', 'Data seed', trainedConfig.data.seed, currentConfig.data.seed);

    addItem(
        items,
        'features',
        'activeFeatures',
        'Active features',
        activeFeatureLabel(trainedConfig.features),
        activeFeatureLabel(currentConfig.features),
    );

    addItem(items, 'network', 'inputSize', 'Input size', trainedConfig.network.inputSize, currentConfig.network.inputSize);
    addItem(items, 'network', 'hiddenLayers', 'Hidden layers', trainedConfig.network.hiddenLayers, currentConfig.network.hiddenLayers);
    addItem(items, 'network', 'outputSize', 'Output size', trainedConfig.network.outputSize, currentConfig.network.outputSize);
    addItem(items, 'network', 'activation', 'Hidden activation', trainedConfig.network.activation, currentConfig.network.activation);
    addItem(items, 'network', 'outputActivation', 'Output activation', trainedConfig.network.outputActivation, currentConfig.network.outputActivation);
    addItem(items, 'network', 'weightInit', 'Weight init', trainedConfig.network.weightInit, currentConfig.network.weightInit);
    addItem(items, 'network', 'seed', 'Weight seed', trainedConfig.network.seed, currentConfig.network.seed);

    addItem(items, 'training', 'learningRate', 'Learning rate', trainedConfig.training.learningRate, currentConfig.training.learningRate);
    addItem(items, 'training', 'batchSize', 'Batch size', trainedConfig.training.batchSize, currentConfig.training.batchSize);
    addItem(items, 'training', 'lossType', 'Loss', trainedConfig.training.lossType, currentConfig.training.lossType);
    addItem(items, 'training', 'optimizer', 'Optimizer', trainedConfig.training.optimizer, currentConfig.training.optimizer);
    addItem(items, 'training', 'momentum', 'Momentum', trainedConfig.training.momentum, currentConfig.training.momentum);
    addItem(items, 'training', 'regularization', 'Regularization', trainedConfig.training.regularization, currentConfig.training.regularization);
    addItem(items, 'training', 'regularizationRate', 'Regularization rate', trainedConfig.training.regularizationRate, currentConfig.training.regularizationRate);
    addItem(items, 'training', 'gradientClip', 'Gradient clip', trainedConfig.training.gradientClip, currentConfig.training.gradientClip);
    addItem(items, 'training', 'adamBeta1', 'Adam beta 1', trainedConfig.training.adamBeta1, currentConfig.training.adamBeta1);
    addItem(items, 'training', 'adamBeta2', 'Adam beta 2', trainedConfig.training.adamBeta2, currentConfig.training.adamBeta2);
    addItem(items, 'training', 'adamEps', 'Adam epsilon', trainedConfig.training.adamEps, currentConfig.training.adamEps);
    addItem(items, 'training', 'huberDelta', 'Huber delta', trainedConfig.training.huberDelta, currentConfig.training.huberDelta);
    addItem(items, 'training', 'lrSchedule', 'LR schedule', trainedConfig.training.lrSchedule, currentConfig.training.lrSchedule);

    const hasDrift = identity
        ? identity.currentRecipeFingerprint === null
            || identity.trainedRecipeFingerprint !== identity.currentRecipeFingerprint
        : items.length > 0;
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
