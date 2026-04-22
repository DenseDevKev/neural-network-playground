/// <reference lib="dom" />
// ── URL serialization for shareable state ──
import type {
    FeatureFlags,
    DataConfig,
    TrainingConfig,
    DatasetType,
    ActivationType,
    LossType,
    OptimizerType,
    RegularizationType,
    WeightInitType,
} from '@nn-playground/engine';
import type { AppConfig, UIConfig } from './types.js';
import {
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    MAX_HIDDEN_LAYERS,
    MAX_NEURONS_PER_LAYER,
} from './constants.js';
import { countActiveFeatures, isLossCompatible } from '@nn-playground/engine';

const VALID_DATASETS = new Set<DatasetType>([
    'circle',
    'xor',
    'gauss',
    'spiral',
    'moons',
    'checkerboard',
    'rings',
    'heart',
    'reg-plane',
    'reg-gauss',
]);

const VALID_ACTIVATIONS = new Set<ActivationType>([
    'relu',
    'tanh',
    'sigmoid',
    'linear',
    'leakyRelu',
    'elu',
    'swish',
    'softplus',
]);

const VALID_WEIGHT_INITS = new Set<WeightInitType>(['xavier', 'he', 'uniform', 'zeros']);
const VALID_LOSSES = new Set<LossType>(['mse', 'crossEntropy', 'huber']);
const VALID_OPTIMIZERS = new Set<OptimizerType>(['sgd', 'sgdMomentum', 'adam']);
const VALID_REGULARIZATION = new Set<RegularizationType>(['none', 'l1', 'l2']);
const VALID_PROBLEM_TYPES = new Set<DataConfig['problemType']>(['classification', 'regression']);

export interface ImportedConfigValidationResult {
    config: AppConfig | null;
    error: string | null;
}

/**
 * Encode app config into a URL hash string.
 * Uses short keys for compactness.
 */
export function encodeUrlState(config: AppConfig): string {
    const p = new URLSearchParams();

    // Data
    p.set('d', config.data.dataset);
    p.set('pt', config.data.problemType);
    p.set('r', String(config.data.trainTestRatio));
    p.set('n', String(config.data.noise));
    p.set('ns', String(config.data.numSamples));
    p.set('s', String(config.data.seed));

    // Network
    p.set('hl', config.network.hiddenLayers.join(','));
    p.set('a', config.network.activation);
    p.set('oa', config.network.outputActivation);
    p.set('wi', config.network.weightInit);
    p.set('ws', String(config.network.seed));

    // Training
    p.set('lr', String(config.training.learningRate));
    p.set('bs', String(config.training.batchSize));
    p.set('l', config.training.lossType);
    p.set('o', config.training.optimizer);
    p.set('rg', config.training.regularization);
    p.set('rr', String(config.training.regularizationRate));

    // Features (encode as a bitfield: x,y,x²,y²,xy,sinx,siny,cosx,cosy)
    const featureBits = [
        config.features.x, config.features.y,
        config.features.xSquared, config.features.ySquared,
        config.features.xy,
        config.features.sinX, config.features.sinY,
        config.features.cosX, config.features.cosY,
    ].map((b) => (b ? '1' : '0')).join('');
    p.set('f', featureBits);

    // UI
    if (config.ui.showTestData) p.set('st', '1');
    if (config.ui.discretizeOutput) p.set('do', '1');

    return p.toString();
}

/**
 * Decode URL hash into app config, using defaults for missing values.
 */
export function decodeUrlState(hash: string): AppConfig {
    const p = new URLSearchParams(hash.replace(/^#/, ''));

    const featureBits = (p.get('f') || '110000000').padEnd(9, '0');
    let features: FeatureFlags = {
        x: featureBits[0] === '1',
        y: featureBits[1] === '1',
        xSquared: featureBits[2] === '1',
        ySquared: featureBits[3] === '1',
        xy: featureBits[4] === '1',
        sinX: featureBits[5] === '1',
        sinY: featureBits[6] === '1',
        cosX: featureBits[7] === '1',
        cosY: featureBits[8] === '1',
    };

    if (countActiveFeatures(features) === 0) {
        features = { ...DEFAULT_FEATURES };
    }

    const hiddenLayersStr = p.get('hl');
    const hiddenLayers = hiddenLayersStr
        ? hiddenLayersStr
            .split(',')
            .map(Number)
            .filter((n: number) => !isNaN(n) && n > 0)
            .slice(0, MAX_HIDDEN_LAYERS)
            .map((n) => Math.min(MAX_NEURONS_PER_LAYER, n))
        : [...DEFAULT_NETWORK.hiddenLayers];

    const inputSize = countActiveFeatures(features);

    const data: DataConfig = {
        dataset: getValidValue(p.get('d'), VALID_DATASETS, DEFAULT_DATA.dataset),
        problemType: getValidValue(p.get('pt'), VALID_PROBLEM_TYPES, DEFAULT_DATA.problemType),
        trainTestRatio: parseNum(p.get('r'), DEFAULT_DATA.trainTestRatio),
        noise: parseNum(p.get('n'), DEFAULT_DATA.noise),
        numSamples: parseNum(p.get('ns'), DEFAULT_DATA.numSamples),
        seed: parseNum(p.get('s'), DEFAULT_DATA.seed),
    };

    const training: TrainingConfig = {
        learningRate: parseNum(p.get('lr'), DEFAULT_TRAINING.learningRate),
        batchSize: parseNum(p.get('bs'), DEFAULT_TRAINING.batchSize),
        lossType: getValidValue(p.get('l'), VALID_LOSSES, DEFAULT_TRAINING.lossType),
        optimizer: getValidValue(p.get('o'), VALID_OPTIMIZERS, DEFAULT_TRAINING.optimizer),
        momentum: DEFAULT_TRAINING.momentum,
        regularization: getValidValue(p.get('rg'), VALID_REGULARIZATION, DEFAULT_TRAINING.regularization),
        regularizationRate: parseNum(p.get('rr'), DEFAULT_TRAINING.regularizationRate),
        gradientClip: null,
    };

    const ui: UIConfig = {
        showTestData: p.get('st') === '1',
        discretizeOutput: p.get('do') === '1',
    };

    const outputActivation = getCompatibleOutputActivation(
        training.lossType,
        getValidValue(p.get('oa'), VALID_ACTIVATIONS, DEFAULT_NETWORK.outputActivation),
    );

    return {
        network: {
            inputSize,
            hiddenLayers,
            outputSize: 1,
            activation: getValidValue(p.get('a'), VALID_ACTIVATIONS, DEFAULT_NETWORK.activation),
            outputActivation,
            weightInit: getValidValue(p.get('wi'), VALID_WEIGHT_INITS, DEFAULT_NETWORK.weightInit),
            seed: parseNum(p.get('ws'), DEFAULT_NETWORK.seed),
        },
        training,
        data,
        features,
        ui,
    };
}

function getValidValue<T extends string>(value: string | null, validValues: Set<T>, fallback: T): T {
    return value != null && validValues.has(value as T) ? (value as T) : fallback;
}

function parseNum(val: string | null, fallback: number): number {
    if (val == null) return fallback;
    const n = Number(val);
    return isNaN(n) ? fallback : n;
}

function getCompatibleOutputActivation(
    lossType: LossType,
    outputActivation: ActivationType,
): ActivationType {
    if (isLossCompatible(lossType, outputActivation)) {
        return outputActivation;
    }
    return lossType === 'crossEntropy' ? 'sigmoid' : 'linear';
}

/** Export full config as a JSON string. */
export function exportConfigJson(config: AppConfig): string {
    return JSON.stringify(config, null, 2);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function validateFeatureFlags(value: unknown): value is FeatureFlags {
    if (!isRecord(value)) return false;

    const keys = Object.keys(DEFAULT_FEATURES) as Array<keyof FeatureFlags>;
    return keys.every((key) => typeof value[key] === 'boolean');
}

export function validateImportedConfig(candidate: unknown): ImportedConfigValidationResult {
    if (!isRecord(candidate)) {
        return { config: null, error: 'Configuration must be a JSON object.' };
    }

    const { network, training, data, features, ui } = candidate;

    if (!isRecord(network) || !isRecord(training) || !isRecord(data) || !validateFeatureFlags(features)) {
        return { config: null, error: 'Configuration is missing required sections.' };
    }

    if (!VALID_DATASETS.has(data.dataset as DatasetType)) {
        return { config: null, error: 'Configuration contains an unsupported dataset.' };
    }

    if (data.problemType !== 'classification' && data.problemType !== 'regression') {
        return { config: null, error: 'Configuration contains an unsupported problem type.' };
    }

    if (!isFiniteNumber(data.trainTestRatio) || data.trainTestRatio <= 0 || data.trainTestRatio >= 1) {
        return { config: null, error: 'Train/test ratio must be between 0 and 1.' };
    }

    if (!isFiniteNumber(data.noise) || data.noise < 0 || data.noise > 100) {
        return { config: null, error: 'Noise must be between 0 and 100.' };
    }

    if (!isFiniteNumber(data.numSamples) || data.numSamples < 1 || data.numSamples > 10000) {
        return { config: null, error: 'Sample count must be between 1 and 10000.' };
    }

    if (!isFiniteNumber(data.seed)) {
        return { config: null, error: 'Data seed must be a valid number.' };
    }

    if (!Array.isArray(network.hiddenLayers) || network.hiddenLayers.length > MAX_HIDDEN_LAYERS) {
        return { config: null, error: 'Hidden layer configuration is invalid.' };
    }

    if (network.hiddenLayers.some((value) => !isFiniteNumber(value) || value < 1 || value > MAX_NEURONS_PER_LAYER)) {
        return { config: null, error: 'Hidden layer sizes must be between 1 and 32.' };
    }

    if (!VALID_ACTIVATIONS.has(network.activation as ActivationType) || !VALID_ACTIVATIONS.has(network.outputActivation as ActivationType)) {
        return { config: null, error: 'Configuration contains an unsupported activation function.' };
    }

    if (!VALID_WEIGHT_INITS.has(network.weightInit as WeightInitType)) {
        return { config: null, error: 'Configuration contains an unsupported weight initialization method.' };
    }

    if (!isFiniteNumber(network.seed)) {
        return { config: null, error: 'Network seed must be a valid number.' };
    }

    if (!VALID_LOSSES.has(training.lossType as LossType)) {
        return { config: null, error: 'Configuration contains an unsupported loss function.' };
    }

    if (!VALID_OPTIMIZERS.has(training.optimizer as OptimizerType)) {
        return { config: null, error: 'Configuration contains an unsupported optimizer.' };
    }

    if (!VALID_REGULARIZATION.has(training.regularization as RegularizationType)) {
        return { config: null, error: 'Configuration contains an unsupported regularization mode.' };
    }

    if (!isFiniteNumber(training.learningRate) || training.learningRate <= 0 || training.learningRate > 10) {
        return { config: null, error: 'Learning rate must be greater than 0 and at most 10.' };
    }

    if (!isFiniteNumber(training.batchSize) || training.batchSize < 1 || training.batchSize > 512) {
        return { config: null, error: 'Batch size must be between 1 and 512.' };
    }

    if (!isFiniteNumber(training.momentum) || training.momentum < 0 || training.momentum > 1) {
        return { config: null, error: 'Momentum must be between 0 and 1.' };
    }

    if (!isFiniteNumber(training.regularizationRate) || training.regularizationRate < 0 || training.regularizationRate > 1) {
        return { config: null, error: 'Regularization rate must be between 0 and 1.' };
    }

    if (training.gradientClip !== null && (!isFiniteNumber(training.gradientClip) || training.gradientClip <= 0)) {
        return { config: null, error: 'Gradient clip must be null or a positive number.' };
    }

    const inputSize = countActiveFeatures(features);
    if (inputSize === 0) {
        return { config: null, error: 'At least one input feature must be enabled.' };
    }

    const normalizedUi: UIConfig = isRecord(ui)
        ? {
            showTestData: Boolean(ui.showTestData),
            discretizeOutput: Boolean(ui.discretizeOutput),
        }
        : {
            showTestData: false,
            discretizeOutput: false,
        };

    const lossType = training.lossType as LossType;
    const outputActivation = getCompatibleOutputActivation(
        lossType,
        network.outputActivation as ActivationType,
    );

    return {
        config: {
            data: {
                dataset: data.dataset as DatasetType,
                problemType: data.problemType as DataConfig['problemType'],
                trainTestRatio: data.trainTestRatio as number,
                noise: data.noise as number,
                numSamples: data.numSamples as number,
                seed: data.seed as number,
            },
            network: {
                inputSize,
                hiddenLayers: [...(network.hiddenLayers as number[])],
                outputSize: isFiniteNumber(network.outputSize) ? network.outputSize : 1,
                activation: network.activation as ActivationType,
                outputActivation,
                weightInit: network.weightInit as WeightInitType,
                seed: network.seed as number,
            },
            training: {
                learningRate: training.learningRate as number,
                batchSize: training.batchSize as number,
                lossType,
                optimizer: training.optimizer as OptimizerType,
                momentum: training.momentum as number,
                regularization: training.regularization as RegularizationType,
                regularizationRate: training.regularizationRate as number,
                gradientClip: training.gradientClip as number | null,
            },
            features,
            ui: normalizedUi,
        },
        error: null,
    };
}

/** Import config from JSON string. Returns null if invalid. */
export function importConfigJson(json: string): AppConfig | null {
    try {
        const parsed = JSON.parse(json);
        return validateImportedConfig(parsed).config;
    } catch {
        return null;
    }
}
