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
    MAX_TRAIN_TEST_RATIO,
    MIN_TRAIN_TEST_RATIO,
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

export interface NormalizeAppConfigOptions {
    mode?: 'strict' | 'lenient';
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

    const candidate: AppConfig = {
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

    return normalizeAppConfig(candidate, { mode: 'lenient' }).config ?? fallbackConfig();
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

function isStrict(options?: NormalizeAppConfigOptions): boolean {
    return options?.mode !== 'lenient';
}

function fallbackConfig(): AppConfig {
    const features = { ...DEFAULT_FEATURES };
    return {
        data: { ...DEFAULT_DATA },
        network: {
            ...DEFAULT_NETWORK,
            inputSize: countActiveFeatures(features),
            outputSize: 1,
        },
        training: { ...DEFAULT_TRAINING },
        features,
        ui: { showTestData: false, discretizeOutput: false },
    };
}

function strictNumber(
    value: unknown,
    predicate: (n: number) => boolean,
    error: string,
): { value: number | null; error: string | null } {
    if (!isFiniteNumber(value) || !predicate(value)) {
        return { value: null, error };
    }
    return { value, error: null };
}

function lenientNumber(
    value: unknown,
    fallback: number,
    min: number,
    max: number,
    opts: { exclusive?: boolean; integer?: boolean } = {},
): number {
    if (!isFiniteNumber(value)) return fallback;
    if (opts.exclusive && (value <= min || value >= max)) return fallback;
    if (!opts.exclusive && value < min) return fallback;

    const clamped = opts.exclusive
        ? value
        : Math.min(max, value);
    return opts.integer ? Math.trunc(clamped) : clamped;
}

function validateFeatureFlags(value: unknown): value is FeatureFlags {
    if (!isRecord(value)) return false;

    const keys = Object.keys(DEFAULT_FEATURES) as Array<keyof FeatureFlags>;
    return keys.every((key) => typeof value[key] === 'boolean');
}

export function normalizeAppConfig(
    candidate: unknown,
    options: NormalizeAppConfigOptions = {},
): ImportedConfigValidationResult {
    const strict = isStrict(options);
    const defaults = fallbackConfig();

    if (!isRecord(candidate)) {
        return strict
            ? { config: null, error: 'Configuration must be a JSON object.' }
            : { config: defaults, error: null };
    }

    const { network, training, data, features, ui } = candidate;

    if (!isRecord(network) || !isRecord(training) || !isRecord(data) || !validateFeatureFlags(features)) {
        return strict
            ? { config: null, error: 'Configuration is missing required sections.' }
            : { config: defaults, error: null };
    }

    if (strict && !VALID_DATASETS.has(data.dataset as DatasetType)) {
        return { config: null, error: 'Configuration contains an unsupported dataset.' };
    }

    if (strict && data.problemType !== 'classification' && data.problemType !== 'regression') {
        return { config: null, error: 'Configuration contains an unsupported problem type.' };
    }

    if (
        strict &&
        (
            !isFiniteNumber(data.trainTestRatio) ||
            data.trainTestRatio < MIN_TRAIN_TEST_RATIO ||
            data.trainTestRatio > MAX_TRAIN_TEST_RATIO
        )
    ) {
        return { config: null, error: 'Train/test ratio must be between 0.1 and 0.9.' };
    }

    if (strict && (!isFiniteNumber(data.noise) || data.noise < 0 || data.noise > 100)) {
        return { config: null, error: 'Noise must be between 0 and 100.' };
    }

    if (strict && (!isFiniteNumber(data.numSamples) || !Number.isInteger(data.numSamples) || data.numSamples < 1 || data.numSamples > 10000)) {
        return { config: null, error: 'Sample count must be between 1 and 10000.' };
    }

    if (strict && !isFiniteNumber(data.seed)) {
        return { config: null, error: 'Data seed must be a valid number.' };
    }

    if (!Array.isArray(network.hiddenLayers) || network.hiddenLayers.length > MAX_HIDDEN_LAYERS) {
        if (strict) {
            return { config: null, error: 'Hidden layer configuration is invalid.' };
        }
    }

    const hiddenLayerCandidates = Array.isArray(network.hiddenLayers) ? network.hiddenLayers : defaults.network.hiddenLayers;
    if (strict && hiddenLayerCandidates.some((value) => (
        !isFiniteNumber(value) ||
        !Number.isInteger(value) ||
        value < 1 ||
        value > MAX_NEURONS_PER_LAYER
    ))) {
        return { config: null, error: 'Hidden layer sizes must be between 1 and 32.' };
    }

    if (strict && (!VALID_ACTIVATIONS.has(network.activation as ActivationType) || !VALID_ACTIVATIONS.has(network.outputActivation as ActivationType))) {
        return { config: null, error: 'Configuration contains an unsupported activation function.' };
    }

    if (strict && !VALID_WEIGHT_INITS.has(network.weightInit as WeightInitType)) {
        return { config: null, error: 'Configuration contains an unsupported weight initialization method.' };
    }

    if (strict && !isFiniteNumber(network.seed)) {
        return { config: null, error: 'Network seed must be a valid number.' };
    }

    if (
        strict &&
        (
            !isFiniteNumber(network.outputSize) ||
            !Number.isInteger(network.outputSize) ||
            network.outputSize !== 1
        )
    ) {
        return { config: null, error: 'Only single-output networks are supported.' };
    }

    if (strict && !VALID_LOSSES.has(training.lossType as LossType)) {
        return { config: null, error: 'Configuration contains an unsupported loss function.' };
    }

    if (strict && !VALID_OPTIMIZERS.has(training.optimizer as OptimizerType)) {
        return { config: null, error: 'Configuration contains an unsupported optimizer.' };
    }

    if (strict && !VALID_REGULARIZATION.has(training.regularization as RegularizationType)) {
        return { config: null, error: 'Configuration contains an unsupported regularization mode.' };
    }

    if (strict && (!isFiniteNumber(training.learningRate) || training.learningRate <= 0 || training.learningRate > 10)) {
        return { config: null, error: 'Learning rate must be greater than 0 and at most 10.' };
    }

    if (strict && (!isFiniteNumber(training.batchSize) || !Number.isInteger(training.batchSize) || training.batchSize < 1 || training.batchSize > 512)) {
        return { config: null, error: 'Batch size must be between 1 and 512.' };
    }

    if (strict && (!isFiniteNumber(training.momentum) || training.momentum < 0 || training.momentum > 1)) {
        return { config: null, error: 'Momentum must be between 0 and 1.' };
    }

    if (strict && (!isFiniteNumber(training.regularizationRate) || training.regularizationRate < 0 || training.regularizationRate > 1)) {
        return { config: null, error: 'Regularization rate must be between 0 and 1.' };
    }

    if (strict && training.gradientClip !== null && (!isFiniteNumber(training.gradientClip) || training.gradientClip <= 0)) {
        return { config: null, error: 'Gradient clip must be null or a positive number.' };
    }

    if (strict && training.huberDelta !== undefined && (!isFiniteNumber(training.huberDelta) || training.huberDelta <= 0)) {
        return { config: null, error: 'Huber delta must be a positive finite number.' };
    }

    const inputSize = countActiveFeatures(features);
    if (inputSize === 0) {
        return strict
            ? { config: null, error: 'At least one input feature must be enabled.' }
            : { config: defaults, error: null };
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

    const hiddenLayers = hiddenLayerCandidates
        .filter((value): value is number => isFiniteNumber(value) && value > 0)
        .slice(0, MAX_HIDDEN_LAYERS)
        .map((value) => Math.min(MAX_NEURONS_PER_LAYER, Math.trunc(value)));

    const lossType = getValidValue(training.lossType as string | null, VALID_LOSSES, DEFAULT_TRAINING.lossType);
    const outputActivation = getCompatibleOutputActivation(
        lossType,
        getValidValue(network.outputActivation as string | null, VALID_ACTIVATIONS, DEFAULT_NETWORK.outputActivation),
    );

    const gradientClip = training.gradientClip === null || training.gradientClip === undefined
        ? null
        : strict
            ? training.gradientClip as number
            : isFiniteNumber(training.gradientClip) && training.gradientClip > 0
                ? training.gradientClip
                : DEFAULT_TRAINING.gradientClip;

    const safeData = strict
        ? {
            trainTestRatio: strictNumber(
                data.trainTestRatio,
                (n) => n >= MIN_TRAIN_TEST_RATIO && n <= MAX_TRAIN_TEST_RATIO,
                'Train/test ratio must be between 0.1 and 0.9.',
            ).value!,
            noise: strictNumber(data.noise, (n) => n >= 0 && n <= 100, 'Noise must be between 0 and 100.').value!,
            numSamples: strictNumber(data.numSamples, (n) => Number.isInteger(n) && n >= 1 && n <= 10000, 'Sample count must be between 1 and 10000.').value!,
            seed: data.seed as number,
        }
        : {
            trainTestRatio: lenientNumber(data.trainTestRatio, DEFAULT_DATA.trainTestRatio, MIN_TRAIN_TEST_RATIO, MAX_TRAIN_TEST_RATIO),
            noise: lenientNumber(data.noise, DEFAULT_DATA.noise, 0, 100),
            numSamples: lenientNumber(data.numSamples, DEFAULT_DATA.numSamples, 1, 10000, { integer: true }),
            seed: isFiniteNumber(data.seed) ? data.seed : DEFAULT_DATA.seed,
        };

    const safeTraining = strict
        ? {
            learningRate: training.learningRate as number,
            batchSize: strictNumber(training.batchSize, (n) => Number.isInteger(n) && n >= 1 && n <= 512, 'Batch size must be between 1 and 512.').value!,
            momentum: training.momentum as number,
            regularizationRate: training.regularizationRate as number,
        }
        : {
            learningRate: lenientNumber(training.learningRate, DEFAULT_TRAINING.learningRate, 0, 10),
            batchSize: lenientNumber(training.batchSize, DEFAULT_TRAINING.batchSize, 1, 512, { integer: true }),
            momentum: lenientNumber(training.momentum, DEFAULT_TRAINING.momentum, 0, 1),
            regularizationRate: lenientNumber(training.regularizationRate, DEFAULT_TRAINING.regularizationRate, 0, 1),
        };

    const huberDelta = isFiniteNumber(training.huberDelta) && training.huberDelta > 0
        ? training.huberDelta
        : undefined;

    return {
        config: {
            data: {
                dataset: getValidValue(data.dataset as string | null, VALID_DATASETS, DEFAULT_DATA.dataset),
                problemType: getValidValue(data.problemType as string | null, VALID_PROBLEM_TYPES, DEFAULT_DATA.problemType),
                trainTestRatio: safeData.trainTestRatio,
                noise: safeData.noise,
                numSamples: safeData.numSamples,
                seed: safeData.seed,
            },
            network: {
                inputSize,
                hiddenLayers,
                outputSize: isFiniteNumber(network.outputSize) && Number.isInteger(network.outputSize) && network.outputSize === 1
                    ? network.outputSize
                    : 1,
                activation: getValidValue(network.activation as string | null, VALID_ACTIVATIONS, DEFAULT_NETWORK.activation),
                outputActivation,
                weightInit: getValidValue(network.weightInit as string | null, VALID_WEIGHT_INITS, DEFAULT_NETWORK.weightInit),
                seed: isFiniteNumber(network.seed) ? network.seed : DEFAULT_NETWORK.seed,
            },
            training: {
                learningRate: safeTraining.learningRate,
                batchSize: safeTraining.batchSize,
                lossType,
                optimizer: getValidValue(training.optimizer as string | null, VALID_OPTIMIZERS, DEFAULT_TRAINING.optimizer),
                momentum: safeTraining.momentum,
                regularization: getValidValue(training.regularization as string | null, VALID_REGULARIZATION, DEFAULT_TRAINING.regularization),
                regularizationRate: safeTraining.regularizationRate,
                gradientClip,
                ...(huberDelta !== undefined ? { huberDelta } : {}),
            },
            features,
            ui: normalizedUi,
        },
        error: null,
    };
}

export function validateImportedConfig(candidate: unknown): ImportedConfigValidationResult {
    return normalizeAppConfig(candidate, { mode: 'strict' });
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
