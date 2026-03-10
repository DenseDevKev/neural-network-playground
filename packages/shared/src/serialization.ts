// ── URL serialization for shareable state ──
import type { FeatureFlags, DataConfig, TrainingConfig, DatasetType } from '@nn-playground/engine';
import type { AppConfig, UIConfig } from './types.js';
import { DEFAULT_NETWORK, DEFAULT_TRAINING, DEFAULT_DATA } from './constants.js';
import { countActiveFeatures } from '@nn-playground/engine';

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

    const featureBits = p.get('f') || '110000000';
    const features: FeatureFlags = {
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

    const hiddenLayersStr = p.get('hl');
    const hiddenLayers = hiddenLayersStr
        ? hiddenLayersStr.split(',').map(Number).filter((n) => !isNaN(n) && n > 0)
        : [...DEFAULT_NETWORK.hiddenLayers];

    const inputSize = countActiveFeatures(features);

    const data: DataConfig = {
        dataset: (p.get('d') as DatasetType) || DEFAULT_DATA.dataset,
        problemType: (p.get('pt') as 'classification' | 'regression') || DEFAULT_DATA.problemType,
        trainTestRatio: parseNum(p.get('r'), DEFAULT_DATA.trainTestRatio),
        noise: parseNum(p.get('n'), DEFAULT_DATA.noise),
        numSamples: parseNum(p.get('ns'), DEFAULT_DATA.numSamples),
        seed: parseNum(p.get('s'), DEFAULT_DATA.seed),
    };

    const training: TrainingConfig = {
        learningRate: parseNum(p.get('lr'), DEFAULT_TRAINING.learningRate),
        batchSize: parseNum(p.get('bs'), DEFAULT_TRAINING.batchSize),
        lossType: (p.get('l') as TrainingConfig['lossType']) || DEFAULT_TRAINING.lossType,
        optimizer: (p.get('o') as TrainingConfig['optimizer']) || DEFAULT_TRAINING.optimizer,
        momentum: DEFAULT_TRAINING.momentum,
        regularization: (p.get('rg') as TrainingConfig['regularization']) || DEFAULT_TRAINING.regularization,
        regularizationRate: parseNum(p.get('rr'), DEFAULT_TRAINING.regularizationRate),
        gradientClip: null,
    };

    const ui: UIConfig = {
        showTestData: p.get('st') === '1',
        discretizeOutput: p.get('do') === '1',
        animationSpeed: 1,
    };

    return {
        network: {
            inputSize,
            hiddenLayers,
            outputSize: 1,
            activation: (p.get('a') as any) || DEFAULT_NETWORK.activation,
            outputActivation: (p.get('oa') as any) || DEFAULT_NETWORK.outputActivation,
            weightInit: (p.get('wi') as any) || DEFAULT_NETWORK.weightInit,
            seed: parseNum(p.get('ws'), DEFAULT_NETWORK.seed),
        },
        training,
        data,
        features,
        ui,
    };
}

function parseNum(val: string | null, fallback: number): number {
    if (val == null) return fallback;
    const n = Number(val);
    return isNaN(n) ? fallback : n;
}

/** Export full config as a JSON string. */
export function exportConfigJson(config: AppConfig): string {
    return JSON.stringify(config, null, 2);
}

/** Import config from JSON string. Returns null if invalid. */
export function importConfigJson(json: string): AppConfig | null {
    try {
        const parsed = JSON.parse(json);
        if (parsed && parsed.network && parsed.training && parsed.data && parsed.features) {
            return parsed as AppConfig;
        }
        return null;
    } catch {
        return null;
    }
}
