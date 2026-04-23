import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    decodeUrlState,
    validateImportedConfig,
    normalizeAppConfig,
    encodeUrlState,
    exportConfigJson,
    importConfigJson
} from '../index.js';
import type { AppConfig } from '../types.js';

const validConfig: AppConfig = {
    data: { ...DEFAULT_DATA },
    network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
    training: { ...DEFAULT_TRAINING },
    features: { ...DEFAULT_FEATURES },
    ui: { showTestData: false, discretizeOutput: false },
};

describe('URL State Serialization', () => {
    it('round-trips a valid configuration', () => {
        const encoded = encodeUrlState(validConfig);
        const decoded = decodeUrlState(encoded);

        // Because encodeUrlState does not serialize the 'noise' default or some other specifics,
        // we might just need to check if the objects match deeply.
        expect(decoded).toEqual(validConfig);
    });

    it('decodes an empty hash to defaults', () => {
        const decoded = decodeUrlState('');

        expect(decoded.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(decoded.network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
        expect(decoded.training.learningRate).toBe(DEFAULT_TRAINING.learningRate);
    });
});

describe('JSON Serialization', () => {
    it('round-trips a valid configuration', () => {
        const json = exportConfigJson(validConfig);
        const imported = importConfigJson(json);

        expect(imported).toEqual(validConfig);
    });

    it('returns null for invalid JSON string', () => {
        const imported = importConfigJson('invalid json');
        expect(imported).toBeNull();
    });

    it('returns null for structurally invalid JSON object', () => {
        const imported = importConfigJson(JSON.stringify({ notAConfig: true }));
        expect(imported).toBeNull();
    });
});

describe('validateImportedConfig', () => {
    it('accepts a valid configuration and normalizes input size', () => {
        const result = validateImportedConfig(validConfig);

        expect(result.error).toBeNull();
        expect(result.config).not.toBeNull();
        expect(result.config?.network.inputSize).toBe(2);
    });

    it('rejects unsupported activation functions', () => {
        const result = validateImportedConfig({
            ...validConfig,
            network: {
                ...validConfig.network,
                activation: 'magic',
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Configuration contains an unsupported activation function.');
    });

    it('rejects invalid learning-rate ranges', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                learningRate: 0,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Learning rate must be greater than 0 and at most 10.');
    });

    it('rejects configurations with no active features', () => {
        const result = validateImportedConfig({
            ...validConfig,
            features: {
                x: false,
                y: false,
                xSquared: false,
                ySquared: false,
                xy: false,
                sinX: false,
                sinY: false,
                cosX: false,
                cosY: false,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('At least one input feature must be enabled.');
    });
});

describe('decodeUrlState', () => {
    it('falls back to defaults for unsupported enum-like URL values', () => {
        const decoded = decodeUrlState('d=invalid&pt=broken&a=magic&oa=wild&wi=bad&l=nope&o=fast&rg=ghost');

        expect(decoded.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(decoded.data.problemType).toBe(DEFAULT_DATA.problemType);
        expect(decoded.network.activation).toBe(DEFAULT_NETWORK.activation);
        expect(decoded.network.outputActivation).toBe(DEFAULT_NETWORK.outputActivation);
        expect(decoded.network.weightInit).toBe(DEFAULT_NETWORK.weightInit);
        expect(decoded.training.lossType).toBe(DEFAULT_TRAINING.lossType);
        expect(decoded.training.optimizer).toBe(DEFAULT_TRAINING.optimizer);
        expect(decoded.training.regularization).toBe(DEFAULT_TRAINING.regularization);
    });

    it('restores default features when the URL disables every feature', () => {
        const decoded = decodeUrlState('f=000000000');

        expect(decoded.features).toEqual(DEFAULT_FEATURES);
        expect(decoded.network.inputSize).toBe(2);
    });

    it('normalizes incompatible loss/output pairs from URL state', () => {
        const decoded = decodeUrlState('l=mse&oa=sigmoid');

        expect(decoded.training.lossType).toBe('mse');
        expect(decoded.network.outputActivation).toBe('linear');
    });

    it('keeps malicious numeric URL values within safe runtime limits', () => {
        const decoded = decodeUrlState(
            'ns=999999999&bs=0&lr=Infinity&r=2&n=-10&s=NaN&ws=Infinity&hl=9999,0,-3,nope',
        );

        expect(decoded.data.numSamples).toBe(10000);
        expect(decoded.training.batchSize).toBe(DEFAULT_TRAINING.batchSize);
        expect(decoded.training.learningRate).toBe(DEFAULT_TRAINING.learningRate);
        expect(decoded.data.trainTestRatio).toBe(DEFAULT_DATA.trainTestRatio);
        expect(decoded.data.noise).toBe(DEFAULT_DATA.noise);
        expect(decoded.data.seed).toBe(DEFAULT_DATA.seed);
        expect(decoded.network.seed).toBe(DEFAULT_NETWORK.seed);
        expect(decoded.network.hiddenLayers).toEqual([32]);
    });
});

describe('compatibility normalization', () => {
    it('normalizes imported configs with incompatible loss/output pairs', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                lossType: 'mse',
            },
            network: {
                ...validConfig.network,
                outputActivation: 'sigmoid',
            },
        });

        expect(result.error).toBeNull();
        expect(result.config?.training.lossType).toBe('mse');
        expect(result.config?.network.outputActivation).toBe('linear');
    });

    it('strictly rejects unsafe imported numeric ranges through the shared normalizer', () => {
        const result = normalizeAppConfig({
            ...validConfig,
            data: {
                ...validConfig.data,
                numSamples: 10001,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Sample count must be between 1 and 10000.');
    });
});
