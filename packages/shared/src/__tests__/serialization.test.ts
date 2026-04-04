import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    validateImportedConfig,
    encodeUrlState,
    decodeUrlState,
    exportConfigJson,
    importConfigJson
} from '../index.js';
import type { AppConfig } from '../types.js';

const validConfig: AppConfig = {
    data: { ...DEFAULT_DATA },
    network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
    training: { ...DEFAULT_TRAINING },
    features: { ...DEFAULT_FEATURES },
    ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
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
