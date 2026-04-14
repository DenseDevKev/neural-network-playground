import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    decodeUrlState,
    validateImportedConfig,
} from '../index.js';

const validConfig = {
    data: { ...DEFAULT_DATA },
    network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
    training: { ...DEFAULT_TRAINING },
    features: { ...DEFAULT_FEATURES },
    ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
};

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
});
