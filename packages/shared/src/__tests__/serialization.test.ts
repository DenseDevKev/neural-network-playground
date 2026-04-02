import { describe, it, expect } from 'vitest';
import { encodeUrlState, decodeUrlState, exportConfigJson, importConfigJson } from '../serialization.js';
import type { AppConfig } from '../types.js';
import { DEFAULT_NETWORK, DEFAULT_TRAINING, DEFAULT_DATA } from '../constants.js';

describe('serialization', () => {
    describe('URL State', () => {
        it('should correctly roundtrip a complete configuration', () => {
            const config: AppConfig = {
                network: {
                    inputSize: 2,
                    hiddenLayers: [4, 4],
                    outputSize: 1,
                    activation: 'relu',
                    outputActivation: 'sigmoid',
                    weightInit: 'xavier',
                    seed: 42,
                },
                training: {
                    learningRate: 0.05,
                    batchSize: 32,
                    lossType: 'mse',
                    optimizer: 'adam',
                    momentum: 0.9,
                    regularization: 'l2',
                    regularizationRate: 0.001,
                    gradientClip: null,
                },
                data: {
                    dataset: 'spiral',
                    problemType: 'classification',
                    trainTestRatio: 0.7,
                    noise: 0.1,
                    numSamples: 500,
                    seed: 123,
                },
                features: {
                    x: true,
                    y: true,
                    xSquared: false,
                    ySquared: false,
                    xy: true,
                    sinX: false,
                    sinY: false,
                    cosX: false,
                    cosY: false,
                },
                ui: {
                    showTestData: true,
                    discretizeOutput: false,
                    animationSpeed: 1,
                },
            };

            const hash = encodeUrlState(config);
            const decoded = decodeUrlState(hash);

            // We only compare the fields that are actually serialized
            expect(decoded.network.hiddenLayers).toEqual(config.network.hiddenLayers);
            expect(decoded.network.activation).toEqual(config.network.activation);
            expect(decoded.network.outputActivation).toEqual(config.network.outputActivation);
            expect(decoded.network.weightInit).toEqual(config.network.weightInit);
            expect(decoded.network.seed).toEqual(config.network.seed);

            expect(decoded.training.learningRate).toEqual(config.training.learningRate);
            expect(decoded.training.batchSize).toEqual(config.training.batchSize);
            expect(decoded.training.lossType).toEqual(config.training.lossType);
            expect(decoded.training.optimizer).toEqual(config.training.optimizer);
            expect(decoded.training.regularization).toEqual(config.training.regularization);
            expect(decoded.training.regularizationRate).toEqual(config.training.regularizationRate);
            // Note: momentum and gradientClip are not serialized

            expect(decoded.data.dataset).toEqual(config.data.dataset);
            expect(decoded.data.problemType).toEqual(config.data.problemType);
            expect(decoded.data.trainTestRatio).toEqual(config.data.trainTestRatio);
            expect(decoded.data.noise).toEqual(config.data.noise);
            expect(decoded.data.numSamples).toEqual(config.data.numSamples);
            expect(decoded.data.seed).toEqual(config.data.seed);

            expect(decoded.features).toEqual(config.features);

            expect(decoded.ui.showTestData).toEqual(config.ui.showTestData);
            expect(decoded.ui.discretizeOutput).toEqual(config.ui.discretizeOutput);
        });

        it('should handle decoding an empty string using defaults', () => {
            const decoded = decodeUrlState('');

            expect(decoded.network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
            expect(decoded.network.activation).toEqual(DEFAULT_NETWORK.activation);

            expect(decoded.data.dataset).toEqual(DEFAULT_DATA.dataset);
            expect(decoded.data.problemType).toEqual(DEFAULT_DATA.problemType);
            expect(decoded.data.noise).toEqual(DEFAULT_DATA.noise);

            expect(decoded.training.learningRate).toEqual(DEFAULT_TRAINING.learningRate);

            // Default features bitfield is '110000000' (x and y are true)
            expect(decoded.features.x).toBe(true);
            expect(decoded.features.y).toBe(true);
            expect(decoded.features.xSquared).toBe(false);
        });

        it('should handle partial or invalid data in URL hash', () => {
            const hash = 'a=invalid_activation&lr=not_a_number&hl=bad,data';
            const decoded = decodeUrlState(hash);

            // Bad string values fall back to whatever is passed, or type assertions handle it
            expect(decoded.network.activation).toEqual('invalid_activation');

            // Bad numbers should fall back to defaults
            expect(decoded.training.learningRate).toEqual(DEFAULT_TRAINING.learningRate);

            // Bad hidden layers fall back to filtering out NaN. In this code, if user gives 'bad,data',
            // `.split(',').map(Number).filter((n: number) => !isNaN(n) && n > 0)`
            // will give an empty array [] since both are NaN
            expect(decoded.network.hiddenLayers).toEqual([]);
        });

        it('should remove leading # from hash', () => {
            const hashWithoutHash = encodeUrlState({
                network: DEFAULT_NETWORK,
                training: DEFAULT_TRAINING,
                data: DEFAULT_DATA,
                features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
                ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 }
            });
            const hashWithHash = '#' + hashWithoutHash;

            const decoded1 = decodeUrlState(hashWithoutHash);
            const decoded2 = decodeUrlState(hashWithHash);

            expect(decoded1).toEqual(decoded2);
        });
    });

    describe('JSON State', () => {
        it('should export and import config as JSON correctly', () => {
            const config: AppConfig = {
                network: {
                    inputSize: 2,
                    hiddenLayers: [4, 4],
                    outputSize: 1,
                    activation: 'relu',
                    outputActivation: 'sigmoid',
                    weightInit: 'xavier',
                    seed: 42,
                },
                training: {
                    learningRate: 0.05,
                    batchSize: 32,
                    lossType: 'mse',
                    optimizer: 'adam',
                    momentum: 0.9,
                    regularization: 'l2',
                    regularizationRate: 0.001,
                    gradientClip: null,
                },
                data: {
                    dataset: 'spiral',
                    problemType: 'classification',
                    trainTestRatio: 0.7,
                    noise: 0.1,
                    numSamples: 500,
                    seed: 123,
                },
                features: {
                    x: true,
                    y: true,
                    xSquared: false,
                    ySquared: false,
                    xy: true,
                    sinX: false,
                    sinY: false,
                    cosX: false,
                    cosY: false,
                },
                ui: {
                    showTestData: true,
                    discretizeOutput: false,
                    animationSpeed: 1,
                },
            };

            const json = JSON.stringify(config, null, 2);

            const exportedJson = exportConfigJson(config);
            expect(exportedJson).toEqual(json);

            const imported = importConfigJson(exportedJson);
            expect(imported).toEqual(config);
        });

        it('should handle invalid JSON import gracefully', () => {
            expect(importConfigJson('invalid json')).toBeNull();
            expect(importConfigJson('{"network": {}}')).toBeNull(); // Missing required fields
        });
    });
});
