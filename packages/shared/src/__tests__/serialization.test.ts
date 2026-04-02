import { describe, it, expect } from 'vitest';
import { encodeUrlState, decodeUrlState, exportConfigJson, importConfigJson } from '../serialization.js';
import type { AppConfig } from '../types.js';
import { DEFAULT_NETWORK, DEFAULT_TRAINING, DEFAULT_DATA, DEFAULT_FEATURES } from '../constants.js';

const mockConfig: AppConfig = {
    network: {
        ...DEFAULT_NETWORK,
        inputSize: 2,
    },
    training: DEFAULT_TRAINING,
    data: DEFAULT_DATA,
    features: DEFAULT_FEATURES,
    ui: {
        showTestData: false,
        discretizeOutput: false,
        animationSpeed: 1,
    },
};

describe('serialization', () => {
    describe('encodeUrlState and decodeUrlState', () => {
        it('should round-trip a config correctly', () => {
            const encoded = encodeUrlState(mockConfig);
            const decoded = decodeUrlState(encoded);

            // We expect the decoded config to match the original,
            // though some default values might be filled in.
            expect(decoded.network).toEqual(mockConfig.network);
            expect(decoded.training).toEqual(mockConfig.training);
            expect(decoded.data).toEqual(mockConfig.data);
            expect(decoded.features).toEqual(mockConfig.features);
            expect(decoded.ui.showTestData).toBe(mockConfig.ui.showTestData);
            expect(decoded.ui.discretizeOutput).toBe(mockConfig.ui.discretizeOutput);
        });

        it('should handle empty hash by returning defaults', () => {
            const decoded = decodeUrlState('');
            expect(decoded.network.activation).toBe(DEFAULT_NETWORK.activation);
            expect(decoded.data.dataset).toBe(DEFAULT_DATA.dataset);
        });
    });

    describe('exportConfigJson and importConfigJson', () => {
        it('should round-trip a config via JSON', () => {
            const json = exportConfigJson(mockConfig);
            const imported = importConfigJson(json);
            expect(imported).toEqual(mockConfig);
        });

        it('should return null for invalid JSON string', () => {
            const invalidJson = '{ invalid json }';
            const result = importConfigJson(invalidJson);
            expect(result).toBeNull();
        });

        it('should return null for JSON missing required fields', () => {
            const incompleteJson = JSON.stringify({
                network: {},
                // missing training, data, features
            });
            const result = importConfigJson(incompleteJson);
            expect(result).toBeNull();
        });

        it('should return null for non-object JSON', () => {
            const result = importConfigJson('123');
            expect(result).toBeNull();
        });

        it('should return null for null JSON', () => {
            const result = importConfigJson('null');
            expect(result).toBeNull();
        });
    });
});
