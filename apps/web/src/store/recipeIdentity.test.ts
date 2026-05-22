import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
} from '@nn-playground/shared';
import { getRecipeDrift, summarizeRecipe } from './recipeIdentity.ts';

function makeConfig(overrides: Partial<AppConfig> = {}): AppConfig {
    return {
        data: overrides.data ?? { ...DEFAULT_DATA },
        features: overrides.features ?? { ...DEFAULT_FEATURES },
        network: overrides.network ?? {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            seed: DEFAULT_DATA.seed,
        },
        training: overrides.training ?? { ...DEFAULT_TRAINING },
        ui: overrides.ui ?? { showTestData: false, discretizeOutput: false },
    };
}

describe('recipe identity drift', () => {
    it('reports no drift for identical experiment recipes', () => {
        const current = makeConfig();
        const drift = getRecipeDrift(current, makeConfig());

        expect(drift.hasDrift).toBe(false);
        expect(drift.items).toEqual([]);
        expect(drift.groupLabels).toEqual([]);
        expect(drift.resolution).toBe('Evidence is aligned with the current recipe.');
    });

    it('detects data, feature, network, and training changes with readable labels', () => {
        const trained = makeConfig();
        const current = makeConfig({
            data: { ...DEFAULT_DATA, dataset: 'xor', seed: 99 },
            features: { ...DEFAULT_FEATURES, xSquared: true },
            network: {
                ...DEFAULT_NETWORK,
                inputSize: 3,
                hiddenLayers: [8, 4],
                seed: DEFAULT_DATA.seed,
            },
            training: { ...DEFAULT_TRAINING, learningRate: 0.1 },
        });

        const drift = getRecipeDrift(trained, current);

        expect(drift.hasDrift).toBe(true);
        expect(drift.headline).toBe('Current recipe differs from trained snapshot.');
        expect(drift.groupLabels).toEqual(['Dataset', 'Features', 'Network', 'Training']);
        expect(drift.items.map((item) => item.label)).toEqual([
            'Dataset',
            'Data seed',
            'Active features',
            'Input size',
            'Hidden layers',
            'Learning rate',
        ]);
        expect(drift.visibleItems.map((item) => item.label)).toEqual([
            'Dataset',
            'Data seed',
            'Active features',
        ]);
        expect(drift.remainingCount).toBe(3);
        expect(drift.items.find((item) => item.label === 'Hidden layers')).toMatchObject({
            snapshotValue: '4 x 4',
            currentValue: '8 x 4',
        });
    });

    it('ignores visualization and panel UI state when detecting recipe drift', () => {
        const trained = makeConfig({
            ui: { showTestData: false, discretizeOutput: false },
        });
        const current = makeConfig({
            ui: { showTestData: true, discretizeOutput: true },
        });

        expect(getRecipeDrift(trained, current).hasDrift).toBe(false);
    });

    it('summarizes the current recipe for compact cards', () => {
        const summary = summarizeRecipe(
            makeConfig({
                data: { ...DEFAULT_DATA, dataset: 'xor', problemType: 'classification' },
                network: {
                    ...DEFAULT_NETWORK,
                    inputSize: 2,
                    hiddenLayers: [6, 3],
                    activation: 'relu',
                    seed: DEFAULT_DATA.seed,
                },
                training: {
                    ...DEFAULT_TRAINING,
                    optimizer: 'adam',
                    lossType: 'crossEntropy',
                    learningRate: 0.003,
                },
            }),
        );

        expect(summary.dataset).toBe('XOR classification');
        expect(summary.architecture).toBe('2 -> 6 x 3 -> 1, relu');
        expect(summary.training).toBe('Adam, lr 0.003');
        expect(summary.lossAndBatch).toBe('cross entropy, batch 10');
        expect(summary.features).toBe('x, y');
        expect(summary.featureCount).toBe('2 features');
    });
});
