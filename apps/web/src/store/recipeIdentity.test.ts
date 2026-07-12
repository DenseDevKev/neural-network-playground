import { describe, expect, it } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    PREPARED_PRESETS,
    validateExperimentDocument,
    type ExperimentDocumentV2,
    type ValidatedStandardExperimentRecipeV2,
} from '@nn-playground/shared';
import {
    getRecipeDrift,
    isSameCanonicalRecipe,
    summarizeRecipe,
} from './recipeIdentity.ts';

function makeRecipe(
    edit?: (document: ExperimentDocumentV2) => void,
): ValidatedStandardExperimentRecipeV2 {
    const document = structuredClone(DEFAULT_EXPERIMENT_DOCUMENT) as ExperimentDocumentV2;
    edit?.(document);
    const result = validateExperimentDocument(document);
    if (!result.ok) throw new Error(result.issues.map((issue) => issue.message).join('; '));
    return result.value.recipe;
}

describe('recipe identity drift', () => {
    it('uses canonical recipe keys for current/catalog equality', () => {
        const first = PREPARED_PRESETS[0].prepared;
        const equivalent = {
            ...first,
            identities: { ...first.identities },
        };

        expect(isSameCanonicalRecipe(first, equivalent)).toBe(true);
        expect(isSameCanonicalRecipe(first, PREPARED_PRESETS[1].prepared)).toBe(false);
        expect(isSameCanonicalRecipe(null, first)).toBe(false);
    });

    it('uses exact fingerprints for trained/current drift instead of fragment equality', () => {
        const sameRecipe = makeRecipe();
        const trained = PREPARED_PRESETS[0].prepared.identities.recipeFingerprint;
        const current = PREPARED_PRESETS[1].prepared.identities.recipeFingerprint;

        const drift = getRecipeDrift(sameRecipe, sameRecipe, 3, {
            trainedRecipeFingerprint: trained,
            currentRecipeFingerprint: current,
        });

        expect(drift.hasDrift).toBe(true);
        expect(drift.headline).toBe('Current recipe differs from trained snapshot.');

        const differentRecipe = makeRecipe((document) => {
            document.recipe.training.learningRate = 0.2;
        });
        expect(getRecipeDrift(sameRecipe, differentRecipe, 3, {
            trainedRecipeFingerprint: trained,
            currentRecipeFingerprint: trained,
        }).hasDrift).toBe(false);
    });

    it('reports no drift for identical validated recipes', () => {
        const current = makeRecipe();
        const drift = getRecipeDrift(current, makeRecipe());

        expect(drift.hasDrift).toBe(false);
        expect(drift.items).toEqual([]);
        expect(drift.groupLabels).toEqual([]);
        expect(drift.resolution).toBe('Evidence is aligned with the current recipe.');
    });

    it('detects data, feature, network, and training changes with readable labels', () => {
        const trained = makeRecipe();
        const current = makeRecipe((document) => {
            document.recipe.task.dataset = 'xor';
            document.recipe.data.seed = 99;
            document.recipe.inputs.featureIds = ['x', 'y', 'xSquared'];
            document.recipe.model.hiddenLayers = [8, 4];
            document.recipe.training.learningRate = 0.1;
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

    it('summarizes the exact validated recipe for compact cards', () => {
        const summary = summarizeRecipe(makeRecipe((document) => {
            document.recipe.task.dataset = 'xor';
            document.recipe.model.hiddenLayers = [6, 3];
            document.recipe.model.hiddenActivation = 'relu';
            document.recipe.training.optimizer = {
                kind: 'adam',
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            };
            document.recipe.training.learningRate = 0.003;
        }));

        expect(summary.dataset).toBe('XOR classification');
        expect(summary.architecture).toBe('2 -> 6 x 3 -> 1, relu');
        expect(summary.training).toBe('Adam, lr 0.003');
        expect(summary.lossAndBatch).toBe('binary cross entropy, batch 10');
        expect(summary.features).toBe('x, y');
        expect(summary.featureCount).toBe('2 features');
    });
});
