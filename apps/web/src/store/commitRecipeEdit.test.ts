import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    type PreparedExperimentDocumentV2,
    type SchemaResult,
} from '@nn-playground/shared';
import { commitRecipeEdit, formatRecipeEditIssue } from './commitRecipeEdit.ts';
import { setFeatureIds, setNoise, toggleFeature } from './recipeEdits.ts';
import {
    usePlaygroundStore,
    type StoreRecipeEditResult,
} from './usePlaygroundStore.ts';
import { useTrainingStore } from './useTrainingStore.ts';
import { currentPreparedForTest } from '../test/playgroundStoreTestUtils.ts';

const originalEditRecipe = usePlaygroundStore.getState().editRecipe;

function resetTrainingTransaction() {
    useTrainingStore.setState({
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
        presetConfigLoading: false,
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
    });
}

describe('commitRecipeEdit', () => {
    beforeEach(async () => {
        usePlaygroundStore.setState({ editRecipe: originalEditRecipe });
        const restored = await usePlaygroundStore.getState()
            .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);
        resetTrainingTransaction();
    });

    it('awaits and recognizes only the exact published result', async () => {
        const published = await commitRecipeEdit(
            'data',
            (recipe) => setNoise(recipe, 17),
        );

        expect(published).toBe(true);
        expect(currentPreparedForTest()?.document.recipe.data.noise).toBe(17);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'data',
            dataConfigLoading: true,
            configError: null,
        });
    });

    it('reports a local typed-edit failure and retains exact prepared state', async () => {
        const oneFeature = await originalEditRecipe(
            (recipe) => setFeatureIds(recipe, ['x']),
        );
        expect(oneFeature.ok).toBe(true);
        const before = currentPreparedForTest();

        const published = await commitRecipeEdit(
            'features',
            (recipe) => toggleFeature(recipe, 'x'),
        );

        expect(published).toBe(false);
        expect(currentPreparedForTest()).toBe(before);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: null,
            featuresConfigLoading: false,
            configErrorSource: 'features',
            configError: 'recipe.inputs.featureIds: cannot remove the final active feature x',
        });
    });

    it('formats schema preparation issues and retains the current experiment', async () => {
        const before = currentPreparedForTest();
        usePlaygroundStore.setState({
            editRecipe: vi.fn(async () => ({
                ok: false as const,
                issues: [{
                    code: 'invalid-field' as const,
                    path: 'recipe.training.batchSize',
                    message: 'batch size must not exceed the training population',
                }],
            })),
        });

        const published = await commitRecipeEdit(
            'training',
            (recipe) => setNoise(recipe, 4),
        );

        expect(published).toBe(false);
        expect(currentPreparedForTest()).toBe(before);
        expect(useTrainingStore.getState()).toMatchObject({
            configErrorSource: 'training',
            configError: 'recipe.training.batchSize: batch size must not exceed the training population',
        });
    });

    it('does not let an older failure clear a newer transaction', async () => {
        let resolveOlder!: (result: StoreRecipeEditResult) => void;
        const older = new Promise<StoreRecipeEditResult>((resolve) => {
            resolveOlder = resolve;
        });
        usePlaygroundStore.setState({ editRecipe: vi.fn(() => older) });

        const pendingCommit = commitRecipeEdit(
            'data',
            (recipe) => setNoise(recipe, 9),
        );
        useTrainingStore.getState().beginConfigChange('network');
        const newer = await originalEditRecipe((recipe) => setNoise(recipe, 11));
        expect(newer.ok).toBe(true);
        resolveOlder({
            ok: false,
            issues: [{ code: 'invalid-field', path: 'recipe', message: 'obsolete failure' }],
        });

        expect(await pendingCommit).toBe(false);
        expect(currentPreparedForTest()?.document.recipe.data.noise).toBe(11);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'network',
            networkConfigLoading: true,
            configError: null,
        });
    });

    it('does not report an older rejected promise over a newer transaction', async () => {
        let rejectOlder!: (reason: unknown) => void;
        const older = new Promise<SchemaResult<PreparedExperimentDocumentV2>>((_, reject) => {
            rejectOlder = reject;
        });
        usePlaygroundStore.setState({ editRecipe: vi.fn(() => older) });

        const pendingCommit = commitRecipeEdit(
            'data',
            (recipe) => setNoise(recipe, 9),
        );
        useTrainingStore.getState().beginConfigChange('network');
        const newer = await originalEditRecipe((recipe) => setNoise(recipe, 13));
        expect(newer.ok).toBe(true);
        rejectOlder(new Error('obsolete rejection'));

        expect(await pendingCommit).toBe(false);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'network',
            networkConfigLoading: true,
            configError: null,
        });
    });

    it('does not overwrite a newer local edit error when preparation request IDs are unchanged', async () => {
        let resolveOlder!: (result: StoreRecipeEditResult) => void;
        const older = new Promise<StoreRecipeEditResult>((resolve) => {
            resolveOlder = resolve;
        });
        const editRecipe = vi.fn()
            .mockImplementationOnce(() => older)
            .mockResolvedValueOnce({
                ok: false as const,
                issue: { kind: 'empty-feature-set' as const },
            });
        usePlaygroundStore.setState({ editRecipe });

        const first = commitRecipeEdit('data', (recipe) => setNoise(recipe, 9));
        const second = commitRecipeEdit('features', (recipe) => setNoise(recipe, 10));
        expect(await second).toBe(false);
        expect(useTrainingStore.getState()).toMatchObject({
            configErrorSource: 'features',
            configError: 'recipe.inputs.featureIds: at least one feature is required',
        });

        resolveOlder({
            ok: false,
            issues: [{ code: 'invalid-field', path: 'recipe.data.noise', message: 'obsolete' }],
        });
        expect(await first).toBe(false);
        expect(useTrainingStore.getState()).toMatchObject({
            configErrorSource: 'features',
            configError: 'recipe.inputs.featureIds: at least one feature is required',
        });
    });
});

describe('formatRecipeEditIssue', () => {
    it.each([
        [
            { kind: 'batch-exceeds-training-population' as const, attemptedEdit: 'batch-size' as const, batchSize: 64, trainCount: 30 },
            'recipe.training.batchSize: batch size 64 exceeds training population 30',
        ],
        [
            { kind: 'seed-overflow' as const, target: 'data' as const, seed: 4_294_967_295 },
            'recipe.data.seed: cannot reshuffle seed 4294967295 beyond the uint32 maximum',
        ],
        [
            { kind: 'hidden-layer-index-out-of-range' as const, layerIndex: 4, layerCount: 2 },
            'recipe.model.hiddenLayers: layer index 4 is outside 2 hidden layers',
        ],
    ])('formats %o as a structured path and message', (issue, expected) => {
        expect(formatRecipeEditIssue(issue)).toBe(expected);
    });
});
