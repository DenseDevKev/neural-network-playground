import { describe, expect, it } from 'vitest';
import {
    PREPARED_PRESETS,
    prepareExperimentDocument,
    type PreparedExperimentDocumentV2,
    type StandardExperimentRecipeV2,
} from '@nn-playground/shared';
import {
    reshuffleDataSeed,
    setBatchSize,
    setDataSeed,
    setFeatureIds,
    setGradientClipping,
    setHiddenActivation,
    setHiddenLayers,
    setHiddenLayerWidth,
    setInitialization,
    setLearningRate,
    setModelSeed,
    setNoise,
    setOptimizer,
    setPenalty,
    setRegressionDataLoss,
    setSampleCount,
    setSchedule,
    setTrainFraction,
    switchDataset,
    toggleFeature,
    type RecipeEditResult,
} from './recipeEdits';

const BINARY_RECIPE = PREPARED_PRESETS.find((entry) => entry.id === 'single-neuron')!.recipe;
const HIDDEN_RECIPE = PREPARED_PRESETS.find((entry) => entry.id === 'xor-hidden')!.recipe;
const REGRESSION_RECIPE = PREPARED_PRESETS.find((entry) => entry.id === 'regression-plane')!.recipe;

function unwrap(result: RecipeEditResult): StandardExperimentRecipeV2 {
    if (!result.ok) throw new Error(`Expected successful edit, received ${result.issue.kind}`);
    return result.recipe;
}

async function prepare(
    recipe: StandardExperimentRecipeV2,
): Promise<PreparedExperimentDocumentV2> {
    const result = await prepareExperimentDocument({
        kind: 'nn-playground-experiment',
        schemaVersion: 2,
        recipe,
        view: { showTestData: false, discretizeOutput: false },
    });
    if (!result.ok) {
        throw new Error(result.issues.map((issue) => `${issue.path}: ${issue.message}`).join('; '));
    }
    return result.value;
}

async function expectPreparationIssue(
    result: RecipeEditResult,
    path: string,
): Promise<void> {
    const recipe = unwrap(result);
    const preparation = await prepareExperimentDocument({
        kind: 'nn-playground-experiment',
        schemaVersion: 2,
        recipe,
        view: { showTestData: false, discretizeOutput: false },
    });
    expect(preparation.ok).toBe(false);
    if (preparation.ok) throw new Error('Expected preparation to fail');
    expect(preparation.issues.map((issue) => issue.path)).toContain(path);
}

describe('recipe edits', () => {
    it('switches a binary recipe to regression atomically without mutating the input', () => {
        const input = BINARY_RECIPE;
        const snapshot = structuredClone(input);

        const result = switchDataset(input, 'reg-plane');

        expect(result).toEqual({
            ok: true,
            recipe: {
                ...input,
                task: { kind: 'regression', dataset: 'reg-plane' },
                objective: {
                    ...input.objective,
                    dataLoss: { kind: 'mean-squared-error' },
                },
            },
        });
        expect(input).toEqual(snapshot);
        expect(BINARY_RECIPE).toBe(input);
    });

    it('installs exact task objectives across every transition and preserves penalty', () => {
        const penalty = { kind: 'l2', coefficient: 0.037, applyTo: 'weights' } as const;
        const binary = unwrap(setPenalty(BINARY_RECIPE, penalty));
        const regression = unwrap(switchDataset(binary, 'reg-gauss'));
        const multiclass = unwrap(switchDataset(regression, 'three-class-clusters'));
        const binaryAgain = unwrap(switchDataset(multiclass, 'xor'));

        expect(regression.task).toEqual({ kind: 'regression', dataset: 'reg-gauss' });
        expect(regression.objective).toEqual({
            dataLoss: { kind: 'mean-squared-error' },
            penalty,
            reduction: 'mean-per-sample',
        });
        expect(multiclass.task).toEqual({
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
        });
        expect(multiclass.objective).toEqual({
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
            penalty,
            reduction: 'mean-per-sample',
        });
        expect(binaryAgain.task).toEqual({ kind: 'binary-classification', dataset: 'xor' });
        expect(binaryAgain.objective).toEqual({
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty,
            reduction: 'mean-per-sample',
        });
        expect(binaryAgain).not.toHaveProperty('problemType');
        expect(binaryAgain.model).not.toHaveProperty('outputSize');
        expect(binaryAgain.model).not.toHaveProperty('outputActivation');
        expect(binaryAgain.objective).not.toHaveProperty('lossType');
    });

    it('uses exact schedule variants and removes inactive fields', () => {
        const step = unwrap(setSchedule(BINARY_RECIPE, {
            kind: 'step',
            interval: 17,
            gamma: 0.63,
        }));
        const cosine = unwrap(setSchedule(step, {
            kind: 'cosine',
            totalSteps: 901,
            minimumRate: 0.004,
        }));
        const contaminatedConstant = Object.assign({ kind: 'constant' } as const, {
            interval: 999,
            gamma: 0.01,
            totalSteps: 999,
            minimumRate: 0.01,
        });
        const constant = unwrap(setSchedule(cosine, contaminatedConstant));

        expect(step.training.schedule).toEqual({ kind: 'step', interval: 17, gamma: 0.63 });
        expect(cosine.training.schedule).toEqual({
            kind: 'cosine',
            totalSteps: 901,
            minimumRate: 0.004,
        });
        expect(constant.training.schedule).toEqual({ kind: 'constant' });
        expect(Object.keys(constant.training.schedule)).toEqual(['kind']);
    });

    it('uses exact optimizer variants and removes inactive fields', () => {
        const momentum = unwrap(setOptimizer(BINARY_RECIPE, {
            kind: 'sgd-momentum',
            momentum: 0.37,
        }));
        const adam = unwrap(setOptimizer(momentum, {
            kind: 'adam',
            beta1: 0.81,
            beta2: 0.971,
            epsilon: 0.000_000_7,
        }));
        const contaminatedSgd = Object.assign({ kind: 'sgd' } as const, {
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
        const sgd = unwrap(setOptimizer(adam, contaminatedSgd));

        expect(momentum.training.optimizer).toEqual({
            kind: 'sgd-momentum',
            momentum: 0.37,
        });
        expect(adam.training.optimizer).toEqual({
            kind: 'adam',
            beta1: 0.81,
            beta2: 0.971,
            epsilon: 0.000_000_7,
        });
        expect(sgd.training.optimizer).toEqual({ kind: 'sgd' });
        expect(Object.keys(sgd.training.optimizer)).toEqual(['kind']);
    });

    it('uses exact clipping, regression-loss, and penalty variants', () => {
        const clipped = unwrap(setGradientClipping(REGRESSION_RECIPE, {
            kind: 'global-norm',
            maximumNorm: 4.25,
            scope: 'total-objective-gradient',
        }));
        const noClip = unwrap(setGradientClipping(clipped, Object.assign(
            { kind: 'none' } as const,
            { maximumNorm: 99, scope: 'total-objective-gradient' },
        )));
        const huber = unwrap(setRegressionDataLoss(noClip, { kind: 'huber', delta: 1.75 }));
        const mse = unwrap(setRegressionDataLoss(huber, Object.assign(
            { kind: 'mean-squared-error' } as const,
            { delta: 99 },
        )));
        const l1 = unwrap(setPenalty(mse, {
            kind: 'l1',
            coefficient: 0.012,
            applyTo: 'weights',
        }));
        const none = unwrap(setPenalty(l1, Object.assign(
            { kind: 'none' } as const,
            { coefficient: 0.9, applyTo: 'weights' },
        )));

        expect(clipped.training.gradientClipping).toEqual({
            kind: 'global-norm',
            maximumNorm: 4.25,
            scope: 'total-objective-gradient',
        });
        expect(noClip.training.gradientClipping).toEqual({ kind: 'none' });
        expect(huber.objective.dataLoss).toEqual({ kind: 'huber', delta: 1.75 });
        expect(mse.objective.dataLoss).toEqual({ kind: 'mean-squared-error' });
        expect(l1.objective.penalty).toEqual({
            kind: 'l1',
            coefficient: 0.012,
            applyTo: 'weights',
        });
        expect(none.objective.penalty).toEqual({ kind: 'none' });
    });

    it('does not repair invalid active clipping or penalty literals', async () => {
        const invalidClipping = JSON.parse(JSON.stringify({
            kind: 'global-norm',
            maximumNorm: 2,
            scope: 'data-gradient-only',
        }));
        const invalidPenalty = JSON.parse(JSON.stringify({
            kind: 'l1',
            coefficient: 0.01,
            applyTo: 'biases',
        }));
        const clippingResult = setGradientClipping(BINARY_RECIPE, invalidClipping);
        const penaltyResult = setPenalty(BINARY_RECIPE, invalidPenalty);

        expect(unwrap(clippingResult).training.gradientClipping).toEqual(invalidClipping);
        expect(unwrap(penaltyResult).objective.penalty).toEqual(invalidPenalty);
        await expectPreparationIssue(
            clippingResult,
            'recipe.training.gradientClipping.scope',
        );
        await expectPreparationIssue(penaltyResult, 'recipe.objective.penalty.applyTo');
    });

    it('canonicalizes feature order regardless of toggle order', () => {
        let recipe = unwrap(setFeatureIds(BINARY_RECIPE, ['x']));
        recipe = unwrap(toggleFeature(recipe, 'cosY'));
        recipe = unwrap(toggleFeature(recipe, 'ySquared'));
        recipe = unwrap(toggleFeature(recipe, 'y'));

        expect(recipe.inputs.featureIds).toEqual(['x', 'y', 'ySquared', 'cosY']);
        expect(new Set(recipe.inputs.featureIds).size).toBe(recipe.inputs.featureIds.length);
    });

    it('rejects unknown, duplicate, empty, and last-feature removal edits without a candidate', () => {
        const oneFeature = unwrap(setFeatureIds(BINARY_RECIPE, ['x']));

        expect(toggleFeature(oneFeature, 'not-a-feature')).toEqual({
            ok: false,
            issue: { kind: 'unknown-feature', featureId: 'not-a-feature' },
        });
        expect(setFeatureIds(BINARY_RECIPE, ['x', 'x'])).toEqual({
            ok: false,
            issue: { kind: 'duplicate-feature', featureId: 'x' },
        });
        expect(setFeatureIds(BINARY_RECIPE, [])).toEqual({
            ok: false,
            issue: { kind: 'empty-feature-set' },
        });
        expect(toggleFeature(oneFeature, 'x')).toEqual({
            ok: false,
            issue: { kind: 'last-feature-removal', featureId: 'x' },
        });
    });

    it('rejects uint32 seed overflow instead of wrapping', () => {
        const maxSeed = unwrap(setDataSeed(BINARY_RECIPE, 4_294_967_295));

        expect(reshuffleDataSeed(maxSeed)).toEqual({
            ok: false,
            issue: {
                kind: 'seed-overflow',
                target: 'data',
                seed: 4_294_967_295,
            },
        });
        expect(maxSeed.data.seed).toBe(4_294_967_295);
    });

    it('rejects valid batch edits that exceed the derived training population', () => {
        expect(setBatchSize(BINARY_RECIPE, 151)).toEqual({
            ok: false,
            issue: {
                kind: 'batch-exceeds-training-population',
                attemptedEdit: 'batch-size',
                batchSize: 151,
                trainCount: 150,
            },
        });
        expect(setSampleCount(BINARY_RECIPE, 18)).toEqual({
            ok: false,
            issue: {
                kind: 'batch-exceeds-training-population',
                attemptedEdit: 'sample-count',
                batchSize: 10,
                trainCount: 9,
            },
        });
    });

    it('rejects a task switch when its preserved batch already exceeds training count', () => {
        const invalidInput = {
            ...BINARY_RECIPE,
            data: { ...BINARY_RECIPE.data, sampleCount: 18 },
        };

        expect(switchDataset(invalidInput, 'reg-plane')).toEqual({
            ok: false,
            issue: {
                kind: 'batch-exceeds-training-population',
                attemptedEdit: 'dataset',
                batchSize: 10,
                trainCount: 9,
            },
        });
    });

    it('returns invalid ordinary candidates unchanged for structured preparation errors', async () => {
        const invalidSchedule = setSchedule(BINARY_RECIPE, {
            kind: 'step',
            interval: 0,
            gamma: 0.5,
        });
        const invalidRate = setLearningRate(BINARY_RECIPE, 0);
        const invalidArchitecture = setHiddenLayers(BINARY_RECIPE, [100, 100]);

        expect(unwrap(invalidSchedule).training.schedule).toEqual({
            kind: 'step',
            interval: 0,
            gamma: 0.5,
        });
        expect(unwrap(invalidRate).training.learningRate).toBe(0);
        expect(unwrap(invalidArchitecture).model.hiddenLayers).toEqual([100, 100]);
        await expectPreparationIssue(invalidSchedule, 'recipe.training.schedule.interval');
        await expectPreparationIssue(invalidRate, 'recipe.training.learningRate');
        await expectPreparationIssue(invalidArchitecture, 'recipe.model.hiddenLayers[0]');
    });

    it('rejects a hidden-width edit only when the requested layer does not exist', () => {
        expect(setHiddenLayerWidth(BINARY_RECIPE, 0, 7)).toEqual({
            ok: false,
            issue: {
                kind: 'hidden-layer-index-out-of-range',
                layerIndex: 0,
                layerCount: 0,
            },
        });
        expect(unwrap(setHiddenLayerWidth(HIDDEN_RECIPE, 0, 17)).model.hiddenLayers).toEqual([17, 4]);
    });

    it('rejects regression loss edits for classification recipes', () => {
        expect(setRegressionDataLoss(BINARY_RECIPE, { kind: 'huber', delta: 1 })).toEqual({
            ok: false,
            issue: {
                kind: 'incompatible-task-edit',
                editor: 'regression-data-loss',
                taskKind: 'binary-classification',
            },
        });
    });

    it('prepares a representative valid candidate from every editor without mutation or freezing', async () => {
        const binaryEditors = [
            (recipe: StandardExperimentRecipeV2) => switchDataset(recipe, 'xor'),
            (recipe: StandardExperimentRecipeV2) => setSampleCount(recipe, 320),
            (recipe: StandardExperimentRecipeV2) => setTrainFraction(recipe, 0.6),
            (recipe: StandardExperimentRecipeV2) => setNoise(recipe, 7.5),
            (recipe: StandardExperimentRecipeV2) => setDataSeed(recipe, 123),
            (recipe: StandardExperimentRecipeV2) => reshuffleDataSeed(recipe),
            (recipe: StandardExperimentRecipeV2) => setFeatureIds(recipe, ['y', 'xy']),
            (recipe: StandardExperimentRecipeV2) => toggleFeature(recipe, 'xSquared'),
            (recipe: StandardExperimentRecipeV2) => setHiddenLayers(recipe, [5, 3]),
            (recipe: StandardExperimentRecipeV2) => setHiddenActivation(recipe, 'swish'),
            (recipe: StandardExperimentRecipeV2) => setInitialization(recipe, 'he'),
            (recipe: StandardExperimentRecipeV2) => setModelSeed(recipe, 321),
            (recipe: StandardExperimentRecipeV2) => setLearningRate(recipe, 0.02),
            (recipe: StandardExperimentRecipeV2) => setBatchSize(recipe, 12),
            (recipe: StandardExperimentRecipeV2) => setSchedule(recipe, {
                kind: 'cosine',
                totalSteps: 500,
                minimumRate: 0.003,
            }),
            (recipe: StandardExperimentRecipeV2) => setOptimizer(recipe, {
                kind: 'adam',
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            }),
            (recipe: StandardExperimentRecipeV2) => setGradientClipping(recipe, {
                kind: 'global-norm',
                maximumNorm: 2,
                scope: 'total-objective-gradient',
            }),
            (recipe: StandardExperimentRecipeV2) => setPenalty(recipe, {
                kind: 'l2',
                coefficient: 0.01,
                applyTo: 'weights',
            }),
        ];

        for (const edit of binaryEditors) {
            const input = structuredClone(BINARY_RECIPE);
            const inputReference = input;
            const snapshot = structuredClone(input);
            expect(Object.isFrozen(input)).toBe(false);

            const result = edit(input);

            await prepare(unwrap(result));
            expect(input).toBe(inputReference);
            expect(input).toEqual(snapshot);
            expect(Object.isFrozen(input)).toBe(false);
        }

        const widthInput = structuredClone(HIDDEN_RECIPE);
        const widthSnapshot = structuredClone(widthInput);
        await prepare(unwrap(setHiddenLayerWidth(widthInput, 1, 6)));
        expect(widthInput).toEqual(widthSnapshot);
        expect(Object.isFrozen(widthInput)).toBe(false);

        const regressionInput = structuredClone(REGRESSION_RECIPE);
        const regressionSnapshot = structuredClone(regressionInput);
        await prepare(unwrap(setRegressionDataLoss(regressionInput, {
            kind: 'huber',
            delta: 1.25,
        })));
        expect(regressionInput).toEqual(regressionSnapshot);
        expect(Object.isFrozen(regressionInput)).toBe(false);
    });

    it('preserves dataset and objective identities for an orthogonal model edit', async () => {
        const before = await prepare(structuredClone(BINARY_RECIPE));
        const after = await prepare(unwrap(setHiddenActivation(BINARY_RECIPE, 'elu')));

        expect(after.identities.datasetKey).toBe(before.identities.datasetKey);
        expect(after.identities.objectiveKey).toBe(before.identities.objectiveKey);
        expect(after.identities.recipeFingerprint).not.toBe(before.identities.recipeFingerprint);
    });
});
