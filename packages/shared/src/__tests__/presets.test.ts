import { describe, expect, it } from 'vitest';
import * as presets from '../presets.js';
import {
    PREPARED_PRESETS,
    resolveRecipe,
} from '../index.js';
import type {
    RecipeCatalogEntry,
    RecipeRef,
    StandardExperimentRecipeV2,
} from '../index.js';

const EXPECTED_REFS = [
    { id: 'single-neuron', revision: 1 },
    { id: 'xor-hidden', revision: 1 },
    { id: 'circle-one-layer', revision: 1 },
    { id: 'spiral-deep', revision: 1 },
    { id: 'regression-plane', revision: 1 },
    { id: 'feature-engineering', revision: 1 },
    { id: 'three-class-clusters', revision: 1 },
] as const satisfies readonly RecipeRef[];

type ExpectedRecipeId = typeof EXPECTED_REFS[number]['id'];

const BASE_RECIPE = {
    data: {
        sampleCount: 300,
        trainFraction: 0.5,
        noise: 0,
        seed: 42,
    },
    inputs: {
        featureIds: ['x', 'y'],
    },
    model: {
        hiddenLayers: [],
        hiddenActivation: 'tanh',
        initialization: 'xavier',
        seed: 42,
    },
    training: {
        batchSize: 10,
        learningRate: 0.03,
        schedule: { kind: 'constant' },
        optimizer: { kind: 'sgd' },
        gradientClipping: { kind: 'none' },
    },
} as const;

const BINARY_OBJECTIVE = {
    dataLoss: { kind: 'binary-cross-entropy-with-logits' },
    penalty: { kind: 'none' },
    reduction: 'mean-per-sample',
} as const;

const EXPECTED_RECIPES = {
    'single-neuron': {
        ...BASE_RECIPE,
        task: { kind: 'binary-classification', dataset: 'gauss' },
        objective: BINARY_OBJECTIVE,
    },
    'xor-hidden': {
        ...BASE_RECIPE,
        model: { ...BASE_RECIPE.model, hiddenLayers: [4, 4] },
        task: { kind: 'binary-classification', dataset: 'xor' },
        objective: BINARY_OBJECTIVE,
    },
    'circle-one-layer': {
        ...BASE_RECIPE,
        model: {
            ...BASE_RECIPE.model,
            hiddenLayers: [6],
            hiddenActivation: 'relu',
        },
        task: { kind: 'binary-classification', dataset: 'circle' },
        objective: BINARY_OBJECTIVE,
    },
    'spiral-deep': {
        ...BASE_RECIPE,
        model: { ...BASE_RECIPE.model, hiddenLayers: [8, 8, 4] },
        task: { kind: 'binary-classification', dataset: 'spiral' },
        objective: BINARY_OBJECTIVE,
    },
    'regression-plane': {
        ...BASE_RECIPE,
        data: { ...BASE_RECIPE.data, noise: 5 },
        training: { ...BASE_RECIPE.training, learningRate: 0.01 },
        task: { kind: 'regression', dataset: 'reg-plane' },
        objective: {
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        },
    },
    'feature-engineering': {
        ...BASE_RECIPE,
        inputs: { featureIds: ['x', 'y', 'xSquared', 'ySquared'] },
        task: { kind: 'binary-classification', dataset: 'circle' },
        objective: BINARY_OBJECTIVE,
    },
    'three-class-clusters': {
        ...BASE_RECIPE,
        data: { ...BASE_RECIPE.data, noise: 0.05 },
        model: { ...BASE_RECIPE.model, hiddenLayers: [6, 6] },
        task: {
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
        },
        objective: {
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        },
    },
} as const satisfies Record<ExpectedRecipeId, StandardExperimentRecipeV2>;

const EXPECTED_FINGERPRINTS: Record<ExpectedRecipeId, string> = {
    'single-neuron': 'r2.1.qoFTcasi-Ml9rion3oAAtahrvXsmVt_5Qh9hKwXrodI',
    'xor-hidden': 'r2.1.s7KkpuX9x5Ct1FqICqBkuk3veDCjyFcslR9HVKOwIv4',
    'circle-one-layer': 'r2.1.2-OG05gxjMt1HrOB63U-X01bQkpbwsCGWYdLx2jcz2o',
    'spiral-deep': 'r2.1.w5FqLcy63bSrw0VZAeRylAwe5vh9lAfn-C0Xn8Q0ntM',
    'regression-plane': 'r2.1.nnCQPsCK4Sod5OJJogEvi3vy6gE3Uyc9TG7rO-LC0t8',
    'feature-engineering': 'r2.1.AJTeOfsnIOiodeA6xZOYO5Eex3T5CNO_9gx1mkqTeW4',
    'three-class-clusters': 'r2.1.4EVeEMIUv4zdaWIk0GQRVg9UKkktYAdCix5PPECMozU',
};

const EXPECTED_METADATA = {
    'single-neuron': {
        title: 'Single Neuron Linear Separator',
        description: 'A single neuron can only learn a linear boundary.',
        learningGoal: 'Understand that a single neuron computes a weighted sum and can only separate linearly.',
        difficulty: 'beginner',
    },
    'xor-hidden': {
        title: 'XOR Needs Hidden Layers',
        description: 'XOR is not linearly separable — you need at least one hidden layer.',
        learningGoal: 'See that hidden layers enable non-linear decision boundaries.',
        difficulty: 'beginner',
    },
    'circle-one-layer': {
        title: 'Circle with One Hidden Layer',
        description: 'A simple circle dataset with a single hidden layer.',
        learningGoal: 'One hidden layer with enough neurons can learn a circular boundary.',
        difficulty: 'intermediate',
    },
    'spiral-deep': {
        title: 'Spiral with Deeper Network',
        description: 'The spiral dataset requires a deeper network to learn.',
        learningGoal: 'Deeper networks can learn more complex decision boundaries.',
        difficulty: 'advanced',
    },
    'regression-plane': {
        title: 'Regression with No Hidden Layer',
        description: 'A simple linear regression on a plane surface.',
        learningGoal: 'Linear regression can perfectly fit a plane with no hidden layers.',
        difficulty: 'beginner',
    },
    'feature-engineering': {
        title: 'Feature Engineering Helps',
        description: 'Adding x² and y² features makes the circle trivially separable.',
        learningGoal: 'Good features can simplify a problem — even without hidden layers.',
        difficulty: 'intermediate',
    },
    'three-class-clusters': {
        title: 'Three-Class Softmax Lab',
        description: 'Three compact clusters teach how softmax chooses between competing classes.',
        learningGoal: 'See that multiclass classification uses one output per class, and the largest softmax output wins.',
        difficulty: 'advanced',
    },
} as const;

const NEUTRAL_VIEW = {
    showTestData: false,
    discretizeOutput: false,
} as const;

function expectDeeplyFrozen(value: unknown, seen = new Set<object>()): void {
    if (typeof value !== 'object' || value === null || seen.has(value)) return;
    seen.add(value);
    expect(Object.isFrozen(value)).toBe(true);
    for (const child of Object.values(value)) expectDeeplyFrozen(child, seen);
}

function collectKeys(value: unknown, keys = new Set<string>()): Set<string> {
    if (typeof value !== 'object' || value === null) return keys;
    for (const [key, child] of Object.entries(value)) {
        keys.add(key);
        collectKeys(child, keys);
    }
    return keys;
}

describe('prepared recipe catalog', () => {
    it('does not expose the removed PRESETS compatibility alias', () => {
        expect(presets).not.toHaveProperty('PRESETS');
    });

    it('exposes exactly the seven revision-1 recipes in canonical order', () => {
        expect(PREPARED_PRESETS.map(({ id, revision }) => ({ id, revision })))
            .toEqual(EXPECTED_REFS);
    });

    it('preserves the educational copy and declares every complete recipe field', () => {
        for (const entry of PREPARED_PRESETS) {
            expect(entry).toMatchObject(EXPECTED_METADATA[entry.id]);
            expect(entry.recipe).toEqual(EXPECTED_RECIPES[entry.id]);
            expect(Object.keys(entry.recipe).sort()).toEqual([
                'data',
                'inputs',
                'model',
                'objective',
                'task',
                'training',
            ]);
        }
    });

    it('pairs each recipe with its actual neutral-view prepared document', () => {
        for (const entry of PREPARED_PRESETS) {
            expect(entry.prepared.document.recipe).toBe(entry.recipe);
            expect(entry.prepared.document.view).toEqual(NEUTRAL_VIEW);
            expect(entry.prepared.identities.recipeFingerprint)
                .toBe(EXPECTED_FINGERPRINTS[entry.id]);
        }
    });

    it('resolves only exact id and revision pairs', () => {
        for (const ref of EXPECTED_REFS) {
            expect(resolveRecipe(ref)).toBe(PREPARED_PRESETS.find((entry) => entry.id === ref.id));
        }

        expect(resolveRecipe({ id: 'xor-hidden', revision: 2 })).toBeUndefined();
        expect(resolveRecipe({ id: 'missing-recipe', revision: 1 })).toBeUndefined();
    });

    it('pins the exact revision-1 recipe fingerprints', () => {
        expect(Object.fromEntries(PREPARED_PRESETS.map((entry) => [
            entry.id,
            entry.prepared.identities.recipeFingerprint,
        ]))).toEqual(EXPECTED_FINGERPRINTS);
    });

    it('does not expose the legacy partial config shape', () => {
        type CatalogHasConfig = 'config' extends keyof RecipeCatalogEntry ? true : false;
        const catalogHasConfig: CatalogHasConfig = false;

        expect(catalogHasConfig).toBe(false);
        for (const entry of PREPARED_PRESETS) {
            expect(entry).not.toHaveProperty('config');
        }
    });

    it('carries no redundant legacy problem, output, loss, or optimizer fields', () => {
        const forbiddenKeys = [
            'problem',
            'problemType',
            'inputSize',
            'output',
            'outputSize',
            'outputActivation',
            'loss',
            'lossType',
            'momentum',
            'regularization',
            'regularizationRate',
            'gradientClip',
        ];

        for (const entry of PREPARED_PRESETS) {
            const keys = collectKeys(entry.recipe);
            for (const key of forbiddenKeys) expect(keys.has(key), `${entry.id}: ${key}`).toBe(false);
        }
    });

    it('recursively freezes catalog values so one application cannot corrupt the next', () => {
        for (const entry of PREPARED_PRESETS) expectDeeplyFrozen(entry);

        const xor = resolveRecipe({ id: 'xor-hidden', revision: 1 });
        expect(xor).toBeDefined();
        if (!xor) throw new Error('Missing xor-hidden@1');
        expect(Reflect.set(xor, 'title', 'corrupted')).toBe(false);
        expect(Reflect.set(xor.recipe.model.hiddenLayers, '0', 99)).toBe(false);

        const reapplied = resolveRecipe({ id: 'xor-hidden', revision: 1 });
        expect(reapplied?.title).toBe(EXPECTED_METADATA['xor-hidden'].title);
        expect(reapplied?.recipe.model.hiddenLayers).toEqual([4, 4]);
    });
});
