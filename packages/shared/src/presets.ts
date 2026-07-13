import { prepareExperimentDocument } from './experimentSchema.js';
import type {
    RecipeCatalogEntry,
    RecipeId,
    StandardExperimentRecipeV2,
} from './types.js';

interface RawRecipeCatalogEntry {
    readonly id: RecipeId;
    readonly revision: 1;
    readonly title: string;
    readonly description: string;
    readonly learningGoal?: string;
    readonly difficulty?: 'beginner' | 'intermediate' | 'advanced';
    readonly recipe: StandardExperimentRecipeV2;
}

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

const RAW_PRESETS = [
    {
        id: 'single-neuron',
        revision: 1,
        title: 'Single Neuron Linear Separator',
        description: 'A single neuron can only learn a linear boundary.',
        learningGoal: 'Understand that a single neuron computes a weighted sum and can only separate linearly.',
        difficulty: 'beginner',
        recipe: {
            ...BASE_RECIPE,
            task: { kind: 'binary-classification', dataset: 'gauss' },
            objective: BINARY_OBJECTIVE,
        },
    },
    {
        id: 'xor-hidden',
        revision: 1,
        title: 'XOR Needs Hidden Layers',
        description: 'XOR is not linearly separable — you need at least one hidden layer.',
        learningGoal: 'See that hidden layers enable non-linear decision boundaries.',
        difficulty: 'beginner',
        recipe: {
            ...BASE_RECIPE,
            model: { ...BASE_RECIPE.model, hiddenLayers: [4, 4] },
            task: { kind: 'binary-classification', dataset: 'xor' },
            objective: BINARY_OBJECTIVE,
        },
    },
    {
        id: 'circle-one-layer',
        revision: 1,
        title: 'Circle with One Hidden Layer',
        description: 'A simple circle dataset with a single hidden layer.',
        learningGoal: 'One hidden layer with enough neurons can learn a circular boundary.',
        difficulty: 'intermediate',
        recipe: {
            ...BASE_RECIPE,
            model: {
                ...BASE_RECIPE.model,
                hiddenLayers: [6],
                hiddenActivation: 'relu',
            },
            task: { kind: 'binary-classification', dataset: 'circle' },
            objective: BINARY_OBJECTIVE,
        },
    },
    {
        id: 'spiral-deep',
        revision: 1,
        title: 'Spiral with Deeper Network',
        description: 'The spiral dataset requires a deeper network to learn.',
        learningGoal: 'Deeper networks can learn more complex decision boundaries.',
        difficulty: 'advanced',
        recipe: {
            ...BASE_RECIPE,
            model: { ...BASE_RECIPE.model, hiddenLayers: [8, 8, 4] },
            task: { kind: 'binary-classification', dataset: 'spiral' },
            objective: BINARY_OBJECTIVE,
        },
    },
    {
        id: 'regression-plane',
        revision: 1,
        title: 'Regression with No Hidden Layer',
        description: 'A simple linear regression on a plane surface.',
        learningGoal: 'Linear regression can perfectly fit a plane with no hidden layers.',
        difficulty: 'beginner',
        recipe: {
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
    },
    {
        id: 'feature-engineering',
        revision: 1,
        title: 'Feature Engineering Helps',
        description: 'Adding x² and y² features makes the circle trivially separable.',
        learningGoal: 'Good features can simplify a problem — even without hidden layers.',
        difficulty: 'intermediate',
        recipe: {
            ...BASE_RECIPE,
            inputs: { featureIds: ['x', 'y', 'xSquared', 'ySquared'] },
            task: { kind: 'binary-classification', dataset: 'circle' },
            objective: BINARY_OBJECTIVE,
        },
    },
    {
        id: 'three-class-clusters',
        revision: 1,
        title: 'Three-Class Softmax Lab',
        description: 'Three compact clusters teach how softmax chooses between competing classes.',
        learningGoal: 'See that multiclass classification uses one output per class, and the largest softmax output wins.',
        difficulty: 'advanced',
        recipe: {
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
    },
] as const satisfies readonly RawRecipeCatalogEntry[];

const NEUTRAL_VIEW = {
    showTestData: false,
    discretizeOutput: false,
} as const;

function deepFreeze<T>(value: T, seen = new WeakSet<object>()): T {
    if (typeof value !== 'object' || value === null || seen.has(value)) return value;
    seen.add(value);
    for (const child of Object.values(value)) deepFreeze(child, seen);
    return Object.freeze(value);
}

function invalidBuiltInMessage(
    candidate: RawRecipeCatalogEntry,
    issues: readonly { code: string; path: string; message: string }[],
): string {
    const details = issues
        .map((issue) => `${issue.path} [${issue.code}]: ${issue.message}`)
        .join('; ');
    return `Built-in recipe invariant failed for ${candidate.id}@${candidate.revision}: ${details}`;
}

async function prepareBuiltInCatalog(): Promise<readonly RecipeCatalogEntry[]> {
    const results = await Promise.all(RAW_PRESETS.map(async (candidate) => ({
        candidate,
        result: await prepareExperimentDocument({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: candidate.recipe,
            view: NEUTRAL_VIEW,
        }),
    })));

    const invalid = results.find(({ result }) => !result.ok);
    if (invalid && !invalid.result.ok) {
        throw new Error(invalidBuiltInMessage(invalid.candidate, invalid.result.issues));
    }

    return deepFreeze(results.map(({ candidate, result }) => {
        if (!result.ok) {
            throw new Error(invalidBuiltInMessage(candidate, result.issues));
        }
        const { recipe: _rawRecipe, ...metadata } = candidate;
        return {
            ...metadata,
            recipe: result.value.document.recipe,
            prepared: result.value,
        };
    }));
}

export const PREPARED_PRESETS: readonly RecipeCatalogEntry[] = await prepareBuiltInCatalog();

const RECIPE_BY_REF = new Map(
    PREPARED_PRESETS.map((entry) => [`${entry.id}@${entry.revision}`, entry]),
);

export function resolveRecipe(
    ref: Readonly<{ id: string; revision: number }>,
): RecipeCatalogEntry | undefined {
    return RECIPE_BY_REF.get(`${ref.id}@${ref.revision}`);
}
