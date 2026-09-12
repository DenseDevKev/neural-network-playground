import type { ExperimentSchemaIssue, PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import type { RecipeEditIssue } from './recipeEdits.ts';
import {
    usePlaygroundStore,
    type PlaygroundStore,
    type StoreRecipeEditResult,
} from './usePlaygroundStore.ts';
import {
    useTrainingStore,
    type ConfigChangeSource,
} from './useTrainingStore.ts';

export type RecipeConfigChangeSource = Exclude<
    ConfigChangeSource,
    null | 'preset'
>;

type RecipeEdit = Parameters<PlaygroundStore['editRecipe']>[0];

let latestRecipeEditTransaction = 0;

function formatSchemaIssues(issues: readonly ExperimentSchemaIssue[]): string {
    return issues.map((issue) => `${issue.path}: ${issue.message}`).join('; ');
}

export function formatRecipeEditIssue(issue: RecipeEditIssue): string {
    switch (issue.kind) {
        case 'batch-exceeds-training-population': {
            const paths = {
                'batch-size': 'recipe.training.batchSize',
                'sample-count': 'recipe.data.sampleCount',
                'train-fraction': 'recipe.data.trainFraction',
                dataset: 'recipe.task.dataset',
            } as const;
            return `${paths[issue.attemptedEdit]}: batch size ${issue.batchSize} exceeds training population ${issue.trainCount}`;
        }
        case 'seed-overflow':
            return `recipe.data.seed: cannot reshuffle seed ${issue.seed} beyond the uint32 maximum`;
        case 'unknown-feature':
            return `recipe.inputs.featureIds: unknown feature ${issue.featureId}`;
        case 'duplicate-feature':
            return `recipe.inputs.featureIds: duplicate feature ${issue.featureId}`;
        case 'empty-feature-set':
            return 'recipe.inputs.featureIds: at least one feature is required';
        case 'last-feature-removal':
            return `recipe.inputs.featureIds: cannot remove the final active feature ${issue.featureId}`;
        case 'hidden-layer-index-out-of-range':
            return `recipe.model.hiddenLayers: layer index ${issue.layerIndex} is outside ${issue.layerCount} hidden layers`;
        case 'incompatible-task-edit':
            return `recipe.objective.dataLoss: ${issue.editor} is unavailable for task ${issue.taskKind}`;
    }
}

function formatFailure(result: Extract<StoreRecipeEditResult, { ok: false }>): string {
    if ('issue' in result) return formatRecipeEditIssue(result.issue);
    return formatSchemaIssues(result.issues) || 'recipe: experiment edit failed';
}

function errorMessage(error: unknown): string {
    return error instanceof Error && error.message
        ? `recipe: ${error.message}`
        : 'recipe: experiment edit failed';
}

/**
 * Runs one canonical recipe edit as an owned training-config transaction.
 * Returns this request's exact prepared object only while it is the current
 * store publication. Stale completions never clear or overwrite newer state.
 */
export async function commitRecipeEditPrepared(
    source: RecipeConfigChangeSource,
    edit: RecipeEdit,
    expectedBaseKey?: string,
): Promise<PreparedExperimentDocumentV2 | null> {
    // Compare before beginning a transaction: stale drafts must not pause or
    // replace the active recipe, nor supersede another pending preparation.
    const current = usePlaygroundStore.getState();
    if (expectedBaseKey !== undefined && (
        current.access.status !== 'ready'
        || current.access.prepared.identities.canonicalRecipeKey !== expectedBaseKey
        || current.preparation.status === 'preparing'
        || useTrainingStore.getState().pendingConfigSource !== null
    )) return null;
    const transactionId = ++latestRecipeEditTransaction;
    const trainingStore = useTrainingStore.getState();
    trainingStore.beginConfigChange(source);
    let requestId = usePlaygroundStore.getState().preparation.requestId;

    try {
        const pendingResult = usePlaygroundStore.getState().editRecipe(edit);
        requestId = usePlaygroundStore.getState().preparation.requestId;
        const result = await pendingResult;
        const playground = usePlaygroundStore.getState();

        if (transactionId !== latestRecipeEditTransaction
            || playground.preparation.requestId !== requestId) return null;
        if (!result.ok) {
            trainingStore.failConfigChange(formatFailure(result));
            return null;
        }
        return playground.access.status === 'ready'
            && playground.access.prepared === result.value ? result.value : null;
    } catch (error) {
        if (transactionId === latestRecipeEditTransaction
            && usePlaygroundStore.getState().preparation.requestId === requestId) {
            trainingStore.failConfigChange(errorMessage(error));
        }
        return null;
    }
}

/** Compatibility facade for inline edits that only need publication success. */
export async function commitRecipeEdit(
    source: RecipeConfigChangeSource,
    edit: RecipeEdit,
    expectedBaseKey?: string,
): Promise<boolean> {
    return (await commitRecipeEditPrepared(source, edit, expectedBaseKey)) !== null;
}
