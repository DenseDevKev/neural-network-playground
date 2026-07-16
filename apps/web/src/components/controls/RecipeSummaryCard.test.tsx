import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PREPARED_PRESETS, type PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { RecipeSummaryCard } from './RecipeSummaryCard.tsx';

function prepared(id = 'xor-hidden'): PreparedExperimentDocumentV2 {
    const match = PREPARED_PRESETS.find((entry) => entry.id === id)?.prepared;
    if (!match) throw new Error(`missing preset ${id}`);
    return match;
}

function installCurrent(next = prepared()) {
    usePlaygroundStore.setState({
        access: { status: 'ready', prepared: next },
        preparation: { status: 'ready', requestId: 0, issues: [] },
    });
}

function installTrained(next = prepared()) {
    useTrainingStore.getState().markTrainedRecipe(
        next.document.recipe,
        'initialize',
        next.identities.recipeFingerprint,
    );
}

function withAdvancedSettings(): PreparedExperimentDocumentV2 {
    const base = prepared('regression-plane');
    return {
        ...base,
        document: {
            ...base.document,
            recipe: {
                ...base.document.recipe,
                training: {
                    ...base.document.recipe.training,
                    schedule: { kind: 'step', interval: 17, gamma: 0.63 },
                    optimizer: { kind: 'sgd-momentum', momentum: 0.81 },
                    gradientClipping: {
                        kind: 'global-norm',
                        maximumNorm: 2.5,
                        scope: 'total-objective-gradient',
                    },
                },
                objective: {
                    ...base.document.recipe.objective,
                    dataLoss: { kind: 'huber', delta: 1.75 },
                    penalty: { kind: 'l2', coefficient: 0.012, applyTo: 'weights' },
                },
            },
        },
    } as PreparedExperimentDocumentV2;
}

describe('RecipeSummaryCard', () => {
    beforeEach(() => {
        installCurrent();
        useTrainingStore.setState({
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
            latestLiveSignal: null,
            latestEvaluation: null,
        });
        useLayoutStore.setState({
            audienceMode: 'explore',
            advancedToolsOpen: false,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
        });
    });

    it('summarizes the exact current V2 recipe when no trained evidence exists', () => {
        render(<RecipeSummaryCard />);
        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByText('XOR classification')).toBeInTheDocument();
        expect(screen.getByText('2 -> 4 x 4 -> 1, tanh')).toBeInTheDocument();
        expect(screen.getByText('2 features')).toBeInTheDocument();
        expect(screen.getByText('binary cross entropy, batch 10')).toBeInTheDocument();
        expect(screen.getByText('No trained snapshot yet.')).toBeInTheDocument();
    });

    it('shows drift from the exact trained V2 recipe', () => {
        installTrained();
        installCurrent(prepared('regression-plane'));
        render(<RecipeSummaryCard />);
        expect(screen.getByText('Current recipe differs from trained snapshot.')).toBeInTheDocument();
        expect(screen.getByText('Dataset, Network, Training')).toBeInTheDocument();
    });

    it('states the exact paired-evaluation age instead of a global stale flag', () => {
        installTrained();
        useTrainingStore.setState({
            latestLiveSignal: {
                model: { generationId: 1, revision: 42, step: 42, epoch: 2 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 7, testCount: 3 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 2, throughStep: 42 },
                dataLoss: 0.4,
            },
            latestEvaluation: {
                evaluationId: 2,
                trigger: 'cadence',
                model: { generationId: 1, revision: 40, step: 40, epoch: 2 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 7, testCount: 3 },
                objectiveKey: 'o',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 7, populationCount: 7 }, values: { dataLoss: 0.3 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 3, populationCount: 3 }, values: { dataLoss: 0.4 } },
                objective: { regularizationPenalty: 0, trainTotalObjective: 0.3 },
            },
        });
        render(<RecipeSummaryCard />);
        expect(screen.getByText('Full evaluation at step 40; batch trend through step 42.')).toBeInTheDocument();
        expect(screen.getByText('Paired train/test evidence is 2 steps behind the current model.')).toBeInTheDocument();
    });

    it('recovers active advanced settings hidden by Beginner without changing the recipe', async () => {
        const user = userEvent.setup();
        installCurrent(withAdvancedSettings());
        useLayoutStore.setState({ audienceMode: 'beginner', advancedToolsOpen: false });
        const recipe = usePlaygroundStore.getState().access.status === 'ready'
            ? usePlaygroundStore.getState().access.prepared.document.recipe
            : null;

        render(<RecipeSummaryCard />);

        const note = screen.getByRole('note', { name: 'Advanced settings active' });
        expect(note).toHaveTextContent('Step interval17');
        expect(note).toHaveTextContent('Step gamma0.63');
        expect(note).toHaveTextContent('Momentum0.81');
        expect(note).toHaveTextContent('Huber delta1.75');
        expect(note).toHaveTextContent('L2 coefficient (weights)0.012');
        expect(note).toHaveTextContent('Global norm clip (total objective gradient)2.5');

        await user.click(screen.getByRole('button', { name: 'Open Advanced Tools' }));

        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
        expect(usePlaygroundStore.getState().access.status).toBe('ready');
        if (usePlaygroundStore.getState().access.status === 'ready') {
            expect(usePlaygroundStore.getState().access.prepared.document.recipe).toBe(recipe);
        }
    });

    it('atomically leaves Run for visible Hyperparameters without changing scientific state', async () => {
        const user = userEvent.setup();
        const next = withAdvancedSettings();
        installCurrent(next);
        const liveSignal = {
            model: { generationId: 4, revision: 20, step: 20, epoch: 2 },
            dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
            objectiveKey: 'o',
            basis: {
                kind: 'mini-batch-ema' as const,
                alpha: 0.1,
                latestBatchSize: 10,
                throughStep: 20,
            },
            dataLoss: 0.4,
        };
        const checkpointTimeline = {
            checkpoints: [],
            maxCheckpoints: 12,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
        };
        useTrainingStore.setState({
            status: 'paused',
            pauseReason: 'manual',
            trainedRecipe: next.document.recipe,
            trainedRecipeFingerprint: next.identities.recipeFingerprint,
            latestLiveSignal: liveSignal,
            checkpointTimeline,
        });
        useLayoutStore.setState({
            view: 'run',
            phase: 'run',
            audienceMode: 'beginner',
            advancedToolsOpen: false,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
        });
        const recipe = next.document.recipe;
        const transitions: Array<ReturnType<typeof useLayoutStore.getState>> = [];
        const unsubscribe = useLayoutStore.subscribe((state) => transitions.push(state));

        render(<RecipeSummaryCard />);
        expect(screen.getByRole('note', { name: 'Advanced settings active' }))
            .toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Open Advanced Tools' }));
        unsubscribe();

        expect(transitions).toHaveLength(1);
        expect(transitions[0]).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'hyperparams',
            activeTabLeft: 'hyperparams',
            advancedToolsOpen: true,
        });
        expect(usePlaygroundStore.getState().access.status).toBe('ready');
        if (usePlaygroundStore.getState().access.status === 'ready') {
            expect(usePlaygroundStore.getState().access.prepared.document.recipe).toBe(recipe);
        }
        expect(useTrainingStore.getState()).toMatchObject({
            status: 'paused',
            pauseReason: 'manual',
        });
        expect(useTrainingStore.getState().latestLiveSignal).toBe(liveSignal);
        expect(useTrainingStore.getState().checkpointTimeline).toBe(checkpointTimeline);
    });

    it('does not warn when Hyperparameters is already visible', () => {
        installCurrent(withAdvancedSettings());
        useLayoutStore.setState({ audienceMode: 'explore', advancedToolsOpen: false });

        render(<RecipeSummaryCard />);

        expect(screen.queryByRole('note', { name: 'Advanced settings active' }))
            .not.toBeInTheDocument();
    });
});
