import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PREPARED_PRESETS, type PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getConceptById } from '../../concepts/conceptCatalog.ts';
import { CurrentRunCard } from './CurrentRunCard.tsx';

function prepared(id = 'xor-hidden'): PreparedExperimentDocumentV2 {
    const match = PREPARED_PRESETS.find((entry) => entry.id === id)?.prepared;
    if (!match) throw new Error(`missing prepared preset ${id}`);
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

function live(step: number, epoch: number) {
    return {
        model: { generationId: 1, revision: step, step, epoch },
        dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
        objectiveKey: 'o',
        basis: { kind: 'mini-batch-ema' as const, alpha: 0.1, latestBatchSize: 10, throughStep: step },
        dataLoss: 0.2,
    };
}

describe('CurrentRunCard', () => {
    beforeEach(() => {
        installCurrent();
        useTrainingStore.setState({
            status: 'idle',
            evidenceGenerationId: null,
            latestLiveSignal: null,
            latestEvaluation: null,
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            workerError: null,
            pauseReason: null,
        });
        useLayoutStore.setState({ audienceMode: 'beginner' });
    });

    it('opens the catalog data-loss definition without replacing the current-run label', async () => {
        const user = userEvent.setup();
        installTrained();
        useTrainingStore.setState({
            latestLiveSignal: live(128, 4),
            latestEvaluation: {
                evaluationId: 3,
                trigger: 'cadence',
                model: { generationId: 1, revision: 128, step: 128, epoch: 4 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 }, values: { dataLoss: 0.22, accuracy: 0.9 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 }, values: { dataLoss: 0.31, accuracy: 0.8 } },
                objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.23 },
            },
        });
        render(<CurrentRunCard />);

        expect(screen.getByText('Current run')).toBeInTheDocument();
        expect(screen.getByText('Train data loss (full split) 0.2200')).toBeInTheDocument();
        expect(screen.getByText('Training objective 0.2300')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Learn about Data loss' }));

        expect(screen.getByText(getConceptById('data-loss')?.plainDefinition ?? ''))
            .toBeInTheDocument();
        expect(screen.getByText(getConceptById('data-loss')?.examples?.[0] ?? ''))
            .toBeInTheDocument();
    });

    it('explains the idle no-evidence state', () => {
        render(<CurrentRunCard />);
        expect(screen.getByRole('region', { name: 'Current run' })).toBeInTheDocument();
        expect(screen.getByText('Ready to train')).toBeInTheDocument();
        expect(screen.getByText('No trained snapshot exists yet.')).toBeInTheDocument();
    });

    it('does not treat a contradictory legacy snapshot as scientific model evidence', () => {
        useTrainingStore.setState({
            latestLiveSignal: null,
            latestEvaluation: null,
            snapshot: { step: 999, epoch: 99, trainLoss: 9, testLoss: 8 },
        } as never);
        render(<CurrentRunCard />);
        expect(screen.getByText('Ready to train')).toBeInTheDocument();
        expect(screen.getByText('No batch trend')).toBeInTheDocument();
        expect(screen.queryByText(/999/)).not.toBeInTheDocument();
    });

    it('identifies live and paired evidence separately', () => {
        installTrained();
        useTrainingStore.setState({
            status: 'running',
            latestLiveSignal: live(128, 4),
            latestEvaluation: {
                evaluationId: 3,
                trigger: 'cadence',
                model: { generationId: 1, revision: 120, step: 120, epoch: 3 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 }, values: { dataLoss: 0.22, accuracy: 0.9 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 }, values: { dataLoss: 0.31, accuracy: 0.8 } },
                objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.23 },
            },
        });

        render(<CurrentRunCard />);
        expect(screen.getByText('Live run')).toBeInTheDocument();
        expect(screen.getByText('Batch trend through step 128')).toBeInTheDocument();
        expect(screen.getByText('Full evaluation 3 at step 120')).toBeInTheDocument();
        expect(screen.getByText('Epoch 4')).toBeInTheDocument();
        expect(screen.getByText('Batch trend (EMA) 0.2000')).toBeInTheDocument();
        expect(screen.getByText('Train data loss (full split) 0.2200')).toBeInTheDocument();
        expect(screen.getByText('Test data loss (full split) 0.3100')).toBeInTheDocument();
        expect(screen.getByText('gap +0.0900')).toBeInTheDocument();
    });

    it('exposes the accepted model identity for browser-level invariant checks', () => {
        installTrained();
        useTrainingStore.setState({
            status: 'paused',
            latestLiveSignal: {
                ...live(128, 4),
                model: { generationId: 7, revision: 131, step: 128, epoch: 4 },
            },
        });

        render(<CurrentRunCard />);

        const currentRun = screen.getByRole('region', { name: 'Current run' });
        expect(currentRun).toHaveAttribute('data-model-generation', '7');
        expect(currentRun).toHaveAttribute('data-model-revision', '131');
        expect(currentRun).toHaveAttribute('data-model-step', '128');
    });

    it('prioritizes pending config sync over generic idle state', () => {
        useTrainingStore.setState({ pendingConfigSource: 'training', trainingConfigLoading: true });
        render(<CurrentRunCard />);
        expect(screen.getByText('Updating training config')).toBeInTheDocument();
    });

    it('explains paused runs with their pause reason', () => {
        installTrained();
        useTrainingStore.setState({ status: 'paused', pauseReason: 'manual', latestLiveSignal: live(64, 3) });
        render(<CurrentRunCard />);
        expect(screen.getByText('Paused run')).toBeInTheDocument();
        expect(screen.getByText('Paused manually.')).toBeInTheDocument();
    });

    it('states evaluation absence without a global stale flag', () => {
        installTrained();
        useTrainingStore.setState({ latestLiveSignal: live(72, 4), latestEvaluation: null });
        render(<CurrentRunCard />);
        expect(screen.getByText('Awaiting full evaluation')).toBeInTheDocument();
        expect(screen.getByText(/Batch trend is current through step 72/i)).toBeInTheDocument();
        expect(screen.queryByText(/metrics stale/i)).not.toBeInTheDocument();
    });

    it('surfaces worker failure as the current run state', () => {
        useTrainingStore.setState({ workerError: 'Worker channel closed unexpectedly.' });
        render(<CurrentRunCard />);
        expect(screen.getByText('Worker connection lost')).toBeInTheDocument();
    });

    it('shows recipe drift against the exact trained recipe', () => {
        installTrained();
        installCurrent(prepared('regression-plane'));
        useTrainingStore.setState({ latestLiveSignal: live(20, 2) });
        render(<CurrentRunCard />);
        expect(screen.getByText('Recipe drift')).toBeInTheDocument();
        expect(screen.getByText('Current recipe differs from trained snapshot.')).toBeInTheDocument();
    });
});
