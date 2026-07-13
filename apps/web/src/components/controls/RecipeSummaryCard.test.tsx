import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PREPARED_PRESETS, type PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
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
});
