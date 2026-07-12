import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { CurrentRunCard } from './CurrentRunCard.tsx';

function makeConfig(overrides: Partial<AppConfig> = {}): AppConfig {
    return {
        data: overrides.data ?? { ...DEFAULT_DATA },
        features: overrides.features ?? { ...DEFAULT_FEATURES },
        network: overrides.network ?? {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            hiddenLayers: [...DEFAULT_NETWORK.hiddenLayers],
            seed: DEFAULT_DATA.seed,
        },
        training: overrides.training ?? { ...DEFAULT_TRAINING },
        ui: overrides.ui ?? { showTestData: false, discretizeOutput: false },
    };
}

describe('CurrentRunCard', () => {
    beforeEach(() => {
        const config = makeConfig();
        usePlaygroundStore.setState(config);
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainedRecipeConfig: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            workerError: null,
            pauseReason: null,
            testMetricsStale: false,
        });
    });

    it('explains the idle no-snapshot state', () => {
        render(<CurrentRunCard />);

        expect(screen.getByRole('region', { name: 'Current run' })).toBeInTheDocument();
        expect(screen.getByText('Ready to train')).toBeInTheDocument();
        expect(screen.getByText('No trained snapshot exists yet.')).toBeInTheDocument();
    });

    it('identifies a live run and trained snapshot step', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({
            status: 'running',
            snapshot: { step: 999, epoch: 99, trainLoss: 9, testLoss: 8 } as any,
            latestLiveSignal: {
                model: { generationId: 1, revision: 128, step: 128, epoch: 4 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 128 },
                dataLoss: 0.2,
            },
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

    it('prioritizes pending config sync over generic idle state', () => {
        useTrainingStore.setState({
            pendingConfigSource: 'training',
            trainingConfigLoading: true,
            snapshot: { step: 12, epoch: 1 } as any,
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Updating training config')).toBeInTheDocument();
        expect(screen.getByText('Evidence still reflects the last trained snapshot until the update completes.')).toBeInTheDocument();
    });

    it('explains paused runs with their pause reason', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({
            status: 'paused',
            pauseReason: 'manual',
            snapshot: { step: 64, epoch: 3 } as any,
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Paused run')).toBeInTheDocument();
        expect(screen.getByText('Paused manually.')).toBeInTheDocument();
    });

    it('states evaluation age without a global stale-metrics claim', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({
            snapshot: { step: 72, epoch: 4 } as any,
            latestLiveSignal: {
                model: { generationId: 1, revision: 72, step: 72, epoch: 4 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 7, testCount: 3 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 2, throughStep: 72 },
                dataLoss: 0.4,
            },
            latestEvaluation: null,
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Awaiting full evaluation')).toBeInTheDocument();
        expect(screen.getByText(/Batch trend is current through step 72/i)).toBeInTheDocument();
        expect(screen.queryByText(/metrics stale/i)).not.toBeInTheDocument();
    });

    it('surfaces worker failure as the current run state', () => {
        useTrainingStore.setState({ workerError: 'Worker channel closed unexpectedly.' });

        render(<CurrentRunCard />);

        expect(screen.getByText('Worker connection lost')).toBeInTheDocument();
        expect(screen.getByText('Worker channel closed unexpectedly. Refresh the page to restart the playground.')).toBeInTheDocument();
    });

    it('shows stale drift when current recipe differs from trained snapshot', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({ snapshot: { step: 20, epoch: 2 } as any });
        usePlaygroundStore.setState({
            ...makeConfig(),
            training: { ...DEFAULT_TRAINING, learningRate: 0.1 },
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Recipe drift')).toBeInTheDocument();
        expect(screen.getByText('Current recipe differs from trained snapshot.')).toBeInTheDocument();
        expect(screen.getByText('Training')).toBeInTheDocument();
    });
});
