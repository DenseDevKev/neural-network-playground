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
            snapshot: { step: 128, epoch: 4, trainLoss: 0.2, testLoss: 0.3 } as any,
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Live run')).toBeInTheDocument();
        expect(screen.getByText('Snapshot step 128')).toBeInTheDocument();
        expect(screen.getByText('Epoch 4')).toBeInTheDocument();
        expect(screen.getByText('train 0.2000')).toBeInTheDocument();
        expect(screen.getByText('test 0.3000')).toBeInTheDocument();
        expect(screen.getByText('gap +0.1000')).toBeInTheDocument();
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

    it('explains stale metrics without treating the recipe as drifted', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({
            snapshot: { step: 72, epoch: 4 } as any,
            testMetricsStale: true,
        });

        render(<CurrentRunCard />);

        expect(screen.getByText('Stale metrics')).toBeInTheDocument();
        expect(screen.getByText('The latest evidence is reusing cached test metrics until a fresh pass completes.')).toBeInTheDocument();
        expect(screen.getByText('metrics stale')).toBeInTheDocument();
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
