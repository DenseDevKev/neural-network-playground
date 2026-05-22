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
import { RecipeSummaryCard } from './RecipeSummaryCard.tsx';

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

describe('RecipeSummaryCard', () => {
    beforeEach(() => {
        const config = makeConfig();
        usePlaygroundStore.setState(config);
        useTrainingStore.setState({
            trainedRecipeConfig: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
        });
    });

    it('summarizes the editable current recipe when no snapshot exists', () => {
        render(<RecipeSummaryCard />);

        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByText('Circle classification')).toBeInTheDocument();
        expect(screen.getByText('2 -> 4 x 4 -> 1, tanh')).toBeInTheDocument();
        expect(screen.getByText('2 features')).toBeInTheDocument();
        expect(screen.getByText('cross entropy, batch 10')).toBeInTheDocument();
        expect(screen.getByText('No trained snapshot yet.')).toBeInTheDocument();
    });

    it('shows calm drift details when the current recipe differs from the trained snapshot', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        usePlaygroundStore.setState({
            ...makeConfig(),
            network: {
                ...DEFAULT_NETWORK,
                inputSize: 2,
                hiddenLayers: [8, 4],
                seed: DEFAULT_DATA.seed,
            },
            training: { ...DEFAULT_TRAINING, learningRate: 0.1 },
        });

        render(<RecipeSummaryCard />);

        expect(screen.getByText('Current recipe differs from trained snapshot.')).toBeInTheDocument();
        expect(screen.getByText('Network, Training')).toBeInTheDocument();
        expect(screen.getByText('Hidden layers')).toBeInTheDocument();
        expect(screen.getByText('4 x 4 -> 8 x 4')).toBeInTheDocument();
    });

    it('does not describe stale evidence as aligned with the current recipe', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'config-sync');
        useTrainingStore.setState({
            snapshot: { step: 42, epoch: 2 } as any,
            testMetricsStale: true,
        });

        render(<RecipeSummaryCard />);

        expect(screen.getByText('Current recipe accepted; evidence metrics are stale.')).toBeInTheDocument();
        expect(screen.getByText('Run, step, or resume to refresh evidence for this recipe.')).toBeInTheDocument();
        expect(screen.queryByText('Evidence is aligned with the current recipe.')).not.toBeInTheDocument();
    });
});
