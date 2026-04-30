import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FeaturesPanel } from './FeaturesPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('FeaturesPanel loading feedback', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            featuresConfigLoading: false,
            trainingConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
        });
    });

    it('shows the inline loading state when features change', async () => {
        const user = userEvent.setup();

        render(<FeaturesPanel />);

        await user.click(screen.getByRole('button', { name: 'X₁²' }));

        expect(screen.getByRole('status')).toHaveTextContent('Updating features...');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('features');
    });

    it('shows feature-specific config errors and allows retrying', async () => {
        const user = userEvent.setup();

        useTrainingStore.setState({
            configError: 'Failed to update features',
            configErrorSource: 'features',
        });

        render(<FeaturesPanel />);

        expect(screen.getByText('Failed to update features')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('features');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });

    it('does not start loading when the feature toggle would be rejected', async () => {
        const user = userEvent.setup();

        usePlaygroundStore.setState((state) => ({
            features: {
                ...state.features,
                x: true,
                y: false,
            },
            network: {
                ...state.network,
                inputSize: 1,
            },
        }));

        render(<FeaturesPanel />);

        await user.click(screen.getByRole('button', { name: 'X₁' }));

        expect(screen.queryByRole('status')).not.toBeInTheDocument();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
    });
});
