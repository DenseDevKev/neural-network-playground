import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { NetworkConfigPanel } from './NetworkConfigPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

describe('NetworkConfigPanel loading feedback', () => {
    beforeEach(() => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                hiddenLayers: [4],
                activation: 'relu',
            },
        }));

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
        });
    });

    it('shows the inline loading state when the network changes', async () => {
        const user = userEvent.setup();

        render(<NetworkConfigPanel />);

        await user.click(screen.getByRole('button', { name: 'Add hidden layer' }));

        expect(screen.getByRole('status')).toHaveTextContent('Initializing network...');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
    });

    it('shows network-specific config errors and allows retrying', async () => {
        const user = userEvent.setup();

        useTrainingStore.setState({
            configError: 'Failed to initialize network',
            configErrorSource: 'network',
        });

        render(<NetworkConfigPanel />);

        expect(screen.getByText('Failed to initialize network')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });

    it('gives the activation select an accessible name', () => {
        render(<NetworkConfigPanel />);
        expect(screen.getByRole('combobox', { name: 'Activation' })).toBeInTheDocument();
    });
});
