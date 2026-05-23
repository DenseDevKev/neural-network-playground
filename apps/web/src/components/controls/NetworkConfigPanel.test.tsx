import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
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

    it('uses typed steppers instead of sliders for neuron counts', async () => {
        const user = userEvent.setup();

        render(<NetworkConfigPanel />);

        expect(screen.queryByRole('slider', { name: 'Neurons in layer 1' })).not.toBeInTheDocument();

        const input = screen.getByRole('spinbutton', { name: 'Neuron count for layer 1' });
        expect(input).toHaveValue(4);

        await user.click(screen.getByRole('button', { name: 'Increase neurons in layer 1' }));
        expect(usePlaygroundStore.getState().network.hiddenLayers[0]).toBe(5);

        await user.clear(input);
        await user.type(input, '12');
        expect(usePlaygroundStore.getState().network.hiddenLayers[0]).toBe(12);

        await user.clear(input);
        await user.type(input, '99');
        expect(usePlaygroundStore.getState().network.hiddenLayers[0]).toBe(16);
    });

    it('does not expose softmax as a hidden-layer activation', async () => {
        const user = userEvent.setup();

        render(<NetworkConfigPanel />);

        const activation = screen.getByRole('combobox', { name: 'Activation' });
        expect(within(activation).queryByRole('option', { name: 'Softmax' })).not.toBeInTheDocument();

        await user.selectOptions(activation, 'tanh');
        await user.click(screen.getByRole('button', { name: 'Add hidden layer' }));

        expect(usePlaygroundStore.getState().network.activation).toBe('tanh');
    });

    it('explains cause and effect in network tooltips', () => {
        render(<NetworkConfigPanel />);

        expect(screen.getByText('Cause: adding a hidden layer adds another learned transformation. Effect: the boundary can bend more, but training may take longer.')).toBeInTheDocument();
        expect(screen.getByText('Cause: activation functions decide when neurons pass signal forward. Effect: tanh/sigmoid smooth the boundary, while ReLU-family choices make sharper bends.')).toBeInTheDocument();
    });
});
