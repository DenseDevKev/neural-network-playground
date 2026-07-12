import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import { setHiddenLayers } from '../../store/recipeEdits.ts';
import {
    usePlaygroundStore,
    type PlaygroundStore,
} from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';
import { NetworkConfigPanel } from './NetworkConfigPanel.tsx';

const originalEditRecipe: PlaygroundStore['editRecipe'] = usePlaygroundStore.getState().editRecipe;

function resetTrainingTransaction() {
    useTrainingStore.setState({
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
        presetConfigLoading: false,
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
        configSyncNonce: 0,
    });
}

describe('NetworkConfigPanel canonical V2 controls', () => {
    beforeEach(async () => {
        usePlaygroundStore.setState({ editRecipe: originalEditRecipe });
        const restored = await usePlaygroundStore.getState()
            .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);
        resetTrainingTransaction();
    });

    afterEach(() => {
        act(() => usePlaygroundStore.setState({ editRecipe: originalEditRecipe }));
        vi.restoreAllMocks();
    });

    it('reads canonical model values and presents task output as derived labels', () => {
        render(<NetworkConfigPanel />);

        expect(screen.getByText('2 inputs')).toBeInTheDocument();
        expect(screen.getByText('1 output')).toBeInTheDocument();
        expect(screen.getByText('sigmoid')).toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: /output activation/i }))
            .not.toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Hidden activation' }))
            .toHaveValue('tanh');
        expect(screen.getByRole('combobox', { name: 'Weight initialization' }))
            .toHaveValue('xavier');
    });

    it('awaits a complete add-layer edit and exposes loading state', async () => {
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);

        await user.click(screen.getByRole('button', { name: 'Add hidden layer' }));

        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model.hiddenLayers)
                .toEqual([4, 4, 4]);
        });
        expect(screen.getByRole('status')).toHaveTextContent('Initializing network...');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
    });

    it('commits exact widths without clamping invalid numeric input', async () => {
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);

        const input = screen.getByRole('spinbutton', { name: 'Neuron count for layer 1' });
        await user.click(screen.getByRole('button', { name: 'Increase neurons in layer 1' }));
        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model.hiddenLayers[0])
                .toBe(5);
        });

        await user.clear(input);
        await user.type(input, '12');
        await user.tab();
        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model.hiddenLayers[0])
                .toBe(12);
        });
        const beforeInvalid = currentPreparedForTest();

        await user.click(input);
        await user.clear(input);
        await user.type(input, '99');
        await user.tab();

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'recipe.model.hiddenLayers[0]',
        );
        expect(currentPreparedForTest()).toBe(beforeInvalid);
        expect(input).toHaveValue(99);
    });

    it('uses exact hidden activation and initialization variants', async () => {
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);
        const beforeFingerprint = currentPreparedForTest()?.identities.recipeFingerprint;

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Hidden activation' }),
            'relu',
        );
        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model.hiddenActivation)
                .toBe('relu');
        });
        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Weight initialization' }),
            'he',
        );

        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model)
                .toMatchObject({ hiddenActivation: 'relu', initialization: 'he' });
        });
        expect(currentPreparedForTest()?.identities.recipeFingerprint)
            .not.toBe(beforeFingerprint);
        expect(within(screen.getByRole('combobox', { name: 'Hidden activation' }))
            .queryByRole('option', { name: 'Softmax' })).not.toBeInTheDocument();
    });

    it('does not rebuild the runtime when a numeric field is blurred unchanged', async () => {
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);
        const before = currentPreparedForTest();

        await user.click(screen.getByRole('spinbutton', { name: 'Neuron count for layer 1' }));
        await user.tab();

        expect(currentPreparedForTest()).toBe(before);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
    });

    it('preserves rapid orthogonal width and activation edits', async () => {
        render(<NetworkConfigPanel />);
        const width = screen.getByRole('spinbutton', { name: 'Neuron count for layer 1' });
        const activation = screen.getByRole('combobox', { name: 'Hidden activation' });

        fireEvent.change(width, { target: { value: '7' } });
        fireEvent.blur(width);
        fireEvent.change(activation, { target: { value: 'relu' } });

        await waitFor(() => {
            expect(currentPreparedForTest()?.document.recipe.model)
                .toMatchObject({ hiddenLayers: [7, 4], hiddenActivation: 'relu' });
        });
    });

    it('keeps independently mergeable controls available while a candidate prepares', () => {
        usePlaygroundStore.setState((state) => ({
            preparation: {
                status: 'preparing',
                requestId: state.preparation.requestId + 1,
                issues: [],
            },
        }));
        render(<NetworkConfigPanel />);

        expect(screen.getByRole('spinbutton', { name: 'Neuron count for layer 1' }))
            .not.toBeDisabled();
        expect(screen.getByRole('combobox', { name: 'Hidden activation' }))
            .not.toBeDisabled();
        expect(screen.getByRole('combobox', { name: 'Weight initialization' }))
            .not.toBeDisabled();
    });

    it('disables actions at the six-layer and width boundaries', async () => {
        const maxLayers = await originalEditRecipe(
            (recipe) => setHiddenLayers(recipe, [16, 16, 16, 16, 16, 16]),
        );
        expect(maxLayers.ok).toBe(true);
        render(<NetworkConfigPanel />);

        expect(screen.getByRole('button', { name: 'Add hidden layer' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Increase neurons in layer 1' }))
            .toBeDisabled();
        expect(screen.getAllByRole('spinbutton')).toHaveLength(6);
    });

    it('retains exact state and reports preparation failures', async () => {
        const before = currentPreparedForTest();
        usePlaygroundStore.setState({
            editRecipe: vi.fn(async () => ({
                ok: false as const,
                issues: [{
                    code: 'resource-limit' as const,
                    path: 'recipe.model.hiddenLayers',
                    message: 'architecture exceeds 2000 trainable parameters',
                }],
            })),
        });
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);

        await user.click(screen.getByRole('button', { name: 'Add hidden layer' }));

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'recipe.model.hiddenLayers: architecture exceeds 2000 trainable parameters',
        );
        expect(currentPreparedForTest()).toBe(before);
    });

    it('keeps network errors retryable', async () => {
        useTrainingStore.setState({
            configError: 'Failed to initialize network',
            configErrorSource: 'network',
        });
        const user = userEvent.setup();
        render(<NetworkConfigPanel />);

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });

    it('preserves cause/effect explanations', () => {
        render(<NetworkConfigPanel />);

        expect(screen.getByText(/Cause: adding a hidden layer adds another learned transformation/))
            .toBeInTheDocument();
        expect(screen.getByText(/Cause: activation functions decide when neurons pass signal forward/))
            .toBeInTheDocument();
    });
});
