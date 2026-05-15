import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HyperparamPanel } from './HyperparamPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('HyperparamPanel accessibility', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });
    });

    it('gives each hyperparameter select an accessible name', () => {
        render(<HyperparamPanel />);

        expect(screen.getByRole('combobox', { name: 'Learning rate' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Loss' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Optimizer' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Batch size' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Regularization' })).toBeInTheDocument();
    });

    it('does not expose multiclass-only loss or output activation controls yet', () => {
        render(<HyperparamPanel />);

        const loss = screen.getByRole('combobox', { name: 'Loss' });
        expect(loss).not.toHaveTextContent('Categorical Cross-Entropy');
        expect(within(loss).queryByRole('option', { name: 'Categorical Cross-Entropy' })).not.toBeInTheDocument();

        const outputActivation = screen.getByRole('combobox', { name: 'Output activation' });
        expect(outputActivation).not.toHaveTextContent('Softmax');
        expect(within(outputActivation).queryByRole('option', { name: 'Softmax' })).not.toBeInTheDocument();
    });

    it('keeps public hyperparameter changes on scalar output contracts', async () => {
        const user = userEvent.setup();

        render(<HyperparamPanel />);

        await user.selectOptions(screen.getByRole('combobox', { name: 'Loss' }), 'mse');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Output activation' }), 'linear');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Loss' }), 'crossEntropy');

        const { network, training } = usePlaygroundStore.getState();
        expect(network.outputSize).toBe(1);
        expect(network.outputActivation).toBe('sigmoid');
        expect(network.outputActivation).not.toBe('softmax');
        expect(training.lossType).toBe('crossEntropy');
        expect(training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it('explains cause and effect in hyperparameter tooltips', () => {
        render(<HyperparamPanel />);

        expect(screen.getByText('Cause: larger learning rates take bigger weight updates. Effect: training can move faster, but too large can overshoot and make loss jump.')).toBeInTheDocument();
        expect(screen.getByText('Cause: larger batches average more samples per update. Effect: the path is steadier, but each visible update reacts less often.')).toBeInTheDocument();
    });

    it('shows optimizer-specific controls without losing hidden optimizer values', async () => {
        const user = userEvent.setup();

        render(<HyperparamPanel />);

        expect(screen.queryByRole('combobox', { name: 'Momentum' })).not.toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Adam beta 1' })).not.toBeInTheDocument();

        await user.selectOptions(screen.getByRole('combobox', { name: 'Optimizer' }), 'sgdMomentum');
        expect(screen.getByRole('combobox', { name: 'Momentum' })).toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Adam beta 1' })).not.toBeInTheDocument();

        await user.selectOptions(screen.getByRole('combobox', { name: 'Optimizer' }), 'adam');
        expect(screen.queryByRole('combobox', { name: 'Momentum' })).not.toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Adam beta 1' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Adam beta 2' })).toBeInTheDocument();

        expect(usePlaygroundStore.getState().training.momentum).toBe(DEFAULT_TRAINING.momentum);
    });

    it('summarizes the learning-rate schedule in plain language', async () => {
        const user = userEvent.setup();

        render(<HyperparamPanel />);

        expect(screen.getByText('Uses 0.03 every update.')).toBeInTheDocument();

        await user.selectOptions(screen.getByRole('combobox', { name: 'LR schedule' }), 'step');

        expect(screen.getByText('Starts at 0.03; multiplies by 0.5 every 100 updates.')).toBeInTheDocument();
    });
});
