import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
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

    it('exposes advanced hyperparameter controls when applicable', () => {
        usePlaygroundStore.setState((s) => ({
            ...s,
            training: {
                ...s.training,
                optimizer: 'adam',
                lossType: 'huber',
                gradientClip: 0.5,
                lrSchedule: { type: 'cosine', totalSteps: 500, minLr: 0.001 },
            },
        }));

        render(<HyperparamPanel />);

        expect(screen.getByRole('combobox', { name: 'Momentum' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Gradient clipping' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Adam beta 1' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Adam beta 2' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Huber delta' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'LR schedule' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Weight initialization' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Output activation' })).toBeInTheDocument();
        expect(screen.getByRole('spinbutton', { name: 'Cosine total steps' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Cosine minimum learning rate' })).toBeInTheDocument();
    });

    it('writes advanced control changes into the store', async () => {
        const user = userEvent.setup();

        render(<HyperparamPanel />);

        await user.selectOptions(screen.getByRole('combobox', { name: 'Optimizer' }), 'adam');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Adam beta 1' }), '0.8');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Adam beta 2' }), '0.98');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Loss' }), 'huber');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Huber delta' }), '0.5');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Gradient clipping' }), '1');
        await user.selectOptions(screen.getByRole('combobox', { name: 'LR schedule' }), 'step');
        await user.clear(screen.getByRole('spinbutton', { name: 'Step schedule interval' }));
        await user.type(screen.getByRole('spinbutton', { name: 'Step schedule interval' }), '25');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Step schedule gamma' }), '0.5');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Weight initialization' }), 'he');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Output activation' }), 'linear');

        expect(usePlaygroundStore.getState().training).toMatchObject({
            optimizer: 'adam',
            lossType: 'huber',
            adamBeta1: 0.8,
            adamBeta2: 0.98,
            huberDelta: 0.5,
            gradientClip: 1,
            lrSchedule: { type: 'step', stepSize: 25, gamma: 0.5 },
        });
        expect(usePlaygroundStore.getState().network.weightInit).toBe('he');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('linear');
    });

    it('explains cause and effect in hyperparameter tooltips', () => {
        render(<HyperparamPanel />);

        expect(screen.getByText('Cause: larger learning rates take bigger weight updates. Effect: training can move faster, but too large can overshoot and make loss jump.')).toBeInTheDocument();
        expect(screen.getByText('Cause: larger batches average more samples per update. Effect: the path is steadier, but each visible update reacts less often.')).toBeInTheDocument();
    });
});
