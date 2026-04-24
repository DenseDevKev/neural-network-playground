import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
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
});
