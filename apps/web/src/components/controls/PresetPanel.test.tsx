import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PresetPanel } from './PresetPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('PresetPanel', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
        });
    });

    it('renders presets in a card grid', () => {
        render(<PresetPanel onReset={vi.fn()} />);

        expect(screen.getByRole('list', { name: 'Available presets' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toBeInTheDocument();
    });

    it('applies a preset, resets training, and highlights the selected card', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();

        render(<PresetPanel onReset={onReset} />);

        const button = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        await user.click(button);

        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
        expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual([4, 4]);
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(button).toHaveClass('preset-card--selected');
        expect(button).toHaveAttribute('aria-pressed', 'true');
    });
});
