import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PresetPanel } from './PresetPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    PRESETS,
} from '@nn-playground/shared';

describe('PresetPanel', () => {
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
            presetConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
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
        expect(screen.getByRole('status')).toHaveTextContent('Applying preset...');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
    });

    it('keeps the applied preset highlighted after the panel remounts', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();

        const { unmount } = render(<PresetPanel onReset={onReset} />);

        await user.click(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }));

        unmount();
        render(<PresetPanel onReset={onReset} />);

        const button = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        expect(button).toHaveClass('preset-card--selected');
        expect(button).toHaveAttribute('aria-pressed', 'true');
    });

    it('does not enter pending state when the selected preset already matches', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const xorPreset = PRESETS.find((preset) => preset.id === 'xor-hidden')!;
        usePlaygroundStore.getState().applyPreset(xorPreset);

        render(<PresetPanel onReset={onReset} />);

        await user.click(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }));

        expect(onReset).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('shows preset-specific config errors and allows retrying', async () => {
        const user = userEvent.setup();

        useTrainingStore.setState({
            configError: 'Failed to apply preset',
            configErrorSource: 'preset',
        });

        render(<PresetPanel onReset={vi.fn()} />);

        expect(screen.getByText('Failed to apply preset')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });
});
