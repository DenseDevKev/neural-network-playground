import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GuidedLessonPanel } from './GuidedLessonPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('GuidedLessonPanel', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });
        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('runs through the XOR hidden-layer lesson and clears the highlight on finish', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();

        render(<GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />);

        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
        expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual([4, 4]);
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(onHighlightChange).toHaveBeenLastCalledWith('data');
        expect(screen.getByText('Step 1 of 4')).toBeInTheDocument();
        expect(screen.getByText('XOR Needs Hidden Layers')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('network');
        expect(screen.getByText('Step 2 of 4')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('hyperparams');
        expect(screen.getByText('Step 3 of 4')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('transport');
        expect(screen.getByText('Step 4 of 4')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Finish guided lesson' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeInTheDocument();
    });
});
