import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GuidedLessonPanel } from './GuidedLessonPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    getLessonPreset,
    LESSON_DEFINITIONS,
} from '../../lessons/lessonRegistry.ts';
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
        expect(useLayoutStore.getState().activeTabLeft).toBe('data');
        expect(useLayoutStore.getState().phase).toBe('build');

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('network');
        expect(screen.getByText('Step 2 of 4')).toBeInTheDocument();
        expect(useLayoutStore.getState().activeTabLeft).toBe('network');
        expect(useLayoutStore.getState().phase).toBe('build');

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('hyperparams');
        expect(screen.getByText('Step 3 of 4')).toBeInTheDocument();
        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        expect(useLayoutStore.getState().phase).toBe('build');

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('transport');
        expect(screen.getByText('Step 4 of 4')).toBeInTheDocument();
        expect(useLayoutStore.getState().phase).toBe('run');

        await user.click(screen.getByRole('button', { name: 'Finish guided lesson' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeInTheDocument();
    });

    it('shows the conservative P1 lesson choices and starts each selected preset', async () => {
        const user = userEvent.setup();

        render(<GuidedLessonPanel onReset={vi.fn()} />);

        for (const lesson of LESSON_DEFINITIONS) {
            expect(screen.getByRole('button', { name: `Select lesson: ${lesson.title}` })).toBeInTheDocument();
        }

        for (const lesson of LESSON_DEFINITIONS) {
            usePlaygroundStore.setState({
                data: { ...DEFAULT_DATA },
                network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
                features: { ...DEFAULT_FEATURES },
                training: { ...DEFAULT_TRAINING },
                ui: { showTestData: false, discretizeOutput: false },
            });

            await user.click(screen.getByRole('button', { name: `Select lesson: ${lesson.title}` }));
            await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

            const preset = getLessonPreset(lesson);
            expect(usePlaygroundStore.getState().data.dataset).toBe(preset.config.data?.dataset);
            expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual(preset.config.network?.hiddenLayers);
            expect(screen.getByText(`Step 1 of ${lesson.steps.length}`)).toBeInTheDocument();

            for (let stepIndex = 1; stepIndex < lesson.steps.length; stepIndex += 1) {
                await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
            }
            expect(useLayoutStore.getState().phase).toBe('run');

            await user.click(screen.getByRole('button', { name: 'Finish guided lesson' }));
        }
    });

    it('clears the active highlight when an active lesson unmounts', async () => {
        const user = userEvent.setup();
        const onHighlightChange = vi.fn();

        const { unmount } = render(
            <GuidedLessonPanel onReset={vi.fn()} onHighlightChange={onHighlightChange} />,
        );

        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('data');

        unmount();

        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
    });
});
