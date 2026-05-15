import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GuidedLessonPanel } from './GuidedLessonPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    getLessonDefinition,
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
    function mockCompactLessonDrawer(matches: boolean) {
        Object.defineProperty(window, 'matchMedia', {
            writable: true,
            configurable: true,
            value: vi.fn().mockImplementation((query: string) => ({
                matches,
                media: query,
                onchange: null,
                addEventListener: vi.fn(),
                removeEventListener: vi.fn(),
                addListener: vi.fn(),
                removeListener: vi.fn(),
                dispatchEvent: vi.fn(),
            })),
        });
    }

    beforeEach(() => {
        mockCompactLessonDrawer(false);
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
            activeLessonId: null,
            activeLessonStepIndex: null,
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
        expect(useLayoutStore.getState().activeLessonId).toBe('lesson-xor-hidden-layers');
        expect(useLayoutStore.getState().activeLessonStepIndex).toBe(0);
        expect(screen.getByText('Step 1 of 4')).toBeInTheDocument();
        expect(screen.getByText('XOR Needs Hidden Layers')).toBeInTheDocument();
        expect(useLayoutStore.getState().activeTabLeft).toBe('data');
        expect(useLayoutStore.getState().phase).toBe('build');

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('network');
        expect(useLayoutStore.getState().activeLessonStepIndex).toBe(1);
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
        expect(useLayoutStore.getState().activeLessonId).toBeNull();
        expect(useLayoutStore.getState().activeLessonStepIndex).toBeNull();
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeInTheDocument();
    });

    it('lists registry lessons and starts the selected lesson preset', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const regressionLesson = getLessonDefinition('lesson-regression-plane-baseline')!;

        render(<GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />);

        const selector = screen.getByRole('combobox', { name: 'Guided lesson' });
        for (const lesson of LESSON_DEFINITIONS) {
            expect(screen.getByRole('option', { name: lesson.title })).toBeInTheDocument();
        }

        await user.selectOptions(selector, regressionLesson.id);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        expect(usePlaygroundStore.getState().data.dataset).toBe('reg-plane');
        expect(usePlaygroundStore.getState().training.lossType).toBe('mse');
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(onHighlightChange).toHaveBeenLastCalledWith(regressionLesson.steps[0].target);
        expect(screen.getByText(`Step 1 of ${regressionLesson.steps.length}`)).toBeInTheDocument();
        expect(screen.getByText(regressionLesson.steps[0].title)).toBeInTheDocument();
    });

    it('starts each registry lesson preset from a fresh selector render', () => {
        for (const lesson of LESSON_DEFINITIONS) {
            usePlaygroundStore.setState({
                data: { ...DEFAULT_DATA },
                network: { ...DEFAULT_NETWORK, inputSize: 2, seed: DEFAULT_DATA.seed },
                features: { ...DEFAULT_FEATURES },
                training: { ...DEFAULT_TRAINING },
                ui: { showTestData: false, discretizeOutput: false },
            });

            const onReset = vi.fn();
            const onHighlightChange = vi.fn();
            const { unmount } = render(
                <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
            );

            fireEvent.change(screen.getByRole('combobox', { name: 'Guided lesson' }), {
                target: { value: lesson.id },
            });
            fireEvent.click(screen.getByRole('button', { name: 'Start guided lesson' }));

            const preset = getLessonPreset(lesson);
            expect(usePlaygroundStore.getState().data.dataset).toBe(preset.config.data?.dataset);
            expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual(preset.config.network?.hiddenLayers);
            expect(onReset).toHaveBeenCalledTimes(1);
            expect(onHighlightChange).toHaveBeenLastCalledWith(lesson.steps[0].target);
            expect(useLayoutStore.getState().activeLessonId).toBe(lesson.id);
            expect(useLayoutStore.getState().activeLessonStepIndex).toBe(0);
            expect(screen.getByText(`Step 1 of ${lesson.steps.length}`)).toBeInTheDocument();
            expect(screen.getByText(lesson.steps[0].title)).toBeInTheDocument();

            unmount();
        }
    });

    it('collapses and expands the docked lesson drawer without losing selected lesson state', async () => {
        const user = userEvent.setup();

        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Guided lesson' }),
            'lesson-regression-plane-baseline',
        );

        await user.click(screen.getByRole('button', { name: 'Collapse guided lesson drawer' }));

        expect(screen.getByRole('button', { name: 'Expand guided lesson drawer' })).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('button', { name: 'Start guided lesson' })).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Expand guided lesson drawer' }));

        expect(screen.getByRole('button', { name: 'Collapse guided lesson drawer' })).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByRole('combobox', { name: 'Guided lesson' })).toHaveValue('lesson-regression-plane-baseline');
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeInTheDocument();
    });

    it('defaults the lesson drawer to collapsed on compact screens', () => {
        mockCompactLessonDrawer(true);

        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Expand guided lesson drawer' })).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('button', { name: 'Start guided lesson' })).not.toBeInTheDocument();
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
