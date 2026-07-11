import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    PREPARED_PRESETS,
    type PreparedExperimentDocumentV2,
    type SchemaResult,
} from '@nn-playground/shared';
import {
    getLessonDefinition,
    getLessonRecipe,
    LESSON_DEFINITIONS,
} from '../../lessons/lessonRegistry.ts';
import {
    usePlaygroundStore,
    type PlaygroundStore,
} from '../../store/usePlaygroundStore.ts';
import { setNoise } from '../../store/recipeEdits.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { GuidedLessonPanel } from './GuidedLessonPanel.tsx';

type ApplyResult = SchemaResult<PreparedExperimentDocumentV2>;

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

function resetLayout() {
    useLayoutStore.setState({
        view: 'build',
        activeRecipeSection: 'data',
        activeEvidenceView: 'boundary',
        layout: 'dock',
        phase: 'build',
        activeTabLeft: 'data',
        activeTabRight: 'boundary',
        activeLessonId: null,
        activeLessonStepIndex: null,
    });
}

function resetTrainingTransactionState() {
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

function deferApplyCompletion(apply: PlaygroundStore['applyRecipe']) {
    let release!: () => void;
    let pending: Promise<ApplyResult> | null = null;
    const gate = new Promise<void>((resolve) => {
        release = resolve;
    });
    const applyRecipe = vi.fn((entry: Parameters<PlaygroundStore['applyRecipe']>[0]) => {
        pending = (async () => {
            const result = await apply(entry);
            await gate;
            return result;
        })();
        return pending;
    });

    return {
        applyRecipe,
        release,
        wait: async () => {
            if (!pending) throw new Error('Deferred apply was not started');
            return pending;
        },
    };
}

describe('GuidedLessonPanel', () => {
    let originalApplyRecipe: PlaygroundStore['applyRecipe'];

    beforeEach(async () => {
        mockCompactLessonDrawer(false);
        originalApplyRecipe = usePlaygroundStore.getState().applyRecipe;
        const restored = await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);
        resetLayout();
        resetTrainingTransactionState();
    });

    afterEach(() => {
        act(() => {
            usePlaygroundStore.setState({ applyRecipe: originalApplyRecipe });
        });
        vi.restoreAllMocks();
    });

    it('awaits the exact XOR recipe before reset, step activation, layout, and highlight', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const target = getLessonRecipe(getLessonDefinition()!);

        render(<GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
                .toBe(target.prepared.identities.canonicalRecipeKey);
        });
        expect(usePlaygroundStore.getState().prepared?.identities.recipeFingerprint)
            .toBe(target.prepared.identities.recipeFingerprint);
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(onHighlightChange).toHaveBeenLastCalledWith('data');
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: 'lesson-xor-hidden-layers',
            activeLessonStepIndex: 0,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
            view: 'build',
            phase: 'build',
        });
        expect(screen.getByText('Step 1 of 4')).toBeInTheDocument();
    });

    it('runs through lesson navigation and clears the highlight on finish', async () => {
        const user = userEvent.setup();
        const onHighlightChange = vi.fn();
        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={onHighlightChange} />);

        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await screen.findByText('Step 1 of 4');

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('network');
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonStepIndex: 1,
            activeRecipeSection: 'network',
            activeTabLeft: 'network',
            view: 'build',
            phase: 'build',
        });

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('hyperparams');
        expect(screen.getByText('Step 3 of 4')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Next lesson step' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith('transport');
        expect(useLayoutStore.getState()).toMatchObject({ view: 'run', phase: 'run' });

        await user.click(screen.getByRole('button', { name: 'Finish guided lesson' }));
        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeInTheDocument();
    });

    it('starts selected regression and multiclass lessons from their compiled contracts', async () => {
        const user = userEvent.setup();
        const { unmount } = render(
            <GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />,
        );

        const regression = getLessonDefinition('lesson-regression-plane-baseline')!;
        await user.selectOptions(screen.getByRole('combobox', { name: 'Guided lesson' }), regression.id);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.compiled.task.kind).toBe('regression');
        });
        expect(usePlaygroundStore.getState().prepared?.compiled.task).toMatchObject({
            dataset: 'reg-plane',
            outputSize: 1,
            outputActivation: 'linear',
        });

        unmount();
        resetLayout();
        resetTrainingTransactionState();
        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);
        const multiclass = getLessonDefinition('lesson-three-class-softmax')!;
        await user.selectOptions(screen.getByRole('combobox', { name: 'Guided lesson' }), multiclass.id);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.compiled.task.kind)
                .toBe('multiclass-classification');
        });
        expect(usePlaygroundStore.getState().prepared?.compiled.task).toMatchObject({
            dataset: 'three-class-clusters',
            outputSize: 3,
            outputActivation: 'softmax',
        });
        expect(screen.getByText('Three-Class Softmax Lab')).toBeInTheDocument();
    });

    it('ends at the exact lesson destination for all 70 catalog-source transitions', async () => {
        const user = userEvent.setup();
        const viewResult = await usePlaygroundStore.getState().editView(() => ({
            showTestData: true,
            discretizeOutput: true,
        }));
        expect(viewResult.ok).toBe(true);
        const canonicalApplyRecipe = vi.fn((entry: Parameters<PlaygroundStore['applyRecipe']>[0]) => (
            originalApplyRecipe(entry)
        ));
        usePlaygroundStore.setState({ applyRecipe: canonicalApplyRecipe });

        for (const source of PREPARED_PRESETS) {
            for (const lesson of LESSON_DEFINITIONS) {
                const sourceResult = await usePlaygroundStore.getState().applyRecipe(source);
                expect(sourceResult.ok, `source ${source.id}`).toBe(true);
                resetLayout();
                resetTrainingTransactionState();

                const target = getLessonRecipe(lesson);
                const onReset = vi.fn();
                const onHighlightChange = vi.fn();
                const { unmount } = render(
                    <GuidedLessonPanel
                        onReset={onReset}
                        onHighlightChange={onHighlightChange}
                    />,
                );
                await user.selectOptions(
                    screen.getByRole('combobox', { name: 'Guided lesson' }),
                    lesson.id,
                );
                await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

                await waitFor(() => {
                    expect(
                        usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey,
                        `${source.id}@${source.revision} -> ${lesson.id} (${target.id}@${target.revision})`,
                    ).toBe(target.prepared.identities.canonicalRecipeKey);
                });
                expect(usePlaygroundStore.getState().prepared?.identities.recipeFingerprint)
                    .toBe(target.prepared.identities.recipeFingerprint);
                expect(usePlaygroundStore.getState().prepared?.document.view).toEqual({
                    showTestData: true,
                    discretizeOutput: true,
                });
                expect(onReset).toHaveBeenCalledTimes(1);
                expect(onHighlightChange).toHaveBeenLastCalledWith(lesson.steps[0].target);
                expect(useLayoutStore.getState()).toMatchObject({
                    activeLessonId: lesson.id,
                    activeLessonStepIndex: 0,
                });
                expect(canonicalApplyRecipe).toHaveBeenLastCalledWith(target);

                unmount();
                useTrainingStore.getState().finishConfigChange();
            }
        }
    }, 30_000);

    it('deduplicates double start and changes no lesson UI before preparation resolves', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const lesson = getLessonDefinition()!;
        const target = getLessonRecipe(lesson);
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });
        useLayoutStore.setState({
            view: 'run',
            phase: 'run',
            activeRecipeSection: 'features',
            activeTabLeft: 'features',
        });

        const { container } = render(
            <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
        );
        const startButton = screen.getByRole('button', { name: 'Start guided lesson' });
        await user.click(startButton);

        expect(deferred.applyRecipe).toHaveBeenCalledTimes(1);
        expect(deferred.applyRecipe).toHaveBeenCalledWith(target);
        expect(startButton).toBeDisabled();
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'preset',
            presetConfigLoading: true,
        });
        expect(onReset).not.toHaveBeenCalled();
        expect(onHighlightChange).not.toHaveBeenCalled();
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
            activeRecipeSection: 'features',
            view: 'run',
        });
        expect(container.querySelector('.guided-lesson')).not.toHaveClass('guided-lesson--active');

        await user.click(startButton);
        expect(deferred.applyRecipe).toHaveBeenCalledTimes(1);

        await act(async () => {
            deferred.release();
            await deferred.wait();
        });

        expect(onReset).toHaveBeenCalledTimes(1);
        expect(onHighlightChange).toHaveBeenLastCalledWith('data');
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: lesson.id,
            activeLessonStepIndex: 0,
            activeRecipeSection: 'data',
            view: 'build',
        });
        expect(container.querySelector('.guided-lesson')).toHaveClass('guided-lesson--active');
    });

    it('does not activate a stale lesson start after a newer recipe edit wins', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const target = getLessonRecipe(getLessonDefinition()!);
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        const { container } = render(
            <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
                .toBe(target.prepared.identities.canonicalRecipeKey);
        });
        let newerEdit!: Awaited<ReturnType<PlaygroundStore['editRecipe']>>;
        await act(async () => {
            useTrainingStore.getState().beginConfigChange('network');
            newerEdit = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setNoise(recipe, recipe.data.noise + 1),
            );
        });
        expect(newerEdit.ok).toBe(true);

        await act(async () => {
            deferred.release();
            await deferred.wait();
        });

        expect(onReset).not.toHaveBeenCalled();
        expect(onHighlightChange).not.toHaveBeenCalled();
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'network',
            networkConfigLoading: true,
            configError: null,
        });
        expect(container.querySelector('.guided-lesson')).not.toHaveClass('guided-lesson--active');
    });

    it('does not activate an older same-target lesson result', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const target = getLessonRecipe(getLessonDefinition()!);
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        const { container } = render(
            <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
                .toBe(target.prepared.identities.canonicalRecipeKey);
        });
        await act(async () => {
            const newerResult = await originalApplyRecipe(target);
            expect(newerResult.ok).toBe(true);
        });

        await act(async () => {
            deferred.release();
            await deferred.wait();
        });

        expect(onReset).not.toHaveBeenCalled();
        expect(onHighlightChange).not.toHaveBeenCalled();
        expect(container.querySelector('.guided-lesson')).not.toHaveClass('guided-lesson--active');
    });

    it('does not report an older failure over a newer config transaction', async () => {
        const user = userEvent.setup();
        let resolveApply!: (result: ApplyResult) => void;
        const pending = new Promise<ApplyResult>((resolve) => {
            resolveApply = resolve;
        });
        usePlaygroundStore.setState({ applyRecipe: vi.fn(() => pending) });

        render(<GuidedLessonPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await act(async () => {
            useTrainingStore.getState().beginConfigChange('network');
            const newerEdit = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setNoise(recipe, recipe.data.noise + 1),
            );
            expect(newerEdit.ok).toBe(true);
        });

        await act(async () => {
            resolveApply({
                ok: false,
                issues: [{ code: 'invalid-field', path: 'recipe', message: 'Obsolete failure' }],
            });
            await pending;
        });

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'network',
            networkConfigLoading: true,
            presetConfigLoading: false,
            configError: null,
        });
    });

    it('does not activate or highlight after a pending lesson start unmounts', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const target = getLessonRecipe(getLessonDefinition()!);
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        const { unmount } = render(
            <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
                .toBe(target.prepared.identities.canonicalRecipeKey);
        });
        unmount();

        await act(async () => {
            deferred.release();
            await deferred.wait();
        });

        expect(onReset).not.toHaveBeenCalled();
        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });

    it('records a still-current preparation failure after the lesson panel unmounts', async () => {
        const user = userEvent.setup();
        let resolveApply!: (result: ApplyResult) => void;
        const pending = new Promise<ApplyResult>((resolve) => {
            resolveApply = resolve;
        });
        usePlaygroundStore.setState({ applyRecipe: vi.fn(() => pending) });

        const { unmount } = render(<GuidedLessonPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        unmount();

        await act(async () => {
            resolveApply({
                ok: false,
                issues: [{ code: 'invalid-field', path: 'recipe', message: 'Current failure' }],
            });
            await pending;
        });

        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: null,
            presetConfigLoading: false,
            configErrorSource: 'preset',
            configError: 'recipe: Current failure',
        });
    });

    it('keeps the prior experiment and inactive UI with a persistent accessible error on failure', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onHighlightChange = vi.fn();
        const priorPrepared = usePlaygroundStore.getState().prepared;
        usePlaygroundStore.setState({
            applyRecipe: vi.fn(async () => ({
                ok: false as const,
                issues: [{ code: 'invalid-field' as const, path: 'recipe', message: 'Deliberate failure' }],
            })),
        });
        useLayoutStore.setState({
            view: 'run',
            phase: 'run',
            activeRecipeSection: 'features',
            activeTabLeft: 'features',
        });

        const { container, rerender } = render(
            <GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />,
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('Deliberate failure');
        expect(usePlaygroundStore.getState().prepared).toBe(priorPrepared);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: null,
            presetConfigLoading: false,
            configErrorSource: 'preset',
        });
        expect(onReset).not.toHaveBeenCalled();
        expect(onHighlightChange).not.toHaveBeenCalled();
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
            activeRecipeSection: 'features',
            view: 'run',
        });
        expect(container.querySelector('.guided-lesson')).not.toHaveClass('guided-lesson--active');

        rerender(<GuidedLessonPanel onReset={onReset} onHighlightChange={onHighlightChange} />);
        expect(screen.getByRole('alert')).toHaveTextContent('Deliberate failure');
        expect(screen.getByRole('button', { name: 'Start guided lesson' })).toBeEnabled();
    });

    it('preserves explicit button semantics and keyboard lesson navigation', async () => {
        const user = userEvent.setup();
        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);

        const startButton = screen.getByRole('button', { name: 'Start guided lesson' });
        expect(startButton).toHaveAttribute('type', 'button');
        await user.click(startButton);

        const nextButton = await screen.findByRole('button', { name: 'Next lesson step' });
        expect(screen.getByRole('button', { name: 'Back' })).toHaveAttribute('type', 'button');
        nextButton.focus();
        await user.keyboard('{Enter}');
        expect(screen.getByText('Give the model capacity')).toBeInTheDocument();

        screen.getByRole('button', { name: 'Next lesson step' }).focus();
        await user.keyboard(' ');
        expect(screen.getByText('Use steady updates')).toBeInTheDocument();
    });

    it('collapses and expands without losing the selected revision-pinned lesson', async () => {
        const user = userEvent.setup();
        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Guided lesson' }),
            'lesson-regression-plane-baseline',
        );
        await user.click(screen.getByRole('button', { name: 'Collapse guided lesson drawer' }));
        expect(screen.getByRole('button', { name: 'Expand guided lesson drawer' }))
            .toHaveAttribute('aria-expanded', 'false');

        await user.click(screen.getByRole('button', { name: 'Expand guided lesson drawer' }));
        expect(screen.getByRole('combobox', { name: 'Guided lesson' }))
            .toHaveValue('lesson-regression-plane-baseline');
    });

    it('defaults to collapsed on compact screens', () => {
        mockCompactLessonDrawer(true);
        render(<GuidedLessonPanel onReset={vi.fn()} onHighlightChange={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Expand guided lesson drawer' }))
            .toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('button', { name: 'Start guided lesson' })).not.toBeInTheDocument();
    });

    it('clears the active highlight when an active lesson unmounts', async () => {
        const user = userEvent.setup();
        const onHighlightChange = vi.fn();
        const { unmount } = render(
            <GuidedLessonPanel onReset={vi.fn()} onHighlightChange={onHighlightChange} />,
        );

        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));
        await waitFor(() => expect(onHighlightChange).toHaveBeenLastCalledWith('data'));
        unmount();

        expect(onHighlightChange).toHaveBeenLastCalledWith(null);
        expect(useLayoutStore.getState()).toMatchObject({
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });
});
