import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    PREPARED_PRESETS,
    resolveRecipe,
    type PreparedExperimentDocumentV2,
    type SchemaResult,
} from '@nn-playground/shared';
import { setNoise } from '../../store/recipeEdits.ts';
import {
    usePlaygroundStore,
    type PlaygroundStore,
} from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { PresetPanel } from './PresetPanel';

type ApplyResult = SchemaResult<PreparedExperimentDocumentV2>;

function resetTrainingTransactionState() {
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

describe('PresetPanel', () => {
    let originalApplyRecipe: PlaygroundStore['applyRecipe'];

    beforeEach(async () => {
        originalApplyRecipe = usePlaygroundStore.getState().applyRecipe;
        const restored = await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);
        resetTrainingTransactionState();
    });

    afterEach(() => {
        act(() => {
            usePlaygroundStore.setState({ applyRecipe: originalApplyRecipe });
        });
        vi.restoreAllMocks();
    });

    it('renders the complete prepared catalog in an accessible card grid', () => {
        render(<PresetPanel onReset={vi.fn()} />);

        expect(screen.getByRole('list', { name: 'Available presets' })).toBeInTheDocument();
        expect(screen.getAllByRole('listitem')).toHaveLength(PREPARED_PRESETS.length);
        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toBeInTheDocument();
    });

    it('awaits an exact recipe, preserves view, then resets and highlights', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onApplied = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const viewResult = await usePlaygroundStore.getState().editView(() => ({
            showTestData: true,
            discretizeOutput: true,
        }));
        expect(viewResult.ok).toBe(true);

        render(<PresetPanel onReset={onReset} onApplied={onApplied} />);

        const button = screen.getByRole('button', { name: `Apply preset: ${target.title}` });
        await user.click(button);

        await waitFor(() => {
            expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
                .toBe(target.prepared.identities.canonicalRecipeKey);
        });
        expect(usePlaygroundStore.getState().prepared?.identities.recipeFingerprint)
            .toBe(target.prepared.identities.recipeFingerprint);
        expect(usePlaygroundStore.getState().prepared?.document.view).toEqual({
            showTestData: true,
            discretizeOutput: true,
        });
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(onApplied).toHaveBeenCalledTimes(1);
        expect(button).toHaveClass('preset-card--selected');
        expect(button).toHaveAttribute('aria-pressed', 'true');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
    });

    it('ends at the exact destination identity for all 49 catalog transitions', async () => {
        const user = userEvent.setup();
        const viewResult = await usePlaygroundStore.getState().editView(() => ({
            showTestData: true,
            discretizeOutput: false,
        }));
        expect(viewResult.ok).toBe(true);

        for (const source of PREPARED_PRESETS) {
            for (const target of PREPARED_PRESETS) {
                const sourceResult = await usePlaygroundStore.getState().applyRecipe(source);
                expect(sourceResult.ok, `source ${source.id}`).toBe(true);
                useTrainingStore.getState().finishConfigChange();

                const onReset = vi.fn();
                const { unmount } = render(<PresetPanel onReset={onReset} />);
                const targetButton = screen.getByRole('button', {
                    name: `Apply preset: ${target.title}`,
                });

                await user.click(targetButton);

                await waitFor(() => {
                    expect(
                        usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey,
                        `${source.id}@${source.revision} -> ${target.id}@${target.revision}`,
                    ).toBe(target.prepared.identities.canonicalRecipeKey);
                });
                expect(usePlaygroundStore.getState().prepared?.identities.recipeFingerprint)
                    .toBe(target.prepared.identities.recipeFingerprint);
                expect(usePlaygroundStore.getState().prepared?.document.view).toEqual({
                    showTestData: true,
                    discretizeOutput: false,
                });
                expect(targetButton).toHaveAttribute('aria-pressed', 'true');
                expect(onReset).toHaveBeenCalledTimes(source === target ? 0 : 1);

                unmount();
                useTrainingStore.getState().finishConfigChange();
            }
        }
    }, 30_000);

    it('keeps exact selection across remounts', async () => {
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const applied = await usePlaygroundStore.getState().applyRecipe(target);
        expect(applied.ok).toBe(true);
        const onReset = vi.fn();

        const { unmount } = render(<PresetPanel onReset={onReset} />);
        expect(screen.getByRole('button', { name: `Apply preset: ${target.title}` }))
            .toHaveAttribute('aria-pressed', 'true');

        unmount();
        render(<PresetPanel onReset={onReset} />);

        expect(screen.getByRole('button', { name: `Apply preset: ${target.title}` }))
            .toHaveAttribute('aria-pressed', 'true');
    });

    it('removes selection after a one-field canonical recipe drift', async () => {
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const applied = await usePlaygroundStore.getState().applyRecipe(target);
        expect(applied.ok).toBe(true);
        render(<PresetPanel onReset={vi.fn()} />);
        const button = screen.getByRole('button', { name: `Apply preset: ${target.title}` });
        expect(button).toHaveAttribute('aria-pressed', 'true');

        await act(async () => {
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setNoise(recipe, 0.1),
            );
            expect(edited.ok).toBe(true);
        });

        await waitFor(() => expect(button).toHaveAttribute('aria-pressed', 'false'));
    });

    it('does not enter the loading transaction when the exact recipe is already selected', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const applied = await usePlaygroundStore.getState().applyRecipe(target);
        expect(applied.ok).toBe(true);

        render(<PresetPanel onReset={onReset} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));

        expect(onReset).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('disables cards while pending and does not reset or highlight before resolution', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        let resolveApply!: (result: ApplyResult) => void;
        const pending = new Promise<ApplyResult>((resolve) => {
            resolveApply = resolve;
        });
        const applyRecipe = vi.fn(() => pending);
        usePlaygroundStore.setState({ applyRecipe });

        render(<PresetPanel onReset={onReset} />);
        const targetButton = screen.getByRole('button', { name: `Apply preset: ${target.title}` });
        await user.click(targetButton);

        expect(applyRecipe).toHaveBeenCalledTimes(1);
        expect(applyRecipe).toHaveBeenCalledWith(target);
        expect(targetButton).toBeDisabled();
        expect(targetButton).toHaveAttribute('aria-pressed', 'false');
        expect(onReset).not.toHaveBeenCalled();

        await user.click(targetButton);
        expect(applyRecipe).toHaveBeenCalledTimes(1);

        await act(async () => {
            resolveApply({
                ok: false,
                issues: [{ code: 'invalid-field', path: 'recipe', message: 'Deliberate failure' }],
            });
            await pending;
        });

        expect(onReset).not.toHaveBeenCalled();
    });

    it('uses the persistent config error path and retains the prior identity after failure', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onApplied = vi.fn();
        const prior = usePlaygroundStore.getState().prepared!;
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        usePlaygroundStore.setState({
            applyRecipe: vi.fn(async () => ({
                ok: false as const,
                issues: [{ code: 'invalid-field' as const, path: 'recipe', message: 'Deliberate failure' }],
            })),
        });

        render(<PresetPanel onReset={onReset} onApplied={onApplied} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Deliberate failure');
        expect(useTrainingStore.getState().configErrorSource).toBe('preset');
        expect(useTrainingStore.getState().configError).toContain('Deliberate failure');
        expect(usePlaygroundStore.getState().prepared).toBe(prior);
        expect(onReset).not.toHaveBeenCalled();
        expect(onApplied).not.toHaveBeenCalled();
        expect(screen.getByRole('button', { name: `Apply preset: ${target.title}` }))
            .toHaveAttribute('aria-pressed', 'false');
    });

    it('does not reset or report a stale successful apply after a newer edit wins', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onApplied = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        render(<PresetPanel onReset={onReset} onApplied={onApplied} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));
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
        expect(onApplied).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().prepared?.identities.canonicalRecipeKey)
            .not.toBe(target.prepared.identities.canonicalRecipeKey);
        expect(useTrainingStore.getState()).toMatchObject({
            pendingConfigSource: 'network',
            networkConfigLoading: true,
            configError: null,
        });
    });

    it('does not report an older same-target preset result', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onApplied = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        render(<PresetPanel onReset={onReset} onApplied={onApplied} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));
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
        expect(onApplied).not.toHaveBeenCalled();
    });

    it('records a still-current preparation failure after the preset panel unmounts', async () => {
        const user = userEvent.setup();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        let resolveApply!: (result: ApplyResult) => void;
        const pending = new Promise<ApplyResult>((resolve) => {
            resolveApply = resolve;
        });
        usePlaygroundStore.setState({ applyRecipe: vi.fn(() => pending) });

        const { unmount } = render(<PresetPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));
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

    it('does not report an older failure over a newer config transaction', async () => {
        const user = userEvent.setup();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        let resolveApply!: (result: ApplyResult) => void;
        const pending = new Promise<ApplyResult>((resolve) => {
            resolveApply = resolve;
        });
        usePlaygroundStore.setState({ applyRecipe: vi.fn(() => pending) });

        render(<PresetPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));
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

    it('does not invoke callbacks after a pending apply unmounts', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        const onApplied = vi.fn();
        const target = resolveRecipe({ id: 'xor-hidden', revision: 1 })!;
        const deferred = deferApplyCompletion(originalApplyRecipe);
        usePlaygroundStore.setState({ applyRecipe: deferred.applyRecipe });

        const { unmount } = render(<PresetPanel onReset={onReset} onApplied={onApplied} />);
        await user.click(screen.getByRole('button', { name: `Apply preset: ${target.title}` }));
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
        expect(onApplied).not.toHaveBeenCalled();
    });

    it('keeps preset-specific config errors retryable', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({
            configError: 'Failed to apply preset',
            configErrorSource: 'preset',
        });

        render(<PresetPanel onReset={vi.fn()} />);

        expect(screen.getByRole('alert')).toHaveTextContent('Failed to apply preset');
        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });
});
