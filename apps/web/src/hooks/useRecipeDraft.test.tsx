import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import { useRecipeDraft } from './useRecipeDraft.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import * as recipeCommit from '../store/commitRecipeEdit.ts';
const originalCommitPrepared = recipeCommit.commitRecipeEditPrepared;
afterEach(() => vi.restoreAllMocks());
const originalEdit = usePlaygroundStore.getState().editRecipe;
function prepared() { const access = usePlaygroundStore.getState().access; if(access.status !== 'ready') throw Error('not ready'); return access.prepared; }
async function acknowledge() {
    const p = prepared();
    act(() => {
        useTrainingStore.getState().markTrainedRecipe(p.document.recipe,'config-sync',p.identities.recipeFingerprint);
        useTrainingStore.getState().finishConfigChange();
    });
}
beforeEach(async () => {
    usePlaygroundStore.setState({editRecipe:originalEdit});
    await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
    useTrainingStore.getState().finishConfigChange();
    useTrainingStore.setState({configError:null,configErrorSource:null,trainedRecipeFingerprint:null,trainedRecipeSource:null});
});
describe('session recipe draft', () => {
    it('keeps cross-section changes local and cancels all of them', () => {
        const original=prepared();const {result}=renderHook(useRecipeDraft);
        act(() => {result.current.commands.number('data.noise','12');result.current.commands.number('model.seed','11');result.current.commands.number('training.learningRate','0.1');});
        expect(result.current.dirty).toBe(true);expect(prepared()).toBe(original);
        act(() => result.current.commands.cancel());
        expect(result.current.dirty).toBe(false);expect(result.current.recipe).toEqual(original.document.recipe);
    });
    it('allows incomplete numbers and cross-field conflicts until corrected', () => {
        const {result}=renderHook(useRecipeDraft);
        act(() => {result.current.commands.number('data.sampleCount','');result.current.commands.number('training.batchSize','200');});
        expect(result.current.value('data.sampleCount')).toBe('');expect(result.current.valid).toBe(false);
        act(() => result.current.commands.number('data.sampleCount','100'));
        expect(result.current.valid).toBe(false);
        act(() => result.current.commands.number('training.batchSize','10'));
        expect(result.current.valid).toBe(true);
    });
    it('rejects a stale base without publishing or pausing training', async () => {
        const {result}=renderHook(useRecipeDraft);
        act(() => result.current.commands.number('data.noise','12'));
        await act(async () => {await originalEdit((r) => ({ok:true,recipe:{...r,model:{...r.model,seed:123}}}));});
        let accepted=true;await act(async () => {accepted=await result.current.commands.apply();});
        expect(accepted).toBe(false);expect(result.current.error).toContain('active recipe changed');expect(useTrainingStore.getState().pendingConfigSource).toBeNull();expect(result.current.dirty).toBe(true);
    });
    it('submits once and clears only on the matching worker acknowledgement', async () => {
        const spy=vi.fn(originalEdit);usePlaygroundStore.setState({editRecipe:spy});
        const {result}=renderHook(useRecipeDraft);
        act(() => result.current.commands.number('data.noise','12'));
        let apply!:Promise<boolean>;let duplicate!:Promise<boolean>;
        act(() => {apply=result.current.commands.apply();duplicate=result.current.commands.apply();});
        expect(await duplicate).toBe(false);
        await waitFor(() => expect(prepared().document.recipe.data.noise).toBe(12));
        expect(result.current.submitted).toBe(true);expect(result.current.dirty).toBe(true);expect(spy).toHaveBeenCalledTimes(1);
        act(() => useTrainingStore.getState().finishConfigChange());
        expect(result.current.dirty).toBe(true);
        await acknowledge();expect(await apply).toBe(true);
        await waitFor(() => expect(result.current.dirty).toBe(false));
    });
    it('never adopts a superseding publication in the commit continuation microtask', async () => {
        await originalEdit((recipe) => ({ok:true,recipe:{...recipe,data:{...recipe.data,noise:29}}}));
        const superseding = prepared();
        await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        vi.spyOn(recipeCommit, 'commitRecipeEditPrepared').mockImplementation(async (...args) => {
            const publication = await originalCommitPrepared(...args);
            // This replacement runs after publication succeeded but before the
            // setup hook resumes its awaited continuation.
            queueMicrotask(() => {
                usePlaygroundStore.setState({access:{status:'ready',prepared:superseding}});
                useTrainingStore.getState().markTrainedRecipe(superseding.document.recipe,'config-sync',superseding.identities.recipeFingerprint);
                useTrainingStore.getState().finishConfigChange();
            });
            return publication;
        });
        const {result}=renderHook(useRecipeDraft);
        act(() => result.current.commands.number('data.noise','12'));
        await act(async () => {expect(await result.current.commands.apply()).toBe(false);});
        expect(prepared()).toBe(superseding);
        expect(result.current.dirty).toBe(true);
        expect(result.current.submitted).toBe(false);
        expect(result.current.value('data.noise')).toBe('12');
        expect(result.current.error).toContain('active experiment changed');
        act(() => result.current.commands.cancel());
        expect(result.current.dirty).toBe(false);
    });
    it('unlocks cancellation when a failed sync is superseded while its error persists', async () => {
        const {result}=renderHook(useRecipeDraft);
        act(() => result.current.commands.number('data.noise','12'));
        let apply!:Promise<boolean>;act(() => {apply=result.current.commands.apply();});
        await waitFor(() => expect(prepared().document.recipe.data.noise).toBe(12));
        act(() => useTrainingStore.getState().failConfigChange('Worker unavailable'));
        expect(await apply).toBe(false);
        expect(result.current.submitted).toBe(true);
        await act(async () => {await originalEdit((recipe) => ({ok:true,recipe:{...recipe,data:{...recipe.data,noise:29}}}));});
        expect(useTrainingStore.getState().configError).toBe('Worker unavailable');
        expect(result.current.submitted).toBe(false);
        expect(result.current.busy).toBe(false);
        expect(result.current.dirty).toBe(true);
        act(() => result.current.commands.cancel());
        expect(result.current.dirty).toBe(false);
        expect(result.current.recipe?.data.noise).toBe(29);
    });
    it('preserves input after preparation failure', async () => {
        usePlaygroundStore.setState({editRecipe:vi.fn(async () => ({ok:false as const,issues:[{code:'invalid-field' as const,path:'recipe',message:'Preparation failed'}]}))});
        const {result}=renderHook(useRecipeDraft);act(() => result.current.commands.number('data.noise','12'));
        await act(async () => {expect(await result.current.commands.apply()).toBe(false);});
        expect(result.current.value('data.noise')).toBe('12');expect(result.current.busy).toBe(false);expect(result.current.error).toContain('Preparation failed');
    });
    it('retains submitted state through a sync failure and existing retry', async () => {
        const {result}=renderHook(useRecipeDraft);act(() => result.current.commands.number('data.noise','12'));
        let apply!:Promise<boolean>;act(() => {apply=result.current.commands.apply();});
        await waitFor(() => expect(prepared().document.recipe.data.noise).toBe(12));
        act(() => useTrainingStore.getState().failConfigChange('Worker unavailable'));
        expect(await apply).toBe(false);expect(result.current.submitted).toBe(true);expect(result.current.dirty).toBe(true);
        act(() => result.current.commands.retry());expect(useTrainingStore.getState().pendingConfigSource).toBe('setup');
        await acknowledge();await waitFor(() => expect(result.current.dirty).toBe(false));
    });
});

it('stages every catalog transition exactly without preparing and clears stale raw input', () => {
    const original = prepared();
    const spy = vi.fn(originalEdit);
    usePlaygroundStore.setState({ editRecipe: spy });
    const { result } = renderHook(useRecipeDraft);
    for (const source of PREPARED_PRESETS) {
        for (const destination of PREPARED_PRESETS) {
            act(() => {
                result.current.commands.preset(source.prepared.document.recipe);
                result.current.commands.number('data.sampleCount', '');
                result.current.commands.preset(destination.prepared.document.recipe);
            });
            expect(result.current.recipe).toEqual(destination.prepared.document.recipe);
            expect(result.current.value('data.sampleCount')).toBe(destination.prepared.document.recipe.data.sampleCount);
            expect(result.current.valid).toBe(true);
        }
    }
    expect(prepared()).toBe(original);
    expect(spy).not.toHaveBeenCalled();
    expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
    act(() => result.current.commands.cancel());
    expect(result.current.recipe).toEqual(original.document.recipe);
    expect(result.current.dirty).toBe(false);
});

it('applies a preset plus cross-tab edits once and waits for exact acknowledgement', async () => {
    const { result } = renderHook(useRecipeDraft);
    const chosen = PREPARED_PRESETS[0].prepared.document.recipe;
    act(() => {
        result.current.commands.preset(chosen);
        result.current.commands.number('data.noise', '12.3456789');
        result.current.commands.number('model.seed', '998');
        result.current.commands.number('training.learningRate', '0.004321');
    });
    const candidate = structuredClone(result.current.recipe);
    let apply!: Promise<boolean>;
    act(() => { apply = result.current.commands.apply(); });
    await waitFor(() => expect(prepared().document.recipe).toEqual(candidate));
    act(() => result.current.commands.preset(PREPARED_PRESETS[1].prepared.document.recipe));
    expect(result.current.recipe).toEqual(candidate);
    expect(result.current.submitted).toBe(true);
    await acknowledge();
    expect(await apply).toBe(true);
    await waitFor(() => expect(result.current.dirty).toBe(false));
});

it('preset replacement retains the original draft base when another recipe wins', async () => {
    const { result } = renderHook(useRecipeDraft);
    act(() => result.current.commands.number('data.noise', '12'));
    await act(async () => { await originalEdit((recipe) => ({ ok: true, recipe: { ...recipe, model: { ...recipe.model, seed: 123 } } })); });
    act(() => result.current.commands.preset(PREPARED_PRESETS[0].prepared.document.recipe));
    await act(async () => { expect(await result.current.commands.apply()).toBe(false); });
    expect(result.current.error).toContain('active recipe changed');
    expect(prepared().document.recipe.model.seed).toBe(123);
});
