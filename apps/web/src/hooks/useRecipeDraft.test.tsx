import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import { useRecipeDraft } from './useRecipeDraft.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
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
