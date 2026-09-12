import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, LEGACY_LAYOUT_STORAGE_KEY } from './useLayoutStore.ts';

describe('Atelier navigation', () => {
    beforeEach(() => window.localStorage.clear());
    afterEach(() => vi.restoreAllMocks());
    it('starts new visitors on Network and stores navigation outside the experiment URL', () => {
        const store = createLayoutStore();
        const original = location.hash;
        expect(store.getState().workspaceTab).toBe('network');
        store.getState().navigate('saved-runs');
        store.getState().openSetup('training');
        store.getState().setInspectTab('gradients');
        expect(location.hash).toBe(original);
        expect(createLayoutStore().getState()).toMatchObject({destination:'playground',workspaceTab:'setup',setupTab:'training',inspectTab:'gradients'});
    });
    it.each([['build','hyperparams','boundary','setup','training','boundary'],['run','network','loss','results','network','learning'],['run','data','inspection','inspect','dataset','boundary']])('migrates %s/%s/%s without deleting the old key', (view,recipe,evidence,workspaceTab,setupTab,resultsTab) => {
        const legacy = JSON.stringify({state:{view,activeRecipeSection:recipe,activeEvidenceView:evidence,codeExportTab:'numpy',lessonCueDismissed:true},version:0});
        window.localStorage.setItem(LEGACY_LAYOUT_STORAGE_KEY,legacy);
        const store = createLayoutStore();
        expect(store.getState()).toMatchObject({workspaceTab,setupTab,resultsTab,codeExportTab:'numpy',lessonCueDismissed:true});
        store.getState().navigate('lessons');
        expect(window.localStorage.getItem(LEGACY_LAYOUT_STORAGE_KEY)).toBe(legacy);
        expect(window.localStorage.getItem(LAYOUT_STORAGE_KEY)).not.toBeNull();
    });
    it('keeps navigation and guidance usable when storage writes fail', () => {
        const store = createLayoutStore();
        vi.spyOn(Storage.prototype,'setItem').mockImplementation(() => { throw new Error('quota'); });
        expect(() => store.getState().navigate('playground','inspect')).not.toThrow();
        store.getState().setAudienceMode('beginner');
        expect(store.getState().workspaceTab).toBe('inspect');
    });
});
