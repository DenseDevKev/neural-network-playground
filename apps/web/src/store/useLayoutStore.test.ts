import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, LEGACY_LAYOUT_STORAGE_KEY } from './useLayoutStore.ts';

describe('local layout preferences', () => {
    beforeEach(() => window.localStorage.clear());
    it('starts on Network with Standard guidance and no duplicate legacy state', () => {
        const state = createLayoutStore().getState();
        expect(state).toMatchObject({ destination:'playground', workspaceTab:'network', audienceMode:'explore' });
        for (const key of ['view','phase','activeTabLeft','activeTabRight','advancedToolsOpen','buildContextOpen']) expect(state).not.toHaveProperty(key);
    });
    it.each(['dataset','network','training'] as const)('opens %s setup atomically without affecting the lesson session', (tab) => {
        const store = createLayoutStore();
        store.getState().setActiveLessonStep('lesson-xor-hidden-layers', 1);
        store.getState().navigate('saved-runs');
        let transitions = 0;
        const unsubscribe = store.subscribe(() => { transitions++; });
        store.getState().openSetup(tab); unsubscribe();
        expect(transitions).toBe(1);
        expect(store.getState()).toMatchObject({destination:'playground',workspaceTab:'setup',setupTab:tab,activeLessonId:'lesson-xor-hidden-layers',activeLessonStepIndex:1});
    });
    it.each(['beginner','explore','lab'] as const)('guidance %s preserves selected diagnostic and setup tabs', (mode) => {
        const store = createLayoutStore();
        store.getState().openSetup('training');
        store.getState().navigate('playground','inspect');
        store.getState().setInspectTab('gradients');
        let transitions = 0;
        const unsubscribe = store.subscribe(() => { transitions++; });
        store.getState().setAudienceMode(mode); unsubscribe();
        expect(transitions).toBe(1);
        expect(store.getState()).toMatchObject({audienceMode:mode,workspaceTab:'inspect',inspectTab:'gradients',setupTab:'training'});
    });
    it('persists current navigation and code choice but excludes transient execution and overlays', () => {
        const store = createLayoutStore();
        store.getState().setResultsTab('errors'); store.getState().setCodeExportTab('numpy');
        store.getState().navigate('saved-runs'); store.getState().requestExport('code');
        store.getState().setActiveLessonStep('lesson-xor-hidden-layers',2); store.getState().dismissLessonCue();
        const stored = JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY)!);
        expect(stored.state).toMatchObject({destination:'saved-runs',resultsTab:'errors',codeExportTab:'numpy',lessonCueDismissed:true,hasStartedLesson:true});
        for (const key of ['exportRequest','activeLessonId','activeLessonStepIndex','view','advancedToolsOpen','draft','comparison']) expect(stored.state).not.toHaveProperty(key);
        expect(createLayoutStore().getState()).toMatchObject({...stored.state,exportRequest:null,activeLessonId:null,activeLessonStepIndex:null});
    });
    it('dismisses the invitation and exits a lesson without erasing its completed-start preference', () => {
        const store = createLayoutStore();
        store.getState().setActiveLessonStep('lesson-circle-hidden-layer',1);
        store.getState().dismissLessonCue(); store.getState().clearActiveLessonStep();
        expect(store.getState()).toMatchObject({activeLessonId:null,activeLessonStepIndex:null,hasStartedLesson:true,lessonCueDismissed:true,workspaceTab:'network'});
    });
    it('uses a fresh export request object and leaves underlying workspace selection intact', () => {
        const store = createLayoutStore(); store.getState().navigate('playground','inspect');
        store.getState().requestExport('setup'); const first = store.getState().exportRequest;
        store.getState().requestExport('code'); expect(store.getState().exportRequest!.id).toBeGreaterThan(first!.id);
        expect(store.getState().workspaceTab).toBe('inspect'); store.getState().clearExportRequest(); expect(store.getState().exportRequest).toBeNull();
    });
    const legacyCases: Array<[Record<string, unknown>, Record<string, unknown>]> = [
        [{view:'build',activeRecipeSection:'features'}, {workspaceTab:'setup',setupTab:'network'}],
        [{phase:'run',activeTabLeft:'network',activeTabRight:'confusion',layout:'split',codeExportTab:'tfjs'}, {workspaceTab:'results',setupTab:'network',resultsTab:'errors',codeExportTab:'tfjs'}],
        [{view:'run',activeEvidenceView:'inspection',audienceMode:'beginner',advancedToolsOpen:false}, {workspaceTab:'inspect',audienceMode:'beginner'}],
        [{view:'build',activeRecipeSection:'hyperparams',audienceMode:'lab',advancedToolsOpen:false,buildContextOpen:true}, {workspaceTab:'setup',setupTab:'training',audienceMode:'lab'}],
        [{view:'run',activeEvidenceView:'history'}, {destination:'saved-runs'}],
        [{view:'run',activeEvidenceView:'loss'}, {workspaceTab:'results',resultsTab:'learning'}],
        [{view:'run',activeRecipeSection:'features',activeEvidenceView:'code',codeExportTab:'numpy'}, {workspaceTab:'results',setupTab:'network',codeExportTab:'numpy',exportRequest:{mode:'code',id:1}}],
        [{view:'build',activeRecipeSection:'config'}, {workspaceTab:'setup',exportRequest:{mode:'setup',id:1}}],
        [{view:'build',activeRecipeSection:'config',activeEvidenceView:'code'}, {workspaceTab:'setup',exportRequest:{mode:'setup',id:1}}],
        [{view:'run',activeRecipeSection:'config',activeEvidenceView:'code'}, {workspaceTab:'results',exportRequest:{mode:'code',id:1}}],
        [{phase:'build',activeTabLeft:'config',activeTabRight:'code'}, {workspaceTab:'setup',exportRequest:{mode:'setup',id:1}}],
        [{phase:'run',activeTabLeft:'config',activeTabRight:'code'}, {workspaceTab:'results',exportRequest:{mode:'code',id:1}}],
    ];
    it.each(legacyCases)('migrates old selections %j without writing or deleting the old key', (state, expected) => {
        const serialized = JSON.stringify({state,version:0}); localStorage.setItem(LEGACY_LAYOUT_STORAGE_KEY,serialized);
        const store = createLayoutStore(); expect(store.getState()).toMatchObject(expected);
        store.getState().navigate('lessons'); expect(localStorage.getItem(LEGACY_LAYOUT_STORAGE_KEY)).toBe(serialized);
    });
    it('prefers current explicit navigation over stale legacy aliases', () => {
        localStorage.setItem(LAYOUT_STORAGE_KEY,JSON.stringify({state:{destination:'playground',workspaceTab:'inspect',inspectTab:'gradients',view:'run',activeEvidenceView:'code'},version:0}));
        expect(createLayoutStore().getState()).toMatchObject({workspaceTab:'inspect',inspectTab:'gradients',exportRequest:null});
    });
    it('sanitizes invalid enums and boolean flags while ignoring persisted active lessons', () => {
        localStorage.setItem(LAYOUT_STORAGE_KEY,JSON.stringify({state:{view:'debug',phase:'debug',workspaceTab:'missing',setupTab:'missing',resultsTab:'missing',inspectTab:'missing',codeExportTab:'missing',audienceMode:'expert',lessonCueDismissed:'true',hasStartedLesson:true,activeLessonId:'stale',activeLessonStepIndex:4},version:0}));
        expect(createLayoutStore().getState()).toMatchObject({workspaceTab:'network',setupTab:'dataset',resultsTab:'boundary',inspectTab:'trace',codeExportTab:'pseudocode',audienceMode:'explore',lessonCueDismissed:false,hasStartedLesson:true,activeLessonId:null,activeLessonStepIndex:null});
    });
    it('ignores malformed legacy JSON and keeps fresh defaults', () => {
        localStorage.setItem(LEGACY_LAYOUT_STORAGE_KEY,'{broken'); expect(createLayoutStore().getState().workspaceTab).toBe('network');
    });
});
