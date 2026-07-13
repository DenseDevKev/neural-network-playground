import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, useLayoutStore } from './useLayoutStore.ts';

describe('useLayoutStore', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
            codeExportTab: 'pseudocode',
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });

    it('defaults to the Build view and boundary evidence', () => {
        const state = useLayoutStore.getState();
        expect(state.view).toBe('build');
        expect(state.activeRecipeSection).toBe('data');
        expect(state.activeEvidenceView).toBe('boundary');
    });

    it('setView toggles between Build and Run while keeping legacy phase in sync', () => {
        const { setView } = useLayoutStore.getState();

        setView('run');
        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().phase).toBe('run');

        setView('build');
        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().phase).toBe('build');
    });

    it('tracks the active recipe section', () => {
        useLayoutStore.getState().setActiveRecipeSection('network');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('network');
        expect(useLayoutStore.getState().activeTabLeft).toBe('network');

        useLayoutStore.getState().setActiveRecipeSection('hyperparams');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
    });

    it('tracks the active evidence view', () => {
        useLayoutStore.getState().setActiveEvidenceView('loss');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');

        useLayoutStore.getState().setActiveEvidenceView('code');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('code');
        expect(useLayoutStore.getState().activeTabRight).toBe('code');
    });

    it('persists the code export tab across panel remounts', () => {
        useLayoutStore.getState().setCodeExportTab('numpy');
        expect(useLayoutStore.getState().codeExportTab).toBe('numpy');

        useLayoutStore.getState().setCodeExportTab('tfjs');
        expect(useLayoutStore.getState().codeExportTab).toBe('tfjs');
    });

    it('persists only app-local Build/Run workspace state to localStorage', () => {
        useLayoutStore.getState().setView('run');
        useLayoutStore.getState().setActiveRecipeSection('features');
        useLayoutStore.getState().setActiveEvidenceView('inspection');
        useLayoutStore.getState().setActiveLessonStep('lesson-xor-hidden-layers', 1);

        const stored = JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? '{}');
        expect(stored.state?.view).toBe('run');
        expect(stored.state?.activeRecipeSection).toBe('features');
        expect(stored.state?.activeEvidenceView).toBe('inspection');
        expect(stored.state?.layout).toBeUndefined();
        expect(stored.state?.phase).toBeUndefined();
        expect(stored.state?.activeLessonId).toBeUndefined();
        expect(stored.state?.activeLessonStepIndex).toBeUndefined();
    });

    it('tracks active lesson step as transient UI state', () => {
        useLayoutStore.getState().setActiveLessonStep('lesson-xor-hidden-layers', 1);
        expect(useLayoutStore.getState().activeLessonId).toBe('lesson-xor-hidden-layers');
        expect(useLayoutStore.getState().activeLessonStepIndex).toBe(1);

        useLayoutStore.getState().clearActiveLessonStep();
        expect(useLayoutStore.getState().activeLessonId).toBeNull();
        expect(useLayoutStore.getState().activeLessonStepIndex).toBeNull();
    });

    it('rehydrates state from localStorage with a fresh store instance', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    view: 'run',
                    activeRecipeSection: 'features',
                    activeEvidenceView: 'code',
                    codeExportTab: 'numpy',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().view).toBe('run');
        expect(freshStore.getState().activeRecipeSection).toBe('features');
        expect(freshStore.getState().activeEvidenceView).toBe('code');
        expect(freshStore.getState().phase).toBe('run');
        expect(freshStore.getState().activeTabLeft).toBe('features');
        expect(freshStore.getState().activeTabRight).toBe('code');
        expect(freshStore.getState().codeExportTab).toBe('numpy');
    });

    it('migrates old persisted phase and tab state while ignoring old layout', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    layout: 'split',
                    phase: 'run',
                    activeTabLeft: 'network',
                    activeTabRight: 'confusion',
                    codeExportTab: 'tfjs',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().view).toBe('run');
        expect(freshStore.getState().activeRecipeSection).toBe('network');
        expect(freshStore.getState().activeEvidenceView).toBe('confusion');
        expect(freshStore.getState().layout).toBe('dock');
        expect(freshStore.getState().codeExportTab).toBe('tfjs');
    });

    it('sanitizes invalid persisted layout state on rehydrate', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    view: 'debug',
                    layout: 'wide-open',
                    phase: 'debug',
                    activeRecipeSection: 'missing',
                    activeEvidenceView: 'also-missing',
                    codeExportTab: 'also-missing',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().view).toBe('build');
        expect(freshStore.getState().activeRecipeSection).toBe('data');
        expect(freshStore.getState().activeEvidenceView).toBe('boundary');
        expect(freshStore.getState().phase).toBe('build');
        expect(freshStore.getState().activeTabLeft).toBe('data');
        expect(freshStore.getState().activeTabRight).toBe('boundary');
        expect(freshStore.getState().codeExportTab).toBe('pseudocode');
    });
});
