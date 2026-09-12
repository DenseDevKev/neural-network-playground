import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, useLayoutStore } from './useLayoutStore.ts';

describe('useLayoutStore', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
            codeExportTab: 'pseudocode',
            activeLessonId: null,
            activeLessonStepIndex: null,
            lessonCueDismissed: false,
            hasStartedLesson: false,
        });
    });

    it('defaults to the Build view, Explore mode, and closed Advanced Tools', () => {
        const state = useLayoutStore.getState();
        expect(state.view).toBe('build');
        expect(state.activeRecipeSection).toBe('data');
        expect(state.activeEvidenceView).toBe('boundary');
        expect(state.audienceMode).toBe('explore');
        expect(state.advancedToolsOpen).toBe(false);
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
        expect(useLayoutStore.getState().exportRequest?.mode).toBe('code');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
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
        useLayoutStore.getState().setAudienceMode('lab');
        useLayoutStore.getState().setAdvancedToolsOpen(false);
        useLayoutStore.getState().setActiveLessonStep('lesson-xor-hidden-layers', 1);

        const stored = JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? '{}');
        expect(stored.state?.view).toBe('run');
        expect(stored.state?.activeRecipeSection).toBe('features');
        expect(stored.state?.activeEvidenceView).toBe('boundary');
        expect(stored.state?.audienceMode).toBe('lab');
        expect(stored.state?.advancedToolsOpen).toBe(false);
        expect(stored.state?.layout).toBeUndefined();
        expect(stored.state?.phase).toBeUndefined();
        expect(stored.state?.activeLessonId).toBeUndefined();
        expect(stored.state?.activeLessonStepIndex).toBeUndefined();
    });

    it('applies the selected profile disclosure default without changing the profile for navigation', () => {
        useLayoutStore.getState().setAudienceMode('lab');
        expect(useLayoutStore.getState().audienceMode).toBe('lab');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);

        useLayoutStore.getState().setAudienceMode('beginner');
        expect(useLayoutStore.getState().audienceMode).toBe('beginner');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);

        useLayoutStore.getState().setActiveRecipeSection('features');
        expect(useLayoutStore.getState().audienceMode).toBe('beginner');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
    });

    it('falls back hidden targets and synchronizes aliases in one disclosure transition', () => {
        useLayoutStore.getState().setAudienceMode('beginner');
        useLayoutStore.getState().setAdvancedToolsOpen(true);
        useLayoutStore.getState().setActiveRecipeSection('config');
        useLayoutStore.getState().setActiveEvidenceView('inspection');

        let notifications = 0;
        const unsubscribe = useLayoutStore.subscribe(() => {
            notifications += 1;
        });
        useLayoutStore.getState().setAdvancedToolsOpen(false);
        unsubscribe();

        const state = useLayoutStore.getState();
        expect(notifications).toBe(1);
        expect(state.advancedToolsOpen).toBe(false);
        expect(state.activeRecipeSection).toBe('data');
        expect(state.activeTabLeft).toBe('data');
        expect(state.activeEvidenceView).toBe('boundary');
        expect(state.activeTabRight).toBe('boundary');
    });

    it('falls back targets hidden by a mode default in one atomic transition', () => {
        useLayoutStore.getState().setAudienceMode('lab');
        useLayoutStore.getState().setActiveRecipeSection('config');
        useLayoutStore.getState().setActiveEvidenceView('code');

        let notifications = 0;
        const unsubscribe = useLayoutStore.subscribe(() => {
            notifications += 1;
        });
        useLayoutStore.getState().setAudienceMode('beginner');
        unsubscribe();

        const state = useLayoutStore.getState();
        expect(notifications).toBe(1);
        expect(state.audienceMode).toBe('beginner');
        expect(state.advancedToolsOpen).toBe(false);
        expect(state.activeRecipeSection).toBe('data');
        expect(state.activeTabLeft).toBe('data');
        expect(state.activeEvidenceView).toBe('boundary');
        expect(state.activeTabRight).toBe('boundary');
    });

    it('opens Advanced Tools for hidden canonical and legacy navigation targets', () => {
        useLayoutStore.getState().setAudienceMode('beginner');

        useLayoutStore.getState().setActiveTabLeft('hyperparams');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
        expect(useLayoutStore.getState().audienceMode).toBe('beginner');

        useLayoutStore.getState().setAdvancedToolsOpen(false);
        useLayoutStore.getState().setActiveTabRight('confusion');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('confusion');
        expect(useLayoutStore.getState().activeTabRight).toBe('confusion');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
        expect(useLayoutStore.getState().audienceMode).toBe('beginner');
    });

    it('opens an advanced recipe section in Build with one atomic notification', () => {
        useLayoutStore.setState({
            view: 'run',
            phase: 'run',
            audienceMode: 'beginner',
            advancedToolsOpen: false,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
        });
        const transitions: Array<ReturnType<typeof useLayoutStore.getState>> = [];
        const unsubscribe = useLayoutStore.subscribe((state) => transitions.push(state));

        useLayoutStore.getState().openAdvancedRecipeSection('hyperparams');
        unsubscribe();

        expect(transitions).toHaveLength(1);
        expect(transitions[0]).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'hyperparams',
            activeTabLeft: 'hyperparams',
            advancedToolsOpen: true,
        });
    });

    it('keeps legacy History navigation available without opening Advanced Tools', () => {
        useLayoutStore.getState().setAudienceMode('beginner');
        useLayoutStore.getState().setActiveEvidenceView('history');

        expect(useLayoutStore.getState().activeEvidenceView).toBe('history');
        expect(useLayoutStore.getState().activeTabRight).toBe('history');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
    });

    it('does not rewrite a visible legacy History alias during profile or disclosure changes', () => {
        useLayoutStore.getState().setActiveEvidenceView('history');

        useLayoutStore.getState().setAudienceMode('lab');
        useLayoutStore.getState().setAdvancedToolsOpen(false);

        expect(useLayoutStore.getState().activeEvidenceView).toBe('history');
        expect(useLayoutStore.getState().activeTabRight).toBe('history');
    });

    it('tracks active lesson step as transient UI state', () => {
        useLayoutStore.getState().setActiveLessonStep('lesson-xor-hidden-layers', 1);
        expect(useLayoutStore.getState().activeLessonId).toBe('lesson-xor-hidden-layers');
        expect(useLayoutStore.getState().activeLessonStepIndex).toBe(1);
        expect(useLayoutStore.getState().hasStartedLesson).toBe(true);

        useLayoutStore.getState().clearActiveLessonStep();
        expect(useLayoutStore.getState().activeLessonId).toBeNull();
        expect(useLayoutStore.getState().activeLessonStepIndex).toBeNull();
        expect(useLayoutStore.getState().hasStartedLesson).toBe(true);
    });

    it('dismisses the first-visit lesson cue without changing other layout state', () => {
        const before = useLayoutStore.getState();

        before.dismissLessonCue();

        expect(useLayoutStore.getState()).toMatchObject({
            lessonCueDismissed: true,
            hasStartedLesson: false,
            view: before.view,
            audienceMode: before.audienceMode,
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });

    it('persists sticky lesson history while keeping active lesson fields transient', async () => {
        useLayoutStore.getState().dismissLessonCue();
        useLayoutStore.getState().setActiveLessonStep('lesson-xor-hidden-layers', 2);

        const stored = JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? '{}');
        expect(stored.version).toBe(0);
        expect(stored.state?.lessonCueDismissed).toBe(true);
        expect(stored.state?.hasStartedLesson).toBe(true);
        expect(stored.state?.activeLessonId).toBeUndefined();
        expect(stored.state?.activeLessonStepIndex).toBeUndefined();

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());
        expect(freshStore.getState()).toMatchObject({
            lessonCueDismissed: true,
            hasStartedLesson: true,
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });

    it('sanitizes sticky lesson flags as actual booleans', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    lessonCueDismissed: 'true',
                    hasStartedLesson: true,
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().lessonCueDismissed).toBe(false);
        expect(freshStore.getState().hasStartedLesson).toBe(true);
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
        expect(freshStore.getState().exportRequest?.mode).toBe('code');
        expect(freshStore.getState().activeEvidenceView).toBe('boundary');
        expect(freshStore.getState().phase).toBe('run');
        expect(freshStore.getState().activeTabLeft).toBe('features');
        expect(freshStore.getState().activeTabRight).toBe('boundary');
        expect(freshStore.getState().codeExportTab).toBe('numpy');
        expect(freshStore.getState().audienceMode).toBe('explore');
        expect(freshStore.getState().advancedToolsOpen).toBe(true);
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
        expect(freshStore.getState().audienceMode).toBe('explore');
        expect(freshStore.getState().advancedToolsOpen).toBe(false);
    });

    it('uses each profile default when persisted disclosure state is missing', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    audienceMode: 'lab',
                    activeRecipeSection: 'data',
                    activeEvidenceView: 'boundary',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().audienceMode).toBe('lab');
        expect(freshStore.getState().advancedToolsOpen).toBe(true);
    });

    it('preserves a valid explicit collapsed Lab state across hydration', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    audienceMode: 'lab',
                    advancedToolsOpen: false,
                    activeRecipeSection: 'hyperparams',
                    activeEvidenceView: 'confusion',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().audienceMode).toBe('lab');
        expect(freshStore.getState().advancedToolsOpen).toBe(false);
        expect(freshStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(freshStore.getState().activeEvidenceView).toBe('confusion');
    });

    it('opens disclosure on hydration to preserve a hidden persisted target', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    audienceMode: 'beginner',
                    advancedToolsOpen: false,
                    activeRecipeSection: 'features',
                    activeEvidenceView: 'inspection',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().audienceMode).toBe('beginner');
        expect(freshStore.getState().advancedToolsOpen).toBe(true);
        expect(freshStore.getState().activeRecipeSection).toBe('features');
        expect(freshStore.getState().activeTabLeft).toBe('features');
        expect(freshStore.getState().activeEvidenceView).toBe('inspection');
        expect(freshStore.getState().activeTabRight).toBe('inspection');
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
                    audienceMode: 'expert',
                    advancedToolsOpen: 'yes',
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
        expect(freshStore.getState().audienceMode).toBe('explore');
        expect(freshStore.getState().advancedToolsOpen).toBe(false);
    });
});
