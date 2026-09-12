// Local navigation and guidance never enter the shareable experiment document.
import { useStore } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';
import type { Destination, WorkspaceTab, SetupTab, ResultsTab, InspectTab } from '../productShell/atelierTypes.ts';
import { CODE_EXPORT_TABS, EVIDENCE_VIEW_IDS, RECIPE_SECTION_IDS, type CodeExportTab } from '../productShell/shellTypes.ts';
import { isAudienceMode, type AudienceMode } from '../productShell/audienceProfiles.ts';

export type { CodeExportTab, EvidenceViewId, RecipeSectionId, LeftTabId, RightTabId, PhaseMode } from '../productShell/shellTypes.ts';
export const LAYOUT_STORAGE_KEY = 'nn-playground-layout-v2';
export const LEGACY_LAYOUT_STORAGE_KEY = 'nn-playground-layout';

const DEFAULT_LAYOUT_STATE = {
    destination: 'playground' as Destination,
    workspaceTab: 'network' as WorkspaceTab,
    setupTab: 'dataset' as SetupTab,
    resultsTab: 'boundary' as ResultsTab,
    inspectTab: 'trace' as InspectTab,
    audienceMode: 'explore' as AudienceMode,
    codeExportTab: 'pseudocode' as CodeExportTab,
    exportRequest: null as { mode: 'setup' | 'code'; id: number } | null,
    activeLessonId: null as string | null,
    activeLessonStepIndex: null as number | null,
    lessonCueDismissed: false,
    hasStartedLesson: false,
};

export interface LayoutStore extends Readonly<typeof DEFAULT_LAYOUT_STATE> {
    navigate(destination: Destination, tab?: WorkspaceTab): void;
    openSetup(tab: SetupTab): void;
    setSetupTab(tab: SetupTab): void;
    setResultsTab(tab: ResultsTab): void;
    setInspectTab(tab: InspectTab): void;
    setAudienceMode(mode: AudienceMode): void;
    requestExport(mode: 'setup' | 'code'): void;
    clearExportRequest(): void;
    setCodeExportTab(tab: CodeExportTab): void;
    setActiveLessonStep(lessonId: string, stepIndex: number): void;
    clearActiveLessonStep(): void;
    dismissLessonCue(): void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}
function isOneOf<T extends string>(value: unknown, options: readonly T[]): value is T {
    return typeof value === 'string' && options.includes(value as T);
}

/** Old field names are read only here, never maintained as duplicate live state. */
function sanitizePersistedLayoutState(value: unknown): typeof DEFAULT_LAYOUT_STATE {
    const state = isRecord(value) && isRecord(value.state) ? value.state : value;
    if (!isRecord(state)) return { ...DEFAULT_LAYOUT_STATE };
    const view = isOneOf(state.view, ['build', 'run']) ? state.view : isOneOf(state.phase, ['build', 'run']) ? state.phase : null;
    const recipe = isOneOf(state.activeRecipeSection, RECIPE_SECTION_IDS) ? state.activeRecipeSection
        : isOneOf(state.activeTabLeft, RECIPE_SECTION_IDS) ? state.activeTabLeft : 'data';
    const evidence = isOneOf(state.activeEvidenceView, EVIDENCE_VIEW_IDS) ? state.activeEvidenceView
        : isOneOf(state.activeTabRight, EVIDENCE_VIEW_IDS) ? state.activeTabRight : 'boundary';
    const hasCurrentNavigation = isOneOf(state.workspaceTab, ['setup', 'network', 'results', 'inspect']);
    return {
        ...DEFAULT_LAYOUT_STATE,
        destination: isOneOf(state.destination, ['playground', 'saved-runs', 'lessons']) ? state.destination : view === 'run' && evidence === 'history' ? 'saved-runs' : 'playground',
        workspaceTab: hasCurrentNavigation ? state.workspaceTab as WorkspaceTab : view === 'build' ? 'setup' : view === 'run' ? evidence === 'inspection' ? 'inspect' : 'results' : 'network',
        setupTab: isOneOf(state.setupTab, ['dataset', 'network', 'training']) ? state.setupTab : recipe === 'hyperparams' ? 'training' : recipe === 'network' || recipe === 'features' ? 'network' : 'dataset',
        resultsTab: isOneOf(state.resultsTab, ['boundary', 'learning', 'errors']) ? state.resultsTab : evidence === 'loss' ? 'learning' : evidence === 'confusion' ? 'errors' : 'boundary',
        inspectTab: isOneOf(state.inspectTab, ['trace', 'activations', 'gradients']) ? state.inspectTab : 'trace',
        audienceMode: isAudienceMode(state.audienceMode) ? state.audienceMode : 'explore',
        codeExportTab: isOneOf(state.codeExportTab, CODE_EXPORT_TABS) ? state.codeExportTab : 'pseudocode',
        exportRequest: !hasCurrentNavigation ? view === 'run' && evidence === 'code' ? { mode: 'code', id: 1 } : recipe === 'config' ? { mode: 'setup', id: 1 } : evidence === 'code' ? { mode: 'code', id: 1 } : null : null,
        lessonCueDismissed: typeof state.lessonCueDismissed === 'boolean' ? state.lessonCueDismissed : false,
        hasStartedLesson: typeof state.hasStartedLesson === 'boolean' ? state.hasStartedLesson : false,
    };
}

export function createLayoutStore() {
    let initial = DEFAULT_LAYOUT_STATE;
    try {
        if (typeof window !== 'undefined' && !window.localStorage.getItem(LAYOUT_STORAGE_KEY)) {
            const legacy = window.localStorage.getItem(LEGACY_LAYOUT_STORAGE_KEY);
            if (legacy) initial = sanitizePersistedLayoutState(JSON.parse(legacy));
        }
    } catch { /* Unavailable preferences cannot prevent an experiment from opening. */ }
    return createStore<LayoutStore>()(persist((set) => ({
        ...initial,
        navigate: (destination, workspaceTab) => set((state) => ({ destination, workspaceTab: workspaceTab ?? state.workspaceTab })),
        openSetup: (setupTab) => set({ destination: 'playground', workspaceTab: 'setup', setupTab }),
        setSetupTab: (setupTab) => set({ setupTab }),
        setResultsTab: (resultsTab) => set({ resultsTab }),
        setInspectTab: (inspectTab) => set({ inspectTab }),
        setAudienceMode: (audienceMode) => set({ audienceMode }),
        requestExport: (mode) => set((state) => ({ exportRequest: { mode, id: (state.exportRequest?.id ?? 0) + 1 } })),
        clearExportRequest: () => set({ exportRequest: null }),
        setCodeExportTab: (codeExportTab) => set({ codeExportTab }),
        setActiveLessonStep: (activeLessonId, activeLessonStepIndex) => set({ activeLessonId, activeLessonStepIndex, hasStartedLesson: true }),
        clearActiveLessonStep: () => set({ activeLessonId: null, activeLessonStepIndex: null }),
        dismissLessonCue: () => set({ lessonCueDismissed: true }),
    }), {
        name: LAYOUT_STORAGE_KEY,
        storage: createJSONStorage(() => ({
            getItem: (name) => { try { return window.localStorage.getItem(name); } catch { return null; } },
            setItem: (name, value) => { try { window.localStorage.setItem(name, value); } catch { /* Retain this session's choice. */ } },
            removeItem: (name) => { try { window.localStorage.removeItem(name); } catch { /* Retain this session's choice. */ } },
        })),
        partialize: (state) => ({
            destination: state.destination, workspaceTab: state.workspaceTab, setupTab: state.setupTab,
            resultsTab: state.resultsTab, inspectTab: state.inspectTab, audienceMode: state.audienceMode,
            codeExportTab: state.codeExportTab, lessonCueDismissed: state.lessonCueDismissed, hasStartedLesson: state.hasStartedLesson,
        }),
        version: 0,
        merge: (persistedState, currentState) => ({ ...currentState, ...(persistedState == null ? {} : sanitizePersistedLayoutState(persistedState)) }),
    }));
}

const layoutStore = createLayoutStore();
type LayoutStoreHook = {
    (): LayoutStore;
    <T>(selector: (state: LayoutStore) => T): T;
} & typeof layoutStore;
const boundUseLayoutStore = ((selector?: (state: LayoutStore) => unknown) => (
    selector ? useStore(layoutStore, selector) : useStore(layoutStore)
)) as LayoutStoreHook;
export const useLayoutStore = Object.assign(boundUseLayoutStore, layoutStore);
