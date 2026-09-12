// ── Layout Store ──
// Manages local Signal Atelier navigation and legacy layout adapters.
// Persisted to localStorage so the user's workspace view survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { useStore } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';
import type { Destination, WorkspaceTab, SetupTab, ResultsTab, InspectTab } from '../productShell/atelierTypes.ts';
import {
    CODE_EXPORT_TABS,
    EVIDENCE_VIEW_IDS,
    RECIPE_SECTION_IDS,
    WORKSPACE_VIEWS,
    type CodeExportTab,
    type EvidenceViewId,
    type LayoutVariant,
    type LeftTabId,
    type PhaseMode,
    type RecipeSectionId,
    type RightTabId,
    type WorkspaceView,
} from '../productShell/shellTypes.ts';
import {
    getAudienceProfile,
    isAudienceMode,
    type AudienceMode,
    type BuildModuleId,
} from '../productShell/audienceProfiles.ts';
import {
    isEvidenceViewVisible,
    isRecipeSectionVisible,
    resolveVisibleEvidenceView,
    resolveVisibleRecipeSection,
} from '../productShell/visibleShell.ts';

export const LAYOUT_STORAGE_KEY = 'nn-playground-layout-v2';
export const LEGACY_LAYOUT_STORAGE_KEY = 'nn-playground-layout';

export type {
    CodeExportTab,
    EvidenceViewId,
    LayoutVariant,
    LeftTabId,
    PhaseMode,
    RecipeSectionId,
    RightTabId,
    WorkspaceView,
} from '../productShell/shellTypes.ts';

const DEFAULT_LAYOUT_STATE = {
    exportRequest: null as { mode: 'setup' | 'code'; id: number } | null,
    destination: 'playground' as Destination,
    workspaceTab: 'network' as WorkspaceTab,
    setupTab: 'dataset' as SetupTab,
    resultsTab: 'boundary' as ResultsTab,
    inspectTab: 'trace' as InspectTab,
    view: 'build' as WorkspaceView,
    activeRecipeSection: 'data' as RecipeSectionId,
    activeEvidenceView: 'boundary' as EvidenceViewId,
    audienceMode: 'explore' as AudienceMode,
    advancedToolsOpen: false,
    buildContextOpen: false,

    // Deprecated compatibility fields for older helper modules and tests.
    layout: 'dock' as LayoutVariant,
    phase: 'build' as PhaseMode,
    activeTabLeft: 'data' as LeftTabId,
    activeTabRight: 'boundary' as RightTabId,
    codeExportTab: 'pseudocode' as CodeExportTab,
    activeLessonId: null as string | null,
    activeLessonStepIndex: null as number | null,
    lessonCueDismissed: false as boolean,
    hasStartedLesson: false as boolean,
};

const VALID_PHASES = WORKSPACE_VIEWS;
const VALID_RECIPE_SECTIONS = RECIPE_SECTION_IDS;
const VALID_EVIDENCE_VIEWS = EVIDENCE_VIEW_IDS;
const VALID_LEFT_TABS = VALID_RECIPE_SECTIONS;
const VALID_RIGHT_TABS = VALID_EVIDENCE_VIEWS;
const VALID_CODE_EXPORT_TABS = CODE_EXPORT_TABS;

export interface LayoutStore {
    exportRequest: { mode: 'setup' | 'code'; id: number } | null;
    requestExport: (mode: 'setup' | 'code') => void;
    clearExportRequest: () => void;
    destination: Destination;
    workspaceTab: WorkspaceTab;
    setupTab: SetupTab;
    resultsTab: ResultsTab;
    inspectTab: InspectTab;
    navigate: (destination: Destination, tab?: WorkspaceTab) => void;
    openSetup: (tab: SetupTab) => void;
    setSetupTab: (tab: SetupTab) => void;
    setResultsTab: (tab: ResultsTab) => void;
    setInspectTab: (tab: InspectTab) => void;
    view: WorkspaceView;
    activeRecipeSection: RecipeSectionId;
    activeEvidenceView: EvidenceViewId;
    audienceMode: AudienceMode;
    advancedToolsOpen: boolean;
    buildContextOpen: boolean;

    // Deprecated compatibility fields. User-facing UI should prefer destination,
    // workspaceTab, setupTab, resultsTab, and inspectTab.
    layout: LayoutVariant;
    phase: PhaseMode;
    activeTabLeft: LeftTabId;
    activeTabRight: RightTabId;
    codeExportTab: CodeExportTab;
    activeLessonId: string | null;
    activeLessonStepIndex: number | null;
    lessonCueDismissed: boolean;
    hasStartedLesson: boolean;

    setView: (view: WorkspaceView) => void;
    setActiveRecipeSection: (section: RecipeSectionId) => void;
    setActiveEvidenceView: (view: EvidenceViewId) => void;
    setAudienceMode: (mode: AudienceMode) => void;
    setAdvancedToolsOpen: (open: boolean) => void;
    setBuildContextOpen: (open: boolean) => void;
    selectBuildContext: (section: RecipeSectionId) => void;
    openAdvancedRecipeSection: (section: BuildModuleId) => void;

    setLayout: (layout: LayoutVariant) => void;
    setPhase: (phase: PhaseMode) => void;
    setActiveTabLeft: (tab: LeftTabId) => void;
    setActiveTabRight: (tab: RightTabId) => void;
    setCodeExportTab: (tab: CodeExportTab) => void;
    setActiveLessonStep: (lessonId: string, stepIndex: number) => void;
    clearActiveLessonStep: () => void;
    dismissLessonCue: () => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isOneOf<T extends string>(value: unknown, options: readonly T[]): value is T {
    return typeof value === 'string' && options.includes(value as T);
}

function setupFor(section: RecipeSectionId): SetupTab {
    return section === 'hyperparams' ? 'training' : section === 'network' || section === 'features' ? 'network' : 'dataset';
}
function evidenceFor(view: EvidenceViewId): { workspaceTab: WorkspaceTab; resultsTab: ResultsTab } {
    return { workspaceTab: view === 'inspection' ? 'inspect' : 'results', resultsTab: view === 'loss' ? 'learning' : view === 'confusion' ? 'errors' : 'boundary' };
}

function sanitizePersistedLayoutState(value: unknown): typeof DEFAULT_LAYOUT_STATE {
    const state = isRecord(value) && isRecord(value.state) ? value.state : value;
    if (!isRecord(state)) return { ...DEFAULT_LAYOUT_STATE };
    const view = isOneOf(state.view, VALID_PHASES)
        ? state.view
        : isOneOf(state.phase, VALID_PHASES)
            ? state.phase
            : DEFAULT_LAYOUT_STATE.view;
    const activeRecipeSection = isOneOf(state.activeRecipeSection, VALID_RECIPE_SECTIONS)
        ? state.activeRecipeSection
        : isOneOf(state.activeTabLeft, VALID_LEFT_TABS)
            ? state.activeTabLeft
            : DEFAULT_LAYOUT_STATE.activeRecipeSection;
    const activeEvidenceView = isOneOf(state.activeEvidenceView, VALID_EVIDENCE_VIEWS)
        ? state.activeEvidenceView
        : isOneOf(state.activeTabRight, VALID_RIGHT_TABS)
            ? state.activeTabRight
            : DEFAULT_LAYOUT_STATE.activeEvidenceView;
    const audienceMode = isAudienceMode(state.audienceMode)
        ? state.audienceMode
        : DEFAULT_LAYOUT_STATE.audienceMode;
    const requestedAdvancedToolsOpen = typeof state.advancedToolsOpen === 'boolean'
        ? state.advancedToolsOpen
        : getAudienceProfile(audienceMode).advancedDefaultOpen;
    const advancedToolsOpen = requestedAdvancedToolsOpen
        || !isRecipeSectionVisible(audienceMode, false, activeRecipeSection)
        || !isEvidenceViewVisible(audienceMode, false, activeEvidenceView);

    return {
        exportRequest: view === 'run' && activeEvidenceView === 'code' ? { mode: 'code', id: 1 } : activeRecipeSection === 'config' ? { mode: 'setup', id: 1 } : activeEvidenceView === 'code' ? { mode: 'code', id: 1 } : null,
        destination: isOneOf(state.destination, ['playground', 'saved-runs', 'lessons']) ? state.destination : 'playground',
        workspaceTab: isOneOf(state.workspaceTab, ['setup', 'network', 'results', 'inspect']) ? state.workspaceTab
            : view === 'build' ? 'setup' : evidenceFor(activeEvidenceView).workspaceTab,
        setupTab: isOneOf(state.setupTab, ['dataset', 'network', 'training']) ? state.setupTab : setupFor(activeRecipeSection),
        resultsTab: isOneOf(state.resultsTab, ['boundary', 'learning', 'errors']) ? state.resultsTab : evidenceFor(activeEvidenceView).resultsTab,
        inspectTab: isOneOf(state.inspectTab, ['trace', 'activations', 'gradients']) ? state.inspectTab : 'trace',
        view,
        activeRecipeSection: activeRecipeSection === 'config' ? 'data' : activeRecipeSection,
        activeEvidenceView: activeEvidenceView === 'code' ? 'boundary' : activeEvidenceView,
        audienceMode,
        advancedToolsOpen,
        buildContextOpen: false,
        layout: DEFAULT_LAYOUT_STATE.layout,
        phase: view,
        activeTabLeft: activeRecipeSection === 'config' ? 'data' : activeRecipeSection,
        activeTabRight: activeEvidenceView === 'code' ? 'boundary' : activeEvidenceView,
        codeExportTab: isOneOf(state.codeExportTab, VALID_CODE_EXPORT_TABS)
            ? state.codeExportTab
            : DEFAULT_LAYOUT_STATE.codeExportTab,
        activeLessonId: null,
        activeLessonStepIndex: null,
        lessonCueDismissed: typeof state.lessonCueDismissed === 'boolean'
            ? state.lessonCueDismissed
            : DEFAULT_LAYOUT_STATE.lessonCueDismissed,
        hasStartedLesson: typeof state.hasStartedLesson === 'boolean'
            ? state.hasStartedLesson
            : DEFAULT_LAYOUT_STATE.hasStartedLesson,
    };
}

export function createLayoutStore() {
    let initial = DEFAULT_LAYOUT_STATE;
    try {
        if (typeof window !== 'undefined' && !window.localStorage.getItem(LAYOUT_STORAGE_KEY)) {
            const legacy = window.localStorage.getItem(LEGACY_LAYOUT_STORAGE_KEY);
            if (legacy) initial = sanitizePersistedLayoutState(JSON.parse(legacy));
        }
    } catch { /* Preferences must never prevent an experiment from opening. */ }
    return createStore<LayoutStore>()(
        persist(
            (set) => ({
                ...initial,
                requestExport: (mode) => set((state) => ({ exportRequest: { mode, id: (state.exportRequest?.id ?? 0) + 1 } })),
                clearExportRequest: () => set({ exportRequest: null }),
                navigate: (destination, workspaceTab) => set((state) => ({
                    destination, workspaceTab: workspaceTab ?? state.workspaceTab,
                    ...(workspaceTab ? { view: workspaceTab === 'setup' ? 'build' as const : 'run' as const, phase: workspaceTab === 'setup' ? 'build' as const : 'run' as const } : {}),
                })),
                openSetup: (setupTab) => set({ destination: 'playground', workspaceTab: 'setup', setupTab, view: 'build', phase: 'build' }),
                setSetupTab: (setupTab) => set({ setupTab }),
                setResultsTab: (resultsTab) => set({ resultsTab, activeEvidenceView: resultsTab === 'learning' ? 'loss' : resultsTab === 'errors' ? 'confusion' : 'boundary' }),
                setInspectTab: (inspectTab) => set({ inspectTab }),

                setView: (view) => set((state) => ({
                    destination: 'playground', workspaceTab: view === 'build' ? 'setup' : evidenceFor(state.activeEvidenceView).workspaceTab,
                    view,
                    phase: view,
                    buildContextOpen: view === 'run' ? false : state.buildContextOpen,
                })),
                setActiveRecipeSection: (activeRecipeSection) => set((state) => activeRecipeSection === 'config' ? { exportRequest: { mode: 'setup' as const, id: (state.exportRequest?.id ?? 0) + 1 } } : ({
                    destination: 'playground', workspaceTab: 'setup', setupTab: setupFor(activeRecipeSection),
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                    buildContextOpen: true,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isRecipeSectionVisible(state.audienceMode, false, activeRecipeSection),
                })),
                setActiveEvidenceView: (activeEvidenceView) => set((state) => activeEvidenceView === 'code' ? { exportRequest: { mode: 'code' as const, id: (state.exportRequest?.id ?? 0) + 1 } } : ({
                    destination: 'playground', ...evidenceFor(activeEvidenceView),
                    activeEvidenceView,
                    activeTabRight: activeEvidenceView,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isEvidenceViewVisible(state.audienceMode, false, activeEvidenceView),
                })),
                setAudienceMode: (audienceMode) => set((state) => {
                    const advancedToolsOpen = getAudienceProfile(audienceMode).advancedDefaultOpen;
                    const activeRecipeSection = resolveVisibleRecipeSection(
                        audienceMode,
                        advancedToolsOpen,
                        state.activeRecipeSection,
                    );
                    const activeEvidenceView = isEvidenceViewVisible(
                        audienceMode,
                        advancedToolsOpen,
                        state.activeEvidenceView,
                    )
                        ? state.activeEvidenceView
                        : resolveVisibleEvidenceView(
                            audienceMode,
                            advancedToolsOpen,
                            state.activeEvidenceView,
                        );
                    return {
                        audienceMode,
                        advancedToolsOpen,
                        activeRecipeSection,
                        activeTabLeft: activeRecipeSection,
                        activeEvidenceView,
                        activeTabRight: activeEvidenceView,
                    };
                }),
                setAdvancedToolsOpen: (advancedToolsOpen) => set((state) => {
                    if (advancedToolsOpen) return { advancedToolsOpen: true };
                    const activeRecipeSection = resolveVisibleRecipeSection(
                        state.audienceMode,
                        false,
                        state.activeRecipeSection,
                    );
                    const activeEvidenceView = isEvidenceViewVisible(
                        state.audienceMode,
                        false,
                        state.activeEvidenceView,
                    )
                        ? state.activeEvidenceView
                        : resolveVisibleEvidenceView(
                            state.audienceMode,
                            false,
                            state.activeEvidenceView,
                        );
                    return {
                        advancedToolsOpen: false,
                        activeRecipeSection,
                        activeTabLeft: activeRecipeSection,
                        activeEvidenceView,
                        activeTabRight: activeEvidenceView,
                    };
                }),
                setBuildContextOpen: (buildContextOpen) => set({ buildContextOpen }),
                selectBuildContext: (activeRecipeSection) => set((state) => activeRecipeSection === 'config' ? { exportRequest: { mode: 'setup' as const, id: (state.exportRequest?.id ?? 0) + 1 } } : ({
                    destination: 'playground', workspaceTab: 'setup', setupTab: setupFor(activeRecipeSection),
                    view: 'build',
                    phase: 'build',
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                    buildContextOpen: true,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isRecipeSectionVisible(state.audienceMode, false, activeRecipeSection),
                })),
                openAdvancedRecipeSection: (activeRecipeSection) => set(activeRecipeSection === 'config' ? { exportRequest: { mode: 'setup', id: Date.now() } } : {
                    destination: 'playground', workspaceTab: 'setup', setupTab: setupFor(activeRecipeSection),
                    view: 'build',
                    phase: 'build',
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                    buildContextOpen: true,
                    advancedToolsOpen: true,
                }),

                setLayout: (layout) => set({ layout }),
                setPhase: (phase) => set((state) => ({
                    destination: 'playground', workspaceTab: phase === 'build' ? 'setup' : evidenceFor(state.activeEvidenceView).workspaceTab,
                    view: phase,
                    phase,
                    buildContextOpen: phase === 'run' ? false : state.buildContextOpen,
                })),
                setActiveTabLeft: (activeTabLeft) => set((state) => activeTabLeft === 'config' ? { exportRequest: { mode: 'setup' as const, id: (state.exportRequest?.id ?? 0) + 1 } } : ({
                    destination: 'playground', workspaceTab: 'setup', setupTab: setupFor(activeTabLeft),
                    activeRecipeSection: activeTabLeft,
                    activeTabLeft,
                    buildContextOpen: true,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isRecipeSectionVisible(state.audienceMode, false, activeTabLeft),
                })),
                setActiveTabRight: (activeTabRight) => set((state) => activeTabRight === 'code' ? { exportRequest: { mode: 'code' as const, id: (state.exportRequest?.id ?? 0) + 1 } } : ({
                    destination: 'playground', ...evidenceFor(activeTabRight),
                    activeEvidenceView: activeTabRight,
                    activeTabRight,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isEvidenceViewVisible(state.audienceMode, false, activeTabRight),
                })),
                setCodeExportTab: (codeExportTab) => set({ codeExportTab }),
                setActiveLessonStep: (activeLessonId, activeLessonStepIndex) => set({
                    activeLessonId,
                    activeLessonStepIndex,
                    hasStartedLesson: true,
                }),
                clearActiveLessonStep: () => set({
                    activeLessonId: null,
                    activeLessonStepIndex: null,
                }),
                dismissLessonCue: () => set({ lessonCueDismissed: true }),
            }),
            {
                name: LAYOUT_STORAGE_KEY,
                storage: createJSONStorage(() => ({
                    getItem: (name) => { try { return window.localStorage.getItem(name); } catch { return null; } },
                    setItem: (name, value) => { try { window.localStorage.setItem(name, value); } catch { /* Keep navigation usable for this session. */ } },
                    removeItem: (name) => { try { window.localStorage.removeItem(name); } catch { /* Session-only preferences remain usable. */ } },
                })),
                partialize: (state) => ({
                    destination: state.destination, workspaceTab: state.workspaceTab, setupTab: state.setupTab, resultsTab: state.resultsTab, inspectTab: state.inspectTab,
                    view: state.view,
                    activeRecipeSection: state.activeRecipeSection,
                    activeEvidenceView: state.activeEvidenceView,
                    codeExportTab: state.codeExportTab,
                    audienceMode: state.audienceMode,
                    advancedToolsOpen: state.advancedToolsOpen,
                    lessonCueDismissed: state.lessonCueDismissed,
                    hasStartedLesson: state.hasStartedLesson,
                }),
                version: 0,
                merge: (persistedState, currentState) => ({
                    ...currentState,
                    ...(persistedState == null ? {} : sanitizePersistedLayoutState(persistedState)),
                }),
            },
        ),
    );
}

const layoutStore = createLayoutStore();

type LayoutSelector<T> = (state: LayoutStore) => T;
type LayoutStoreHook = {
    (): LayoutStore;
    <T>(selector: LayoutSelector<T>): T;
} & typeof layoutStore;

const boundUseLayoutStore = ((selector?: LayoutSelector<unknown>) => (
    selector ? useStore(layoutStore, selector) : useStore(layoutStore)
)) as LayoutStoreHook;

export const useLayoutStore = Object.assign(boundUseLayoutStore, layoutStore);
