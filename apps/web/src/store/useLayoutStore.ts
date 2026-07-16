// ── Layout Store ──
// Manages pure UI state for the Build / Run instrument shell.
// Persisted to localStorage so the user's workspace view survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { useStore } from 'zustand';
import { persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';
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

export const LAYOUT_STORAGE_KEY = 'nn-playground-layout';

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
    view: 'build' as WorkspaceView,
    activeRecipeSection: 'data' as RecipeSectionId,
    activeEvidenceView: 'boundary' as EvidenceViewId,
    audienceMode: 'explore' as AudienceMode,
    advancedToolsOpen: false,

    // Deprecated compatibility fields for older helper modules and tests.
    layout: 'dock' as LayoutVariant,
    phase: 'build' as PhaseMode,
    activeTabLeft: 'data' as LeftTabId,
    activeTabRight: 'boundary' as RightTabId,
    codeExportTab: 'pseudocode' as CodeExportTab,
    activeLessonId: null as string | null,
    activeLessonStepIndex: null as number | null,
};

const VALID_PHASES = WORKSPACE_VIEWS;
const VALID_RECIPE_SECTIONS = RECIPE_SECTION_IDS;
const VALID_EVIDENCE_VIEWS = EVIDENCE_VIEW_IDS;
const VALID_LEFT_TABS = VALID_RECIPE_SECTIONS;
const VALID_RIGHT_TABS = VALID_EVIDENCE_VIEWS;
const VALID_CODE_EXPORT_TABS = CODE_EXPORT_TABS;

export interface LayoutStore {
    view: WorkspaceView;
    activeRecipeSection: RecipeSectionId;
    activeEvidenceView: EvidenceViewId;
    audienceMode: AudienceMode;
    advancedToolsOpen: boolean;

    // Deprecated compatibility fields. User-facing UI should prefer view,
    // activeRecipeSection, and activeEvidenceView.
    layout: LayoutVariant;
    phase: PhaseMode;
    activeTabLeft: LeftTabId;
    activeTabRight: RightTabId;
    codeExportTab: CodeExportTab;
    activeLessonId: string | null;
    activeLessonStepIndex: number | null;

    setView: (view: WorkspaceView) => void;
    setActiveRecipeSection: (section: RecipeSectionId) => void;
    setActiveEvidenceView: (view: EvidenceViewId) => void;
    setAudienceMode: (mode: AudienceMode) => void;
    setAdvancedToolsOpen: (open: boolean) => void;
    openAdvancedRecipeSection: (section: BuildModuleId) => void;

    setLayout: (layout: LayoutVariant) => void;
    setPhase: (phase: PhaseMode) => void;
    setActiveTabLeft: (tab: LeftTabId) => void;
    setActiveTabRight: (tab: RightTabId) => void;
    setCodeExportTab: (tab: CodeExportTab) => void;
    setActiveLessonStep: (lessonId: string, stepIndex: number) => void;
    clearActiveLessonStep: () => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function isOneOf<T extends string>(value: unknown, options: readonly T[]): value is T {
    return typeof value === 'string' && options.includes(value as T);
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
        view,
        activeRecipeSection,
        activeEvidenceView,
        audienceMode,
        advancedToolsOpen,
        layout: DEFAULT_LAYOUT_STATE.layout,
        phase: view,
        activeTabLeft: activeRecipeSection,
        activeTabRight: activeEvidenceView,
        codeExportTab: isOneOf(state.codeExportTab, VALID_CODE_EXPORT_TABS)
            ? state.codeExportTab
            : DEFAULT_LAYOUT_STATE.codeExportTab,
        activeLessonId: null,
        activeLessonStepIndex: null,
    };
}

export function createLayoutStore() {
    return createStore<LayoutStore>()(
        persist(
            (set) => ({
                ...DEFAULT_LAYOUT_STATE,

                setView: (view) => set({ view, phase: view }),
                setActiveRecipeSection: (activeRecipeSection) => set((state) => ({
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isRecipeSectionVisible(state.audienceMode, false, activeRecipeSection),
                })),
                setActiveEvidenceView: (activeEvidenceView) => set((state) => ({
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
                openAdvancedRecipeSection: (activeRecipeSection) => set({
                    view: 'build',
                    phase: 'build',
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                    advancedToolsOpen: true,
                }),

                setLayout: (layout) => set({ layout }),
                setPhase: (phase) => set({ view: phase, phase }),
                setActiveTabLeft: (activeTabLeft) => set((state) => ({
                    activeRecipeSection: activeTabLeft,
                    activeTabLeft,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isRecipeSectionVisible(state.audienceMode, false, activeTabLeft),
                })),
                setActiveTabRight: (activeTabRight) => set((state) => ({
                    activeEvidenceView: activeTabRight,
                    activeTabRight,
                    advancedToolsOpen: state.advancedToolsOpen
                        || !isEvidenceViewVisible(state.audienceMode, false, activeTabRight),
                })),
                setCodeExportTab: (codeExportTab) => set({ codeExportTab }),
                setActiveLessonStep: (activeLessonId, activeLessonStepIndex) => set({
                    activeLessonId,
                    activeLessonStepIndex,
                }),
                clearActiveLessonStep: () => set({
                    activeLessonId: null,
                    activeLessonStepIndex: null,
                }),
            }),
            {
                name: LAYOUT_STORAGE_KEY,
                partialize: (state) => ({
                    view: state.view,
                    activeRecipeSection: state.activeRecipeSection,
                    activeEvidenceView: state.activeEvidenceView,
                    codeExportTab: state.codeExportTab,
                    audienceMode: state.audienceMode,
                    advancedToolsOpen: state.advancedToolsOpen,
                }),
                merge: (persistedState, currentState) => ({
                    ...currentState,
                    ...sanitizePersistedLayoutState(persistedState),
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
