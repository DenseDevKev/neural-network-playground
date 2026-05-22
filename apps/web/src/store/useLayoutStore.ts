// ── Layout Store ──
// Manages pure UI state for the Build / Run instrument shell.
// Persisted to localStorage so the user's workspace view survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { useStore } from 'zustand';
import { persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';

export const LAYOUT_STORAGE_KEY = 'nn-playground-layout';

export type WorkspaceView = 'build' | 'run';
export type LayoutVariant = 'dock' | 'focus' | 'grid' | 'split';
export type PhaseMode = 'build' | 'run';

export type RecipeSectionId = 'presets' | 'data' | 'features' | 'network' | 'hyperparams' | 'config';
export type EvidenceViewId = 'boundary' | 'loss' | 'confusion' | 'inspection' | 'code' | 'history';
export type LeftTabId = RecipeSectionId;
export type RightTabId = EvidenceViewId;
export type CodeExportTab = 'pseudocode' | 'numpy' | 'tfjs';

const DEFAULT_LAYOUT_STATE = {
    view: 'build' as WorkspaceView,
    activeRecipeSection: 'data' as RecipeSectionId,
    activeEvidenceView: 'boundary' as EvidenceViewId,

    // Deprecated compatibility fields for older helper modules and tests.
    layout: 'dock' as LayoutVariant,
    phase: 'build' as PhaseMode,
    activeTabLeft: 'data' as LeftTabId,
    activeTabRight: 'boundary' as RightTabId,
    codeExportTab: 'pseudocode' as CodeExportTab,
    activeLessonId: null as string | null,
    activeLessonStepIndex: null as number | null,
};

const VALID_PHASES: readonly PhaseMode[] = ['build', 'run'];
const VALID_RECIPE_SECTIONS: readonly RecipeSectionId[] = ['presets', 'data', 'features', 'network', 'hyperparams', 'config'];
const VALID_EVIDENCE_VIEWS: readonly EvidenceViewId[] = ['boundary', 'loss', 'confusion', 'inspection', 'code', 'history'];
const VALID_LEFT_TABS = VALID_RECIPE_SECTIONS;
const VALID_RIGHT_TABS = VALID_EVIDENCE_VIEWS;
const VALID_CODE_EXPORT_TABS: readonly CodeExportTab[] = ['pseudocode', 'numpy', 'tfjs'];

export interface LayoutStore {
    view: WorkspaceView;
    activeRecipeSection: RecipeSectionId;
    activeEvidenceView: EvidenceViewId;

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

    return {
        view,
        activeRecipeSection,
        activeEvidenceView,
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
                setActiveRecipeSection: (activeRecipeSection) => set({
                    activeRecipeSection,
                    activeTabLeft: activeRecipeSection,
                }),
                setActiveEvidenceView: (activeEvidenceView) => set({
                    activeEvidenceView,
                    activeTabRight: activeEvidenceView,
                }),

                setLayout: (layout) => set({ layout }),
                setPhase: (phase) => set({ view: phase, phase }),
                setActiveTabLeft: (activeTabLeft) => set({
                    activeRecipeSection: activeTabLeft,
                    activeTabLeft,
                }),
                setActiveTabRight: (activeTabRight) => set({
                    activeEvidenceView: activeTabRight,
                    activeTabRight,
                }),
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
