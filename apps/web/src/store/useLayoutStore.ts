// ── Layout Store ──
// Manages UI layout state: which variant is active, phase, and tab selections.
// Persisted to localStorage so the user's layout choice survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { useStore } from 'zustand';
import { persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';

export const LAYOUT_STORAGE_KEY = 'nn-playground-layout';

export type LayoutVariant = 'dock' | 'focus' | 'grid' | 'split';
export type PhaseMode = 'build' | 'run';

export type LeftTabId = 'presets' | 'data' | 'features' | 'network' | 'hyperparams' | 'config';
export type RightTabId = 'boundary' | 'loss' | 'confusion' | 'inspection' | 'code' | 'history';
export type CodeExportTab = 'pseudocode' | 'numpy' | 'tfjs';

const DEFAULT_LAYOUT_STATE = {
    layout: 'dock' as LayoutVariant,
    phase: 'build' as PhaseMode,
    activeTabLeft: 'data' as LeftTabId,
    activeTabRight: 'boundary' as RightTabId,
    codeExportTab: 'pseudocode' as CodeExportTab,
    activeLessonId: null as string | null,
    activeLessonStepIndex: null as number | null,
};

const VALID_LAYOUTS: readonly LayoutVariant[] = ['dock', 'focus', 'grid', 'split'];
const VALID_PHASES: readonly PhaseMode[] = ['build', 'run'];
const VALID_LEFT_TABS: readonly LeftTabId[] = ['presets', 'data', 'features', 'network', 'hyperparams', 'config'];
const VALID_RIGHT_TABS: readonly RightTabId[] = ['boundary', 'loss', 'confusion', 'inspection', 'code', 'history'];
const VALID_CODE_EXPORT_TABS: readonly CodeExportTab[] = ['pseudocode', 'numpy', 'tfjs'];

export interface LayoutStore {
    layout: LayoutVariant;
    phase: PhaseMode;
    activeTabLeft: LeftTabId;
    activeTabRight: RightTabId;
    codeExportTab: CodeExportTab;
    activeLessonId: string | null;
    activeLessonStepIndex: number | null;

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

    return {
        layout: isOneOf(state.layout, VALID_LAYOUTS) ? state.layout : DEFAULT_LAYOUT_STATE.layout,
        phase: isOneOf(state.phase, VALID_PHASES) ? state.phase : DEFAULT_LAYOUT_STATE.phase,
        activeTabLeft: isOneOf(state.activeTabLeft, VALID_LEFT_TABS)
            ? state.activeTabLeft
            : DEFAULT_LAYOUT_STATE.activeTabLeft,
        activeTabRight: isOneOf(state.activeTabRight, VALID_RIGHT_TABS)
            ? state.activeTabRight
            : DEFAULT_LAYOUT_STATE.activeTabRight,
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

                setLayout: (layout) => set({ layout }),
                setPhase: (phase) => set({ phase }),
                setActiveTabLeft: (activeTabLeft) => set({ activeTabLeft }),
                setActiveTabRight: (activeTabRight) => set({ activeTabRight }),
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
                    layout: state.layout,
                    phase: state.phase,
                    activeTabLeft: state.activeTabLeft,
                    activeTabRight: state.activeTabRight,
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
