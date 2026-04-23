// ── Layout Store ──
// Manages UI layout state: which variant is active, phase, and tab selections.
// Persisted to localStorage so the user's layout choice survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { useStore } from 'zustand';
import { persist } from 'zustand/middleware';
import { createStore } from 'zustand/vanilla';

export const LAYOUT_STORAGE_KEY = 'nn-playground-layout';

export type LayoutVariant = 'dock' | 'grid' | 'split';
export type PhaseMode = 'build' | 'run';

export type LeftTabId = 'presets' | 'data' | 'features' | 'network' | 'hyperparams' | 'config';
export type RightTabId = 'boundary' | 'loss' | 'confusion' | 'inspection' | 'code';

export interface LayoutStore {
    layout: LayoutVariant;
    phase: PhaseMode;
    activeTabLeft: LeftTabId;
    activeTabRight: RightTabId;

    setLayout: (layout: LayoutVariant) => void;
    setPhase: (phase: PhaseMode) => void;
    setActiveTabLeft: (tab: LeftTabId) => void;
    setActiveTabRight: (tab: RightTabId) => void;
}

export function createLayoutStore() {
    return createStore<LayoutStore>()(
        persist(
            (set) => ({
                layout: 'dock',
                phase: 'build',
                activeTabLeft: 'data',
                activeTabRight: 'boundary',

                setLayout: (layout) => set({ layout }),
                setPhase: (phase) => set({ phase }),
                setActiveTabLeft: (activeTabLeft) => set({ activeTabLeft }),
                setActiveTabRight: (activeTabRight) => set({ activeTabRight }),
            }),
            { name: LAYOUT_STORAGE_KEY },
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
