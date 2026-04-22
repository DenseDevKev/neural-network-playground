// ── Layout Store ──
// Manages UI layout state: which variant is active, phase, tab selections.
// Persisted to localStorage so the user's layout choice survives reloads.
// Deliberately separated from usePlaygroundStore (config) and
// useTrainingStore (runtime) — layout is a pure UI concern.

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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

export const useLayoutStore = create<LayoutStore>()(
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
        { name: 'nn-playground-layout' },
    ),
);
