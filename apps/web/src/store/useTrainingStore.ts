// ── Training Store ──
// Volatile runtime state that changes every frame during training.
// Separated from usePlaygroundStore to prevent sidebar re-renders during training.

import { create } from 'zustand';
import type {
    NetworkSnapshot,
    HistoryPoint,
    DataPoint,
} from '@nn-playground/engine';
import type { TrainingStatus } from '@nn-playground/shared';

export interface TrainingStore {
    // ── Runtime State ──
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    history: HistoryPoint[];
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    /** Steps of training to run per animation frame. */
    stepsPerFrame: number;

    // ── Actions ──
    setStatus: (s: TrainingStatus) => void;
    setSnapshot: (snap: NetworkSnapshot) => void;
    addHistoryPoint: (point: HistoryPoint) => void;
    resetHistory: () => void;
    setTrainPoints: (pts: DataPoint[]) => void;
    setTestPoints: (pts: DataPoint[]) => void;
    setStepsPerFrame: (n: number) => void;
}

export const useTrainingStore = create<TrainingStore>((set) => ({
    status: 'idle',
    snapshot: null,
    history: [],
    trainPoints: [],
    testPoints: [],
    stepsPerFrame: 5,

    setStatus: (status) => set({ status }),
    setSnapshot: (snapshot) => set({ snapshot }),
    addHistoryPoint: (point) => set((s) => {
        const next = [...s.history, point];
        // Cap at 2000 points — downsample first half to keep chart readable
        if (next.length > 2000) {
            const half = Math.floor(next.length / 2);
            const downsampled = next.filter((_, i) => i >= half || i % 2 === 0);
            return { history: downsampled };
        }
        return { history: next };
    }),
    resetHistory: () => set({ history: [], snapshot: null }),
    setTrainPoints: (trainPoints) => set({ trainPoints }),
    setTestPoints: (testPoints) => set({ testPoints }),
    setStepsPerFrame: (n) => set({ stepsPerFrame: Math.max(1, Math.min(100, n)) }),
}));
