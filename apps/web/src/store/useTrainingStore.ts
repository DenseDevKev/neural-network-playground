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
import {
    appendHistoryPoint,
    resetHistoryBuffer,
    seedHistory,
} from './historyBuffer.ts';

type ConfigChangeSource = 'data' | 'network' | null;

export interface TrainingStore {
    // ── Runtime State ──
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    /**
     * @deprecated Kept for test compatibility only. Production code reads
     * training history from `historyBuffer.readHistory()` and subscribes
     * to `historyVersion` below. Writes to this field via setState are
     * mirrored into the ring buffer (see `seedHistory`).
     */
    history: HistoryPoint[];
    /** Monotonic counter — bumped every time `historyBuffer` is mutated. */
    historyVersion: number;
    frameVersion: number;
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    /** Steps of training to run per animation frame. */
    stepsPerFrame: number;
    dataConfigLoading: boolean;
    networkConfigLoading: boolean;
    pendingConfigSource: ConfigChangeSource;
    configError: string | null;
    configErrorSource: ConfigChangeSource;
    configSyncNonce: number;
    workerError: string | null;
    /** True when the most recent streamed snapshot reused cached test metrics. */
    testMetricsStale: boolean;

    // ── Actions ──
    setStatus: (s: TrainingStatus) => void;
    setSnapshot: (snap: NetworkSnapshot) => void;
    addHistoryPoint: (point: HistoryPoint) => void;
    resetHistory: () => void;
    setFrameVersion: (version: number) => void;
    setTrainPoints: (pts: DataPoint[]) => void;
    setTestPoints: (pts: DataPoint[]) => void;
    setStepsPerFrame: (n: number) => void;
    beginConfigChange: (source: Exclude<ConfigChangeSource, null>) => void;
    finishConfigChange: () => void;
    failConfigChange: (message: string) => void;
    retryConfigSync: () => void;
    setWorkerError: (message: string) => void;
    clearWorkerError: () => void;
    setTestMetricsStale: (stale: boolean) => void;
}

// Stable reference-equal empty array — handed out as the default `history`
// field so consumers that still read it see a consistent identity until
// they migrate to the ring-buffer API.
const EMPTY_HISTORY: readonly HistoryPoint[] = Object.freeze([]);

export const useTrainingStore = create<TrainingStore>((set) => ({
    status: 'idle',
    snapshot: null,
    history: EMPTY_HISTORY as HistoryPoint[],
    historyVersion: 0,
    frameVersion: 0,
    trainPoints: [],
    testPoints: [],
    stepsPerFrame: 5,
    dataConfigLoading: false,
    networkConfigLoading: false,
    pendingConfigSource: null,
    configError: null,
    configErrorSource: null,
    configSyncNonce: 0,
    workerError: null,
    testMetricsStale: false,

    setStatus: (status) => set({ status }),
    setSnapshot: (snapshot) => set({ snapshot }),
    addHistoryPoint: (point) => {
        // Append to the packed ring buffer and publish the new version.
        // No array is allocated per frame; chart components pull data
        // from historyBuffer.readHistory() on their own cadence.
        const version = appendHistoryPoint(point);
        set({ historyVersion: version });
    },
    resetHistory: () => {
        const version = resetHistoryBuffer();
        set({ historyVersion: version, history: EMPTY_HISTORY as HistoryPoint[] });
    },
    setFrameVersion: (frameVersion) => set({ frameVersion }),
    setTrainPoints: (trainPoints) => set({ trainPoints }),
    setTestPoints: (testPoints) => set({ testPoints }),
    setStepsPerFrame: (n) => set({ stepsPerFrame: Math.max(1, Math.min(100, n)) }),
    beginConfigChange: (source) => set({
        pendingConfigSource: source,
        dataConfigLoading: source === 'data',
        networkConfigLoading: source === 'network',
        configError: null,
        configErrorSource: null,
        workerError: null,
    }),
    finishConfigChange: () => set({
        pendingConfigSource: null,
        dataConfigLoading: false,
        networkConfigLoading: false,
    }),
    failConfigChange: (message) => set((state) => ({
        pendingConfigSource: null,
        dataConfigLoading: false,
        networkConfigLoading: false,
        configError: message,
        configErrorSource: state.pendingConfigSource,
    })),
    retryConfigSync: () => set((state) => {
        if (!state.configErrorSource) {
            return {};
        }

        return {
            pendingConfigSource: state.configErrorSource,
            dataConfigLoading: state.configErrorSource === 'data',
            networkConfigLoading: state.configErrorSource === 'network',
            configError: null,
            configErrorSource: null,
            configSyncNonce: state.configSyncNonce + 1,
        };
    }),
    setWorkerError: (message) => set({ workerError: message }),
    clearWorkerError: () => set({ workerError: null }),
    setTestMetricsStale: (testMetricsStale) => set({ testMetricsStale }),
}));

// ── Ring-buffer mirror for direct history writes ────────────────────────────
// Tests still drive the store via `setState({ history: [...] })`. To keep
// them working without edits — and to keep LossChart always reading from
// the packed buffer — any direct write to `history` is mirrored into the
// buffer here. Production code uses `addHistoryPoint` / `resetHistory`
// which bypass `history` entirely, so this path is inert at runtime.
useTrainingStore.subscribe((state, prevState) => {
    if (state.history !== prevState.history) {
        const version = seedHistory(state.history);
        if (state.historyVersion !== version) {
            // Propagate the new version without triggering the subscriber
            // to recurse: history ref is unchanged on this follow-up set.
            useTrainingStore.setState({ historyVersion: version });
        }
    }
});
