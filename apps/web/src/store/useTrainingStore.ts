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
} from './historyBuffer.ts';
import { normalizeTrainingSpeed } from '../worker/trainingLoop.ts';
import type { FrameVersions } from '../worker/frameBuffer.ts';

export type ConfigChangeSource = 'data' | 'network' | 'features' | 'training' | null;

export interface TrainingStore {
    // ── Runtime State ──
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    /** Monotonic counter — bumped every time `historyBuffer` is mutated. */
    historyVersion: number;
    frameVersion: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    /** Steps of training to run per animation frame. */
    stepsPerFrame: number;
    dataConfigLoading: boolean;
    networkConfigLoading: boolean;
    featuresConfigLoading: boolean;
    trainingConfigLoading: boolean;
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
    applyStreamedSnapshot: (payload: {
        snapshot: NetworkSnapshot;
        frameVersions: FrameVersions;
        testMetricsStale: boolean;
    }) => void;
    resetHistory: () => void;
    setFrameVersion: (version: number) => void;
    setFrameVersions: (versions: FrameVersions) => void;
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

export const useTrainingStore = create<TrainingStore>((set) => ({
    status: 'idle',
    snapshot: null,
    historyVersion: 0,
    frameVersion: 0,
    outputGridVersion: 0,
    neuronGridsVersion: 0,
    paramsVersion: 0,
    layerStatsVersion: 0,
    confusionMatrixVersion: 0,
    trainPoints: [],
    testPoints: [],
    stepsPerFrame: 5,
    dataConfigLoading: false,
    networkConfigLoading: false,
    featuresConfigLoading: false,
    trainingConfigLoading: false,
    pendingConfigSource: null,
    configError: null,
    configErrorSource: null,
    configSyncNonce: 0,
    workerError: null,
    testMetricsStale: false,

    setStatus: (status) => set({ status }),
    setSnapshot: (snapshot) => set({ snapshot }),
    applyStreamedSnapshot: ({ snapshot, frameVersions, testMetricsStale }) => {
        set((state) => {
            const historyVersion = snapshot.historyPoint
                ? appendHistoryPoint(snapshot.historyPoint)
                : state.historyVersion;

            return {
                snapshot,
                frameVersion: frameVersions.frameVersion,
                outputGridVersion: frameVersions.outputGridVersion,
                neuronGridsVersion: frameVersions.neuronGridsVersion,
                paramsVersion: frameVersions.paramsVersion,
                layerStatsVersion: frameVersions.layerStatsVersion,
                confusionMatrixVersion: frameVersions.confusionMatrixVersion,
                historyVersion,
                testMetricsStale,
                workerError: null,
            };
        });
    },
    addHistoryPoint: (point) => {
        // Append to the packed ring buffer and publish the new version.
        // No array is allocated per frame; chart components pull data
        // from historyBuffer.readHistory() on their own cadence.
        const version = appendHistoryPoint(point);
        set({ historyVersion: version });
    },
    resetHistory: () => {
        const version = resetHistoryBuffer();
        set({ historyVersion: version });
    },
    setFrameVersion: (frameVersion) => set({ frameVersion }),
    setFrameVersions: (versions) => set({
        frameVersion: versions.frameVersion,
        outputGridVersion: versions.outputGridVersion,
        neuronGridsVersion: versions.neuronGridsVersion,
        paramsVersion: versions.paramsVersion,
        layerStatsVersion: versions.layerStatsVersion,
        confusionMatrixVersion: versions.confusionMatrixVersion,
    }),
    setTrainPoints: (trainPoints) => set({ trainPoints }),
    setTestPoints: (testPoints) => set({ testPoints }),
    setStepsPerFrame: (n) => set({ stepsPerFrame: normalizeTrainingSpeed(n) }),
    beginConfigChange: (source) => set({
        pendingConfigSource: source,
        dataConfigLoading: source === 'data',
        networkConfigLoading: source === 'network',
        featuresConfigLoading: source === 'features',
        trainingConfigLoading: source === 'training',
        configError: null,
        configErrorSource: null,
        workerError: null,
    }),
    finishConfigChange: () => set({
        pendingConfigSource: null,
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
    }),
    failConfigChange: (message) => set((state) => ({
        pendingConfigSource: null,
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
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
            featuresConfigLoading: state.configErrorSource === 'features',
            trainingConfigLoading: state.configErrorSource === 'training',
            configError: null,
            configErrorSource: null,
            configSyncNonce: state.configSyncNonce + 1,
        };
    }),
    setWorkerError: (message) => set({ workerError: message }),
    clearWorkerError: () => set({ workerError: null }),
    setTestMetricsStale: (testMetricsStale) => set({ testMetricsStale }),
}));
