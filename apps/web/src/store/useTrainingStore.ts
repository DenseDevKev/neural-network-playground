// ── Training Store ──
// Volatile runtime state that changes every frame during training.
// Separated from usePlaygroundStore to prevent sidebar re-renders during training.

import { create } from 'zustand';
import type {
    NetworkSnapshot,
    HistoryPoint,
    DataPoint,
} from '@nn-playground/engine';
import type {
    ArenaModelSummary,
    CheckpointTimeline,
    LiveTrainingSignal,
    PairedEvaluation,
    PauseReason,
    RecipeFingerprint,
    TrainingStatus,
} from '@nn-playground/shared';
import {
    canonicalizeJson,
    parseWorkerEvidenceMessageV2,
    type AppConfig,
} from '@nn-playground/shared';
import {
    appendHistoryPoint,
    resetHistoryBuffer,
} from './historyBuffer.ts';
import { metricHistoryBuffer } from './metricHistoryBuffer.ts';
import { normalizeTrainingSpeed } from '../worker/trainingLoop.ts';
import type { FrameVersions } from '../worker/frameBuffer.ts';

export type ConfigChangeSource = 'data' | 'network' | 'features' | 'training' | 'preset' | null;
export type TrainedRecipeSource = 'initialize' | 'config-sync' | 'reset' | 'restore';

function cloneAppConfig(config: AppConfig): AppConfig {
    return structuredClone(config);
}

export interface TrainingStore {
    // ── Runtime State ──
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    /** Monotonic counter — bumped every time `historyBuffer` is mutated. */
    historyVersion: number;
    /** Worker-authored scientific evidence generation currently accepted by the UI. */
    evidenceGenerationId: number | null;
    latestLiveSignal: LiveTrainingSignal | null;
    latestEvaluation: PairedEvaluation | null;
    /** Independent packed-series versions; charts read the singleton buffer by version. */
    trainingTrendVersion: number;
    evaluationHistoryVersion: number;
    frameVersion: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
    activationHistogramsVersion: number;
    multiclassBoundaryVersion: number;
    arenaSummariesVersion: number;
    /** Bounded scalar summaries only; arena model arrays stay in the worker/frame buffer. */
    arenaSummaries: ArenaModelSummary[] | null;
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    /** Steps of training to run per animation frame. */
    stepsPerFrame: number;
    dataConfigLoading: boolean;
    networkConfigLoading: boolean;
    featuresConfigLoading: boolean;
    trainingConfigLoading: boolean;
    presetConfigLoading: boolean;
    pendingConfigSource: ConfigChangeSource;
    configError: string | null;
    configErrorSource: ConfigChangeSource;
    configSyncNonce: number;
    workerError: string | null;
    pauseReason: PauseReason | null;
    /** True when the most recent streamed snapshot reused cached test metrics. */
    testMetricsStale: boolean;
    /** Lightweight checkpoint timeline metadata only; model payloads stay in the worker. */
    checkpointTimeline: CheckpointTimeline;
    /** App-local recipe identity for the snapshot/evidence currently shown in the UI. */
    trainedRecipeConfig: AppConfig | null;
    trainedRecipeFingerprint: RecipeFingerprint | null;
    trainedRecipeRecordedAt: number | null;
    trainedRecipeSource: TrainedRecipeSource | null;

    // ── Actions ──
    setStatus: (s: TrainingStatus) => void;
    setSnapshot: (snap: NetworkSnapshot) => void;
    addHistoryPoint: (point: HistoryPoint) => void;
    applyStreamedSnapshot: (payload: {
        snapshot: NetworkSnapshot;
        frameVersion: number;
        frameVersions?: FrameVersions;
        testMetricsStale: boolean;
        checkpointTimeline?: CheckpointTimeline;
    }) => void;
    resetHistory: () => void;
    /** Validate and atomically publish a strict protocol-V2 evidence message. */
    applyEvidence: (message: unknown) => void;
    /** Clear both scientific series and their generation in one publication. */
    resetEvidence: () => void;
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
    setPauseReason: (reason: PauseReason | null) => void;
    clearPauseReason: () => void;
    setTestMetricsStale: (stale: boolean) => void;
    setCheckpointTimeline: (timeline: CheckpointTimeline) => void;
    setArenaSummaries: (summaries: ArenaModelSummary[] | null) => void;
    markTrainedRecipe: (
        config: AppConfig,
        source: TrainedRecipeSource,
        recipeFingerprint?: RecipeFingerprint | null,
    ) => void;
}

const EMPTY_CHECKPOINT_TIMELINE: CheckpointTimeline = {
    checkpoints: [],
    maxCheckpoints: 8,
    evictedCount: 0,
    liveCheckpointId: null,
    restoredCheckpointId: null,
};

function evidenceGenerationId(message: ReturnType<typeof parseWorkerEvidenceMessageV2>): number {
    if (message.liveSignal) return message.liveSignal.model.generationId;
    if (message.latestEvaluation) return message.latestEvaluation.model.generationId;
    const artifact = message.artifacts && Object.values(message.artifacts)[0];
    if (!artifact) throw new TypeError('evidence message has no generation identity');
    return artifact.model.generationId;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
    status: 'idle',
    snapshot: null,
    historyVersion: 0,
    evidenceGenerationId: null,
    latestLiveSignal: null,
    latestEvaluation: null,
    trainingTrendVersion: metricHistoryBuffer.versions.trendVersion,
    evaluationHistoryVersion: metricHistoryBuffer.versions.evaluationVersion,
    frameVersion: 0,
    outputGridVersion: 0,
    neuronGridsVersion: 0,
    paramsVersion: 0,
    layerStatsVersion: 0,
    confusionMatrixVersion: 0,
    activationHistogramsVersion: 0,
    multiclassBoundaryVersion: 0,
    arenaSummariesVersion: 0,
    arenaSummaries: null,
    trainPoints: [],
    testPoints: [],
    stepsPerFrame: 5,
    dataConfigLoading: false,
    networkConfigLoading: false,
    featuresConfigLoading: false,
    trainingConfigLoading: false,
    presetConfigLoading: false,
    pendingConfigSource: null,
    configError: null,
    configErrorSource: null,
    configSyncNonce: 0,
    workerError: null,
    pauseReason: null,
    testMetricsStale: false,
    checkpointTimeline: EMPTY_CHECKPOINT_TIMELINE,
    trainedRecipeConfig: null,
    trainedRecipeFingerprint: null,
    trainedRecipeRecordedAt: null,
    trainedRecipeSource: null,

    setStatus: (status) => set({ status }),
    setSnapshot: (snapshot) => set({ snapshot }),
    applyStreamedSnapshot: ({ snapshot, frameVersion, frameVersions, testMetricsStale, checkpointTimeline }) => {
        set((state) => {
            const versions = frameVersions ?? {
                frameVersion,
                outputGridVersion: state.outputGridVersion,
                neuronGridsVersion: state.neuronGridsVersion,
                paramsVersion: state.paramsVersion,
                layerStatsVersion: state.layerStatsVersion,
                confusionMatrixVersion: state.confusionMatrixVersion,
                activationHistogramsVersion: state.activationHistogramsVersion,
                multiclassBoundaryVersion: state.multiclassBoundaryVersion,
                arenaSummariesVersion: state.arenaSummariesVersion,
            };

            return {
                snapshot,
                frameVersion: versions.frameVersion,
                outputGridVersion: versions.outputGridVersion,
                neuronGridsVersion: versions.neuronGridsVersion,
                paramsVersion: versions.paramsVersion,
                layerStatsVersion: versions.layerStatsVersion,
                confusionMatrixVersion: versions.confusionMatrixVersion,
                activationHistogramsVersion: versions.activationHistogramsVersion,
                multiclassBoundaryVersion: versions.multiclassBoundaryVersion,
                arenaSummariesVersion: versions.arenaSummariesVersion,
                historyVersion: state.historyVersion,
                testMetricsStale,
                checkpointTimeline: checkpointTimeline ?? state.checkpointTimeline,
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
    applyEvidence: (value) => {
        // Parse into a detached, deeply-frozen snapshot before inspecting or
        // mutating any client state. This is the app's worker trust boundary.
        const message = parseWorkerEvidenceMessageV2(value);
        const generationId = evidenceGenerationId(message);
        const current = get();
        if (current.evidenceGenerationId !== null
            && current.evidenceGenerationId !== generationId) {
            throw new TypeError(
                `evidence generation ${generationId} does not match active generation ${current.evidenceGenerationId}`,
            );
        }

        // Preflight both channels before touching either packed series. This
        // makes a bundle all-or-nothing even when one half is a replay or a
        // conflict and the other half is new.
        let appendLive = false;
        if (message.liveSignal) {
            const latest = current.latestLiveSignal;
            if (latest && latest.model.revision === message.liveSignal.model.revision) {
                if (canonicalizeJson(latest) !== canonicalizeJson(message.liveSignal)) {
                    throw new TypeError(
                        `conflicting live signal for generation ${generationId} revision ${message.liveSignal.model.revision}`,
                    );
                }
            } else {
                appendLive = latest === null
                    || message.liveSignal.model.revision > latest.model.revision;
            }
        }

        let appendEvaluation = message.latestEvaluation !== undefined;
        if (message.latestEvaluation && current.latestEvaluation
            && message.latestEvaluation.evaluationId === current.latestEvaluation.evaluationId) {
            if (canonicalizeJson(current.latestEvaluation)
                !== canonicalizeJson(message.latestEvaluation)) {
                throw new TypeError(
                    `conflicting evaluationId ${message.latestEvaluation.evaluationId}`,
                );
            }
        }

        const establishesGeneration = current.evidenceGenerationId === null;
        if (!appendLive && !appendEvaluation && !establishesGeneration) return;

        let trainingTrendVersion = current.trainingTrendVersion;
        let evaluationHistoryVersion = current.evaluationHistoryVersion;
        // Evaluation goes first: its bounded replay map can still reject a
        // conflicting retained ID. The already-parsed live append cannot fail,
        // so a rejected bundle never partially appends its live half.
        if (appendEvaluation && message.latestEvaluation) {
            const result = metricHistoryBuffer.appendEvaluation(message.latestEvaluation);
            evaluationHistoryVersion = result.version;
            appendEvaluation = result.appended;
        }
        if (appendLive && message.liveSignal) {
            trainingTrendVersion = metricHistoryBuffer.appendTrend(message.liveSignal);
        }

        if (!appendLive && !appendEvaluation && !establishesGeneration) return;

        set({
            evidenceGenerationId: generationId,
            latestLiveSignal: appendLive && message.liveSignal
                ? message.liveSignal
                : current.latestLiveSignal,
            latestEvaluation: appendEvaluation && message.latestEvaluation
                ? message.latestEvaluation
                : current.latestEvaluation,
            trainingTrendVersion,
            evaluationHistoryVersion,
        });
    },
    resetEvidence: () => {
        const versions = metricHistoryBuffer.reset();
        set({
            evidenceGenerationId: null,
            latestLiveSignal: null,
            latestEvaluation: null,
            trainingTrendVersion: versions.trendVersion,
            evaluationHistoryVersion: versions.evaluationVersion,
        });
    },
    setFrameVersion: (frameVersion) => set({ frameVersion }),
    setFrameVersions: (versions) => set({
        frameVersion: versions.frameVersion,
        outputGridVersion: versions.outputGridVersion,
        neuronGridsVersion: versions.neuronGridsVersion,
        paramsVersion: versions.paramsVersion,
        layerStatsVersion: versions.layerStatsVersion,
        confusionMatrixVersion: versions.confusionMatrixVersion,
        activationHistogramsVersion: versions.activationHistogramsVersion,
        multiclassBoundaryVersion: versions.multiclassBoundaryVersion,
        arenaSummariesVersion: versions.arenaSummariesVersion,
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
        presetConfigLoading: source === 'preset',
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
        presetConfigLoading: false,
    }),
    failConfigChange: (message) => set((state) => ({
        pendingConfigSource: null,
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
        presetConfigLoading: false,
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
            presetConfigLoading: state.configErrorSource === 'preset',
            configError: null,
            configErrorSource: null,
            configSyncNonce: state.configSyncNonce + 1,
        };
    }),
    setWorkerError: (message) => set({ workerError: message }),
    clearWorkerError: () => set({ workerError: null }),
    setPauseReason: (pauseReason) => set({ pauseReason }),
    clearPauseReason: () => set({ pauseReason: null }),
    setTestMetricsStale: (testMetricsStale) => set({ testMetricsStale }),
    setCheckpointTimeline: (checkpointTimeline) => set({ checkpointTimeline }),
    setArenaSummaries: (arenaSummaries) => set({ arenaSummaries }),
    markTrainedRecipe: (config, source, recipeFingerprint = null) => set({
        trainedRecipeConfig: cloneAppConfig(config),
        trainedRecipeFingerprint: recipeFingerprint,
        trainedRecipeRecordedAt: Date.now(),
        trainedRecipeSource: source,
    }),
}));
