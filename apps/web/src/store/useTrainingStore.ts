// ── Training Store ──
// Volatile runtime state that changes every frame during training.
// Separated from usePlaygroundStore to prevent sidebar re-renders during training.

import { create } from 'zustand';
import type { DataPoint } from '@nn-playground/engine';
import type {
    CheckpointTimeline,
    LiveTrainingSignal,
    PairedEvaluation,
    PauseReason,
    RecipeFingerprint,
    TrainingStatus,
    WorkerEvidenceMessageV2,
} from '@nn-playground/shared';
import {
    canonicalizeJson,
    parseWorkerEvidenceMessageV2,
    type ValidatedStandardExperimentRecipeV2,
} from '@nn-playground/shared';
import {
    metricHistoryBuffer,
    type PreparedMetricHistoryAppend,
    type PreparedMetricHistoryReplacement,
} from './metricHistoryBuffer.ts';
import { normalizeTrainingSpeed } from '../worker/trainingLoop.ts';
import type { FrameVersions } from '../worker/frameBuffer.ts';

export type ConfigChangeSource = 'data' | 'network' | 'features' | 'training' | 'preset' | null;
export type TrainedRecipeSource = 'initialize' | 'config-sync' | 'reset' | 'restore';

export interface PreparedTrainingEvidenceReplacement {
    readonly evidence: WorkerEvidenceMessageV2;
    readonly generationId: number;
    readonly history: PreparedMetricHistoryReplacement;
}

export interface PreparedTrainingEvidenceAppend {
    readonly generationId: number;
    readonly latestLiveSignal: LiveTrainingSignal | null;
    readonly latestEvaluation: PairedEvaluation | null;
    readonly history: PreparedMetricHistoryAppend;
    readonly publish: boolean;
}

function cloneValidatedRecipe(
    recipe: ValidatedStandardExperimentRecipeV2,
): ValidatedStandardExperimentRecipeV2 {
    return structuredClone(recipe);
}

export interface TrainingStore {
    // ── Runtime State ──
    status: TrainingStatus;
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
    /** Lightweight checkpoint timeline metadata only; model payloads stay in the worker. */
    checkpointTimeline: CheckpointTimeline;
    /** App-local recipe identity for the snapshot/evidence currently shown in the UI. */
    trainedRecipe: ValidatedStandardExperimentRecipeV2 | null;
    trainedRecipeFingerprint: RecipeFingerprint | null;
    trainedRecipeRecordedAt: number | null;
    trainedRecipeSource: TrainedRecipeSource | null;

    // ── Actions ──
    setStatus: (s: TrainingStatus) => void;
    applyStreamedFrame: (payload: {
        frameVersions: FrameVersions;
        checkpointTimeline?: CheckpointTimeline;
    }) => void;
    /** Validate and atomically publish a strict protocol-V2 evidence message. */
    applyEvidence: (message: unknown) => void;
    /** Preflight a fresh generation without mutating accepted evidence/history. */
    prepareEvidenceReplacement: (
        message: WorkerEvidenceMessageV2,
    ) => PreparedTrainingEvidenceReplacement;
    /** Validation-free publication of a prepared fresh generation. */
    commitEvidenceReplacement: (prepared: PreparedTrainingEvidenceReplacement) => void;
    prepareEvidenceAppend: (message: unknown) => PreparedTrainingEvidenceAppend;
    commitEvidenceAppend: (prepared: PreparedTrainingEvidenceAppend) => void;
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
    setCheckpointTimeline: (timeline: CheckpointTimeline) => void;
    markTrainedRecipe: (
        recipe: ValidatedStandardExperimentRecipeV2,
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

const committedEvidenceReplacements = new WeakSet<PreparedTrainingEvidenceReplacement>();
const committedEvidenceAppends = new WeakSet<PreparedTrainingEvidenceAppend>();

export const useTrainingStore = create<TrainingStore>((set, get) => ({
    status: 'idle',
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
    checkpointTimeline: EMPTY_CHECKPOINT_TIMELINE,
    trainedRecipe: null,
    trainedRecipeFingerprint: null,
    trainedRecipeRecordedAt: null,
    trainedRecipeSource: null,

    setStatus: (status) => set({ status }),
    applyStreamedFrame: ({ frameVersions, checkpointTimeline }) => {
        set((state) => {
            return {
                frameVersion: frameVersions.frameVersion,
                outputGridVersion: frameVersions.outputGridVersion,
                neuronGridsVersion: frameVersions.neuronGridsVersion,
                paramsVersion: frameVersions.paramsVersion,
                layerStatsVersion: frameVersions.layerStatsVersion,
                confusionMatrixVersion: frameVersions.confusionMatrixVersion,
                activationHistogramsVersion: frameVersions.activationHistogramsVersion,
                multiclassBoundaryVersion: frameVersions.multiclassBoundaryVersion,
                checkpointTimeline: checkpointTimeline ?? state.checkpointTimeline,
            };
        });
    },
    applyEvidence: (value) => {
        const prepared = get().prepareEvidenceAppend(value);
        get().commitEvidenceAppend(prepared);
    },
    prepareEvidenceReplacement: (evidence) => {
        const generationId = evidenceGenerationId(evidence);
        const history = metricHistoryBuffer.prepareReplacement(
            evidence.liveSignal,
            evidence.latestEvaluation,
        );
        return Object.freeze({ evidence, generationId, history });
    },
    commitEvidenceReplacement: (prepared) => {
        if (committedEvidenceReplacements.has(prepared)) return;
        prepared.history.commit();
        set({
            evidenceGenerationId: prepared.generationId,
            latestLiveSignal: prepared.evidence.liveSignal ?? null,
            latestEvaluation: prepared.evidence.latestEvaluation ?? null,
            trainingTrendVersion: prepared.history.versions.trendVersion,
            evaluationHistoryVersion: prepared.history.versions.evaluationVersion,
        });
        committedEvidenceReplacements.add(prepared);
    },
    prepareEvidenceAppend: (value) => {
        const message = parseWorkerEvidenceMessageV2(value);
        const generationId = evidenceGenerationId(message);
        const current = get();
        if (current.evidenceGenerationId !== null
            && current.evidenceGenerationId !== generationId) {
            throw new TypeError(
                `evidence generation ${generationId} does not match active generation ${current.evidenceGenerationId}`,
            );
        }
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
            appendEvaluation = false;
        }
        const history = metricHistoryBuffer.prepareAppend(
            appendLive ? message.liveSignal : undefined,
            appendEvaluation ? message.latestEvaluation : undefined,
        );
        appendEvaluation = history.evaluationAppended;
        const establishesGeneration = current.evidenceGenerationId === null;
        return Object.freeze({
            generationId,
            latestLiveSignal: appendLive && message.liveSignal
                ? message.liveSignal
                : current.latestLiveSignal,
            latestEvaluation: appendEvaluation && message.latestEvaluation
                ? message.latestEvaluation
                : current.latestEvaluation,
            history,
            publish: appendLive || appendEvaluation || establishesGeneration,
        });
    },
    commitEvidenceAppend: (prepared) => {
        if (!prepared.publish || committedEvidenceAppends.has(prepared)) return;
        prepared.history.commit();
        set({
            evidenceGenerationId: prepared.generationId,
            latestLiveSignal: prepared.latestLiveSignal,
            latestEvaluation: prepared.latestEvaluation,
            trainingTrendVersion: prepared.history.versions.trendVersion,
            evaluationHistoryVersion: prepared.history.versions.evaluationVersion,
        });
        committedEvidenceAppends.add(prepared);
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
    setCheckpointTimeline: (checkpointTimeline) => set({ checkpointTimeline }),
    markTrainedRecipe: (recipe, source, recipeFingerprint = null) => set({
        trainedRecipe: cloneValidatedRecipe(recipe),
        trainedRecipeFingerprint: recipeFingerprint,
        trainedRecipeRecordedAt: Date.now(),
        trainedRecipeSource: source,
    }),
}));
