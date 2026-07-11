// ── useTraining hook ──
// Manages the training loop, worker communication, and data synchronization.
// Phase 3: Uses useTrainingStore for runtime state, usePlaygroundStore for config.

import { useEffect, useRef, useCallback } from 'react';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import {
    useTrainingStore,
    type TrainedRecipeSource,
    type TrainingStore,
} from '../store/useTrainingStore.ts';
import { projectPreparedExperiment } from '../store/legacyProjection.ts';
import {
    getWorkerApi,
    setupStreamChannel,
    postStreamCommand,
    startRenderLoop,
    stopRenderLoop,
    onSnapshot,
    newRunTo,
    terminateWorker,
} from '../worker/workerBridge.ts';
import {
    getFrameBuffer,
    getFrameVersions,
    updateFrameBuffer,
    type FrameVersions,
} from '../worker/frameBuffer.ts';
import {
    flattenBiases,
    flattenNeuronGrids,
    flattenWeights,
} from '../worker/frameBufferLayout.ts';
import type {
    DataConfig,
    DataPoint,
    FeatureFlags,
    NetworkConfig,
    NetworkSnapshot,
    TrainingConfig,
} from '@nn-playground/engine';
import type {
    ArenaScalarSnapshot,
    CheckpointTimeline,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerEvidenceMessageV2,
    WorkerExperimentRequestV2,
    WorkerArenaSnapshotMessage,
    WorkerProtocolErrorMessageV2,
    WorkerSnapshotMessage,
    WorkerToMainMessage,
} from '@nn-playground/shared';
import { WORKER_PROTOCOL_VERSION } from '@nn-playground/shared';

export interface LiveArenaModelInput {
    label?: string;
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
}

export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => Promise<void>;
    reset: () => Promise<void>;
    restoreCheckpoint: (id: number) => Promise<void>;
    initializeArena: (modelA: LiveArenaModelInput, modelB: LiveArenaModelInput) => Promise<void>;
    stepArena: (iterations?: number) => Promise<void>;
}

function getErrorMessage(error: unknown, fallback: string): string {
    if (error instanceof Error) return error.message;
    return typeof error === 'string' && error.length > 0 ? error : fallback;
}

function requirePreparedExperiment(): PreparedExperimentDocumentV2 {
    const prepared = usePlaygroundStore.getState().prepared;
    if (prepared) return prepared;
    throw new Error(
        'Training is unavailable because the shared experiment URL is incompatible with version 2.',
    );
}

export function createWorkerExperimentRequestV2(
    prepared: PreparedExperimentDocumentV2,
    requestId: number,
): WorkerExperimentRequestV2 {
    if (!Number.isSafeInteger(requestId) || requestId < 1) {
        throw new RangeError('worker requestId must be a positive safe integer');
    }
    return {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId,
        document: prepared.document,
        claimedIdentities: prepared.identities,
    };
}

/**
 * Direct manual-step RPCs can resolve before cadence evidence already queued
 * on the ordered MessagePort. Never jump the evaluation series over that gap.
 */
export function evidenceForDirectStep(
    evidence: WorkerEvidenceMessageV2,
    latestEvaluation: PairedEvaluation | null,
): WorkerEvidenceMessageV2 | null {
    const evaluation = evidence.latestEvaluation;
    if (!evaluation) return evidence;
    const latestId = latestEvaluation?.evaluationId ?? 0;
    if (evaluation.evaluationId === latestId
        || evaluation.evaluationId === latestId + 1) {
        return evidence;
    }
    if (!evidence.liveSignal && !evidence.artifacts) return null;
    return {
        type: 'evidence',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        ...(evidence.liveSignal ? { liveSignal: evidence.liveSignal } : {}),
        ...(evidence.artifacts ? { artifacts: evidence.artifacts } : {}),
    };
}

function getTotalNeuronCount(layerSizes: number[]): number {
    let total = 0;
    for (let i = 1; i < layerSizes.length; i++) {
        total += layerSizes[i];
    }
    return total;
}

function syncSnapshotToFrameBuffer(snapshot: NetworkSnapshot): FrameVersions {
    const outputGrid = snapshot.outputGrid.length > 0
        ? (snapshot.outputGrid instanceof Float32Array ? snapshot.outputGrid : new Float32Array(snapshot.outputGrid))
        : null;
    const { buffer: weights, layerSizes } = flattenWeights(snapshot.weights);
    const biases = flattenBiases(snapshot.biases);

    let neuronGrids: Float32Array | null = null;
    let neuronGridLayout: { count: number; gridSize: number } | null = null;
    if (snapshot.neuronGrids && snapshot.neuronGrids.length > 0) {
        if (snapshot.neuronGrids instanceof Float32Array) {
            neuronGrids = snapshot.neuronGrids;
            neuronGridLayout = {
                count: getTotalNeuronCount(layerSizes),
                gridSize: snapshot.gridSize,
            };
        } else {
            const flattened = flattenNeuronGrids(snapshot.neuronGrids, snapshot.gridSize);
            neuronGrids = flattened.buffer;
            neuronGridLayout = flattened.layout;
        }
    }
    const framePatch: Parameters<typeof updateFrameBuffer>[0] = {
        outputGrid,
        gridSize: snapshot.gridSize,
        neuronGrids,
        neuronGridLayout,
        weights,
        biases,
        weightLayout: { layerSizes },
        layerStats: snapshot.layerStats ?? null,
        confusionMatrix: snapshot.testMetrics.confusionMatrix ?? null,
    };
    const currentFrame = getFrameBuffer();
    if (snapshot.multiclassBoundary) {
        framePatch.multiclassClassGrid = snapshot.multiclassBoundary.classGrid;
        framePatch.multiclassConfidenceGrid = snapshot.multiclassBoundary.confidenceGrid;
        framePatch.multiclassBoundaryLayout = {
            gridSize: snapshot.multiclassBoundary.gridSize,
            classCount: 3,
            classLabels: [0, 1, 2],
        };
    } else if (
        (
            currentFrame.multiclassClassGrid !== null ||
            currentFrame.multiclassConfidenceGrid !== null ||
            currentFrame.multiclassBoundaryLayout !== null
        )
    ) {
        framePatch.multiclassClassGrid = null;
        framePatch.multiclassConfidenceGrid = null;
        framePatch.multiclassBoundaryLayout = null;
    }
    if (currentFrame.multiclassConfusionMatrix !== null) {
        framePatch.multiclassConfusionMatrix = null;
    }

    if (snapshot.activationHistograms) {
        framePatch.activationHistogramBins = snapshot.activationHistograms.bins;
        framePatch.activationHistogramLayout = {
            binCount: snapshot.activationHistograms.layers[0]?.binCount ?? 0,
            layers: snapshot.activationHistograms.layers,
        };
    }

    updateFrameBuffer(framePatch);
    return getFrameVersions();
}

function snapshotForReactState(snapshot: NetworkSnapshot): NetworkSnapshot {
    if (!snapshot.activationHistograms) return snapshot;
    const { activationHistograms: _activationHistograms, ...rest } = snapshot;
    return rest;
}

function applyFreshSnapshotToStore(ts: TrainingStore, snapshot: NetworkSnapshot): void {
    ts.setSnapshot(snapshotForReactState(snapshot));
    ts.setTestMetricsStale(snapshot.testMetricsStale === true);
    ts.setFrameVersions(syncSnapshotToFrameBuffer(snapshot));
}

const EMPTY_CHECKPOINT_TIMELINE: CheckpointTimeline = {
    checkpoints: [],
    maxCheckpoints: 8,
    evictedCount: 0,
    liveCheckpointId: null,
    restoredCheckpointId: null,
};

function applyArenaSnapshotToStore(snapshot: ArenaScalarSnapshot | WorkerArenaSnapshotMessage): void {
    const summaries = snapshot.summaries.map((summary) => ({ ...summary }));
    updateFrameBuffer({ arenaSummaries: summaries });
    const ts = useTrainingStore.getState();
    ts.setFrameVersions(getFrameVersions());
    ts.setArenaSummaries(summaries);
}

function createStreamSnapshot(
    msg: WorkerSnapshotMessage,
    previousSnapshot: NetworkSnapshot | null,
): NetworkSnapshot {
    return {
        step: msg.scalars.step,
        epoch: msg.scalars.epoch,
        trainLoss: msg.scalars.trainLoss,
        testLoss: msg.scalars.testLoss,
        trainMetrics: {
            loss: msg.scalars.trainLoss,
            accuracy: msg.scalars.trainAccuracy,
        },
        testMetrics: {
            loss: msg.scalars.testLoss,
            accuracy: msg.scalars.testAccuracy,
            confusionMatrix: msg.confusionMatrix ?? (
                msg.scalars.testMetricsStale === false
                    ? undefined
                    : previousSnapshot?.testMetrics.confusionMatrix
            ),
        },
        weights: previousSnapshot?.weights ?? [],
        biases: previousSnapshot?.biases ?? [],
        outputGrid: msg.outputGrid !== undefined && msg.outputGrid.length === 0
            ? []
            : previousSnapshot?.outputGrid ?? [],
        gridSize: msg.scalars.gridSize,
        neuronGrids: msg.neuronGrids !== undefined && msg.neuronGrids.length === 0
            ? undefined
            : previousSnapshot?.neuronGrids,
        layerStats: previousSnapshot?.layerStats,
        historyPoint: msg.historyPoint,
    };
}

export function useTraining(): TrainingHook {
    // All refs first (stable hook order)
    const mountedRef = useRef(true);
    const initializedRef = useRef(false);
    const prevPreparedRef = useRef<PreparedExperimentDocumentV2 | null>(null);
    const prevConfigSyncNonceRef = useRef(0);
    const requestIdRef = useRef(0);
    const activeRequestIdRef = useRef<number | null>(null);
    const pendingPreparationRequestIdRef = useRef<number | null>(null);
    const rejectedPreparationRequestIdsRef = useRef(new Set<number>());
    const lifecycleEpochRef = useRef(0);
    const streamSetupPromiseRef = useRef<Promise<void> | null>(null);
    const initializationPromiseRef = useRef<Promise<boolean> | null>(null);
    const mutationPausePromiseRef = useRef<Promise<void> | null>(null);
    const mutationPauseResolveRef = useRef<(() => void) | null>(null);
    const mutationPauseRejectRef = useRef<((error: Error) => void) | null>(null);
    const manualActionPendingRef = useRef(false);
    const stepsPerFrameRef = useRef(5);
    const isPlayingRef = useRef(false);
    const configSyncSeqRef = useRef(0);
    const activeConfigSyncSeqRef = useRef(0);
    const configSyncPendingRef = useRef(false);
    const restoreBarrierRef = useRef<Promise<void> | null>(null);

    // Config selectors (from playground store — stable, rarely changes)
    const prepared = usePlaygroundStore((s) => s.prepared);
    const demand = usePlaygroundStore((s) => s.demand);
    const webgpuGrid = usePlaygroundStore((s) => s.featuresUI.webgpuGrid);

    // Runtime selectors (from training store — volatile)
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const configSyncNonce = useTrainingStore((s) => s.configSyncNonce);

    const reportWorkerError = useCallback((error: unknown, fallback: string) => {
        if (!mountedRef.current) return;
        const message = getErrorMessage(error, fallback);
        mutationPauseRejectRef.current?.(new Error(message));
        mutationPausePromiseRef.current = null;
        mutationPauseResolveRef.current = null;
        mutationPauseRejectRef.current = null;
        const ts = useTrainingStore.getState();
        isPlayingRef.current = false;
        stopRenderLoop();
        ts.setWorkerError(message);
        ts.setPauseReason('error');
        ts.setStatus('paused');
        initializedRef.current = false;
    }, []);

    const beginConfigSync = useCallback(() => {
        const seq = configSyncSeqRef.current + 1;
        configSyncSeqRef.current = seq;
        activeConfigSyncSeqRef.current = seq;
        configSyncPendingRef.current = true;
        return seq;
    }, []);

    const isCurrentConfigSync = useCallback((seq: number) => (
        configSyncPendingRef.current && activeConfigSyncSeqRef.current === seq
    ), []);

    const finishConfigSyncIfCurrent = useCallback((seq: number) => {
        if (activeConfigSyncSeqRef.current === seq) {
            configSyncPendingRef.current = false;
        }
    }, []);

    const nextRequest = useCallback((nextPrepared: PreparedExperimentDocumentV2) => {
        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        activeRequestIdRef.current = requestId;
        pendingPreparationRequestIdRef.current = requestId;
        return createWorkerExperimentRequestV2(nextPrepared, requestId);
    }, []);

    const ensureStreamChannel = useCallback((): Promise<void> => {
        if (!streamSetupPromiseRef.current) {
            const promise = setupStreamChannel().catch((error) => {
                if (streamSetupPromiseRef.current === promise) {
                    streamSetupPromiseRef.current = null;
                }
                throw error;
            });
            streamSetupPromiseRef.current = promise;
        }
        return streamSetupPromiseRef.current;
    }, []);

    const pauseForMutation = useCallback((): Promise<void> => {
        if (mutationPausePromiseRef.current) return mutationPausePromiseRef.current;
        if (!isPlayingRef.current) return Promise.resolve();
        isPlayingRef.current = false;
        stopRenderLoop();
        let rejectPause!: (error: Error) => void;
        const promise = new Promise<void>((resolve, reject) => {
            mutationPauseResolveRef.current = resolve;
            mutationPauseRejectRef.current = reject;
            rejectPause = reject;
        });
        mutationPausePromiseRef.current = promise;
        try {
            postStreamCommand({ type: 'stopTraining' });
        } catch (error) {
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            rejectPause(error instanceof Error ? error : new Error(String(error)));
        }
        return promise;
    }, []);

    const applyFreshV2Run = useCallback((
        result: Awaited<ReturnType<ReturnType<typeof getWorkerApi>['initializeExperimentV2']>>,
    ) => {
        const ts = useTrainingStore.getState();
        newRunTo(result.runId);
        ts.resetEvidence();
        ts.applyEvidence(result.evidence);
        applyFreshSnapshotToStore(ts, result.snapshot);
        ts.setCheckpointTimeline(EMPTY_CHECKPOINT_TIMELINE);
        ts.clearWorkerError();
        ts.clearPauseReason();
    }, []);

    const publishCommittedV2Run = useCallback((
        result: Awaited<ReturnType<ReturnType<typeof getWorkerApi>['initializeExperimentV2']>>,
        owner: PreparedExperimentDocumentV2,
        source: TrainedRecipeSource,
    ): boolean => {
        const ts = useTrainingStore.getState();
        if (ts.evidenceGenerationId !== null
            && result.runId <= ts.evidenceGenerationId) return false;
        applyFreshV2Run(result);
        ts.setTrainPoints([]);
        ts.setTestPoints([]);
        ts.markTrainedRecipe(
            projectPreparedExperiment(owner),
            source,
            owner.identities.recipeFingerprint,
        );
        ts.setStatus('idle');
        initializedRef.current = false;
        return true;
    }, [applyFreshV2Run]);

    const initializePrepared = useCallback(async (
        requestedPrepared: PreparedExperimentDocumentV2 = requirePreparedExperiment(),
    ): Promise<boolean> => {
        const lifecycleEpoch = lifecycleEpochRef.current;
        prevPreparedRef.current = requestedPrepared;
        const request = nextRequest(requestedPrepared);
        const api = getWorkerApi();
        let result;
        try {
            await ensureStreamChannel();
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || activeRequestIdRef.current !== request.requestId) {
                return false;
            }
            result = await api.initializeExperimentV2(request);
        } catch (error) {
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || activeRequestIdRef.current !== request.requestId) {
                return false;
            }
            pendingPreparationRequestIdRef.current = null;
            rejectedPreparationRequestIdsRef.current.delete(request.requestId);
            throw error;
        }
        if (!mountedRef.current || lifecycleEpochRef.current !== lifecycleEpoch) return false;
        if (rejectedPreparationRequestIdsRef.current.delete(request.requestId)) return false;
        publishCommittedV2Run(result, requestedPrepared, 'initialize');
        if (activeRequestIdRef.current !== request.requestId) return false;
        pendingPreparationRequestIdRef.current = null;
        if (usePlaygroundStore.getState().prepared !== requestedPrepared) return false;
        const ts = useTrainingStore.getState();

        // Hydration does not own generation identity. A failure below reports
        // a runtime error, but never rolls the committed run back to stale UI.
        const trainPts = await api.getTrainPoints();
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        const testPts = await api.getTestPoints();
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        const latestState = usePlaygroundStore.getState();
        if (latestState.prepared !== requestedPrepared) return false;

        // Send initial demand
        await api.updateDemand(latestState.demand);
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;

        // AS-4: tell the worker whether the user has opted in to the
        // WebGPU grid path. Capability detection still gates this; the
        // worker silently falls back to CPU when the device isn't
        // available or the network shape exceeds the shader caps.
        try {
            await api.setWebGpuEnabled(usePlaygroundStore.getState().featuresUI.webgpuGrid);
        } catch {
            // Older worker bundles won't expose setWebGpuEnabled — ignore.
        }

        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        if (usePlaygroundStore.getState().prepared !== requestedPrepared) return false;
        ts.setTrainPoints(trainPts);
        ts.setTestPoints(testPts);
        initializedRef.current = true;
        return true;
    }, [ensureStreamChannel, nextRequest, publishCommittedV2Run]);

    const initializeWorker = useCallback((
        requestedPrepared: PreparedExperimentDocumentV2 = requirePreparedExperiment(),
    ): Promise<boolean> => {
        const inFlight = initializationPromiseRef.current;
        if (inFlight && prevPreparedRef.current === requestedPrepared) return inFlight;
        const promise = initializePrepared(requestedPrepared);
        initializationPromiseRef.current = promise;
        const clear = () => {
            if (initializationPromiseRef.current === promise) {
                initializationPromiseRef.current = null;
            }
        };
        void promise.then(clear, clear);
        return promise;
    }, [initializePrepared]);

    // Keep ref in sync so streaming commands use current speed.
    useEffect(() => {
        stepsPerFrameRef.current = stepsPerFrame;
        // If currently playing, update the worker's speed
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateSpeed', stepsPerFrame });
        }
    }, [stepsPerFrame]);

    // ── Snapshot handler: applies streamed snapshots to training store ──
    useEffect(() => {
        const unsubscribe = onSnapshot((msg: WorkerToMainMessage) => {
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();

            if (msg.type === 'evidence') {
                try {
                    ts.applyEvidence(msg);
                } catch (error) {
                    reportWorkerError(error, 'Received invalid scientific evidence from the worker.');
                }
            } else if (msg.type === 'worker-error') {
                const error = msg as WorkerProtocolErrorMessageV2;
                if (error.generationId === null) {
                    if (error.requestId === null
                        || error.requestId !== pendingPreparationRequestIdRef.current) return;
                    pendingPreparationRequestIdRef.current = null;
                    rejectedPreparationRequestIdsRef.current.add(error.requestId);
                    activeRequestIdRef.current = null;
                    if (configSyncPendingRef.current) {
                        configSyncPendingRef.current = false;
                        ts.failConfigChange(error.message);
                        return;
                    }
                    reportWorkerError(error.message, 'Failed to prepare the experiment.');
                    return;
                }
                if (error.generationId !== ts.evidenceGenerationId) return;
                reportWorkerError(error.message, 'The training worker failed.');
            } else if (msg.type === 'snapshot') {
                const snapshot = createStreamSnapshot(msg, ts.snapshot);
                const frameVersions = getFrameVersions();
                ts.applyStreamedSnapshot({
                    snapshot,
                    frameVersion: frameVersions.frameVersion,
                    frameVersions,
                    testMetricsStale: msg.scalars.testMetricsStale === true,
                });
            } else if (msg.type === 'status') {
                if (msg.status === 'paused') {
                    const resolveMutationPause = mutationPauseResolveRef.current;
                    if (resolveMutationPause) {
                        mutationPausePromiseRef.current = null;
                        mutationPauseResolveRef.current = null;
                        mutationPauseRejectRef.current = null;
                        resolveMutationPause();
                    }
                    if (msg.pauseReason) {
                        isPlayingRef.current = false;
                        stopRenderLoop();
                        ts.setPauseReason(msg.pauseReason);
                    }
                    ts.setStatus('paused');
                } else if (msg.status === 'idle') {
                    isPlayingRef.current = false;
                    stopRenderLoop();
                    ts.clearPauseReason();
                    ts.setStatus(msg.status);
                } else if (msg.status === 'running') {
                    ts.clearPauseReason();
                    ts.setStatus('running');
                }
            } else if (msg.type === 'error') {
                reportWorkerError(msg.message, 'The training worker failed.');
            } else if (msg.type === 'arenaSnapshot') {
                applyArenaSnapshotToStore(msg);
            }
        });

        return unsubscribe;
    }, [reportWorkerError]);

    // Initialize worker on mount
    useEffect(() => {
        lifecycleEpochRef.current++;
        mountedRef.current = true;
        if (!prepared) {
            reportWorkerError(
                new Error('Training is unavailable because the shared experiment URL is incompatible with version 2.'),
                'Failed to initialize training worker.',
            );
            return;
        }
        initializeWorker(prepared).catch((error) => {
            reportWorkerError(error, 'Failed to initialize training worker.');
        });
        // Mount initialization is intentionally one-shot. Later prepared
        // documents are handled by the ordered config transaction below.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Every accepted prepared-document replacement starts a fresh generation.
    useEffect(() => {
        const isRetry = configSyncNonce !== prevConfigSyncNonceRef.current;
        if (!prepared) return;
        if (!isRetry && prevPreparedRef.current === prepared) return;
        const previousPrepared = prevPreparedRef.current;
        prevPreparedRef.current = prepared;
        prevConfigSyncNonceRef.current = configSyncNonce;
        const seq = beginConfigSync();
        const request = nextRequest(prepared);
        const lifecycleEpoch = lifecycleEpochRef.current;

        const sync = async () => {
            const restoreBarrier = restoreBarrierRef.current;
            if (restoreBarrier) {
                await restoreBarrier;
                if (!isCurrentConfigSync(seq)) return;
            }
            // Serialize MessagePort pause with the Comlink mutation. The
            // paused status is the acknowledgement that the old loop and its
            // forced evaluation have completed.
            try {
                await pauseForMutation();
            } catch (error) {
                if (mountedRef.current && isCurrentConfigSync(seq)) {
                    useTrainingStore.getState().failConfigChange(
                        getErrorMessage(error, 'Failed to pause the current experiment.'),
                    );
                    finishConfigSyncIfCurrent(seq);
                }
                return;
            }
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || !isCurrentConfigSync(seq)) return;

            const api = getWorkerApi();
            const ts = useTrainingStore.getState();
            let result: Awaited<ReturnType<typeof api.initializeExperimentV2>>;
            try {
                await ensureStreamChannel();
                if (!mountedRef.current
                    || lifecycleEpochRef.current !== lifecycleEpoch
                    || !isCurrentConfigSync(seq)
                    || activeRequestIdRef.current !== request.requestId) return;
                result = await api.initializeExperimentV2(request);
                if (!mountedRef.current || lifecycleEpochRef.current !== lifecycleEpoch) return;
                if (rejectedPreparationRequestIdsRef.current.delete(request.requestId)) return;
                publishCommittedV2Run(result, prepared, 'config-sync');
                if (!isCurrentConfigSync(seq)
                    || activeRequestIdRef.current !== request.requestId) return;
                pendingPreparationRequestIdRef.current = null;
                const committed = usePlaygroundStore.getState();
                if (committed.prepared !== prepared) return;
            } catch (error) {
                if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
                if (pendingPreparationRequestIdRef.current === request.requestId) {
                    pendingPreparationRequestIdRef.current = null;
                }
                rejectedPreparationRequestIdsRef.current.delete(request.requestId);
                prevPreparedRef.current = previousPrepared;
                ts.failConfigChange(error instanceof Error ? error.message : 'Failed to update configuration');
                finishConfigSyncIfCurrent(seq);
                return;
            }

            // The worker/app identity commit above cannot be rolled back.
            // Auxiliary failures become runtime errors, never config rollback.
            let auxiliaryError: unknown = null;
            let trainPts: DataPoint[] = [];
            let testPts: DataPoint[] = [];
            try {
                usePlaygroundStore.getState().syncToUrl();
            } catch (error) {
                auxiliaryError = error;
            }
            try {
                trainPts = await api.getTrainPoints();
            } catch (error) {
                auxiliaryError = error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            try {
                testPts = await api.getTestPoints();
            } catch (error) {
                auxiliaryError ??= error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            const latest = usePlaygroundStore.getState();
            if (latest.prepared !== prepared) return;
            try {
                await api.updateDemand(latest.demand);
            } catch (error) {
                auxiliaryError ??= error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            try {
                await api.setWebGpuEnabled(latest.featuresUI.webgpuGrid);
            } catch {
                // Capability and older-bundle fallbacks remain non-fatal.
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;

            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);
            initializedRef.current = true;
            ts.finishConfigChange();
            finishConfigSyncIfCurrent(seq);
            if (auxiliaryError !== null) {
                reportWorkerError(auxiliaryError, 'Failed to hydrate the new experiment runtime.');
            }
        };
        void sync();
    }, [
        prepared,
        configSyncNonce,
        beginConfigSync,
        finishConfigSyncIfCurrent,
        ensureStreamChannel,
        isCurrentConfigSync,
        nextRequest,
        pauseForMutation,
        publishCommittedV2Run,
        reportWorkerError,
    ]);

    // Sync demand changes to worker
    useEffect(() => {
        if (!initializedRef.current) return;
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateDemand', demand });
            return;
        }
        void getWorkerApi().updateDemand(demand).catch((error: unknown) => {
            reportWorkerError(error, 'Failed to update visualization demand.');
        });
    }, [demand, reportWorkerError]);

    // AS-4: live-toggle the WebGPU grid path when the user flips the
    // featuresUI flag. Disabling immediately disposes the GPU predictor
    // (frees device memory); enabling lets the next snapshot lazily
    // re-allocate.
    useEffect(() => {
        if (!initializedRef.current) return;
        const api = getWorkerApi();
        api.setWebGpuEnabled(webgpuGrid).catch(() => {
            // Ignore — capability detection inside the worker handles
            // any per-device fallback. A toggle that doesn't reach the
            // worker just means the next snapshot still uses whatever
            // path the worker last knew about.
        });
    }, [webgpuGrid]);

    const play = useCallback(() => {
        if (configSyncPendingRef.current
            || mutationPausePromiseRef.current !== null
            || manualActionPendingRef.current
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }

        const startTraining = () => {
            isPlayingRef.current = true;
            const ts = useTrainingStore.getState();
            ts.clearPauseReason();
            ts.setStatus('running');
            startRenderLoop();
            postStreamCommand({ type: 'startTraining', stepsPerFrame: stepsPerFrameRef.current });
        };

        if (!initializedRef.current) {
            initializeWorker().catch((error) => {
                reportWorkerError(error, 'Failed to initialize training worker.');
            }).then(() => {
                if (initializedRef.current) {
                    startTraining();
                }
            });
            return;
        }
        startTraining();
    }, [initializeWorker, reportWorkerError]);

    const pause = useCallback(() => {
        if (!isPlayingRef.current && useTrainingStore.getState().status !== 'running') {
            return;
        }
        const ts = useTrainingStore.getState();
        ts.setPauseReason('manual');
        ts.setStatus('paused');
        void pauseForMutation().catch((error) => {
            reportWorkerError(error, 'Failed to pause training.');
        });
    }, [pauseForMutation, reportWorkerError]);

    const step = useCallback(async () => {
        if (configSyncPendingRef.current || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        if (manualActionPendingRef.current) return;
        manualActionPendingRef.current = true;
        const wasPlaying = isPlayingRef.current;
        try {
            await pauseForMutation();
            if (configSyncPendingRef.current
                || useTrainingStore.getState().pendingConfigSource !== null) return;
            if (wasPlaying) {
                useTrainingStore.getState().setPauseReason('manual');
                useTrainingStore.getState().setStatus('paused');
            }
            if (!initializedRef.current) {
                await initializeWorker();
                if (!initializedRef.current) return;
            }
            const api = getWorkerApi();
            const result = await api.stepExperimentV2(1);
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();
            if (configSyncPendingRef.current
                || ts.pendingConfigSource !== null
                || result.runId !== ts.evidenceGenerationId) return;
            const resultRevision = result.evidence.liveSignal?.model.revision
                ?? result.evidence.latestEvaluation?.model.revision;
            const currentRevision = ts.latestLiveSignal?.model.revision
                ?? ts.latestEvaluation?.model.revision;
            if (resultRevision !== undefined
                && currentRevision !== undefined
                && resultRevision < currentRevision) return;
            ts.setSnapshot(snapshotForReactState(result.snapshot));
            ts.setTestMetricsStale(result.snapshot.testMetricsStale === true);
            ts.setFrameVersions(syncSnapshotToFrameBuffer(result.snapshot));
            const directEvidence = evidenceForDirectStep(
                result.evidence,
                ts.latestEvaluation,
            );
            if (directEvidence) ts.applyEvidence(directEvidence);
        } catch (error) {
            reportWorkerError(error, 'Failed to run a training step.');
        } finally {
            manualActionPendingRef.current = false;
        }
    }, [initializeWorker, pauseForMutation, reportWorkerError]);

    const reset = useCallback(async () => {
        if (configSyncPendingRef.current
            || manualActionPendingRef.current
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        const seq = beginConfigSync();
        useTrainingStore.getState().clearPauseReason();
        try {
            await pauseForMutation();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            if (!initializedRef.current) {
                await initializeWorker();
                if (!isCurrentConfigSync(seq)) return;
                useTrainingStore.getState().setStatus('idle');
                finishConfigSyncIfCurrent(seq);
                return;
            }
            const api = getWorkerApi();
            const resetPrepared = requirePreparedExperiment();
            const result = await api.resetExperimentV2();
            if (!mountedRef.current) return;
            publishCommittedV2Run(result, resetPrepared, 'reset');
            if (!isCurrentConfigSync(seq)) return;
            if (usePlaygroundStore.getState().prepared !== resetPrepared) return;
            const ts = useTrainingStore.getState();

            // Auxiliary reset hydration cannot roll generation identity back.
            const trainPts = await api.getTrainPoints();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            const testPts = await api.getTestPoints();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            if (usePlaygroundStore.getState().prepared !== resetPrepared) return;
            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);
            finishConfigSyncIfCurrent(seq);
        } catch (error) {
            if (!isCurrentConfigSync(seq)) return;
            reportWorkerError(error, 'Failed to reset training.');
            finishConfigSyncIfCurrent(seq);
        }
    }, [beginConfigSync, finishConfigSyncIfCurrent, initializeWorker, isCurrentConfigSync, pauseForMutation, publishCommittedV2Run, reportWorkerError]);

    const restoreCheckpoint = useCallback(async (id: number) => {
        // A prepared document means this hook owns a strict V2 runtime. Until
        // Task 10 adds versioned checkpoint RPCs, never cross into the legacy
        // model mutator even if client state is manually seeded.
        if (usePlaygroundStore.getState().prepared !== null) return;
        if (configSyncPendingRef.current
            || restoreBarrierRef.current !== null
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        // Protocol V2 does not expose a checkpoint timeline until Task 10.
        // Keep the legacy restoration branch for that migration, but make it
        // unreachable from a strict run unless the store owns the requested ID.
        if (!useTrainingStore.getState().checkpointTimeline.checkpoints.some(
            (checkpoint) => checkpoint.id === id,
        )) return;
        let releaseRestore!: () => void;
        const restoreBarrier = new Promise<void>((resolve) => {
            releaseRestore = resolve;
        });
        restoreBarrierRef.current = restoreBarrier;
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'stopTraining' });
            stopRenderLoop();
            isPlayingRef.current = false;
        }
        try {
            if (!initializedRef.current) {
                await initializeWorker();
                if (!initializedRef.current) return;
            }
            const restoredPrepared = requirePreparedExperiment();
            const result = await getWorkerApi().restoreCheckpoint(id);
            const currentFingerprint = usePlaygroundStore.getState()
                .prepared?.identities.recipeFingerprint ?? null;
            if (configSyncPendingRef.current
                || currentFingerprint !== restoredPrepared.identities.recipeFingerprint) {
                return;
            }
            newRunTo(result.runId);
            const ts = useTrainingStore.getState();
            applyFreshSnapshotToStore(ts, result.snapshot);
            ts.setCheckpointTimeline(result.timeline as CheckpointTimeline);
            ts.markTrainedRecipe(
                projectPreparedExperiment(restoredPrepared),
                'restore',
                restoredPrepared.identities.recipeFingerprint,
            );
            ts.setPauseReason('manual');
            ts.setStatus('paused');
        } catch (error) {
            reportWorkerError(error, 'Failed to restore checkpoint.');
        } finally {
            if (restoreBarrierRef.current === restoreBarrier) {
                restoreBarrierRef.current = null;
            }
            releaseRestore();
        }
    }, [initializeWorker, reportWorkerError]);

    const initializeArena = useCallback(async (modelA: LiveArenaModelInput, modelB: LiveArenaModelInput) => {
        if (configSyncPendingRef.current || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        if (isPlayingRef.current) {
            pause();
        }
        try {
            if (!initializedRef.current) {
                await initializeWorker();
            }
            const snapshot = await getWorkerApi().initializeArena({ modelA, modelB });
            applyArenaSnapshotToStore(snapshot as ArenaScalarSnapshot);
        } catch (error) {
            reportWorkerError(error, 'Failed to initialize live arena.');
        }
    }, [initializeWorker, pause, reportWorkerError]);

    const stepArena = useCallback(async (iterations: number = 1) => {
        if (configSyncPendingRef.current || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        try {
            if (!initializedRef.current) {
                await initializeWorker();
            }
            const snapshot = await getWorkerApi().stepArena(iterations);
            applyArenaSnapshotToStore(snapshot as ArenaScalarSnapshot);
        } catch (error) {
            reportWorkerError(error, 'Failed to step live arena.');
        }
    }, [initializeWorker, reportWorkerError]);

    // Cleanup on unmount
    useEffect(() => {
        const lifecycleEpoch = lifecycleEpochRef;
        const rejectedPreparationRequestIds = rejectedPreparationRequestIdsRef;
        return () => {
            mountedRef.current = false;
            lifecycleEpoch.current++;
            activeRequestIdRef.current = null;
            pendingPreparationRequestIdRef.current = null;
            rejectedPreparationRequestIds.current.clear();
            streamSetupPromiseRef.current = null;
            initializationPromiseRef.current = null;
            mutationPauseRejectRef.current?.(new Error('Training hook unmounted'));
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            activeConfigSyncSeqRef.current = configSyncSeqRef.current + 1;
            configSyncSeqRef.current = activeConfigSyncSeqRef.current;
            configSyncPendingRef.current = false;
            manualActionPendingRef.current = false;
            isPlayingRef.current = false;
            terminateWorker();
            const ts = useTrainingStore.getState();
            ts.resetEvidence();
            const frameVersions = getFrameVersions();
            useTrainingStore.setState({
                status: 'idle',
                snapshot: null,
                frameVersion: frameVersions.frameVersion,
                outputGridVersion: frameVersions.outputGridVersion,
                neuronGridsVersion: frameVersions.neuronGridsVersion,
                paramsVersion: frameVersions.paramsVersion,
                layerStatsVersion: frameVersions.layerStatsVersion,
                confusionMatrixVersion: frameVersions.confusionMatrixVersion,
                activationHistogramsVersion: frameVersions.activationHistogramsVersion,
                multiclassBoundaryVersion: frameVersions.multiclassBoundaryVersion,
                arenaSummariesVersion: frameVersions.arenaSummariesVersion,
                arenaSummaries: null,
                trainPoints: [],
                testPoints: [],
                dataConfigLoading: false,
                networkConfigLoading: false,
                featuresConfigLoading: false,
                trainingConfigLoading: false,
                presetConfigLoading: false,
                pendingConfigSource: null,
                configError: null,
                configErrorSource: null,
                workerError: null,
                pauseReason: null,
                testMetricsStale: false,
                checkpointTimeline: EMPTY_CHECKPOINT_TIMELINE,
                trainedRecipeConfig: null,
                trainedRecipeFingerprint: null,
                trainedRecipeRecordedAt: null,
                trainedRecipeSource: null,
            });
        };
    }, []);

    return { play, pause, step, reset, restoreCheckpoint, initializeArena, stepArena };
}
