// ── useTraining hook ──
// Manages the training loop, worker communication, and data synchronization.
// Phase 3: Uses useTrainingStore for runtime state, usePlaygroundStore for config.

import { useEffect, useRef, useCallback } from 'react';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
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
    flattenBiases,
    flattenNeuronGrids,
    flattenWeights,
    getFrameVersions,
    updateFrameBuffer,
    type FrameVersions,
} from '../worker/frameBuffer.ts';
import type { NetworkSnapshot } from '@nn-playground/engine';
import type { WorkerSnapshotMessage, WorkerToMainMessage } from '@nn-playground/shared';
import { structuralEqual } from '@nn-playground/shared';


export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => void;
    reset: () => void;
}

function getErrorMessage(error: unknown, fallback: string): string {
    return error instanceof Error ? error.message : fallback;
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

    updateFrameBuffer({
        outputGrid,
        gridSize: snapshot.gridSize,
        neuronGrids,
        neuronGridLayout,
        weights,
        biases,
        weightLayout: { layerSizes },
        layerStats: snapshot.layerStats ?? null,
        confusionMatrix: snapshot.testMetrics.confusionMatrix ?? null,
    });
    return getFrameVersions();
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
            confusionMatrix: msg.confusionMatrix ?? previousSnapshot?.testMetrics.confusionMatrix,
        },
        weights: previousSnapshot?.weights ?? [],
        biases: previousSnapshot?.biases ?? [],
        outputGrid: previousSnapshot?.outputGrid ?? [],
        gridSize: msg.scalars.gridSize,
        neuronGrids: previousSnapshot?.neuronGrids,
        layerStats: previousSnapshot?.layerStats,
        historyPoint: msg.historyPoint,
    };
}

export function useTraining(): TrainingHook {
    // All refs first (stable hook order)
    const initializedRef = useRef(false);
    // Snapshot of the last config objects we successfully sent to the worker.
    // Kept by reference for structural comparison; not JSON.stringify'd every tick.
    const prevConfigRef = useRef<{
        network: unknown;
        training: unknown;
        data: unknown;
        features: unknown;
    } | null>(null);
    const prevConfigSyncNonceRef = useRef(0);
    const stepsPerFrameRef = useRef(5);
    const isPlayingRef = useRef(false);
    const configSyncSeqRef = useRef(0);
    const activeConfigSyncSeqRef = useRef(0);
    const configSyncPendingRef = useRef(false);

    // Config selectors (from playground store — stable, rarely changes)
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const demand = usePlaygroundStore((s) => s.demand);
    const webgpuGrid = usePlaygroundStore((s) => s.featuresUI.webgpuGrid);

    // Runtime selectors (from training store — volatile)
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const configSyncNonce = useTrainingStore((s) => s.configSyncNonce);

    const reportWorkerError = useCallback((error: unknown, fallback: string) => {
        const ts = useTrainingStore.getState();
        ts.setWorkerError(getErrorMessage(error, fallback));
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

    const initializeWorker = useCallback(async () => {
        const api = getWorkerApi();
        const ps = usePlaygroundStore.getState();
        const ts = useTrainingStore.getState();
        const config = ps.getConfig();
        const result = await api.initialize(
            config.network,
            config.training,
            config.data,
            config.features,
        );
        ts.clearWorkerError();
        ts.setSnapshot(result.snapshot);
        ts.resetHistory();
        if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
        ts.setFrameVersions(syncSnapshotToFrameBuffer(result.snapshot));
        newRunTo(result.runId);

        // Store points reactively so UI renders immediately
        const trainPts = await api.getTrainPoints();
        const testPts = await api.getTestPoints();
        ts.setTrainPoints(trainPts);
        ts.setTestPoints(testPts);

        // Record the initial config snapshot to prevent duplicate sync
        const latestState = usePlaygroundStore.getState();
        prevConfigRef.current = {
            network: latestState.network,
            training: latestState.training,
            data: latestState.data,
            features: latestState.features,
        };

        // Set up MessageChannel for streaming
        await setupStreamChannel();

        // Send initial demand
        await api.updateDemand(latestState.demand);

        // AS-4: tell the worker whether the user has opted in to the
        // WebGPU grid path. Capability detection still gates this; the
        // worker silently falls back to CPU when the device isn't
        // available or the network shape exceeds the shader caps.
        try {
            await api.setWebGpuEnabled(usePlaygroundStore.getState().featuresUI.webgpuGrid);
        } catch {
            // Older worker bundles won't expose setWebGpuEnabled — ignore.
        }

        initializedRef.current = true;
    }, []);

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
            const ts = useTrainingStore.getState();

            if (msg.type === 'snapshot') {
                ts.applyStreamedSnapshot({
                    snapshot: createStreamSnapshot(msg, ts.snapshot),
                    frameVersions: getFrameVersions(),
                    testMetricsStale: msg.scalars.testMetricsStale === true,
                });
            } else if (msg.type === 'status') {
                if (msg.status === 'paused' || msg.status === 'idle') {
                    ts.setStatus(msg.status);
                } else if (msg.status === 'running') {
                    ts.setStatus('running');
                }
            } else if (msg.type === 'error') {
                ts.setWorkerError(msg.message);
                ts.setStatus('paused');
            }
        });

        return unsubscribe;
    }, []);

    // Initialize worker on mount
    useEffect(() => {
        initializeWorker().catch((error) => {
            reportWorkerError(error, 'Failed to initialize training worker.');
        });
    }, [initializeWorker, reportWorkerError]);

    // Sync config changes to worker (rebuild when needed)
    useEffect(() => {
        if (!initializedRef.current) return;
        const nextSnapshot = { network, training, data, features };
        const isRetry = configSyncNonce !== prevConfigSyncNonceRef.current;
        if (!isRetry && prevConfigRef.current && structuralEqual(nextSnapshot, prevConfigRef.current)) return;
        const previousConfigSnapshot = prevConfigRef.current;
        prevConfigRef.current = nextSnapshot;
        prevConfigSyncNonceRef.current = configSyncNonce;
        const seq = beginConfigSync();

        const sync = async () => {
            // Stop streaming before config change
            if (isPlayingRef.current) {
                postStreamCommand({ type: 'stopTraining' });
                stopRenderLoop();
                isPlayingRef.current = false;
            }

            const api = getWorkerApi();
            const ps = usePlaygroundStore.getState();
            const ts = useTrainingStore.getState();
            try {
                const config = ps.getConfig();
                const result = await api.updateConfig(
                    config.network,
                    config.training,
                    config.data,
                    config.features,
                    false,
                );
                if (!isCurrentConfigSync(seq)) return;
                // Sync to worker's runId AFTER updateConfig returns
                newRunTo(result.runId);
                ts.setSnapshot(result.snapshot);
                ts.resetHistory();
                if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
                ts.setFrameVersions(syncSnapshotToFrameBuffer(result.snapshot));

                // Update points reactively
                const trainPts = await api.getTrainPoints();
                if (!isCurrentConfigSync(seq)) return;
                const testPts = await api.getTestPoints();
                if (!isCurrentConfigSync(seq)) return;
                ts.setTrainPoints(trainPts);
                ts.setTestPoints(testPts);

                ps.syncToUrl();
                ts.finishConfigChange();
                finishConfigSyncIfCurrent(seq);
            } catch (error) {
                if (!isCurrentConfigSync(seq)) return;
                prevConfigRef.current = previousConfigSnapshot;
                ts.failConfigChange(error instanceof Error ? error.message : 'Failed to update configuration');
                finishConfigSyncIfCurrent(seq);
            }
        };
        sync();
    }, [network, training, data, features, configSyncNonce, beginConfigSync, finishConfigSyncIfCurrent, isCurrentConfigSync]);

    // Sync demand changes to worker
    useEffect(() => {
        if (!initializedRef.current) return;
        postStreamCommand({ type: 'updateDemand', demand });
    }, [demand]);

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
        if (configSyncPendingRef.current || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }

        const startTraining = () => {
            isPlayingRef.current = true;
            useTrainingStore.getState().setStatus('running');
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
        isPlayingRef.current = false;
        postStreamCommand({ type: 'stopTraining' });
        stopRenderLoop();
        useTrainingStore.getState().setStatus('paused');
    }, []);

    const step = useCallback(async () => {
        if (isPlayingRef.current) {
            pause();
        }
        try {
            if (!initializedRef.current) {
                await initializeWorker();
            }
            const api = getWorkerApi();
            const snap = await api.step(1);
            const ts = useTrainingStore.getState();
            ts.setSnapshot(snap);
            ts.setFrameVersions(syncSnapshotToFrameBuffer(snap));
            if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);
        } catch (error) {
            reportWorkerError(error, 'Failed to run a training step.');
        }
    }, [initializeWorker, pause, reportWorkerError]);

    const reset = useCallback(async () => {
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'stopTraining' });
            stopRenderLoop();
            isPlayingRef.current = false;
        }
        const seq = beginConfigSync();
        try {
            if (!initializedRef.current) {
                await initializeWorker();
                if (!isCurrentConfigSync(seq)) return;
                useTrainingStore.getState().setStatus('idle');
                finishConfigSyncIfCurrent(seq);
                return;
            }
            const api = getWorkerApi();
            const result = await api.reset();
            if (!isCurrentConfigSync(seq)) return;
            newRunTo(result.runId);
            const ts = useTrainingStore.getState();
            ts.setSnapshot(result.snapshot);
            ts.resetHistory();
            if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
            ts.setFrameVersions(syncSnapshotToFrameBuffer(result.snapshot));
            ts.setStatus('idle');

            // Refresh points
            const trainPts = await api.getTrainPoints();
            if (!isCurrentConfigSync(seq)) return;
            const testPts = await api.getTestPoints();
            if (!isCurrentConfigSync(seq)) return;
            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);
            finishConfigSyncIfCurrent(seq);
        } catch (error) {
            if (!isCurrentConfigSync(seq)) return;
            reportWorkerError(error, 'Failed to reset training.');
            finishConfigSyncIfCurrent(seq);
        }
    }, [beginConfigSync, finishConfigSyncIfCurrent, initializeWorker, isCurrentConfigSync, reportWorkerError]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            activeConfigSyncSeqRef.current = configSyncSeqRef.current + 1;
            configSyncSeqRef.current = activeConfigSyncSeqRef.current;
            configSyncPendingRef.current = false;
            isPlayingRef.current = false;
            terminateWorker();
        };
    }, []);

    return { play, pause, step, reset };
}
