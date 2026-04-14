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
    getFrameVersion,
    updateFrameBuffer,
} from '../worker/frameBuffer.ts';
import type { NetworkSnapshot } from '@nn-playground/engine';
import type { WorkerSnapshotMessage, WorkerToMainMessage } from '@nn-playground/shared';


export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => void;
    reset: () => void;
}

function getTotalNeuronCount(layerSizes: number[]): number {
    let total = 0;
    for (let i = 1; i < layerSizes.length; i++) {
        total += layerSizes[i];
    }
    return total;
}

function syncSnapshotToFrameBuffer(snapshot: NetworkSnapshot): number {
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

    return updateFrameBuffer({
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
            confusionMatrix: msg.confusionMatrix,
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
    const prevConfigRef = useRef<string>('');
    const prevConfigSyncNonceRef = useRef(0);
    const stepsPerFrameRef = useRef(5);
    const isPlayingRef = useRef(false);

    // Config selectors (from playground store — stable, rarely changes)
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const demand = usePlaygroundStore((s) => s.demand);

    // Runtime selectors (from training store — volatile)
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const configSyncNonce = useTrainingStore((s) => s.configSyncNonce);

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
                ts.clearWorkerError();
                ts.setSnapshot(createStreamSnapshot(msg, ts.snapshot));
                ts.setFrameVersion(getFrameVersion());
                if (msg.historyPoint) ts.addHistoryPoint(msg.historyPoint);
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
        const init = async () => {
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
            ts.setSnapshot(result.snapshot);
            ts.resetHistory();
            if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
            ts.setFrameVersion(syncSnapshotToFrameBuffer(result.snapshot));
            newRunTo(result.runId);

            // Store points reactively so UI renders immediately
            const trainPts = await api.getTrainPoints();
            const testPts = await api.getTestPoints();
            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);

            // Set initial config string to prevent duplicate sync
            prevConfigRef.current = JSON.stringify({
                network: ps.network,
                training: ps.training,
                data: ps.data,
                features: ps.features,
            });

            // Set up MessageChannel for streaming
            await setupStreamChannel();

            // Send initial demand
            await api.updateDemand(usePlaygroundStore.getState().demand);

            initializedRef.current = true;
        };
        init();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Sync config changes to worker (rebuild when needed)
    useEffect(() => {
        if (!initializedRef.current) return;
        const configStr = JSON.stringify({ network, training, data, features });
        const isRetry = configSyncNonce !== prevConfigSyncNonceRef.current;
        if (!isRetry && configStr === prevConfigRef.current) return;
        const previousConfigStr = prevConfigRef.current;
        prevConfigRef.current = configStr;
        prevConfigSyncNonceRef.current = configSyncNonce;

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
                // Sync to worker's runId AFTER updateConfig returns
                newRunTo(result.runId);
                ts.setSnapshot(result.snapshot);
                ts.resetHistory();
                if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
                ts.setFrameVersion(syncSnapshotToFrameBuffer(result.snapshot));

                // Update points reactively
                const trainPts = await api.getTrainPoints();
                const testPts = await api.getTestPoints();
                ts.setTrainPoints(trainPts);
                ts.setTestPoints(testPts);

                ps.syncToUrl();
                ts.finishConfigChange();
            } catch (error) {
                prevConfigRef.current = previousConfigStr;
                ts.failConfigChange(error instanceof Error ? error.message : 'Failed to update configuration');
            }
        };
        sync();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [network, training, data, features, configSyncNonce]);

    // Sync demand changes to worker
    useEffect(() => {
        if (!initializedRef.current) return;
        postStreamCommand({ type: 'updateDemand', demand });
    }, [demand]);

    const play = useCallback(() => {
        isPlayingRef.current = true;
        useTrainingStore.getState().setStatus('running');
        startRenderLoop();
        postStreamCommand({ type: 'startTraining', stepsPerFrame: stepsPerFrameRef.current });
    }, []);

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
        const api = getWorkerApi();
        const snap = await api.step(1);
        const ts = useTrainingStore.getState();
        ts.setSnapshot(snap);
        ts.setFrameVersion(syncSnapshotToFrameBuffer(snap));
        if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);
    }, [pause]);

    const reset = useCallback(async () => {
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'stopTraining' });
            stopRenderLoop();
            isPlayingRef.current = false;
        }
        const api = getWorkerApi();
        const result = await api.reset();
        newRunTo(result.runId);
        const ts = useTrainingStore.getState();
        ts.setSnapshot(result.snapshot);
        ts.resetHistory();
        if (result.snapshot.historyPoint) ts.addHistoryPoint(result.snapshot.historyPoint);
        ts.setFrameVersion(syncSnapshotToFrameBuffer(result.snapshot));
        ts.setStatus('idle');

        // Refresh points
        const trainPts = await api.getTrainPoints();
        const testPts = await api.getTestPoints();
        ts.setTrainPoints(trainPts);
        ts.setTestPoints(testPts);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            isPlayingRef.current = false;
            terminateWorker();
        };
    }, []);

    return { play, pause, step, reset };
}
