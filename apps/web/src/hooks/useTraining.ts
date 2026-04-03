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
    newRun,
} from '../worker/workerBridge.ts';
import { updateFrameBuffer } from '../worker/frameBuffer.ts';
import type { WorkerToMainMessage } from '@nn-playground/shared';


export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => void;
    reset: () => void;
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
                // Apply snapshot scalars to training store
                ts.setSnapshot({
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
                    // Placeholders — real data lives in frameBuffer
                    weights: [],
                    biases: [],
                    outputGrid: [],
                    gridSize: msg.scalars.gridSize,
                    historyPoint: msg.historyPoint,
                } as any);
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
            const snap = await api.initialize(
                config.network,
                config.training,
                config.data,
                config.features,
            );
            ts.setSnapshot(snap);
            ts.resetHistory();
            if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);

            // Populate frame buffer from initial snapshot
            updateFrameBuffer({
                outputGrid: snap.outputGrid ? new Float32Array(snap.outputGrid) : null,
                gridSize: snap.gridSize,
                weights: null,
                biases: null,
            });

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
                newRun(); // Invalidate any in-flight snapshots
                const config = ps.getConfig();
                const snap = await api.updateConfig(
                    config.network,
                    config.training,
                    config.data,
                    config.features,
                    true,
                );
                ts.setSnapshot(snap);
                ts.resetHistory();
                if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);

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
        if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);
    }, [pause]);

    const reset = useCallback(async () => {
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'stopTraining' });
            stopRenderLoop();
            isPlayingRef.current = false;
        }
        newRun(); // Invalidate in-flight snapshots
        const api = getWorkerApi();
        const snap = await api.reset();
        const ts = useTrainingStore.getState();
        ts.setSnapshot(snap);
        ts.resetHistory();
        if (snap.historyPoint) ts.addHistoryPoint(snap.historyPoint);
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
            postStreamCommand({ type: 'stopTraining' });
            stopRenderLoop();
        };
    }, []);

    return { play, pause, step, reset };
}
