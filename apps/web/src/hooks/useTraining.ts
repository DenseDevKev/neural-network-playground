// ── useTraining hook ──
// Manages the training loop, worker communication, and data synchronization.

import { useEffect, useRef, useCallback } from 'react';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { getWorkerApi } from '../worker/workerApi.ts';


export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => void;
    reset: () => void;
}

export function useTraining(): TrainingHook {
    // All refs first (stable hook order)
    const rafRef = useRef<number | null>(null);
    const isRunningRef = useRef(false);
    const initializedRef = useRef(false);
    const prevConfigRef = useRef<string>('');
    const stepsPerFrameRef = useRef(5);

    // All store selectors next (stable hook order)
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const stepsPerFrame = usePlaygroundStore((s) => s.stepsPerFrame);

    // Keep ref in sync so the RAF loop always sees the current speed.
    useEffect(() => {
        stepsPerFrameRef.current = stepsPerFrame;
    }, [stepsPerFrame]);

    // Initialize worker on mount
    useEffect(() => {
        const init = async () => {
            const api = getWorkerApi();
            const s = usePlaygroundStore.getState();
            const config = s.getConfig();
            const snap = await api.initialize(
                config.network,
                config.training,
                config.data,
                config.features,
            );
            s.setSnapshot(snap);
            s.resetHistory();
            if (snap.historyPoint) s.addHistoryPoint(snap.historyPoint);

            // Store points reactively so UI renders immediately
            const trainPts = await api.getTrainPoints();
            const testPts = await api.getTestPoints();
            s.setTrainPoints(trainPts);
            s.setTestPoints(testPts);

            // Set initial config string to prevent duplicate sync
            prevConfigRef.current = JSON.stringify({
                network: s.network,
                training: s.training,
                data: s.data,
                features: s.features,
            });

            initializedRef.current = true;
        };
        init();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Sync config changes to worker (rebuild when needed)
    useEffect(() => {
        if (!initializedRef.current) return;
        const configStr = JSON.stringify({ network, training, data, features });
        if (configStr === prevConfigRef.current) return;
        prevConfigRef.current = configStr;

        const sync = async () => {
            const api = getWorkerApi();
            const s = usePlaygroundStore.getState();
            const config = s.getConfig();
            const snap = await api.updateConfig(
                config.network,
                config.training,
                config.data,
                config.features,
                true,
            );
            s.setSnapshot(snap);
            s.resetHistory();
            if (snap.historyPoint) s.addHistoryPoint(snap.historyPoint);

            // Update points reactively
            const trainPts = await api.getTrainPoints();
            const testPts = await api.getTestPoints();
            s.setTrainPoints(trainPts);
            s.setTestPoints(testPts);

            s.syncToUrl();
        };
        sync();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [network, training, data, features]);

    // Training loop
    const trainLoop = useCallback(async () => {
        if (!isRunningRef.current) return;
        try {
            const api = getWorkerApi();
            const snap = await api.trainAndSnapshot(stepsPerFrameRef.current);
            const s = usePlaygroundStore.getState();
            s.setSnapshot(snap);
            if (snap.historyPoint) s.addHistoryPoint(snap.historyPoint);
        } catch {
            isRunningRef.current = false;
            usePlaygroundStore.getState().setStatus('idle');
            return;
        }
        if (isRunningRef.current) {
            rafRef.current = requestAnimationFrame(trainLoop);
        }
    }, []);

    const play = useCallback(() => {
        isRunningRef.current = true;
        usePlaygroundStore.getState().setStatus('running');
        rafRef.current = requestAnimationFrame(trainLoop);
    }, [trainLoop]);

    const pause = useCallback(() => {
        isRunningRef.current = false;
        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        }
        usePlaygroundStore.getState().setStatus('paused');
    }, []);

    const step = useCallback(async () => {
        pause();
        const api = getWorkerApi();
        const snap = await api.step(1);
        const s = usePlaygroundStore.getState();
        s.setSnapshot(snap);
        if (snap.historyPoint) s.addHistoryPoint(snap.historyPoint);
    }, [pause]);

    const reset = useCallback(async () => {
        pause();
        const api = getWorkerApi();
        const snap = await api.reset();
        const s = usePlaygroundStore.getState();
        s.setSnapshot(snap);
        s.resetHistory();
        if (snap.historyPoint) s.addHistoryPoint(snap.historyPoint);
        s.setStatus('idle');

        // Refresh points
        const trainPts = await api.getTrainPoints();
        const testPts = await api.getTestPoints();
        s.setTrainPoints(trainPts);
        s.setTestPoints(testPts);
    }, [pause]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            isRunningRef.current = false;
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
            }
        };
    }, []);

    return { play, pause, step, reset };
}
