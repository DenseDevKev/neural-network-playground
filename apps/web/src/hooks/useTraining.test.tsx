import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { NetworkSnapshot } from '@nn-playground/engine';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { resetFrameBuffer } from '../worker/frameBuffer.ts';

const bridge = vi.hoisted(() => {
    const workerApi = {
        initialize: vi.fn(),
        updateConfig: vi.fn(),
        reset: vi.fn(),
        step: vi.fn(),
        getTrainPoints: vi.fn(),
        getTestPoints: vi.fn(),
        updateDemand: vi.fn(),
        setStreamPort: vi.fn(),
        setWebGpuEnabled: vi.fn(),
    };

    return {
        workerApi,
        setupStreamChannel: vi.fn(),
        postStreamCommand: vi.fn(),
        startRenderLoop: vi.fn(),
        stopRenderLoop: vi.fn(),
        onSnapshot: vi.fn(),
        newRunTo: vi.fn(),
        terminateWorker: vi.fn(),
    };
});

vi.mock('../worker/workerBridge.ts', () => ({
    getWorkerApi: () => bridge.workerApi,
    setupStreamChannel: bridge.setupStreamChannel,
    postStreamCommand: bridge.postStreamCommand,
    startRenderLoop: bridge.startRenderLoop,
    stopRenderLoop: bridge.stopRenderLoop,
    onSnapshot: bridge.onSnapshot,
    newRunTo: bridge.newRunTo,
    terminateWorker: bridge.terminateWorker,
}));

import { useTraining } from './useTraining.ts';

function makeSnapshot(step: number): NetworkSnapshot {
    return {
        step,
        epoch: Math.floor(step / 10),
        weights: [[[0.1, -0.2]]],
        biases: [[0.05]],
        trainLoss: 0.4 - step * 0.01,
        testLoss: 0.5 - step * 0.01,
        trainMetrics: { loss: 0.4 - step * 0.01, accuracy: 0.7 },
        testMetrics: { loss: 0.5 - step * 0.01, accuracy: 0.6 },
        outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
        gridSize: 2,
        neuronGrids: new Float32Array([0.4, 0.3, 0.2, 0.1]),
        historyPoint: {
            step,
            trainLoss: 0.4 - step * 0.01,
            testLoss: 0.5 - step * 0.01,
            trainAccuracy: 0.7,
            testAccuracy: 0.6,
        },
    };
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

function resetStores(): void {
    window.history.replaceState(null, '', '/');
    resetFrameBuffer();

    usePlaygroundStore.setState({
        data: { ...DEFAULT_DATA },
        network: {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            outputSize: 1,
            seed: DEFAULT_DATA.seed,
        },
        features: { ...DEFAULT_FEATURES },
        training: { ...DEFAULT_TRAINING },
        ui: { showTestData: false, discretizeOutput: false },
        featuresUI: { canvasNetworkGraph: true, webgpuGrid: true },
        demand: { ...DEFAULT_DEMAND },
    });

    useTrainingStore.getState().resetHistory();
    useTrainingStore.setState({
        status: 'idle',
        snapshot: null,
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
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
        configSyncNonce: 0,
        workerError: null,
        testMetricsStale: false,
    });
}

describe('useTraining', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        resetStores();

        bridge.workerApi.initialize.mockResolvedValue({ snapshot: makeSnapshot(1), runId: 101 });
        bridge.workerApi.updateConfig.mockResolvedValue({ snapshot: makeSnapshot(2), runId: 102 });
        bridge.workerApi.reset.mockResolvedValue({ snapshot: makeSnapshot(3), runId: 103 });
        bridge.workerApi.step.mockResolvedValue(makeSnapshot(4));
        bridge.workerApi.getTrainPoints.mockResolvedValue([{ x: 0, y: 1, label: 1 }]);
        bridge.workerApi.getTestPoints.mockResolvedValue([{ x: 1, y: 0, label: 0 }]);
        bridge.workerApi.updateDemand.mockResolvedValue(undefined);
        bridge.workerApi.setWebGpuEnabled.mockResolvedValue(undefined);
        bridge.setupStreamChannel.mockResolvedValue(undefined);
        bridge.onSnapshot.mockReturnValue(() => {});
    });

    it('initializes the worker and hydrates runtime state on mount', async () => {
        renderHook(() => useTraining());

        await waitFor(() => expect(bridge.workerApi.initialize).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        expect(bridge.newRunTo).toHaveBeenCalledWith(101);
        expect(bridge.setupStreamChannel).toHaveBeenCalledTimes(1);
        expect(bridge.workerApi.updateDemand).toHaveBeenCalledWith(DEFAULT_DEMAND);
        expect(bridge.workerApi.setWebGpuEnabled).toHaveBeenCalledWith(true);
        expect(useTrainingStore.getState().trainPoints).toEqual([{ x: 0, y: 1, label: 1 }]);
        expect(useTrainingStore.getState().testPoints).toEqual([{ x: 1, y: 0, label: 0 }]);
        expect(useTrainingStore.getState().paramsVersion).toBeGreaterThan(0);
    });

    it('starts and stops the streaming training loop', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });

        expect(useTrainingStore.getState().status).toBe('running');
        expect(bridge.startRenderLoop).toHaveBeenCalledTimes(1);
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({
            type: 'startTraining',
            stepsPerFrame: 5,
        });

        act(() => {
            result.current.pause();
        });

        expect(useTrainingStore.getState().status).toBe('paused');
        expect(bridge.stopRenderLoop).toHaveBeenCalledTimes(1);
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });
    });

    it('reports a worker error when a manual step fails', async () => {
        bridge.workerApi.step.mockRejectedValueOnce(new Error('step exploded'));
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        await act(async () => {
            await result.current.step();
        });

        expect(useTrainingStore.getState().workerError).toBe('step exploded');
        expect(useTrainingStore.getState().status).toBe('paused');
    });

    it('syncs config changes successfully and clears config loading state', async () => {
        const { unmount } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.updateConfig.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            usePlaygroundStore.getState().setNumSamples(DEFAULT_DATA.numSamples + 1);
        });

        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(2));

        expect(bridge.newRunTo).toHaveBeenCalledWith(102);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);
        expect(useTrainingStore.getState().configError).toBeNull();

        unmount();
    });

    it('records config sync failures and keeps the previous config snapshot retryable', async () => {
        const { unmount } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.updateConfig.mockRejectedValueOnce(new Error('bad network'));

        act(() => {
            useTrainingStore.getState().beginConfigChange('network');
            usePlaygroundStore.getState().setHiddenLayers([5, 3]);
        });

        await waitFor(() => expect(useTrainingStore.getState().configError).toBe('bad network'));

        expect(useTrainingStore.getState().configErrorSource).toBe('network');
        expect(useTrainingStore.getState().networkConfigLoading).toBe(false);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();

        bridge.workerApi.updateConfig.mockResolvedValueOnce({ snapshot: makeSnapshot(5), runId: 105 });
        act(() => {
            useTrainingStore.getState().retryConfigSync();
        });

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(5));
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(bridge.newRunTo).toHaveBeenCalledWith(105);

        unmount();
    });

    it('forwards WebGPU grid toggles to the worker after initialization', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.setWebGpuEnabled.mockClear();

        act(() => {
            usePlaygroundStore.setState((state) => ({
                featuresUI: { ...state.featuresUI, webgpuGrid: false },
            }));
        });

        await waitFor(() => expect(bridge.workerApi.setWebGpuEnabled).toHaveBeenCalledWith(false));
    });

    it('drops stale config sync completions after a newer config wins', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        const first = deferred<{ snapshot: NetworkSnapshot; runId: number }>();
        const second = deferred<{ snapshot: NetworkSnapshot; runId: number }>();
        bridge.workerApi.updateConfig
            .mockReset()
            .mockImplementationOnce(() => first.promise)
            .mockImplementationOnce(() => second.promise);
        bridge.newRunTo.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            usePlaygroundStore.getState().setNumSamples(DEFAULT_DATA.numSamples + 1);
        });
        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));

        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            usePlaygroundStore.getState().setNumSamples(DEFAULT_DATA.numSamples + 2);
        });
        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(2));

        await act(async () => {
            second.resolve({ snapshot: makeSnapshot(20), runId: 220 });
            await second.promise;
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));

        await act(async () => {
            first.resolve({ snapshot: makeSnapshot(10), runId: 210 });
            await first.promise;
        });

        expect(useTrainingStore.getState().snapshot?.step).toBe(20);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(bridge.newRunTo).toHaveBeenCalledWith(220);
        expect(bridge.newRunTo).not.toHaveBeenCalledWith(210);
    });

    it('does not start training while a config sync is pending', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        const pending = deferred<{ snapshot: NetworkSnapshot; runId: number }>();
        bridge.workerApi.updateConfig.mockReset().mockReturnValueOnce(pending.promise);
        bridge.postStreamCommand.mockClear();
        bridge.startRenderLoop.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            usePlaygroundStore.getState().setNumSamples(DEFAULT_DATA.numSamples + 1);
        });
        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));

        act(() => {
            result.current.play();
        });

        expect(useTrainingStore.getState().status).toBe('idle');
        expect(bridge.startRenderLoop).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'startTraining' }));

        await act(async () => {
            pending.resolve({ snapshot: makeSnapshot(30), runId: 230 });
            await pending.promise;
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(30));
    });
});
