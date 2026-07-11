import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { NetworkSnapshot } from '@nn-playground/engine';
import {
    type ArenaScalarSnapshot,
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';
import type { WorkerToMainMessage } from '@nn-playground/shared';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { setNoise } from '../store/recipeEdits.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { getFrameBuffer, getFrameVersions, resetFrameBuffer, updateFrameBuffer } from '../worker/frameBuffer.ts';

const bridge = vi.hoisted(() => {
    const workerApi = {
        initialize: vi.fn(),
        updateConfig: vi.fn(),
        reset: vi.fn(),
        step: vi.fn(),
        restoreCheckpoint: vi.fn(),
        initializeArena: vi.fn(),
        stepArena: vi.fn(),
        getCheckpointTimeline: vi.fn(),
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

const INITIAL_PREPARED = usePlaygroundStore.getState().prepared;

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

function withActivationHistograms(snapshot: NetworkSnapshot): NetworkSnapshot {
    return {
        ...snapshot,
        activationHistograms: {
            bins: new Float32Array([1, 3, 0, 2]),
            layers: [
                {
                    layerIndex: 0,
                    binCount: 4,
                    binStart: -1,
                    binWidth: 0.5,
                    minActivation: -0.5,
                    maxActivation: 0.8,
                    totalCount: 6,
                    nearZeroCount: 1,
                    saturatedCount: 2,
                },
            ],
        },
    };
}

function withConfusionMatrix(snapshot: NetworkSnapshot): NetworkSnapshot {
    return {
        ...snapshot,
        testMetrics: {
            ...snapshot.testMetrics,
            confusionMatrix: { tp: 5, tn: 4, fp: 3, fn: 2 },
        },
    };
}

function makeArenaSnapshot(step: number): ArenaScalarSnapshot {
    return {
        runId: 501,
        snapshotId: step,
        summaries: [
            {
                side: 'A',
                label: 'Tuned model',
                status: 'paused',
                pauseReason: 'manual',
                step,
                epoch: 0,
                trainLoss: 0.25,
                testLoss: 0.32,
                trainAccuracy: 0.86,
                testAccuracy: 0.8,
            },
            {
                side: 'B',
                label: 'Baseline',
                status: 'paused',
                pauseReason: 'manual',
                step,
                epoch: 0,
                trainLoss: 0.4,
                testLoss: 0.52,
                trainAccuracy: 0.74,
                testAccuracy: 0.68,
            },
        ],
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
        prepared: INITIAL_PREPARED,
        preparation: { status: 'ready', requestId: 0, issues: [] },
        incompatibleSource: null,
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
        trainedRecipeConfig: null,
        trainedRecipeFingerprint: null,
        trainedRecipeRecordedAt: null,
        trainedRecipeSource: null,
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
        presetConfigLoading: false,
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
        configSyncNonce: 0,
        workerError: null,
        pauseReason: null,
        testMetricsStale: false,
        arenaSummariesVersion: 0,
        arenaSummaries: null,
        multiclassBoundaryVersion: 0,
    });
}

function seedApprovedMulticlassStoreState(): void {
    usePlaygroundStore.setState((state) => ({
        data: {
            ...state.data,
            dataset: 'three-class-clusters',
            problemType: 'classification',
        },
        network: {
            ...state.network,
            outputSize: 3,
            outputActivation: 'softmax',
        },
        training: {
            ...state.training,
            lossType: 'categoricalCrossEntropy',
        },
    }));
}

function getStreamHandler(): (msg: WorkerToMainMessage) => void {
    const handler = bridge.onSnapshot.mock.calls[0]?.[0] as ((msg: WorkerToMainMessage) => void) | undefined;
    expect(handler).toBeTypeOf('function');
    return handler!;
}

describe('useTraining', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        resetStores();

        bridge.workerApi.initialize.mockResolvedValue({ snapshot: makeSnapshot(1), runId: 101 });
        bridge.workerApi.updateConfig.mockResolvedValue({ snapshot: makeSnapshot(2), runId: 102 });
        bridge.workerApi.reset.mockResolvedValue({ snapshot: makeSnapshot(3), runId: 103 });
        bridge.workerApi.step.mockResolvedValue(makeSnapshot(4));
        bridge.workerApi.initializeArena.mockResolvedValue(makeArenaSnapshot(0));
        bridge.workerApi.stepArena.mockResolvedValue(makeArenaSnapshot(1));
        bridge.workerApi.restoreCheckpoint.mockResolvedValue({
            snapshot: makeSnapshot(0),
            runId: 101,
            timeline: {
                checkpoints: [
                    { id: 1, step: 0, epoch: 0, trainLoss: 0.4, testLoss: 0.5, label: 'Step 0' },
                ],
                maxCheckpoints: 8,
                evictedCount: 0,
                liveCheckpointId: 1,
                restoredCheckpointId: 1,
            },
        });
        bridge.workerApi.getCheckpointTimeline.mockResolvedValue({
            checkpoints: [],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
        });
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
        expect(useTrainingStore.getState().trainedRecipeConfig?.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(useTrainingStore.getState().trainedRecipeFingerprint)
            .toBe(INITIAL_PREPARED?.identities.recipeFingerprint);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('initialize');
    });

    it('does not initialize training when strict URL initialization has no prepared document', async () => {
        usePlaygroundStore.setState({
            prepared: null,
            preparation: {
                status: 'error',
                requestId: 0,
                issues: [{
                    code: 'legacy-state',
                    path: '$',
                    message: 'legacy URL',
                }],
            },
            incompatibleSource: { kind: 'url', raw: '#d=xor' },
        });

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().workerError)
            .toMatch(/shared experiment URL is incompatible with version 2/i));
        expect(bridge.workerApi.initialize).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('does not initialize public training with hidden multiclass configs', async () => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...state.training,
                lossType: 'categoricalCrossEntropy',
            },
        }));

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().workerError).toMatch(/multiclass configurations/i));
        expect(bridge.workerApi.initialize).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('initializes approved multiclass configs through the public training hook', async () => {
        seedApprovedMulticlassStoreState();

        renderHook(() => useTraining());

        await waitFor(() => expect(bridge.workerApi.initialize).toHaveBeenCalledTimes(1));
        expect(bridge.workerApi.initialize).toHaveBeenCalledWith(
            expect.objectContaining({ outputSize: 3, outputActivation: 'softmax' }),
            expect.objectContaining({ lossType: 'categoricalCrossEntropy' }),
            expect.objectContaining({ dataset: 'three-class-clusters', problemType: 'classification' }),
            expect.any(Object),
        );
        expect(useTrainingStore.getState().workerError).toBeNull();
    });

    it('hydrates fresh worker snapshots in the existing store update order', async () => {
        const callOrder: string[] = [];
        const original = useTrainingStore.getState();
        useTrainingStore.setState({
            setSnapshot: (snapshot) => {
                callOrder.push('setSnapshot');
                original.setSnapshot(snapshot);
            },
            resetHistory: () => {
                callOrder.push('resetHistory');
                original.resetHistory();
            },
            addHistoryPoint: (point) => {
                callOrder.push('addHistoryPoint');
                original.addHistoryPoint(point);
            },
            setFrameVersions: (versions) => {
                callOrder.push('setFrameVersions');
                original.setFrameVersions(versions);
            },
        });

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        expect(callOrder).toEqual([
            'setSnapshot',
            'resetHistory',
            'addHistoryPoint',
            'setFrameVersions',
        ]);
    });

    it('keeps bounded activation histogram arrays in the frame buffer only', async () => {
        bridge.workerApi.initialize.mockResolvedValue({
            snapshot: withActivationHistograms(makeSnapshot(1)),
            runId: 101,
        });

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        expect(getFrameBuffer().activationHistogramBins).toEqual(new Float32Array([1, 3, 0, 2]));
        expect(getFrameBuffer().activationHistogramLayout?.layers).toHaveLength(1);
        expect(useTrainingStore.getState().snapshot?.activationHistograms).toBeUndefined();
    });

    it('clears stale multiclass boundary frames when applying a fresh direct snapshot', async () => {
        updateFrameBuffer({
            multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
            multiclassConfidenceGrid: new Float32Array([0.8, 0.7, 0.6, 0.5]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
        });
        const initialMulticlassVersion = getFrameBuffer().multiclassBoundaryVersion;
        bridge.workerApi.initialize.mockResolvedValue({
            snapshot: {
                ...makeSnapshot(1),
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
            },
            runId: 101,
        });

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        expect(getFrameBuffer().multiclassClassGrid).toBeNull();
        expect(getFrameBuffer().multiclassConfidenceGrid).toBeNull();
        expect(getFrameBuffer().multiclassBoundaryLayout).toBeNull();
        expect(getFrameBuffer().multiclassBoundaryVersion).toBe(initialMulticlassVersion + 1);
    });

    it('clears stale worker-authored multiclass confusion data when stepping through a direct snapshot', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        updateFrameBuffer({
            multiclassConfusionMatrix: {
                classCount: 3,
                classLabels: [0, 1, 2],
                counts: [2, 1, 0, 0, 3, 1, 1, 0, 4],
            },
        });
        const initialMulticlassConfusionVersion = getFrameBuffer().multiclassConfusionMatrixVersion;

        await act(async () => {
            await result.current.step();
        });

        expect(getFrameBuffer().multiclassConfusionMatrix).toBeNull();
        expect(getFrameBuffer().multiclassConfusionMatrixVersion).toBe(
            initialMulticlassConfusionVersion + 1,
        );
        expect('multiclassConfusionMatrixVersion' in getFrameVersions()).toBe(false);
    });

    it('does not churn multiclass confusion state when direct sync has no stale worker-authored data', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const initialMulticlassConfusionVersion = getFrameBuffer().multiclassConfusionMatrixVersion;

        await act(async () => {
            await result.current.step();
        });

        expect(getFrameBuffer().multiclassConfusionMatrix).toBeNull();
        expect(getFrameBuffer().multiclassConfusionMatrixVersion).toBe(initialMulticlassConfusionVersion);
        expect('multiclassConfusionMatrixVersion' in useTrainingStore.getState()).toBe(false);
    });

    it('applies a freshly computed multiclass boundary from a direct step snapshot', async () => {
        const classGrid = new Uint8Array([0, 1, 2, 1]);
        const confidenceGrid = new Float32Array([0.9, 0.7, 0.8, 0.6]);
        const directSnapshot: NetworkSnapshot = {
            ...makeSnapshot(2),
            multiclassBoundary: {
                classGrid,
                confidenceGrid,
                gridSize: 2,
            },
        };
        bridge.workerApi.step.mockResolvedValue(directSnapshot);
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        await act(async () => {
            await result.current.step();
        });

        expect(getFrameBuffer().multiclassClassGrid).toEqual(classGrid);
        expect(getFrameBuffer().multiclassConfidenceGrid).toEqual(confidenceGrid);
        expect(getFrameBuffer().multiclassBoundaryLayout).toEqual({
            gridSize: 2,
            classCount: 3,
            classLabels: [0, 1, 2],
        });
    });

    it('clears stale worker-authored multiclass confusion data when resetting through a direct snapshot', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        updateFrameBuffer({
            multiclassConfusionMatrix: {
                classCount: 3,
                classLabels: [0, 1, 2],
                counts: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
        });
        const initialMulticlassConfusionVersion = getFrameBuffer().multiclassConfusionMatrixVersion;

        await act(async () => {
            await result.current.reset();
        });

        expect(getFrameBuffer().multiclassConfusionMatrix).toBeNull();
        expect(getFrameBuffer().multiclassConfusionMatrixVersion).toBe(
            initialMulticlassConfusionVersion + 1,
        );
    });

    it('retains the last activation histogram frame data across cadence-skipped snapshots', async () => {
        bridge.workerApi.initialize.mockResolvedValue({
            snapshot: withActivationHistograms(makeSnapshot(1)),
            runId: 101,
        });
        bridge.workerApi.step.mockResolvedValue(makeSnapshot(2));

        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        const histogramBins = getFrameBuffer().activationHistogramBins;
        expect(histogramBins).toEqual(new Float32Array([1, 3, 0, 2]));

        await act(async () => {
            await result.current.step();
        });

        expect(getFrameBuffer().activationHistogramBins).toBe(histogramBins);
        expect(getFrameBuffer().activationHistogramLayout?.layers).toHaveLength(1);
        expect(useTrainingStore.getState().snapshot?.step).toBe(2);
        expect(useTrainingStore.getState().snapshot?.activationHistograms).toBeUndefined();
    });

    it('initializes and steps the scalar live arena through bounded frame-buffer summaries', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        const modelA = {
            label: 'Tuned model',
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: 7 },
            training: { ...DEFAULT_TRAINING, learningRate: 0.03 },
            data: { ...DEFAULT_DATA, seed: 7 },
            features: { ...DEFAULT_FEATURES },
        };
        const modelB = {
            label: 'Baseline',
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: 11 },
            training: { ...DEFAULT_TRAINING, learningRate: 0.01 },
            data: { ...DEFAULT_DATA, seed: 11 },
            features: { ...DEFAULT_FEATURES },
        };

        await act(async () => {
            await result.current.initializeArena(modelA, modelB);
        });

        expect(bridge.workerApi.initializeArena).toHaveBeenCalledWith({ modelA, modelB });
        expect(getFrameBuffer().arenaSummaries?.map((summary) => summary.label)).toEqual(['Tuned model', 'Baseline']);
        expect(useTrainingStore.getState().arenaSummaries?.[0]?.testLoss).toBe(0.32);
        const initializedVersion = useTrainingStore.getState().arenaSummariesVersion;
        expect(initializedVersion).toBeGreaterThan(0);

        await act(async () => {
            await result.current.stepArena(2);
        });

        expect(bridge.workerApi.stepArena).toHaveBeenCalledWith(2);
        expect(useTrainingStore.getState().arenaSummaries?.[0]?.step).toBe(1);
        expect(useTrainingStore.getState().arenaSummariesVersion).toBeGreaterThan(initializedVersion);
        expect(useTrainingStore.getState().snapshot?.step).toBe(1);
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
        expect(useTrainingStore.getState().pauseReason).toBe('manual');
        expect(bridge.stopRenderLoop).toHaveBeenCalledTimes(1);
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });

        act(() => {
            result.current.play();
        });

        expect(useTrainingStore.getState().pauseReason).toBeNull();
    });

    it('does not pause when training is not running', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.stopRenderLoop.mockClear();
        bridge.postStreamCommand.mockClear();

        act(() => {
            result.current.pause();
        });

        expect(useTrainingStore.getState().status).toBe('idle');
        expect(useTrainingStore.getState().pauseReason).toBeNull();
        expect(bridge.stopRenderLoop).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalled();
    });

    it('records automatic worker pause reasons and stops the local render loop', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        bridge.stopRenderLoop.mockClear();

        act(() => {
            getStreamHandler()({
                type: 'status',
                runId: 101,
                status: 'paused',
                pauseReason: 'diverged',
            });
        });

        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('diverged');
        expect(bridge.stopRenderLoop).toHaveBeenCalledTimes(1);
    });

    it('preserves the final streamed snapshot before applying automatic paused status', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const handler = getStreamHandler();

        act(() => {
            handler({
                type: 'snapshot',
                runId: 101,
                snapshotId: 2,
                scalars: {
                    step: 9,
                    epoch: 0,
                    trainLoss: 0.2,
                    testLoss: 0.3,
                    gridSize: 2,
                },
                historyPoint: { step: 9, trainLoss: 0.2, testLoss: 0.3 },
            });
            handler({
                type: 'status',
                runId: 101,
                status: 'paused',
                pauseReason: 'diverged',
            });
        });

        expect(useTrainingStore.getState().snapshot?.step).toBe(9);
        expect(useTrainingStore.getState().pauseReason).toBe('diverged');
    });

    it('clears stale binary confusion when a fresh streamed snapshot omits confusion data', async () => {
        bridge.workerApi.initialize.mockResolvedValue({
            snapshot: withConfusionMatrix(makeSnapshot(1)),
            runId: 101,
        });

        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.testMetrics.confusionMatrix).toEqual({
            tp: 5,
            tn: 4,
            fp: 3,
            fn: 2,
        }));
        const handler = getStreamHandler();

        act(() => {
            handler({
                type: 'snapshot',
                runId: 101,
                snapshotId: 2,
                scalars: {
                    step: 2,
                    epoch: 0,
                    trainLoss: 0.25,
                    testLoss: 0.35,
                    gridSize: 2,
                    testMetricsStale: true,
                },
                historyPoint: { step: 2, trainLoss: 0.25, testLoss: 0.35 },
            });
        });

        expect(useTrainingStore.getState().snapshot?.testMetrics.confusionMatrix).toEqual({
            tp: 5,
            tn: 4,
            fp: 3,
            fn: 2,
        });

        act(() => {
            handler({
                type: 'snapshot',
                runId: 101,
                snapshotId: 3,
                scalars: {
                    step: 3,
                    epoch: 0,
                    trainLoss: 0.2,
                    testLoss: 0.3,
                    gridSize: 2,
                    testMetricsStale: false,
                },
                outputGrid: new Float32Array(),
                confusionMatrix: {
                    tp: 1,
                    tn: 2,
                    fp: 3,
                    fn: 4,
                },
                historyPoint: { step: 3, trainLoss: 0.2, testLoss: 0.3 },
            });
        });

        expect(useTrainingStore.getState().snapshot?.testMetrics.confusionMatrix).toEqual({
            tp: 1,
            tn: 2,
            fp: 3,
            fn: 4,
        });

        act(() => {
            handler({
                type: 'snapshot',
                runId: 101,
                snapshotId: 4,
                scalars: {
                    step: 4,
                    epoch: 0,
                    trainLoss: 0.2,
                    testLoss: 0.3,
                    gridSize: 2,
                    testMetricsStale: false,
                },
                outputGrid: new Float32Array(),
                historyPoint: { step: 4, trainLoss: 0.2, testLoss: 0.3 },
            });
        });

        expect(useTrainingStore.getState().snapshot?.testMetrics.confusionMatrix).toBeUndefined();
    });

    it('does not mark config-sync internal stops as manual pauses', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            usePlaygroundStore.getState().setNumSamples(DEFAULT_DATA.numSamples + 1);
        });

        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));
        expect(useTrainingStore.getState().pauseReason).toBeNull();
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
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('restores a worker checkpoint and applies its lightweight timeline metadata', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        await act(async () => {
            await result.current.restoreCheckpoint(1);
        });

        expect(bridge.workerApi.restoreCheckpoint).toHaveBeenCalledWith(1);
        expect(useTrainingStore.getState().snapshot?.step).toBe(0);
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().checkpointTimeline.restoredCheckpointId).toBe(1);
        expect(getFrameBuffer().weights).toBeInstanceOf(Float32Array);
    });

    it('does not label a deferred restore with a recipe that changed while restoration was pending', async () => {
        const restoreResult = {
            snapshot: makeSnapshot(0),
            runId: 101,
            timeline: {
                checkpoints: [
                    { id: 1, step: 0, epoch: 0, trainLoss: 0.4, testLoss: 0.5, label: 'Step 0' },
                ],
                maxCheckpoints: 8,
                evictedCount: 0,
                liveCheckpointId: 1,
                restoredCheckpointId: 1,
            },
        };
        const restoreGate = deferred<typeof restoreResult>();
        bridge.workerApi.restoreCheckpoint.mockReturnValueOnce(restoreGate.promise);
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.updateConfig.mockClear();

        let pendingRestore!: Promise<void>;
        act(() => {
            pendingRestore = result.current.restoreCheckpoint(1);
        });
        await waitFor(() => expect(bridge.workerApi.restoreCheckpoint).toHaveBeenCalledWith(1));

        await act(async () => {
            await usePlaygroundStore.getState().editRecipe((recipe) => (
                setNoise(recipe, recipe.data.noise + 1)
            ));
        });
        const editedFingerprint = usePlaygroundStore.getState().prepared!.identities.recipeFingerprint;
        await act(async () => {
            await Promise.resolve();
        });
        expect(bridge.workerApi.updateConfig).not.toHaveBeenCalled();

        restoreGate.resolve(restoreResult);
        await act(async () => {
            await pendingRestore;
        });
        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(useTrainingStore.getState().trainedRecipeSource).toBe('config-sync'));

        expect(useTrainingStore.getState().trainedRecipeFingerprint).toBe(editedFingerprint);
        expect(useTrainingStore.getState().snapshot?.step).toBe(2);
    });

    it('clears pause reason on reset', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        useTrainingStore.getState().setPauseReason('diverged');

        await act(async () => {
            await result.current.reset();
        });

        expect(useTrainingStore.getState().status).toBe('idle');
        expect(useTrainingStore.getState().pauseReason).toBeNull();
    });

    it('clears stale test metrics when reset returns a freshly evaluated snapshot', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        useTrainingStore.getState().setTestMetricsStale(true);

        await act(async () => {
            await result.current.reset();
        });

        expect(useTrainingStore.getState().snapshot?.step).toBe(3);
        expect(useTrainingStore.getState().testMetricsStale).toBe(false);
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
        expect(useTrainingStore.getState().trainedRecipeConfig?.data.numSamples).toBe(DEFAULT_DATA.numSamples + 1);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('config-sync');

        unmount();
    });

    it('syncs scalar-to-approved multiclass config changes while running', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        bridge.workerApi.updateConfig.mockClear();
        bridge.postStreamCommand.mockClear();
        bridge.stopRenderLoop.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('data');
            seedApprovedMulticlassStoreState();
        });

        await waitFor(() => expect(bridge.workerApi.updateConfig).toHaveBeenCalledTimes(1));
        expect(bridge.workerApi.updateConfig).toHaveBeenCalledWith(
            expect.objectContaining({ outputSize: 3, outputActivation: 'softmax' }),
            expect.objectContaining({ lossType: 'categoricalCrossEntropy' }),
            expect.objectContaining({ dataset: 'three-class-clusters', problemType: 'classification' }),
            expect.any(Object),
            false,
        );
        expect(bridge.postStreamCommand).toHaveBeenCalledWith(expect.objectContaining({ type: 'stopTraining' }));
        expect(bridge.stopRenderLoop).toHaveBeenCalledTimes(1);
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);
    });

    it('does not sync hidden multiclass configs through the public training hook', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.updateConfig.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('network');
            usePlaygroundStore.setState((state) => ({
                network: {
                    ...state.network,
                    outputSize: 3,
                    outputActivation: 'softmax',
                },
                training: {
                    ...state.training,
                    lossType: 'categoricalCrossEntropy',
                },
            }));
        });

        await waitFor(() => expect(useTrainingStore.getState().configError).toMatch(/multiclass configurations/i));
        expect(bridge.workerApi.updateConfig).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().networkConfigLoading).toBe(false);
    });

    it('validates hidden multiclass sync before sending worker commands while running', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        bridge.workerApi.updateConfig.mockClear();
        bridge.postStreamCommand.mockClear();
        bridge.stopRenderLoop.mockClear();

        act(() => {
            useTrainingStore.getState().beginConfigChange('network');
            usePlaygroundStore.setState((state) => ({
                network: {
                    ...state.network,
                    outputSize: 3,
                    outputActivation: 'softmax',
                },
                training: {
                    ...state.training,
                    lossType: 'categoricalCrossEntropy',
                },
            }));
        });

        await waitFor(() => expect(useTrainingStore.getState().configError).toMatch(/multiclass configurations/i));
        expect(bridge.workerApi.updateConfig).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'stopTraining' }));
        expect(bridge.stopRenderLoop).not.toHaveBeenCalled();
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
        expect(useTrainingStore.getState().trainedRecipeConfig?.network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('initialize');

        bridge.workerApi.updateConfig.mockResolvedValueOnce({ snapshot: makeSnapshot(5), runId: 105 });
        act(() => {
            useTrainingStore.getState().retryConfigSync();
        });

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(5));
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(bridge.newRunTo).toHaveBeenCalledWith(105);
        expect(useTrainingStore.getState().trainedRecipeConfig?.network.hiddenLayers).toEqual([5, 3]);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('config-sync');

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

    it('does not step or reset while config sync is pending', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.step.mockClear();
        bridge.workerApi.reset.mockClear();

        act(() => {
            useTrainingStore.setState({ pendingConfigSource: 'training', trainingConfigLoading: true });
        });

        await act(async () => {
            await result.current.step();
            await result.current.reset();
        });

        expect(bridge.workerApi.step).not.toHaveBeenCalled();
        expect(bridge.workerApi.reset).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().pendingConfigSource).toBe('training');
    });
});
