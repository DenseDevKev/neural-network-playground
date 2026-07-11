import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { NetworkSnapshot } from '@nn-playground/engine';
import {
    type ArenaScalarSnapshot,
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type EvaluationTrigger,
    type WorkerEvidenceMessageV2,
    type WorkerExperimentRequestV2,
} from '@nn-playground/shared';
import type { WorkerToMainMessage } from '@nn-playground/shared';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { setHiddenLayers, setSampleCount } from '../store/recipeEdits.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { getFrameBuffer, getFrameVersions, resetFrameBuffer, updateFrameBuffer } from '../worker/frameBuffer.ts';
import {
    createScientificTrustFixtures,
    type ScientificTrustFixtures,
} from '../test/scientificTrustFixtures.ts';

const bridge = vi.hoisted(() => {
    const workerApi = {
        initializeExperimentV2: vi.fn(),
        resetExperimentV2: vi.fn(),
        stepExperimentV2: vi.fn(),
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
let fixtures: ScientificTrustFixtures;

beforeAll(async () => {
    fixtures = await createScientificTrustFixtures();
});

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

function makeEvidence(
    generationId: number,
    evaluationId: number,
    step: number,
    trigger: EvaluationTrigger,
): WorkerEvidenceMessageV2 {
    const model = { generationId, revision: step, step, epoch: Math.floor(step / 10) };
    const trainDataLoss = 0.4 - step * 0.001;
    return {
        type: 'evidence',
        protocolVersion: 2,
        liveSignal: {
            ...fixtures.liveSignal,
            model,
            basis: { ...fixtures.liveSignal.basis, throughStep: step },
            dataLoss: trainDataLoss,
        },
        latestEvaluation: {
            ...fixtures.evaluation,
            evaluationId,
            trigger,
            model,
            train: {
                ...fixtures.evaluation.train,
                values: { dataLoss: trainDataLoss },
            },
            test: {
                ...fixtures.evaluation.test,
                values: { dataLoss: trainDataLoss + 0.1 },
            },
            objective: {
                regularizationPenalty: 0,
                trainTotalObjective: trainDataLoss,
            },
        },
    };
}

function makeV2Result(
    runId: number,
    step: number,
    evaluationId = 1,
    trigger: EvaluationTrigger = evaluationId === 1 ? 'initial' : 'manual-step',
    snapshot: NetworkSnapshot = makeSnapshot(step),
) {
    return {
        snapshot,
        runId,
        evidence: makeEvidence(runId, evaluationId, step, trigger),
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
    useTrainingStore.getState().resetEvidence();
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
        checkpointTimeline: {
            checkpoints: [],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
        },
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

        bridge.workerApi.initializeExperimentV2
            .mockReset()
            .mockResolvedValueOnce(makeV2Result(101, 1))
            .mockResolvedValue(makeV2Result(102, 2));
        bridge.workerApi.resetExperimentV2.mockReset().mockResolvedValue(makeV2Result(103, 3));
        bridge.workerApi.stepExperimentV2.mockReset().mockResolvedValue(makeV2Result(101, 4, 2));
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

        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
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
        const request = bridge.workerApi.initializeExperimentV2.mock.calls[0]![0] as WorkerExperimentRequestV2;
        expect(request).toEqual({
            type: 'initialize-experiment',
            protocolVersion: 2,
            requestId: 1,
            document: INITIAL_PREPARED!.document,
            claimedIdentities: INITIAL_PREPARED!.identities,
        });
        expect(bridge.workerApi.initialize).not.toHaveBeenCalled();
        expect(bridge.workerApi.updateConfig).not.toHaveBeenCalled();
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
        expect(bridge.workerApi.initializeExperimentV2).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('does not reconstruct the worker request from hidden projection mutations', async () => {
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

        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        const request = bridge.workerApi.initializeExperimentV2.mock.calls[0]![0] as WorkerExperimentRequestV2;
        expect(request.document).toBe(INITIAL_PREPARED!.document);
        expect(request.document.recipe.task.kind).toBe('binary-classification');
        expect(useTrainingStore.getState().trainedRecipeConfig?.network.outputSize).toBe(1);
        expect(useTrainingStore.getState().trainedRecipeFingerprint)
            .toBe(INITIAL_PREPARED!.identities.recipeFingerprint);
        expect(useTrainingStore.getState().workerError).toBeNull();
    });

    it('keys initialization off prepared identity rather than approved-looking projections', async () => {
        seedApprovedMulticlassStoreState();

        renderHook(() => useTraining());

        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        const request = bridge.workerApi.initializeExperimentV2.mock.calls[0]![0] as WorkerExperimentRequestV2;
        expect(request.document).toBe(INITIAL_PREPARED!.document);
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
            resetEvidence: () => {
                callOrder.push('resetEvidence');
                original.resetEvidence();
            },
            applyEvidence: (evidence) => {
                callOrder.push('applyEvidence');
                original.applyEvidence(evidence);
            },
            setFrameVersions: (versions) => {
                callOrder.push('setFrameVersions');
                original.setFrameVersions(versions);
            },
        });

        renderHook(() => useTraining());

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        expect(callOrder).toEqual([
            'resetEvidence',
            'applyEvidence',
            'setSnapshot',
            'setFrameVersions',
        ]);
    });

    it('keeps bounded activation histogram arrays in the frame buffer only', async () => {
        bridge.workerApi.initializeExperimentV2.mockReset().mockResolvedValue(
            makeV2Result(101, 1, 1, 'initial', withActivationHistograms(makeSnapshot(1))),
        );

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
        const emptySnapshot = {
                ...makeSnapshot(1),
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
            };
        bridge.workerApi.initializeExperimentV2.mockReset().mockResolvedValue(
            makeV2Result(101, 1, 1, 'initial', emptySnapshot),
        );

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
        bridge.workerApi.stepExperimentV2.mockResolvedValue(
            makeV2Result(101, 2, 2, 'manual-step', directSnapshot),
        );
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
        bridge.workerApi.initializeExperimentV2.mockReset().mockResolvedValue(
            makeV2Result(101, 1, 1, 'initial', withActivationHistograms(makeSnapshot(1))),
        );
        bridge.workerApi.stepExperimentV2.mockResolvedValue(makeV2Result(101, 2, 2));

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

        act(() => getStreamHandler()({ type: 'status', runId: 101, status: 'paused' }));
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

    it('awaits the pause acknowledgement before a manual step from running', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        act(() => result.current.play());
        bridge.workerApi.stepExperimentV2.mockClear();
        bridge.postStreamCommand.mockClear();

        let stepPromise!: Promise<void>;
        act(() => {
            stepPromise = result.current.step();
        });
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });
        expect(bridge.workerApi.stepExperimentV2).not.toHaveBeenCalled();

        act(() => getStreamHandler()({ type: 'status', runId: 101, status: 'paused' }));
        await act(async () => stepPromise);
        expect(bridge.workerApi.stepExperimentV2).toHaveBeenCalledWith(1);
        expect(useTrainingStore.getState().pauseReason).toBe('manual');
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
                checkpointTimeline: {
                    checkpoints: [
                        { id: 9, step: 9, epoch: 0, trainLoss: 0.2, testLoss: 0.3, label: 'Legacy' },
                    ],
                    maxCheckpoints: 8,
                    evictedCount: 0,
                    liveCheckpointId: 9,
                    restoredCheckpointId: null,
                },
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
        expect(useTrainingStore.getState().checkpointTimeline.checkpoints).toEqual([]);
    });

    it('clears stale binary confusion when a fresh streamed snapshot omits confusion data', async () => {
        bridge.workerApi.initializeExperimentV2.mockReset().mockResolvedValue(
            makeV2Result(101, 1, 1, 'initial', withConfusionMatrix(makeSnapshot(1))),
        );

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
        const handler = getStreamHandler();

        act(() => {
            result.current.play();
        });
        bridge.postStreamCommand.mockClear();
        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        act(() => handler({ type: 'status', runId: 101, status: 'paused' }));

        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(2));
        expect(bridge.postStreamCommand).toHaveBeenCalledTimes(1);
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });
        expect(useTrainingStore.getState().pauseReason).toBeNull();
    });

    it('reports a worker error when a manual step fails', async () => {
        bridge.workerApi.stepExperimentV2.mockRejectedValueOnce(new Error('step exploded'));
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        await act(async () => {
            await result.current.step();
        });

        expect(useTrainingStore.getState().workerError).toBe('step exploded');
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('never reaches the legacy checkpoint mutator from a strict prepared run', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        useTrainingStore.getState().setCheckpointTimeline({
            checkpoints: [
                { id: 1, step: 0, epoch: 0, trainLoss: 0.4, testLoss: 0.5, label: 'Step 0' },
            ],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: 1,
            restoredCheckpointId: null,
        });

        await act(async () => {
            await result.current.restoreCheckpoint(1);
        });

        expect(bridge.workerApi.restoreCheckpoint).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().snapshot?.step).toBe(1);
        expect(useTrainingStore.getState().status).toBe('idle');
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
        bridge.workerApi.initializeExperimentV2.mockClear();

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });

        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(2));

        expect(bridge.newRunTo).toHaveBeenCalledWith(102);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(useTrainingStore.getState().trainedRecipeConfig?.data.numSamples).toBe(DEFAULT_DATA.numSamples + 1);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('config-sync');

        unmount();
    });

    it('does not stop a run for projection-only config mutations', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        bridge.workerApi.initializeExperimentV2.mockClear();
        bridge.postStreamCommand.mockClear();
        bridge.stopRenderLoop.mockClear();

        act(() => {
            seedApprovedMulticlassStoreState();
        });

        await act(async () => Promise.resolve());
        expect(bridge.workerApi.initializeExperimentV2).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'stopTraining' }));
        expect(bridge.stopRenderLoop).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().configError).toBeNull();
    });

    it('ignores hidden projection changes without creating a config transaction', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.initializeExperimentV2.mockClear();

        act(() => {
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

        await act(async () => Promise.resolve());
        expect(bridge.workerApi.initializeExperimentV2).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().networkConfigLoading).toBe(false);
    });

    it('keeps a running worker isolated from hidden projection changes', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        act(() => {
            result.current.play();
        });
        bridge.workerApi.initializeExperimentV2.mockClear();
        bridge.postStreamCommand.mockClear();
        bridge.stopRenderLoop.mockClear();

        act(() => {
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

        await act(async () => Promise.resolve());
        expect(bridge.workerApi.initializeExperimentV2).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'stopTraining' }));
        expect(bridge.stopRenderLoop).not.toHaveBeenCalled();
    });

    it('records config sync failures and keeps the previous config snapshot retryable', async () => {
        const { unmount } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.initializeExperimentV2.mockRejectedValueOnce(new Error('bad network'));

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('network');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setHiddenLayers(recipe, [5, 3]),
            );
            expect(edited.ok).toBe(true);
        });

        await waitFor(() => expect(useTrainingStore.getState().configError).toBe('bad network'));

        expect(useTrainingStore.getState().configErrorSource).toBe('network');
        expect(useTrainingStore.getState().networkConfigLoading).toBe(false);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().trainedRecipeConfig?.network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
        expect(useTrainingStore.getState().trainedRecipeSource).toBe('initialize');

        bridge.workerApi.initializeExperimentV2.mockResolvedValueOnce(makeV2Result(105, 5));
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

        const first = deferred<ReturnType<typeof makeV2Result>>();
        const second = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2
            .mockReset()
            .mockImplementationOnce(() => first.promise)
            .mockImplementationOnce(() => second.promise);
        bridge.newRunTo.mockClear();

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 2),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(2));

        await act(async () => {
            second.resolve(makeV2Result(220, 20));
            await second.promise;
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));

        await act(async () => {
            first.resolve(makeV2Result(210, 10));
            await first.promise;
        });

        expect(useTrainingStore.getState().snapshot?.step).toBe(20);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(bridge.newRunTo).toHaveBeenCalledWith(220);
        expect(bridge.newRunTo).not.toHaveBeenCalledWith(210);
    });

    it('awaits one acknowledged pause before the newest overlapping config mutation', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const handler = getStreamHandler();
        act(() => result.current.play());
        bridge.postStreamCommand.mockClear();
        bridge.workerApi.initializeExperimentV2.mockClear();

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 2),
            );
            expect(edited.ok).toBe(true);
        });

        expect(bridge.postStreamCommand).toHaveBeenCalledTimes(1);
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });
        expect(bridge.workerApi.initializeExperimentV2).not.toHaveBeenCalled();

        act(() => handler({ type: 'status', runId: 101, status: 'paused' }));
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        const request = bridge.workerApi.initializeExperimentV2.mock.calls[0]![0] as WorkerExperimentRequestV2;
        expect(request.requestId).toBe(3);
        expect(request.document.recipe.data.sampleCount).toBe(DEFAULT_DATA.numSamples + 2);
        await waitFor(() => expect(useTrainingStore.getState().pendingConfigSource).toBeNull());
        expect(useTrainingStore.getState().trainedRecipeConfig?.data.numSamples)
            .toBe(DEFAULT_DATA.numSamples + 2);
    });

    it('keeps committed generation identity when auxiliary config hydration fails', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.initializeExperimentV2.mockResolvedValueOnce(makeV2Result(202, 20));
        bridge.workerApi.getTrainPoints.mockRejectedValueOnce(new Error('points hydration failed'));

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        const expectedFingerprint = usePlaygroundStore.getState().prepared!.identities.recipeFingerprint;

        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));
        await waitFor(() => expect(useTrainingStore.getState().workerError).toBe('points hydration failed'));
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(202);
        expect(useTrainingStore.getState().trainedRecipeFingerprint).toBe(expectedFingerprint);
        expect(useTrainingStore.getState().trainedRecipeConfig?.data.numSamples)
            .toBe(DEFAULT_DATA.numSamples + 1);
        expect(useTrainingStore.getState().configError).toBeNull();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().status).toBe('paused');
    });

    it('drops a manual-step completion after a prepared generation wins', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const staleStep = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.stepExperimentV2.mockReset().mockReturnValueOnce(staleStep.promise);
        bridge.workerApi.initializeExperimentV2.mockResolvedValueOnce(makeV2Result(202, 20));

        let stepPromise!: Promise<void>;
        act(() => {
            stepPromise = result.current.step();
        });
        await waitFor(() => expect(bridge.workerApi.stepExperimentV2).toHaveBeenCalledTimes(1));
        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));

        await act(async () => {
            staleStep.resolve(makeV2Result(101, 4, 2));
            await stepPromise;
        });
        expect(useTrainingStore.getState().snapshot?.step).toBe(20);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(202);
        expect(useTrainingStore.getState().workerError).toBeNull();
    });

    it('blocks play during a manual step and drops a result older than streamed revision', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const pendingStep = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.stepExperimentV2.mockReset().mockReturnValueOnce(pendingStep.promise);
        bridge.postStreamCommand.mockClear();

        let stepPromise!: Promise<void>;
        act(() => {
            stepPromise = result.current.step();
        });
        await waitFor(() => expect(bridge.workerApi.stepExperimentV2).toHaveBeenCalledTimes(1));
        act(() => result.current.play());
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(
            expect.objectContaining({ type: 'startTraining' }),
        );

        act(() => {
            getStreamHandler()(makeEvidence(101, 2, 5, 'cadence'));
            getStreamHandler()({
                type: 'snapshot',
                runId: 101,
                snapshotId: 5,
                scalars: {
                    step: 5,
                    epoch: 0,
                    trainLoss: 0.2,
                    testLoss: 0.3,
                    gridSize: 2,
                    testMetricsStale: false,
                },
            });
        });
        expect(useTrainingStore.getState().snapshot?.step).toBe(5);

        await act(async () => {
            pendingStep.resolve(makeV2Result(101, 4, 3));
            await stepPromise;
        });
        expect(useTrainingStore.getState().snapshot?.step).toBe(5);
        expect(useTrainingStore.getState().latestLiveSignal?.model.revision).toBe(5);

        bridge.postStreamCommand.mockClear();
        act(() => result.current.play());
        expect(bridge.postStreamCommand).toHaveBeenCalledWith(
            expect.objectContaining({ type: 'startTraining' }),
        );
    });

    it('does not start training while a config sync is pending', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        const pending = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2.mockReset().mockReturnValueOnce(pending.promise);
        bridge.postStreamCommand.mockClear();
        bridge.startRenderLoop.mockClear();

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));

        act(() => {
            result.current.play();
        });

        expect(useTrainingStore.getState().status).toBe('idle');
        expect(bridge.startRenderLoop).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'startTraining' }));

        await act(async () => {
            pending.resolve(makeV2Result(230, 30));
            await pending.promise;
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(30));
    });

    it('does not step or reset while config sync is pending', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        bridge.workerApi.stepExperimentV2.mockClear();
        bridge.workerApi.resetExperimentV2.mockClear();

        act(() => {
            useTrainingStore.setState({ pendingConfigSource: 'training', trainingConfigLoading: true });
        });

        await act(async () => {
            await result.current.step();
            await result.current.reset();
        });

        expect(bridge.workerApi.stepExperimentV2).not.toHaveBeenCalled();
        expect(bridge.workerApi.resetExperimentV2).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().pendingConfigSource).toBe('training');
    });

    it('uses only strict V2 lifecycle methods and keeps checkpoint restore unreachable', async () => {
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));

        await act(async () => {
            await result.current.step();
        });
        act(() => result.current.play());
        bridge.postStreamCommand.mockClear();
        let resetPromise!: Promise<void>;
        act(() => {
            resetPromise = result.current.reset();
        });
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({ type: 'stopTraining' });
        act(() => getStreamHandler()({ type: 'status', runId: 101, status: 'paused' }));
        await act(async () => resetPromise);
        await act(async () => result.current.restoreCheckpoint(1));

        expect(bridge.workerApi.stepExperimentV2).toHaveBeenCalledWith(1);
        expect(bridge.workerApi.resetExperimentV2).toHaveBeenCalledTimes(1);
        expect(bridge.workerApi.step).not.toHaveBeenCalled();
        expect(bridge.workerApi.reset).not.toHaveBeenCalled();
        expect(bridge.workerApi.restoreCheckpoint).not.toHaveBeenCalled();
        expect(bridge.postStreamCommand).toHaveBeenCalledTimes(1);
        expect(useTrainingStore.getState().checkpointTimeline.checkpoints).toEqual([]);
    });

    it('applies streamed V2 evidence immediately and pauses only for current-generation errors', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().evidenceGenerationId).toBe(101));
        const handler = getStreamHandler();
        const initialEvaluationVersion = useTrainingStore.getState().evaluationHistoryVersion;

        act(() => {
            handler(makeEvidence(101, 2, 2, 'cadence'));
        });
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(2);
        expect(useTrainingStore.getState().evaluationHistoryVersion)
            .toBe(initialEvaluationVersion + 1);

        act(() => {
            handler({
                type: 'worker-error',
                protocolVersion: 2,
                requestId: 1,
                generationId: null,
                code: 'stale-request',
                path: '$.requestId',
                message: 'late preparation failure',
                source: 'preparation',
            });
            handler({
                type: 'worker-error',
                protocolVersion: 2,
                requestId: null,
                generationId: 999,
                code: 'runtime-failure',
                path: '$',
                message: 'stale run failed',
                source: 'runtime',
            });
        });
        expect(useTrainingStore.getState().workerError).toBeNull();
        expect(useTrainingStore.getState().status).toBe('idle');

        act(() => {
            handler({
                type: 'worker-error',
                protocolVersion: 2,
                requestId: null,
                generationId: 101,
                code: 'evaluation-failed',
                path: '$.evaluation',
                message: 'current evaluation failed',
                source: 'evaluation',
            });
        });
        expect(useTrainingStore.getState().workerError).toBe('current evaluation failed');
        expect(useTrainingStore.getState().status).toBe('paused');
        expect(useTrainingStore.getState().pauseReason).toBe('error');
    });

    it('isolates stale and current preparation errors from the last valid run', async () => {
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(1));
        const handler = getStreamHandler();
        const pending = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2.mockReset().mockReturnValueOnce(pending.promise);

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        const request = bridge.workerApi.initializeExperimentV2.mock.calls[0]![0] as WorkerExperimentRequestV2;

        act(() => {
            handler({
                type: 'worker-error',
                protocolVersion: 2,
                requestId: request.requestId - 1,
                generationId: null,
                code: 'stale-request',
                path: '$.requestId',
                message: 'stale preparation failed',
                source: 'preparation',
            });
        });
        expect(useTrainingStore.getState().configError).toBeNull();

        act(() => {
            handler({
                type: 'worker-error',
                protocolVersion: 2,
                requestId: request.requestId,
                generationId: null,
                code: 'invalid-experiment',
                path: '$.document',
                message: 'current preparation failed',
                source: 'preparation',
            });
        });
        expect(useTrainingStore.getState().configError).toBe('current preparation failed');
        expect(useTrainingStore.getState().workerError).toBeNull();
        expect(useTrainingStore.getState().snapshot?.step).toBe(1);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(101);

        await act(async () => {
            pending.resolve(makeV2Result(404, 40));
            await pending.promise;
        });
        expect(useTrainingStore.getState().snapshot?.step).toBe(1);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(101);
    });

    it('does not skip queued cadence evaluations when a direct step result jumps ahead', async () => {
        const direct = makeV2Result(101, 4, 3, 'manual-step');
        bridge.workerApi.stepExperimentV2.mockResolvedValueOnce(direct);
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(1));
        const handler = getStreamHandler();
        const initialEvaluationVersion = useTrainingStore.getState().evaluationHistoryVersion;

        await act(async () => {
            await result.current.step();
        });
        expect(useTrainingStore.getState().latestLiveSignal?.model.step).toBe(4);
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(1);
        expect(useTrainingStore.getState().evaluationHistoryVersion).toBe(initialEvaluationVersion);

        act(() => {
            handler(makeEvidence(101, 2, 4, 'cadence'));
            handler(direct.evidence);
        });
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(3);
        expect(useTrainingStore.getState().evaluationHistoryVersion)
            .toBe(initialEvaluationVersion + 2);

        act(() => handler(direct.evidence));
        expect(useTrainingStore.getState().evaluationHistoryVersion)
            .toBe(initialEvaluationVersion + 2);
    });

    it('ignores a superseded mount rejection after the newest prepared request wins', async () => {
        const first = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2
            .mockReset()
            .mockImplementationOnce(() => first.promise)
            .mockResolvedValueOnce(makeV2Result(202, 20));
        renderHook(() => useTraining());
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));

        await act(async () => {
            first.reject(new Error('superseded mount failed'));
            await first.promise.catch(() => undefined);
        });
        expect(useTrainingStore.getState().snapshot?.step).toBe(20);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(202);
        expect(useTrainingStore.getState().workerError).toBeNull();
        const requests = bridge.workerApi.initializeExperimentV2.mock.calls
            .map((call) => (call[0] as WorkerExperimentRequestV2).requestId);
        expect(requests).toEqual([1, 2]);
    });

    it('keeps the stream channel live when prepared changes during mount hydration', async () => {
        const stalePoints = deferred<Array<{ x: number; y: number; label: number }>>();
        bridge.workerApi.initializeExperimentV2
            .mockReset()
            .mockResolvedValueOnce(makeV2Result(101, 1))
            .mockResolvedValueOnce(makeV2Result(202, 20));
        bridge.workerApi.getTrainPoints
            .mockReset()
            .mockReturnValueOnce(stalePoints.promise)
            .mockResolvedValue([]);
        const { result } = renderHook(() => useTraining());
        await waitFor(() => expect(bridge.workerApi.getTrainPoints).toHaveBeenCalledTimes(1));
        expect(useTrainingStore.getState().snapshot?.step).toBe(1);

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(useTrainingStore.getState().snapshot?.step).toBe(20));
        await waitFor(() => expect(useTrainingStore.getState().pendingConfigSource).toBeNull());

        expect(bridge.setupStreamChannel).toHaveBeenCalledTimes(1);
        expect(bridge.setupStreamChannel.mock.invocationCallOrder[0])
            .toBeLessThan(bridge.workerApi.initializeExperimentV2.mock.invocationCallOrder[0]!);
        bridge.postStreamCommand.mockClear();
        act(() => result.current.play());
        expect(bridge.postStreamCommand).toHaveBeenCalledWith({
            type: 'startTraining',
            stepsPerFrame: 5,
        });

        act(() => getStreamHandler()(makeEvidence(202, 2, 21, 'cadence')));
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(2);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(202);

        await act(async () => {
            stalePoints.resolve([{ x: 9, y: 9, label: 1 }]);
            await stalePoints.promise;
        });
        expect(useTrainingStore.getState().trainPoints).toEqual([]);
    });

    it('does not apply initialization completion after unmount', async () => {
        const pending = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2.mockReset().mockReturnValueOnce(pending.promise);
        const { unmount } = renderHook(() => useTraining());
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        bridge.newRunTo.mockClear();

        unmount();
        await act(async () => {
            pending.resolve(makeV2Result(303, 30));
            await pending.promise;
        });

        expect(bridge.terminateWorker).toHaveBeenCalledTimes(1);
        expect(bridge.newRunTo).not.toHaveBeenCalled();
        expect(useTrainingStore.getState().snapshot).toBeNull();
    });

    it('clears a dead worker session and accepts a restarted lower generation', async () => {
        const first = renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().evidenceGenerationId).toBe(101));
        const pendingConfig = deferred<ReturnType<typeof makeV2Result>>();
        bridge.workerApi.initializeExperimentV2.mockReset().mockReturnValueOnce(pendingConfig.promise);

        await act(async () => {
            useTrainingStore.getState().beginConfigChange('data');
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => setSampleCount(recipe, DEFAULT_DATA.numSamples + 1),
            );
            expect(edited.ok).toBe(true);
        });
        await waitFor(() => expect(bridge.workerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        first.unmount();

        expect(useTrainingStore.getState().evidenceGenerationId).toBeNull();
        expect(useTrainingStore.getState().snapshot).toBeNull();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);
        await act(async () => {
            pendingConfig.resolve(makeV2Result(202, 20));
            await pendingConfig.promise;
        });
        expect(useTrainingStore.getState().snapshot).toBeNull();

        bridge.workerApi.initializeExperimentV2.mockReset().mockResolvedValueOnce(makeV2Result(1, 5));
        renderHook(() => useTraining());
        await waitFor(() => expect(useTrainingStore.getState().evidenceGenerationId).toBe(1));
        expect(useTrainingStore.getState().snapshot?.step).toBe(5);
        expect(useTrainingStore.getState().trainedRecipeFingerprint)
            .toBe(usePlaygroundStore.getState().prepared!.identities.recipeFingerprint);
    });

});
