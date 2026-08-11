import { act, renderHook, waitFor } from '@testing-library/react';
import { DEFAULT_DEMAND, PREPARED_PRESETS, type ModelRevision } from '@nn-playground/shared';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import { createScientificTrustFixtures } from '../../../test/scientificTrustFixtures.ts';
import {
    getFrameVersions,
    resetFrameBuffer,
    updateFrameBuffer,
} from '../../../worker/frameBuffer.ts';
import type {
    BackpropExplanationResponseV2,
    ObjectiveLandscapeResponseV2,
    PredictionTraceResponseV2,
} from '../../../worker/training.worker.ts';
import { useInspectionPanelController } from './useInspectionPanelController.ts';

const workerApi = vi.hoisted(() => ({
    getPredictionTraceV2: vi.fn(),
    getBackpropExplanationV2: vi.fn(),
    getObjectiveLandscapeV2: vi.fn(),
}));

vi.mock('../../../worker/workerBridge.ts', () => ({
    getWorkerApi: async () => workerApi,
}));

const MODEL: ModelRevision = { generationId: 1, revision: 12, step: 12, epoch: 1 };
const DATASET = {
    generatorVersion: 1,
    datasetKey: 'dataset-v2',
    trainCount: 210,
    testCount: 90,
} as const;

function oneLayerPrepared() {
    const result = PREPARED_PRESETS.find((entry) => entry.id === 'circle-one-layer')?.prepared;
    if (!result) throw new Error('missing circle-one-layer');
    return result;
}

function installCurrentEvidence() {
    useTrainingStore.setState({
        latestLiveSignal: {
            model: MODEL,
            dataset: DATASET,
            objectiveKey: 'objective-v2',
            basis: {
                kind: 'mini-batch-ema',
                alpha: 0.1,
                latestBatchSize: 10,
                throughStep: 12,
            },
            dataLoss: 0.4,
        },
        latestEvaluation: null,
    });
}

function advanceActiveModel(revision = 13) {
    useTrainingStore.setState((state) => ({
        latestLiveSignal: state.latestLiveSignal
            ? {
                ...state.latestLiveSignal,
                model: { ...MODEL, revision, step: revision },
            }
            : null,
    }));
}

function predictionTraceResponse(
    model = MODEL,
    source: 'train' | 'test' = 'train',
    index = 0,
): PredictionTraceResponseV2 {
    return {
        runId: 1,
        model,
        dataset: DATASET,
        objectiveKey: 'objective-v2',
        sample: { source, index, x: 0.25, y: -0.5, label: 1 },
        trace: {
            input: [0.25, -0.5],
            target: [1],
            output: [0.82],
            prediction: 0.82,
            sampleDataLoss: 0.19,
            regularizationPenalty: 0.03,
            layers: [{
                layerIndex: 0,
                preActivations: [1.5],
                activations: [0.82],
            }],
        },
    };
}

function backpropResponse(model = MODEL): BackpropExplanationResponseV2 {
    return {
        runId: 1,
        model,
        dataset: DATASET,
        objectiveKey: 'objective-v2',
        basis: { kind: 'next-mini-batch', sampleCount: 5, populationCount: 210 },
        explanation: {
            batchSize: 5,
            learningRate: 0.03,
            objective: {
                dataLoss: 0.12,
                regularizationPenalty: 0.01,
                totalObjective: 0.13,
            },
            gradients: {
                dataGradientNorm: 0.004,
                penaltyGradientNorm: 0.002,
                totalGradientNorm: 0.006,
                clippedGradientNorm: 0.006,
                clipScale: 1,
            },
            summary: 'Backprop preview found 1 healthy layer update.',
            layers: [{
                layerIndex: 0,
                meanAbsErrorSignal: 0.012,
                maxAbsErrorSignal: 0.02,
                meanAbsGradient: 0.003,
                maxAbsGradient: 0.01,
                meanAbsUpdate: 0.0009,
                maxAbsUpdate: 0.002,
                meanActivation: 0.4,
                activationStd: 0.1,
                status: 'healthy',
                note: 'The previewed update is in a moderate range.',
            }],
        },
    };
}

function landscapeResponse(model = MODEL): ObjectiveLandscapeResponseV2 {
    return {
        runId: 1,
        model,
        provenance: {
            model,
            dataset: DATASET,
            objectiveKey: 'objective-v2',
            basis: { kind: 'parameter-grid', sampleCount: 12, parameterPositions: 9 },
        },
        probe: {
            basis: 'training-objective',
            gridSize: 3,
            sampleCount: 12,
            parameterPositionCount: 9,
            radius: 0.1,
            axisA: {
                parameter: {
                    kind: 'weight',
                    layerIndex: 0,
                    neuronIndex: 0,
                    inputIndex: 0,
                    label: 'W1[0,0]',
                },
                offsets: [-0.1, 0, 0.1],
            },
            axisB: {
                parameter: {
                    kind: 'weight',
                    layerIndex: 0,
                    neuronIndex: 1,
                    inputIndex: 0,
                    label: 'W1[1,0]',
                },
                offsets: [-0.1, 0, 0.1],
            },
            objectives: [0.62, 0.58, 0.5, 0.55, 0.49, 0.45, 0.53, 0.44, 0.4],
            centerObjective: 0.49,
            minObjective: 0.4,
            maxObjective: 0.62,
            best: { row: 2, col: 2, objective: 0.4, offsetA: 0.1, offsetB: 0.1 },
            summary: 'Training-objective surface found best objective 0.4000.',
        },
    };
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((nextResolve, nextReject) => {
        resolve = nextResolve;
        reject = nextReject;
    });
    return { promise, resolve, reject };
}

describe('useInspectionPanelController', () => {
    beforeEach(() => {
        Object.values(workerApi).forEach((mock) => mock.mockReset());
        const prepared = oneLayerPrepared();
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared },
            demand: {
                ...DEFAULT_DEMAND,
                needLayerStats: false,
                needActivationHistograms: false,
            },
        });
        useTrainingStore.setState({
            trainPoints: [],
            testPoints: [],
            latestLiveSignal: null,
            latestEvaluation: null,
        });
        resetFrameBuffer();
        useTrainingStore.setState(getFrameVersions());
    });

    afterEach(() => resetFrameBuffer());

    it('does not mutate shell-owned visualization demand while mounted or unmounted', () => {
        const initialDemand = usePlaygroundStore.getState().demand;
        const { unmount } = renderHook(() => useInspectionPanelController());

        expect(usePlaygroundStore.getState().demand).toBe(initialDemand);

        unmount();
        expect(usePlaygroundStore.getState().demand).toBe(initialDemand);
    });

    it('refreshes layer statistics only for the focused layer version', () => {
        const { result } = renderHook(() => useInspectionPanelController());
        const initialModel = result.current.model;

        act(() => useTrainingStore.setState((state) => ({
            frameVersion: state.frameVersion + 1,
        })));
        expect(result.current.model).toBe(initialModel);

        updateFrameBuffer({
            layerStats: [{
                meanActivation: 0.25,
                activationStd: 0.15,
                meanAbsWeight: 0.2,
                meanAbsGradient: 0.01,
            }],
            layerStatsProvenance: {
                model: MODEL,
                dataset: DATASET,
                objectiveKey: 'objective-v2',
                basis: {
                    kind: 'bounded-sample',
                    split: 'train',
                    sampleCount: 128,
                    populationCount: 210,
                },
            },
            layerStatsGradientRevision: 12,
        });
        act(() => useTrainingStore.setState({
            layerStatsVersion: getFrameVersions().layerStatsVersion,
        }));

        expect(result.current.model.layers).toHaveLength(1);
        expect(result.current.model.activationBasis?.label)
            .toBe('Activation statistics across 128 of 210 training examples');
    });

    it('preserves the raw index but clamps the RPC against a command-time point snapshot', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({
            trainPoints: [
                { x: 0.25, y: -0.5, label: 1 },
                { x: -0.25, y: 0.5, label: 0 },
            ],
            testPoints: [
                { x: 0.5, y: -0.75, label: 1 },
                { x: -0.5, y: 0.75, label: 0 },
                { x: 0.1, y: 0.2, label: 1 },
            ],
        });
        workerApi.getPredictionTraceV2.mockResolvedValue(predictionTraceResponse(
            MODEL,
            'test',
            2,
        ));
        const { result } = renderHook(() => useInspectionPanelController());

        let pending!: Promise<void>;
        await act(async () => {
            result.current.commands.selectTraceSource('test');
            result.current.commands.selectSampleIndex(9);
            pending = result.current.commands.requestTrace();
            await pending;
        });
        expect(result.current.model.trace.sampleIndex).toBe(9);
        expect(result.current.model.trace.effectiveSampleIndex).toBe(2);

        expect(workerApi.getPredictionTraceV2).toHaveBeenCalledWith({
            source: 'test',
            index: 2,
        });
    });

    it('invalidates a pending trace when its source or raw index changes', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({
            trainPoints: [{ x: 0.25, y: -0.5, label: 1 }],
            testPoints: [{ x: -0.25, y: 0.5, label: 0 }],
        });
        const first = deferred<PredictionTraceResponseV2>();
        workerApi.getPredictionTraceV2.mockReturnValue(first.promise);
        const { result } = renderHook(() => useInspectionPanelController());

        let pending!: Promise<void>;
        await act(async () => {
            pending = result.current.commands.requestTrace();
            await Promise.resolve();
        });
        await waitFor(() => expect(workerApi.getPredictionTraceV2).toHaveBeenCalledOnce());
        act(() => {
            result.current.commands.selectTraceSource('test');
            result.current.commands.selectSampleIndex(3);
        });
        await act(async () => {
            first.resolve(predictionTraceResponse());
            await pending;
        });

        expect(result.current.model.trace.result).toBeNull();
        expect(result.current.model.trace.errorMessage).toBeNull();
        expect(result.current.model.trace.source).toBe('test');
        expect(result.current.model.trace.sampleIndex).toBe(3);
    });

    it('publishes current trace, backprop, and landscape responses', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        workerApi.getPredictionTraceV2.mockResolvedValue(predictionTraceResponse());
        workerApi.getBackpropExplanationV2.mockResolvedValue(backpropResponse());
        workerApi.getObjectiveLandscapeV2.mockResolvedValue(landscapeResponse());
        const { result } = renderHook(() => useInspectionPanelController());

        await act(async () => result.current.commands.requestTrace());
        await act(async () => result.current.commands.requestBackprop());
        await act(async () => result.current.commands.requestLandscape());

        expect(result.current.model.trace.result?.sampleDataLoss).toBe('0.1900');
        expect(result.current.model.backprop.result?.provenance).toBe('Preview from step 12 / epoch 1');
        expect(result.current.model.landscape.result?.title)
            .toBe('Training objective on a parameter grid');
    });

    it.each([
        ['trace', 'getPredictionTraceV2', 'requestTrace', predictionTraceResponse()],
        ['backprop', 'getBackpropExplanationV2', 'requestBackprop', backpropResponse()],
        ['landscape', 'getObjectiveLandscapeV2', 'requestLandscape', landscapeResponse()],
    ] as const)('drops stale %s success after the active model changes', async (
        _label,
        method,
        command,
        response,
    ) => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        const request = deferred<typeof response>();
        workerApi[method].mockReturnValue(request.promise);
        const { result } = renderHook(() => useInspectionPanelController());

        let pending!: Promise<void>;
        await act(async () => {
            pending = result.current.commands[command]();
            await Promise.resolve();
        });
        await waitFor(() => expect(workerApi[method]).toHaveBeenCalledOnce());
        await act(async () => {
            advanceActiveModel();
            request.resolve(response);
            await pending;
        });

        expect(result.current.model.trace.result).toBeNull();
        expect(result.current.model.backprop.result).toBeNull();
        expect(result.current.model.landscape.result).toBeNull();
    });

    it.each([
        ['trace', 'getPredictionTraceV2', 'requestTrace'],
        ['backprop', 'getBackpropExplanationV2', 'requestBackprop'],
        ['landscape', 'getObjectiveLandscapeV2', 'requestLandscape'],
    ] as const)('drops stale %s failure after the active model changes', async (
        _label,
        method,
        command,
    ) => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        const request = deferred<never>();
        workerApi[method].mockReturnValue(request.promise);
        const { result } = renderHook(() => useInspectionPanelController());

        let pending!: Promise<void>;
        await act(async () => {
            pending = result.current.commands[command]();
            await Promise.resolve();
        });
        await waitFor(() => expect(workerApi[method]).toHaveBeenCalledOnce());
        await act(async () => {
            advanceActiveModel();
            request.reject(new Error('stale failure'));
            await pending;
        });

        expect(result.current.model.trace.errorMessage).toBeNull();
        expect(result.current.model.backprop.statusText).toBe('');
        expect(result.current.model.landscape.statusText).toBe('');
    });

    it.each([
        ['trace', 'getPredictionTraceV2', 'requestTrace', predictionTraceResponse({ ...MODEL, revision: 11 })],
        ['backprop', 'getBackpropExplanationV2', 'requestBackprop', backpropResponse({ ...MODEL, revision: 11 })],
        ['landscape', 'getObjectiveLandscapeV2', 'requestLandscape', landscapeResponse({ ...MODEL, revision: 11 })],
    ] as const)('rejects a %s response whose model does not match the request', async (
        _label,
        method,
        command,
        response,
    ) => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        workerApi[method].mockResolvedValue(response);
        const { result } = renderHook(() => useInspectionPanelController());

        await act(async () => result.current.commands[command]());

        expect(result.current.model.trace.result).toBeNull();
        expect(result.current.model.backprop.result).toBeNull();
        expect(result.current.model.landscape.result).toBeNull();
    });

    it('lets a newer trace request win after source invalidation', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({
            trainPoints: [{ x: 0.25, y: -0.5, label: 1 }],
            testPoints: [{ x: -0.25, y: 0.5, label: 0 }],
        });
        const first = deferred<PredictionTraceResponseV2>();
        const second = deferred<PredictionTraceResponseV2>();
        workerApi.getPredictionTraceV2
            .mockReturnValueOnce(first.promise)
            .mockReturnValueOnce(second.promise);
        const { result } = renderHook(() => useInspectionPanelController());

        let firstPending!: Promise<void>;
        await act(async () => {
            firstPending = result.current.commands.requestTrace();
            await Promise.resolve();
        });
        act(() => result.current.commands.selectTraceSource('test'));
        let secondPending!: Promise<void>;
        await act(async () => {
            secondPending = result.current.commands.requestTrace();
            await Promise.resolve();
        });
        await act(async () => {
            second.resolve(predictionTraceResponse(MODEL, 'test'));
            await secondPending;
        });
        await act(async () => {
            first.resolve(predictionTraceResponse());
            await firstPending;
        });

        expect(result.current.model.trace.result?.provenance)
            .toContain('Trace from test sample 0');
    });

    it('clears current results and reports trace invalidation when the model changes', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        workerApi.getPredictionTraceV2.mockResolvedValue(predictionTraceResponse());
        workerApi.getBackpropExplanationV2.mockResolvedValue(backpropResponse());
        workerApi.getObjectiveLandscapeV2.mockResolvedValue(landscapeResponse());
        const { result } = renderHook(() => useInspectionPanelController());
        await act(async () => result.current.commands.requestTrace());
        await act(async () => result.current.commands.requestBackprop());
        await act(async () => result.current.commands.requestLandscape());

        act(() => advanceActiveModel());

        expect(result.current.model.trace.result).toBeNull();
        expect(result.current.model.trace.errorMessage)
            .toBe('Trace cleared because the active model changed.');
        expect(result.current.model.backprop.result).toBeNull();
        expect(result.current.model.backprop.statusText).toBe('');
        expect(result.current.model.landscape.result).toBeNull();
        expect(result.current.model.landscape.statusText).toBe('');
    });

    it.each([
        ['trace', 'getPredictionTraceV2', 'requestTrace', 'Trace failed: current failure'],
        ['backprop', 'getBackpropExplanationV2', 'requestBackprop', 'Backprop preview failed: current failure'],
        ['landscape', 'getObjectiveLandscapeV2', 'requestLandscape', 'Loss landscape probe failed: current failure'],
    ] as const)('shows a current %s failure', async (_label, method, command, message) => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        workerApi[method].mockRejectedValue(new Error('current failure'));
        const { result } = renderHook(() => useInspectionPanelController());

        await act(async () => result.current.commands[command]());

        const statuses = [
            result.current.model.trace.errorMessage,
            result.current.model.backprop.statusText,
            result.current.model.landscape.statusText,
        ];
        expect(statuses).toContain(message);
    });

    it('uses scientific-selector precedence for the current model', async () => {
        const fixtures = await createScientificTrustFixtures();
        installCurrentEvidence();
        useTrainingStore.setState((state) => ({
            latestLiveSignal: state.latestLiveSignal,
            latestEvaluation: {
                ...fixtures.evaluation,
                model: { ...MODEL, revision: 13, step: 13 },
            },
        }));
        workerApi.getBackpropExplanationV2.mockResolvedValue(
            backpropResponse({ ...MODEL, revision: 13, step: 13 }),
        );
        const { result } = renderHook(() => useInspectionPanelController());

        await act(async () => result.current.commands.requestBackprop());

        expect(result.current.model.backprop.result?.provenance)
            .toBe('Preview from step 13 / epoch 1');
    });

    it('does not call diagnostics without a current model', async () => {
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        const { result } = renderHook(() => useInspectionPanelController());

        await act(async () => result.current.commands.requestTrace());
        await act(async () => result.current.commands.requestBackprop());
        await act(async () => result.current.commands.requestLandscape());

        expect(workerApi.getPredictionTraceV2).not.toHaveBeenCalled();
        expect(workerApi.getBackpropExplanationV2).not.toHaveBeenCalled();
        expect(workerApi.getObjectiveLandscapeV2).not.toHaveBeenCalled();
    });
});
