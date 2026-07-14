// ── Training Integration Tests ──
// Exercises the training loop pipeline through a mocked workerBridge,
// driving synthetic snapshots and asserting store updates.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import { useTrainingStore } from '../store/useTrainingStore';
import { usePlaygroundStore } from '../store/usePlaygroundStore';
import { useLayoutStore } from '../store/useLayoutStore';
import { resetFrameBuffer } from '../worker/frameBuffer';
import { switchDataset } from '../store/recipeEdits.ts';
import {
    GRID_SIZE,
    prepareExperimentDocument,
    type WorkerExperimentRequestV2,
} from '@nn-playground/shared';
import { getDatasetContract } from '@nn-playground/engine';

// ── Fake workerBridge ──

let capturedOnSnapshot: ((msg: unknown) => void) | null = null;
let fakeTerminateWorker: ReturnType<typeof vi.fn>;
let fakePostStreamCommand: ReturnType<typeof vi.fn>;
let fakeStartRenderLoop: ReturnType<typeof vi.fn>;
let fakeStopRenderLoop: ReturnType<typeof vi.fn>;
let fakeNewRunTo: ReturnType<typeof vi.fn>;

const INITIAL_ACCESS = usePlaygroundStore.getState().access;
if (INITIAL_ACCESS.status !== 'ready') throw new Error('missing integration prepared fixture');
const INITIAL_PREPARED = INITIAL_ACCESS.prepared;

async function fakeStrictResultForRequest(
    request: WorkerExperimentRequestV2,
    runId: number,
) {
    const preparedResult = await prepareExperimentDocument(request.document);
    if (!preparedResult.ok) throw new Error('integration fixture request must prepare');
    const prepared = preparedResult.value;
    const sampleCount = prepared.compiled.data.sampleCount;
    const trainCount = Math.min(sampleCount - 1, Math.max(
        1,
        Math.floor(sampleCount * prepared.compiled.data.trainFraction),
    ));
    const testCount = sampleCount - trainCount;
    const model = { generationId: runId, revision: 0, step: 0, epoch: 0 };
    const dataset = {
        generatorVersion: getDatasetContract(prepared.compiled.data.dataset).generatorVersion,
        datasetKey: prepared.identities.datasetKey,
        trainCount,
        testCount,
    };
    const valuesFor = (count: number, dataLoss: number) => {
        if (prepared.compiled.task.kind === 'regression') return { dataLoss };
        if (prepared.compiled.task.kind === 'binary-classification') {
            return {
                dataLoss,
                accuracy: 1,
                confusionMatrix: { tp: count, tn: 0, fp: 0, fn: 0 },
            };
        }
        const first = Math.floor(count / 3);
        const second = Math.floor((count - first) / 2);
        const third = count - first - second;
        return {
            dataLoss,
            accuracy: 1,
            confusionMatrix: {
                classCount: 3 as const,
                classLabels: [0, 1, 2] as const,
                counts: [first, 0, 0, 0, second, 0, 0, 0, third] as const,
            },
        };
    };
    const trainValues = valuesFor(trainCount, 0.3);
    const testValues = valuesFor(testCount, 0.4);
    const layerSizes = [
        prepared.compiled.network.inputSize,
        ...prepared.compiled.network.hiddenLayers,
        prepared.compiled.network.outputSize,
    ];
    const snapshot = {
        step: 0,
        epoch: 0,
        trainLoss: 0.3,
        testLoss: 0.4,
        trainMetrics: { loss: 0.3, accuracy: trainValues.accuracy },
        testMetrics: { loss: 0.4, accuracy: testValues.accuracy },
        weights: layerSizes.slice(1).map((fanOut, layerIndex) => (
            Array.from({ length: fanOut }, () => (
                Array.from({ length: layerSizes[layerIndex] }, () => 0.01)
            ))
        )),
        biases: layerSizes.slice(1).map((fanOut) => Array.from({ length: fanOut }, () => 0)),
        outputGrid: new Float32Array(0),
        gridSize: GRID_SIZE,
    };
    return {
        snapshot,
        runId,
        identities: prepared.identities,
        checkpointTimeline: {
            checkpoints: [{
                id: 1,
                step: 0,
                epoch: 0,
                trainDataLoss: trainValues.dataLoss,
                testDataLoss: testValues.dataLoss,
                ...(trainValues.accuracy === undefined
                    ? {}
                    : { trainAccuracy: trainValues.accuracy }),
                ...(testValues.accuracy === undefined
                    ? {}
                    : { testAccuracy: testValues.accuracy }),
                label: 'Step 0',
            }],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: 1,
            restoredCheckpointId: null,
        },
        evidence: {
            type: 'evidence' as const,
            protocolVersion: 2 as const,
            latestEvaluation: {
                evaluationId: 1,
                trigger: 'initial' as const,
                model,
                dataset,
                objectiveKey: prepared.identities.objectiveKey,
                train: {
                    basis: {
                        kind: 'full-split' as const,
                        split: 'train' as const,
                        sampleCount: trainCount,
                        populationCount: trainCount,
                    },
                    values: trainValues,
                },
                test: {
                    basis: {
                        kind: 'full-split' as const,
                        split: 'test' as const,
                        sampleCount: testCount,
                        populationCount: testCount,
                    },
                    values: testValues,
                },
                objective: {
                    regularizationPenalty: 0,
                    trainTotalObjective: 0.3,
                },
            },
        },
    };
}

const fakeWorkerApi = {
    initializeExperimentV2: vi.fn(),
    resetExperimentV2: vi.fn(),
    stepExperimentV2: vi.fn(),
    getCheckpointTimeline: vi.fn().mockResolvedValue({
        checkpoints: [],
        maxCheckpoints: 8,
        evictedCount: 0,
        liveCheckpointId: null,
        restoredCheckpointId: null,
    }),
    getTrainPointsV2: vi.fn().mockResolvedValue([]),
    getTestPointsV2: vi.fn().mockResolvedValue([]),
    updateDemand: vi.fn().mockResolvedValue(undefined),
    setStreamPort: vi.fn().mockResolvedValue(undefined),
};

vi.mock('../worker/workerBridge.ts', () => ({
    getWorkerApi: async () => fakeWorkerApi,
    setupStreamChannel: vi.fn().mockResolvedValue(undefined),
    postStreamCommand: (...args: unknown[]) => fakePostStreamCommand(...args),
    onSnapshot: (cb: (msg: unknown) => void) => {
        capturedOnSnapshot = cb;
        return () => { capturedOnSnapshot = null; };
    },
    newRunTo: (...args: unknown[]) => fakeNewRunTo(...args),
    terminateWorker: () => fakeTerminateWorker(),
    startRenderLoop: () => fakeStartRenderLoop(),
    stopRenderLoop: () => fakeStopRenderLoop(),
    getCurrentRunId: vi.fn().mockReturnValue(1),
}));

// Stub visual sub-components so tests stay focused on training logic.
vi.mock('../components/layout/Header.tsx', () => ({
    Header: ({ training }: { training: { play: () => void; pause: () => void } }) => (
        <header>
            <button onClick={() => training.play()}>Play</button>
            <button onClick={() => training.pause()}>Pause</button>
        </header>
    ),
}));
vi.mock('../components/layout/MainArea.tsx', () => ({
    MainArea:        () => <main id="main-content" tabIndex={-1}>Main</main>,
    CanvasContent:   () => <div>Canvas</div>,
    BoundaryContent: () => <div>Boundary</div>,
    LossContent:     () => <div>Loss</div>,
    ConfusionContent:() => <div>Confusion</div>,
    InspectContent:  () => <div>Inspect</div>,
    CodeContent:     () => <div>Code</div>,
    HistoryContent:  () => <div>History</div>,
}));
vi.mock('../components/controls/TrainingControls.tsx', () => ({ TrainingControls: () => <div>Controls</div> }));
vi.mock('../components/visualization/NetworkGraph.tsx', () => ({ NetworkGraph: () => <div>Graph</div> }));
vi.mock('../components/controls/PresetPanel.tsx',       () => ({ PresetPanel: () => <div>Presets</div> }));
vi.mock('../components/controls/DataPanel.tsx',         () => ({ DataPanel: () => <div>Data</div> }));
vi.mock('../components/controls/FeaturesPanel.tsx',     () => ({ FeaturesPanel: () => <div>Features</div> }));
vi.mock('../components/controls/NetworkConfigPanel.tsx',() => ({ NetworkConfigPanel: () => <div>Network</div> }));
vi.mock('../components/controls/HyperparamPanel.tsx',   () => ({ HyperparamPanel: () => <div>Hyperparams</div> }));
vi.mock('../components/controls/ConfigPanel.tsx',       () => ({ ConfigPanel: () => <div>Config</div> }));
vi.mock('../components/controls/InspectionPanel.tsx',   () => ({ InspectionPanel: () => <div>Inspection</div> }));
vi.mock('../components/controls/CodeExportPanel.tsx',   () => ({ CodeExportPanel: () => <div>CodeExport</div> }));

describe('Training integration', () => {
    beforeEach(() => {
        capturedOnSnapshot = null;
        fakeTerminateWorker = vi.fn();
        fakePostStreamCommand = vi.fn();
        fakeStartRenderLoop = vi.fn();
        fakeStopRenderLoop = vi.fn();
        fakeNewRunTo = vi.fn();

        fakeWorkerApi.initializeExperimentV2.mockReset().mockImplementation(
            (request: WorkerExperimentRequestV2) => fakeStrictResultForRequest(
                request,
                (useTrainingStore.getState().evidenceGenerationId ?? 0) + 1,
            ),
        );
        fakeWorkerApi.resetExperimentV2.mockReset();
        fakeWorkerApi.stepExperimentV2.mockReset();
        fakeWorkerApi.getCheckpointTimeline.mockResolvedValue({
            checkpoints: [],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
        });
        fakeWorkerApi.getTrainPointsV2.mockResolvedValue([]);
        fakeWorkerApi.getTestPointsV2.mockResolvedValue([]);

        resetFrameBuffer();

        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });

        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: INITIAL_PREPARED },
            preparation: { status: 'ready', requestId: 0, issues: [] },
        });

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            frameVersion: 0,
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
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            checkpointTimeline: {
                checkpoints: [],
                maxCheckpoints: 8,
                evictedCount: 0,
                liveCheckpointId: null,
                restoredCheckpointId: null,
            },
        });
    });

    it('initializes worker on mount and populates the training store', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        expect(fakeWorkerApi.initializeExperimentV2).toHaveBeenCalledTimes(1);
        expect(useTrainingStore.getState().workerError).toBeNull();
        expect(fakeNewRunTo).toHaveBeenCalledWith(1);
        expect(useTrainingStore.getState().latestEvaluation?.train.values.dataLoss).toBe(0.3);
    });

    it('sends startTraining command when Play is clicked', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Play' }));
        });

        expect(fakePostStreamCommand).toHaveBeenCalledWith({
            type: 'startTraining',
            protocolVersion: 2,
            stepsPerFrame: 5,
        });
        expect(fakeStartRenderLoop).toHaveBeenCalled();
    });

    it('sends stopTraining command when Pause is clicked', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        // Start training first
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Play' }));
        });

        fakePostStreamCommand.mockClear();
        fakeStopRenderLoop.mockClear();

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
        });

        expect(fakePostStreamCommand).toHaveBeenCalledWith({
            type: 'stopTraining',
            protocolVersion: 2,
        });
        expect(fakeStopRenderLoop).toHaveBeenCalled();
    });

    it('applies streamed V2 evidence and checkpoint metadata to the training store', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        const initial = useTrainingStore.getState().latestEvaluation;
        if (!initial) throw new Error('expected initial paired evaluation');
        const model = { generationId: 1, revision: 20, step: 20, epoch: 2 };

        await act(async () => {
            capturedOnSnapshot?.({
                type: 'evidence',
                protocolVersion: 2,
                liveSignal: {
                    model,
                    dataset: initial.dataset,
                    objectiveKey: initial.objectiveKey,
                    basis: {
                        kind: 'mini-batch-ema',
                        alpha: 0.1,
                        latestBatchSize: 10,
                        throughStep: 20,
                    },
                    dataLoss: 0.25,
                },
                latestEvaluation: {
                    ...initial,
                    evaluationId: 2,
                    trigger: 'cadence',
                    model,
                    train: {
                        ...initial.train,
                        values: { ...initial.train.values, dataLoss: 0.25 },
                    },
                    test: {
                        ...initial.test,
                        values: { ...initial.test.values, dataLoss: 0.35 },
                    },
                    objective: {
                        regularizationPenalty: 0,
                        trainTotalObjective: 0.25,
                    },
                },
            });
            capturedOnSnapshot?.({
                type: 'snapshot',
                protocolVersion: 2,
                runId: 1,
                snapshotId: 20,
                model,
                scalars: { step: 20, epoch: 2, gridSize: GRID_SIZE },
                checkpointTimeline: {
                    checkpoints: [{
                        id: 20,
                        step: 20,
                        epoch: 2,
                        trainDataLoss: 0.25,
                        testDataLoss: 0.35,
                        label: 'Step 20',
                    }],
                    maxCheckpoints: 8,
                    evictedCount: 0,
                    liveCheckpointId: 20,
                    restoredCheckpointId: null,
                },
            });
        });

        const state = useTrainingStore.getState();
        expect(state.latestLiveSignal?.model).toEqual(model);
        expect(state.latestEvaluation?.test.values.dataLoss).toBe(0.35);
        expect(state.checkpointTimeline.liveCheckpointId).toBe(20);
    });

    it('routes worker error messages to the error overlay', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        const errorMsg = {
            type: 'error' as const,
            protocolVersion: 2 as const,
            runId: 1,
            message: 'Training diverged.',
        };

        await act(async () => {
            capturedOnSnapshot?.(errorMsg);
        });

        expect(useTrainingStore.getState().workerError).toBe('Training diverged.');
        expect(screen.getByText('Worker connection lost')).toBeInTheDocument();
    });
});

describe('Dataset switching scenario', () => {
    beforeEach(() => {
        fakeTerminateWorker = vi.fn();
        fakePostStreamCommand = vi.fn();
        fakeStartRenderLoop = vi.fn();
        fakeStopRenderLoop = vi.fn();
        fakeNewRunTo = vi.fn();

        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });

        fakeWorkerApi.initializeExperimentV2.mockReset().mockImplementation(
            (request: WorkerExperimentRequestV2) => fakeStrictResultForRequest(
                request,
                (useTrainingStore.getState().evidenceGenerationId ?? 0) + 1,
            ),
        );
        fakeWorkerApi.getTrainPointsV2.mockResolvedValue([]);
        fakeWorkerApi.getTestPointsV2.mockResolvedValue([]);
        fakeWorkerApi.updateDemand.mockResolvedValue(undefined);

        resetFrameBuffer();

        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: INITIAL_PREPARED },
            preparation: { status: 'ready', requestId: 0, issues: [] },
        });

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            frameVersion: 0,
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
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
        });
    });

    it('rebuilds from the accepted prepared document on dataset change', async () => {
        await act(async () => {
            render(<App />);
        });
        await waitFor(() => expect(fakeNewRunTo).toHaveBeenCalledWith(1));

        // Mark as initialized so config-sync useEffect runs
        fakeWorkerApi.initializeExperimentV2.mockClear();

        await act(async () => {
            const edited = await usePlaygroundStore.getState().editRecipe(
                (recipe) => switchDataset(recipe, 'xor'),
            );
            expect(edited.ok).toBe(true);
        });

        // Allow async sync to complete
        await act(async () => {
            await Promise.resolve();
        });

        await waitFor(() => expect(fakeWorkerApi.initializeExperimentV2).toHaveBeenCalledTimes(1));
        const request = fakeWorkerApi.initializeExperimentV2.mock.calls[0]![0] as {
            document: { recipe: { task: { dataset: string } } };
        };
        expect(request.document.recipe.task.dataset).toBe('xor');
        const access = usePlaygroundStore.getState().access;
        expect(access.status === 'ready' ? access.prepared.document.recipe.task.dataset : null)
            .toBe('xor');
    });
});
