import { describe, expect, it, vi } from 'vitest';
import { Network } from '@nn-playground/engine';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    isWorkerToMainMessage,
} from '@nn-playground/shared';
import type { WorkerSnapshotMessage } from '@nn-playground/shared';

vi.mock('comlink', () => ({
    expose: vi.fn(),
}));

import { workerApi } from './training.worker.ts';

function containsTypedArray(value: unknown): boolean {
    if (ArrayBuffer.isView(value)) return true;
    if (value instanceof ArrayBuffer || value instanceof SharedArrayBuffer) return true;
    if (value === null || typeof value !== 'object') return false;
    return Object.values(value).some((child) => containsTypedArray(child));
}

function createCapturingPort(): {
    messages: unknown[];
    dispatch: (data: unknown) => void;
    port: MessagePort;
} {
    let listener: ((event: MessageEvent<unknown>) => void) | null = null;
    const messages: unknown[] = [];
    const port = {
        addEventListener: vi.fn((_type: string, callback: (event: MessageEvent<unknown>) => void) => {
            listener = callback;
        }),
        start: vi.fn(),
        postMessage: vi.fn((message: unknown) => {
            messages.push(message);
        }),
    } as unknown as MessagePort;

    return {
        messages,
        dispatch(data: unknown): void {
            if (!listener) throw new Error('stream listener was not registered');
            listener({ data } as MessageEvent<unknown>);
        },
        port,
    };
}

async function advanceOneWorkerTick(): Promise<void> {
    await vi.advanceTimersByTimeAsync(20);
    await Promise.resolve();
}

function capturedSnapshots(messages: unknown[]): WorkerSnapshotMessage[] {
    return messages.filter((message): message is WorkerSnapshotMessage => (
        typeof message === 'object' &&
        message !== null &&
        (message as { type?: unknown }).type === 'snapshot'
    ));
}

describe('training worker loss landscape probe RPC', () => {
    it('rejects before worker initialization', async () => {
        const { workerApi: freshWorkerApi } = await import('./training.worker.ts?loss-landscape-before-init');

        expect(() => freshWorkerApi.getLossLandscapeProbe()).toThrow('Not initialized');
    });

    it('returns bounded serializable loss-grid metadata for the current worker model', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK, hiddenLayers: [3, 2] },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 930, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getLossLandscapeProbe({
            gridSize: 7,
            maxSamples: 64,
            radius: 0.1,
        });

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.epoch).toBe(0);
        expect(response.probe.gridSize).toBe(7);
        expect(response.probe.sampleCount).toBeLessThanOrEqual(64);
        expect(response.probe.losses).toHaveLength(49);
        expect(response.probe.losses.every((loss) => Number.isFinite(loss))).toBe(true);
        expect(Number.isFinite(response.probe.centerLoss)).toBe(true);
        expect(Number.isFinite(response.probe.minLoss)).toBe(true);
        expect(Number.isFinite(response.probe.maxLoss)).toBe(true);
        expect(response.probe.axisA.offsets).toHaveLength(7);
        expect(response.probe.axisB.offsets).toHaveLength(7);
        expect(containsTypedArray(response)).toBe(false);
        expect(JSON.stringify(response)).not.toMatch(/inputs|targets|weights|biases|checkpoint/i);
    });

    it('is deterministic and does not advance the next training step', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 931, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const first = workerApi.getLossLandscapeProbe();
        const second = workerApi.getLossLandscapeProbe();
        const afterProbeStep = workerApi.step(1);

        expect(second).toEqual(first);
        expect(afterProbeStep.step).toBe(first.step + 1);
    });

    it('leaves the next real training step equivalent to a control run without probing', () => {
        const network = { ...DEFAULT_NETWORK, hiddenLayers: [3] };
        const training = { ...DEFAULT_TRAINING, batchSize: 5 };
        const data = { ...DEFAULT_DATA, seed: 932, numSamples: 20 };
        const features = { ...DEFAULT_FEATURES };

        workerApi.initialize(network, training, data, features);
        workerApi.getLossLandscapeProbe();
        const probeThenStep = workerApi.step(1);

        workerApi.initialize(network, training, data, features);
        const controlStep = workerApi.step(1);

        expect(probeThenStep.step).toBe(controlStep.step);
        expect(probeThenStep.trainLoss).toBeCloseTo(controlStep.trainLoss, 12);
        expect(probeThenStep.testLoss).toBeCloseTo(controlStep.testLoss, 12);
        expect(probeThenStep.weights).toEqual(controlStep.weights);
        expect(probeThenStep.biases).toEqual(controlStep.biases);
    });

    it('propagates engine bounds for invalid probe options', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 933, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.getLossLandscapeProbe({ gridSize: 8 })).toThrow(RangeError);
        expect(() => workerApi.getLossLandscapeProbe({ maxSamples: 65 })).toThrow(RangeError);
        expect(() => workerApi.getLossLandscapeProbe({ radius: 1.01 })).toThrow(RangeError);
    });
});

describe('training worker backprop explanation RPC', () => {
    it('rejects before worker initialization', async () => {
        const { workerApi: freshWorkerApi } = await import('./training.worker.ts?backprop-before-init');

        expect(() => freshWorkerApi.getBackpropExplanation()).toThrow('Not initialized');
    });

    it('returns finite bounded scalar summaries for the current worker model', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK, hiddenLayers: [3, 2] },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 920, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getBackpropExplanation();

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.epoch).toBe(0);
        expect(response.explanation.batchSize).toBeGreaterThan(0);
        expect(response.explanation.layers).toHaveLength(3);
        expect(Number.isFinite(response.explanation.loss)).toBe(true);
        expect(Number.isFinite(response.explanation.globalGradientNorm)).toBe(true);
        expect(response.explanation.layers.every((layer) => Number.isFinite(layer.meanAbsUpdate))).toBe(true);
        expect(containsTypedArray(response)).toBe(false);
        expect(JSON.stringify(response)).not.toMatch(/inputs|targets|weights|biases/i);
    });

    it('is deterministic and does not advance the next training step', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 921, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const first = workerApi.getBackpropExplanation();
        const second = workerApi.getBackpropExplanation();
        const afterPreviewStep = workerApi.step(1);

        expect(second).toEqual(first);
        expect(afterPreviewStep.step).toBe(first.step + 1);
    });

    it('leaves the next real training step equivalent to a control run without preview', () => {
        const network = { ...DEFAULT_NETWORK, hiddenLayers: [3] };
        const training = { ...DEFAULT_TRAINING, batchSize: 5 };
        const data = { ...DEFAULT_DATA, seed: 922, numSamples: 20 };
        const features = { ...DEFAULT_FEATURES };

        workerApi.initialize(network, training, data, features);
        workerApi.getBackpropExplanation();
        const previewThenStep = workerApi.step(1);

        workerApi.initialize(network, training, data, features);
        const controlStep = workerApi.step(1);

        expect(previewThenStep.step).toBe(controlStep.step);
        expect(previewThenStep.trainLoss).toBeCloseTo(controlStep.trainLoss, 12);
        expect(previewThenStep.testLoss).toBeCloseTo(controlStep.testLoss, 12);
        expect(previewThenStep.weights).toEqual(controlStep.weights);
        expect(previewThenStep.biases).toEqual(controlStep.biases);
    });

    it('refuses the epoch reshuffle boundary instead of mutating shuffle state during preview', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, batchSize: 5 },
            { ...DEFAULT_DATA, seed: 923, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.step(2);

        expect(() => workerApi.getBackpropExplanation()).toThrow(/epoch shuffle boundary/i);
        expect(workerApi.step(1).step).toBe(3);
    });
});

describe('training worker prediction trace RPC', () => {
    it('returns an on-demand trace for a training sample', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 123, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const response = workerApi.getPredictionTrace({ source: 'train', index: 0 });

        expect(response.runId).toBe(init.runId);
        expect(response.step).toBe(0);
        expect(response.sample.source).toBe('train');
        expect(response.sample.index).toBe(0);
        expect(response.trace.target).toHaveLength(1);
        expect(response.trace.input.length).toBeGreaterThan(0);
        expect(response.trace.output).toHaveLength(1);
        expect(response.trace.layers.length).toBeGreaterThan(0);
    });

    it('rejects out-of-range sample indexes', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 456, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.getPredictionTrace({ source: 'test', index: 999 })).toThrow(RangeError);
    });
});

describe('training worker multiclass target encoding', () => {
    const multiclassNetwork = {
        ...DEFAULT_NETWORK,
        outputSize: 3,
        outputActivation: 'softmax' as const,
    };
    const multiclassTraining = {
        ...DEFAULT_TRAINING,
        lossType: 'categoricalCrossEntropy' as const,
    };
    const approvedMulticlassData = {
        ...DEFAULT_DATA,
        dataset: 'three-class-clusters' as const,
        problemType: 'classification' as const,
    };

    it('encodes classification labels as bounded one-hot targets for worker training and traces', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, seed: 934, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const snapshot = workerApi.step(1);
        const trace = workerApi.getPredictionTrace({ source: 'train', index: 0 });

        expect(Number.isFinite(snapshot.trainLoss)).toBe(true);
        expect(Number.isFinite(snapshot.testLoss)).toBe(true);
        expect(trace.trace.output).toHaveLength(3);
        expect(trace.trace.target).toHaveLength(3);
        expect(trace.trace.target.reduce((sum, value) => sum + value, 0)).toBe(1);
        expect(trace.trace.target[trace.sample.label ?? -1]).toBe(1);
    });

    it('sources approved three-class samples through worker dataset generation', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, seed: 940, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        );

        const labels = [
            ...workerApi.getTrainPoints(),
            ...workerApi.getTestPoints(),
        ].map((point) => point.label);

        expect(new Set(labels)).toEqual(new Set([0, 1, 2]));
    });

    it('updates from scalar training to the approved three-class runtime and rebuilds data', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 943, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        );

        const updated = workerApi.updateConfig(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, seed: 943, numSamples: 60 },
            { ...DEFAULT_FEATURES },
            false,
        );
        const labels = [
            ...workerApi.getTrainPoints(),
            ...workerApi.getTestPoints(),
        ].map((point) => point.label);

        expect(updated.runId).toBeGreaterThan(init.runId);
        expect(updated.snapshot.step).toBe(0);
        expect(new Set(labels)).toEqual(new Set([0, 1, 2]));
    });

    it('rejects unsupported multiclass updates before mutating the active run', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 944, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.updateConfig(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, dataset: 'circle', seed: 944, numSamples: 60 },
            { ...DEFAULT_FEATURES },
            false,
        )).toThrow(/multiclass configurations/i);

        const labels = [
            ...workerApi.getTrainPoints(),
            ...workerApi.getTestPoints(),
        ].map((point) => point.label);
        const stillScalar = workerApi.updateConfig(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 944, numSamples: 60 },
            { ...DEFAULT_FEATURES },
            false,
        );

        expect(stillScalar.runId).toBe(init.runId);
        expect(new Set(labels)).toEqual(new Set([0, 1]));
    });

    it('rejects unsupported multiclass initialization before replacing an existing scalar run', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 945, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        );

        expect(() => workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, dataset: 'circle', seed: 945, numSamples: 60 },
            { ...DEFAULT_FEATURES },
        )).toThrow(/multiclass configurations/i);

        const labels = [
            ...workerApi.getTrainPoints(),
            ...workerApi.getTestPoints(),
        ].map((point) => point.label);
        const stillScalar = workerApi.updateConfig(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 945, numSamples: 60 },
            { ...DEFAULT_FEATURES },
            false,
        );

        expect(stillScalar.runId).toBe(init.runId);
        expect(new Set(labels)).toEqual(new Set([0, 1]));
    });

    it('encodes custom multiclass trace labels and rejects out-of-range classes', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, seed: 935, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const trace = workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 2,
        });

        expect(trace.trace.target).toEqual([0, 0, 1]);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: -1,
        })).toThrow(/class index/i);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 1.5,
        })).toThrow(/class index/i);
        expect(() => workerApi.getPredictionTrace({
            source: 'custom',
            x: 0,
            y: 0,
            label: 3,
        })).toThrow(/class index/i);
    });

    it('keeps one-shot inspection RPCs finite and non-mutating with multiclass targets', () => {
        workerApi.initialize(
            multiclassNetwork,
            { ...multiclassTraining, batchSize: 4 },
            { ...approvedMulticlassData, seed: 939, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );

        const backprop = workerApi.getBackpropExplanation();
        const landscape = workerApi.getLossLandscapeProbe({ gridSize: 5, maxSamples: 16, radius: 0.05 });
        const afterPreviewStep = workerApi.step(1);

        expect(Number.isFinite(backprop.explanation.loss)).toBe(true);
        expect(backprop.explanation.layers.length).toBeGreaterThan(0);
        expect(containsTypedArray(backprop)).toBe(false);
        expect(Number.isFinite(landscape.probe.centerLoss)).toBe(true);
        expect(landscape.probe.losses).toHaveLength(25);
        expect(containsTypedArray(landscape)).toBe(false);
        expect(afterPreviewStep.step).toBe(1);
    });

    it('requires the approved softmax plus categorical cross-entropy worker pairing', () => {
        expect(() => workerApi.initialize(
            {
                ...DEFAULT_NETWORK,
                outputSize: 3,
                outputActivation: 'sigmoid',
            },
            multiclassTraining,
            { ...approvedMulticlassData, seed: 936, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        )).toThrow(/softmax.*categorical cross-entropy/i);
    });

    it.each([
        [
            'two outputs',
            { outputSize: 2, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            {},
        ],
        [
            'four outputs',
            { outputSize: 4, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            {},
        ],
        [
            'softmax with scalar loss',
            { outputSize: 3, outputActivation: 'softmax' as const },
            { lossType: 'crossEntropy' as const },
            {},
        ],
        [
            'categorical loss without softmax',
            { outputSize: 3, outputActivation: 'sigmoid' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            {},
        ],
        [
            'unsupported dataset',
            { outputSize: 3, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { dataset: 'circle' as const },
        ],
        [
            'regression data',
            { outputSize: 3, outputActivation: 'softmax' as const },
            { lossType: 'categoricalCrossEntropy' as const },
            { problemType: 'regression' as const },
        ],
    ])('rejects unsupported multiclass worker config: %s', (_label, networkOverrides, trainingOverrides, dataOverrides) => {
        expect(() => workerApi.initialize(
            {
                ...DEFAULT_NETWORK,
                ...networkOverrides,
            },
            {
                ...DEFAULT_TRAINING,
                ...trainingOverrides,
            },
            {
                ...approvedMulticlassData,
                ...dataOverrides,
                seed: 936,
                numSamples: 24,
            },
            { ...DEFAULT_FEATURES },
        )).toThrow(/output size 3, softmax output activation, and categorical cross-entropy loss/i);
    });

    it('does not emit scalar decision-boundary grids or binary confusion matrices for multiclass snapshots', () => {
        workerApi.initialize(
            multiclassNetwork,
            multiclassTraining,
            { ...approvedMulticlassData, seed: 938, numSamples: 24 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: true,
            needConfusionMatrix: true,
        });

        const snapshot = workerApi.step(1);

        expect(snapshot.outputGrid).toHaveLength(0);
        expect(snapshot.neuronGrids).toBeUndefined();
        expect(snapshot.testMetrics.confusionMatrix).toBeUndefined();
        expect(snapshot.multiclassBoundary).toEqual(expect.objectContaining({
            gridSize: 40,
        }));
        expect(snapshot.multiclassBoundary?.classGrid).toHaveLength(40 * 40);
        expect(snapshot.multiclassBoundary?.confidenceGrid).toHaveLength(40 * 40);
    });

    it('streams worker-authored multiclass confusion only for fresh demanded approved tuple snapshots', async () => {
        vi.useFakeTimers();
        const stream = createCapturingPort();
        try {
            workerApi.initialize(
                multiclassNetwork,
                multiclassTraining,
                { ...approvedMulticlassData, seed: 946, numSamples: 60 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: false,
                needNeuronGrids: false,
                needConfusionMatrix: true,
                testEvalInterval: 2,
            });
            const expectedRows = workerApi.getTestPoints().reduce<[number, number, number]>((rows, point) => {
                rows[point.label as 0 | 1 | 2]++;
                return rows;
            }, [0, 0, 0]);
            const expectedTotal = expectedRows[0] + expectedRows[1] + expectedRows[2];
            workerApi.setStreamPort(stream.port);

            stream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();
            stream.dispatch({ type: 'frameAck' });
            await advanceOneWorkerTick();
            stream.dispatch({ type: 'stopTraining' });

            const snapshots = capturedSnapshots(stream.messages);
            expect(snapshots).toHaveLength(2);

            const freshSnapshot = snapshots[0];
            expect(isWorkerToMainMessage(freshSnapshot)).toBe(true);
            expect(freshSnapshot.scalars.testMetricsStale).toBe(false);
            expect(freshSnapshot.confusionMatrix).toBeUndefined();
            expect(freshSnapshot.multiclassConfusionMatrixVersion).toBeGreaterThan(0);
            expect(freshSnapshot.multiclassConfusionMatrix).toEqual(expect.objectContaining({
                classCount: 3,
                classLabels: [0, 1, 2],
            }));
            const counts = freshSnapshot.multiclassConfusionMatrix?.counts ?? [];
            expect(counts).toHaveLength(9);
            expect(counts.reduce((sum, count) => sum + count, 0)).toBe(expectedTotal);
            expect([
                counts[0] + counts[1] + counts[2],
                counts[3] + counts[4] + counts[5],
                counts[6] + counts[7] + counts[8],
            ]).toEqual(expectedRows);

            const staleSnapshot = snapshots[1];
            expect(isWorkerToMainMessage(staleSnapshot)).toBe(true);
            expect(staleSnapshot.scalars.testMetricsStale).toBe(true);
            expect(staleSnapshot.confusionMatrix).toBeUndefined();
            expect(staleSnapshot.multiclassConfusionMatrix).toBeUndefined();
            expect(staleSnapshot.multiclassConfusionMatrixVersion).toBeUndefined();
        } finally {
            stream.dispatch({ type: 'stopTraining' });
            vi.useRealTimers();
        }
    });

    it('omits streamed multiclass confusion when confusion demand is disabled', async () => {
        vi.useFakeTimers();
        const stream = createCapturingPort();
        try {
            workerApi.initialize(
                multiclassNetwork,
                multiclassTraining,
                { ...approvedMulticlassData, seed: 947, numSamples: 60 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: false,
                needNeuronGrids: false,
                needConfusionMatrix: false,
            });
            workerApi.setStreamPort(stream.port);

            stream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();
            stream.dispatch({ type: 'stopTraining' });

            const [snapshot] = capturedSnapshots(stream.messages);
            expect(snapshot).toBeDefined();
            expect(isWorkerToMainMessage(snapshot)).toBe(true);
            expect(snapshot.confusionMatrix).toBeUndefined();
            expect(snapshot.multiclassConfusionMatrix).toBeUndefined();
            expect(snapshot.multiclassConfusionMatrixVersion).toBeUndefined();
        } finally {
            stream.dispatch({ type: 'stopTraining' });
            vi.useRealTimers();
        }
    });

    it('continues streaming binary confusion matrices without multiclass payloads for scalar classification', async () => {
        vi.useFakeTimers();
        const stream = createCapturingPort();
        try {
            workerApi.initialize(
                { ...DEFAULT_NETWORK },
                { ...DEFAULT_TRAINING },
                { ...DEFAULT_DATA, seed: 948, numSamples: 40 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: false,
                needNeuronGrids: false,
                needConfusionMatrix: true,
            });
            workerApi.setStreamPort(stream.port);

            stream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();
            stream.dispatch({ type: 'stopTraining' });

            const [snapshot] = capturedSnapshots(stream.messages);
            expect(snapshot).toBeDefined();
            expect(isWorkerToMainMessage(snapshot)).toBe(true);
            expect(snapshot.confusionMatrix).toEqual(expect.objectContaining({
                tp: expect.any(Number),
                tn: expect.any(Number),
                fp: expect.any(Number),
                fn: expect.any(Number),
            }));
            expect(snapshot.multiclassConfusionMatrix).toBeUndefined();
            expect(snapshot.multiclassConfusionMatrixVersion).toBeUndefined();
        } finally {
            stream.dispatch({ type: 'stopTraining' });
            vi.useRealTimers();
        }
    });

    it('streams bounded multiclass boundary payloads only when decision-boundary demand is enabled and cadence is due', async () => {
        vi.useFakeTimers();
        const stream = createCapturingPort();
        try {
            workerApi.initialize(
                multiclassNetwork,
                multiclassTraining,
                { ...approvedMulticlassData, seed: 941, numSamples: 24 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: false,
                needNeuronGrids: false,
                needConfusionMatrix: true,
                gridInterval: 2,
            });
            workerApi.setStreamPort(stream.port);

            stream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();
            stream.dispatch({ type: 'stopTraining' });

            const firstSnapshot = stream.messages.find((message): message is WorkerSnapshotMessage => (
                typeof message === 'object' &&
                message !== null &&
                (message as { type?: unknown }).type === 'snapshot'
            ));
            expect(firstSnapshot).toBeDefined();
            expect(firstSnapshot?.outputGrid).toBeUndefined();
            expect(firstSnapshot?.neuronGrids).toBeUndefined();
            expect(firstSnapshot?.multiclassClassGrid).toBeUndefined();
            expect(firstSnapshot?.multiclassConfidenceGrid).toBeUndefined();

            const demandOnStream = createCapturingPort();
            workerApi.initialize(
                multiclassNetwork,
                multiclassTraining,
                { ...approvedMulticlassData, seed: 942, numSamples: 24 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needConfusionMatrix: true,
                gridInterval: 2,
            });
            workerApi.setStreamPort(demandOnStream.port);

            demandOnStream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();
            demandOnStream.dispatch({ type: 'frameAck' });
            await advanceOneWorkerTick();
            demandOnStream.dispatch({ type: 'stopTraining' });

            const snapshots = demandOnStream.messages.filter((message): message is WorkerSnapshotMessage => (
                typeof message === 'object' &&
                message !== null &&
                (message as { type?: unknown }).type === 'snapshot'
            ));

            expect(snapshots).toHaveLength(2);
            expect(snapshots[0].outputGrid).toEqual(new Float32Array(0));
            expect(snapshots[0].neuronGrids).toEqual(new Float32Array(0));
            expect(snapshots[0].multiclassClassGrid).toBeInstanceOf(Uint8Array);
            expect(snapshots[0].multiclassClassGrid).toHaveLength(
                snapshots[0].scalars.gridSize * snapshots[0].scalars.gridSize,
            );
            expect(snapshots[0].multiclassConfidenceGrid).toBeInstanceOf(Float32Array);
            expect(snapshots[0].multiclassConfidenceGrid).toHaveLength(
                snapshots[0].scalars.gridSize * snapshots[0].scalars.gridSize,
            );
            expect(snapshots[0].multiclassBoundaryLayout).toEqual({
                gridSize: snapshots[0].scalars.gridSize,
                classCount: 3,
                classLabels: [0, 1, 2],
            });
            expect(snapshots[0].multiclassBoundaryVersion).toBeGreaterThan(0);
            expect(snapshots[1].multiclassClassGrid).toBeUndefined();
            expect(snapshots[1].multiclassConfidenceGrid).toBeUndefined();
            expect(snapshots[1].multiclassBoundaryLayout).toBeUndefined();
            expect(snapshots[1].multiclassBoundaryVersion).toBeUndefined();
            expect(snapshots[1].outputGrid).toBeUndefined();
            expect(snapshots[1].neuronGrids).toBeUndefined();
        } finally {
            stream.dispatch({ type: 'stopTraining' });
            vi.useRealTimers();
        }
    });

    it('keeps live arena runtime guarded to scalar models', () => {
        expect(() => workerApi.initializeArena({
            modelA: {
                label: 'Multiclass A',
                network: multiclassNetwork,
                training: multiclassTraining,
                data: { ...DEFAULT_DATA, seed: 937, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Scalar B',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 937, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
        })).toThrow(/multiclass live arena/i);
    });
});

describe('training worker activation histogram demand', () => {
    it('omits activation histograms until explicitly requested', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 789, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
            needActivationHistograms: false,
        });
        expect(workerApi.step(1).activationHistograms).toBeUndefined();

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
            needActivationHistograms: true,
            activationHistogramInterval: 2,
        });
        const snapshot = workerApi.step(1);

        expect(snapshot.activationHistograms).toBeDefined();
        expect(snapshot.activationHistograms?.bins).toBeInstanceOf(Float32Array);
        expect(snapshot.activationHistograms?.bins.length).toBeGreaterThan(0);
        expect(snapshot.activationHistograms?.layers.length).toBeGreaterThan(0);
        expect(workerApi.step(1).activationHistograms).toBeUndefined();
    });

    it('does not compute activation histograms for layer-stat demand alone', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 890, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
            needActivationHistograms: false,
        });

        expect(workerApi.step(1).activationHistograms).toBeUndefined();
    });

    it('computes activation histograms on the first rebuild snapshot while requested', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 891, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needActivationHistograms: true,
            activationHistogramInterval: 5,
        });

        const result = workerApi.updateConfig(
            {
                ...DEFAULT_NETWORK,
                hiddenLayers: [3],
            },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 891, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            true,
        );

        expect(result.snapshot.activationHistograms).toBeDefined();
        expect(result.snapshot.activationHistograms?.layers).toHaveLength(2);
    });
});

describe('training worker lifecycle and demand cadence', () => {
    it('pauses the legacy stream when a training loss becomes non-finite', async () => {
        vi.useFakeTimers();
        const stream = createCapturingPort();
        const original = Network.prototype.trainBatch;
        vi.spyOn(Network.prototype, 'trainBatch').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['trainBatch']>
        ) {
            original.apply(this, args);
            return Number.NaN;
        });
        try {
            workerApi.initialize(
                { ...DEFAULT_NETWORK },
                { ...DEFAULT_TRAINING },
                { ...DEFAULT_DATA, seed: 906, numSamples: 20 },
                { ...DEFAULT_FEATURES },
            );
            workerApi.setStreamPort(stream.port);

            stream.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await advanceOneWorkerTick();

            expect(capturedSnapshots(stream.messages)).toContainEqual(expect.objectContaining({
                scalars: expect.objectContaining({ trainLoss: Number.NaN }),
            }));
            expect(stream.messages).toContainEqual(expect.objectContaining({
                type: 'status',
                status: 'paused',
                pauseReason: 'diverged',
            }));
        } finally {
            stream.dispatch({ type: 'stopTraining' });
            vi.restoreAllMocks();
            vi.useRealTimers();
        }
    });

    it('initializes and steps two scalar-only arena model slots sequentially', () => {
        const arena = workerApi.initializeArena({
            modelA: {
                label: 'Capacity 2',
                network: { ...DEFAULT_NETWORK, hiddenLayers: [2] },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 910, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Capacity 4',
                network: { ...DEFAULT_NETWORK, hiddenLayers: [4] },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 910, numSamples: 24 },
                features: { ...DEFAULT_FEATURES },
            },
        });

        expect(arena.summaries.map((summary) => summary.side)).toEqual(['A', 'B']);
        expect(arena.summaries.map((summary) => summary.label)).toEqual(['Capacity 2', 'Capacity 4']);
        expect(arena.summaries.every((summary) => summary.step === 0)).toBe(true);

        const stepped = workerApi.stepArena(3);

        expect(stepped.snapshotId).toBeGreaterThan(arena.snapshotId);
        expect(stepped.summaries.map((summary) => summary.step)).toEqual([3, 3]);
        expect(stepped.summaries.every((summary) => Number.isFinite(summary.trainLoss))).toBe(true);
        expect(stepped.summaries.every((summary) => Number.isFinite(summary.testLoss))).toBe(true);
    });

    it('keeps live arena stepping isolated from the existing single-model worker run', () => {
        const single = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 911, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.initializeArena({
            modelA: {
                label: 'Arena A',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 912, numSamples: 20 },
                features: { ...DEFAULT_FEATURES },
            },
            modelB: {
                label: 'Arena B',
                network: { ...DEFAULT_NETWORK },
                training: { ...DEFAULT_TRAINING },
                data: { ...DEFAULT_DATA, seed: 913, numSamples: 20 },
                features: { ...DEFAULT_FEATURES },
            },
        });
        workerApi.stepArena(2);

        const singleAfterArena = workerApi.step(1);

        expect(singleAfterArena.step).toBe(1);
        expect(workerApi.getCheckpointTimeline().checkpoints[0].step).toBe(single.snapshot.step);
    });

    it('captures bounded checkpoint metadata and restores an earlier checkpoint', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, optimizer: 'adam' },
            { ...DEFAULT_DATA, seed: 906, numSamples: 30 },
            { ...DEFAULT_FEATURES },
        );

        let timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints).toHaveLength(1);
        expect(timeline.checkpoints[0].step).toBe(init.snapshot.step);

        workerApi.step(6);
        timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints.map((checkpoint) => checkpoint.step)).toContain(6);

        const firstCheckpointId = timeline.checkpoints[0].id;
        const restored = workerApi.restoreCheckpoint(firstCheckpointId);
        expect(restored.snapshot.step).toBe(0);
        expect(restored.timeline.restoredCheckpointId).toBe(firstCheckpointId);
        expect(workerApi.step(1).step).toBe(1);
    });

    it('evicts old checkpoints when the runtime timeline reaches its bound', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 907, numSamples: 30 },
            { ...DEFAULT_FEATURES },
        );

        for (let i = 0; i < 12; i++) {
            workerApi.step(5);
        }

        const timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints.length).toBeLessThanOrEqual(timeline.maxCheckpoints);
        expect(timeline.evictedCount).toBeGreaterThan(0);
        expect(timeline.checkpoints[0].step).toBeGreaterThan(0);
    });

    it('preserves the run on training-only config updates and rebuilds for shape changes', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        const trainingOnly = workerApi.updateConfig(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING, learningRate: DEFAULT_TRAINING.learningRate / 2 },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            false,
        );
        expect(trainingOnly.runId).toBe(init.runId);

        const rebuilt = workerApi.updateConfig(
            { ...DEFAULT_NETWORK, hiddenLayers: [2] },
            { ...DEFAULT_TRAINING, learningRate: DEFAULT_TRAINING.learningRate / 2 },
            { ...DEFAULT_DATA, seed: 901, numSamples: 20 },
            { ...DEFAULT_FEATURES },
            false,
        );
        expect(rebuilt.runId).toBeGreaterThan(init.runId);
        expect(rebuilt.snapshot.step).toBe(0);
    });

    it('reset creates a fresh run after manual training steps', () => {
        const init = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 902, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        expect(init.snapshot.testMetricsStale).toBe(false);

        const stepped = workerApi.step(3);
        expect(stepped.step).toBe(3);
        expect(stepped.testMetricsStale).toBe(true);

        const reset = workerApi.reset();
        expect(reset.runId).toBeGreaterThan(init.runId);
        expect(reset.snapshot.step).toBe(0);
        expect(reset.snapshot.epoch).toBe(0);
        expect(reset.snapshot.testMetricsStale).toBe(false);
    });

    it('honors decision-boundary cadence without recomputing every snapshot', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 903, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: false,
            gridInterval: 2,
        });

        const fresh = workerApi.step(1);
        const skipped = workerApi.step(1);

        expect(fresh.outputGrid).toBeInstanceOf(Float32Array);
        expect(fresh.outputGrid).toHaveLength(fresh.gridSize * fresh.gridSize);
        expect(skipped.outputGrid).toHaveLength(0);
        expect(skipped.neuronGrids).toBeUndefined();
    });

    it('emits layer stats only while layer-stat demand is enabled', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 904, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: false,
        });
        expect(workerApi.step(1).layerStats).toBeUndefined();

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
        });
        expect(workerApi.step(1).layerStats?.length).toBeGreaterThan(0);
    });

    it('rejects malformed demand updates without clearing the previous demand', () => {
        workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA, seed: 905, numSamples: 20 },
            { ...DEFAULT_FEATURES },
        );
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: true,
        });

        expect(() => workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needLayerStats: 'yes',
        } as unknown as typeof DEFAULT_DEMAND)).toThrow('Invalid visualization demand.');

        expect(workerApi.step(1).layerStats?.length).toBeGreaterThan(0);
    });
});
