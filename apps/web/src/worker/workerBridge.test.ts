// ── workerBridge Error-Path Tests ──
// Exercises onerror / onmessageerror handlers and stale-run error passthrough.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Stub out Comlink before importing workerBridge ──
vi.mock('comlink', () => ({
    wrap: vi.fn(() => ({
        setStreamPort: vi.fn().mockResolvedValue(undefined),
        initialize: vi.fn(),
        updateConfig: vi.fn(),
    })),
    transfer: vi.fn((_val: unknown, _t: Transferable[]) => _val),
}));

// ── Stub global Worker ──
let fakeWorkerInstance: {
    onerror: ((e: ErrorEvent) => void) | null;
    onmessageerror: (() => void) | null;
    terminate: ReturnType<typeof vi.fn>;
    postMessage: ReturnType<typeof vi.fn>;
} | null = null;

vi.stubGlobal('Worker', class FakeWorker {
    onerror: ((e: ErrorEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    terminate = vi.fn();
    postMessage = vi.fn();
    constructor() {
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        fakeWorkerInstance = this;
    }
});

// ── Stub MessageChannel ──
let fakePort1: {
    addEventListener: ReturnType<typeof vi.fn>;
    onmessageerror: (() => void) | null;
    postMessage: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
};
let fakePort2: { [key: string]: unknown };

vi.stubGlobal('MessageChannel', class FakeMessageChannel {
    port1: typeof fakePort1;
    port2: typeof fakePort2;
    constructor() {
        fakePort1 = {
            addEventListener: vi.fn(),
            onmessageerror: null,
            postMessage: vi.fn(),
            start: vi.fn(),
            close: vi.fn(),
        };
        fakePort2 = {};
        this.port1 = fakePort1;
        this.port2 = fakePort2;
    }
});

// Import after stubs are set up
import {
    getWorkerApi,
    setupStreamChannel,
    onSnapshot,
    newRunTo,
    postStreamCommand,
    startRenderLoop,
    stopRenderLoop,
    terminateWorker,
} from './workerBridge';
import { getFrameBuffer, resetFrameBuffer } from './frameBuffer.ts';
import {
    CTL_FLAGS,
    CTL_SEQ_END,
    CTL_SEQ_START,
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    allocSharedSnapshotViews,
    publishSharedSnapshot,
} from './sharedSnapshot.ts';
import type {
    ArtifactBasis,
    ArtifactProvenance,
    DatasetRevision,
    WorkerArtifactProvenanceV2,
    WorkerSnapshotMessage,
    WorkerToMainMessage,
} from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';

let artifactDataset: DatasetRevision;
let artifactObjectiveKey: string;

beforeAll(async () => {
    const fixtures = await createScientificTrustFixtures();
    artifactDataset = fixtures.evaluation.dataset;
    artifactObjectiveKey = fixtures.evaluation.objectiveKey;
});

describe('workerBridge error paths', () => {
    let receivedMessages: WorkerToMainMessage[];
    let unsub: () => void;

    beforeEach(async () => {
        receivedMessages = [];
        terminateWorker();
        fakeWorkerInstance = null;

        unsub = onSnapshot((msg) => {
            receivedMessages.push(msg as WorkerToMainMessage);
        });

        // Trigger worker creation
        getWorkerApi();
    });

    afterEach(() => {
        unsub();
        terminateWorker();
    });

    it('routes Worker onerror to _onSnapshot as type=error', () => {
        expect(fakeWorkerInstance).toBeTruthy();
        expect(typeof fakeWorkerInstance!.onerror).toBe('function');

        fakeWorkerInstance!.onerror!({ message: 'Script error' } as ErrorEvent);

        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].type).toBe('error');
        expect((receivedMessages[0] as { type: string; message: string }).message).toContain('Script error');
    });

    it('routes Worker onmessageerror to _onSnapshot as type=error', () => {
        expect(fakeWorkerInstance).toBeTruthy();
        expect(typeof fakeWorkerInstance!.onmessageerror).toBe('function');

        fakeWorkerInstance!.onmessageerror!();

        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].type).toBe('error');
    });

    it('routes stream port onmessageerror to _onSnapshot as type=error', async () => {
        await setupStreamChannel();

        expect(typeof fakePort1.onmessageerror).toBe('function');
        fakePort1.onmessageerror!();

        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].type).toBe('error');
    });

    it('stale-run errors pass through the runId filter', async () => {
        // newRunTo(10) makes _currentRunId = 10; an error with runId=5 must still surface.
        const { newRunTo } = await import('./workerBridge');
        newRunTo(10);

        // Simulate a stream message listener call directly by finding and calling
        // the addEventListener callback registered on port1.
        await setupStreamChannel();

        const listenerCall = (fakePort1.addEventListener as ReturnType<typeof vi.fn>).mock.calls.find(
            (call: unknown[]) => call[0] === 'message',
        );
        expect(listenerCall).toBeTruthy();
        if (!listenerCall) {
            throw new Error('Expected stream port message listener to be registered');
        }
        const listener = listenerCall[1] as (event: MessageEvent) => void;

        // Stale error (runId=5 < _currentRunId=10) must surface
        listener({
            data: { type: 'error', runId: 5, message: 'stale error' },
        } as MessageEvent);

        expect(receivedMessages.some((m) => m.type === 'error')).toBe(true);
    });

    it('malformed messages are rejected with an error notification', async () => {
        await setupStreamChannel();

        const listenerCall = (fakePort1.addEventListener as ReturnType<typeof vi.fn>).mock.calls.find(
            (call: unknown[]) => call[0] === 'message',
        );
        expect(listenerCall).toBeTruthy();
        if (!listenerCall) {
            throw new Error('Expected stream port message listener to be registered');
        }
        const listener = listenerCall[1] as (event: MessageEvent) => void;

        listener({
            data: { foo: 'bar' }, // not a valid WorkerToMainMessage
        } as MessageEvent);

        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].type).toBe('error');
    });

    it('delivers current V2 evidence immediately and drops stale generations', async () => {
        const fixtures = await createScientificTrustFixtures();
        await setupStreamChannel();
        const listener = getRegisteredStreamListener();
        newRunTo(fixtures.evidence.latestEvaluation!.model.generationId);

        listener({ data: fixtures.evidence } as MessageEvent);
        expect(receivedMessages).toEqual([fixtures.evidence]);

        newRunTo(fixtures.evidence.latestEvaluation!.model.generationId + 1);
        listener({ data: fixtures.evidence } as MessageEvent);
        expect(receivedMessages).toEqual([fixtures.evidence]);
    });

    it('always surfaces a structured V2 worker error', async () => {
        await setupStreamChannel();
        const listener = getRegisteredStreamListener();
        newRunTo(20);
        const error = {
            type: 'worker-error',
            protocolVersion: 2,
            requestId: 1,
            generationId: 1,
            code: 'runtime-failure',
            path: '$',
            message: 'stale structured failure',
            source: 'runtime',
        } as const;

        listener({ data: error } as MessageEvent);
        expect(receivedMessages).toEqual([error]);
    });
});

function getRegisteredStreamListener(): (event: MessageEvent) => void {
    const listenerCall = (fakePort1.addEventListener as ReturnType<typeof vi.fn>).mock.calls.find(
        (call: unknown[]) => call[0] === 'message',
    );
    expect(listenerCall).toBeTruthy();
    if (!listenerCall) {
        throw new Error('Expected stream port message listener to be registered');
    }
    return listenerCall[1] as (event: MessageEvent) => void;
}

function makeSnapshotMessage(
    snapshotId: number,
    overrides: Partial<WorkerSnapshotMessage> = {},
): WorkerSnapshotMessage {
    return {
        type: 'snapshot',
        runId: 1,
        snapshotId,
        scalars: {
            step: snapshotId * 10,
            epoch: snapshotId,
            trainLoss: 0.4,
            testLoss: 0.5,
            trainAccuracy: 0.7,
            testAccuracy: 0.6,
            gridSize: 2,
        },
        outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
        neuronGrids: new Float32Array([0.4, 0.3, 0.2, 0.1]),
        neuronGridLayout: { count: 1, gridSize: 2 },
        weights: new Float32Array([0.5, -0.25]),
        biases: new Float32Array([0.1]),
        weightLayout: { layerSizes: [2, 1] },
        activationHistogramBins: new Float32Array([1, 2]),
        activationHistogramLayout: {
            binCount: 2,
            layers: [
                {
                    layerIndex: 0,
                    binCount: 2,
                    binStart: 0,
                    binWidth: 0.5,
                    minActivation: 0,
                    maxActivation: 1,
                    totalCount: 3,
                    nearZeroCount: 1,
                    saturatedCount: 0,
                },
            ],
        },
        activationHistogramVersion: 1,
        historyPoint: {
            step: snapshotId * 10,
            trainLoss: 0.4,
            testLoss: 0.5,
            trainAccuracy: 0.7,
            testAccuracy: 0.6,
        },
        ...overrides,
    };
}

function artifactProvenance(
    snapshotId: number,
    basis: ArtifactBasis,
): ArtifactProvenance {
    return {
        model: {
            generationId: 1,
            revision: snapshotId,
            step: snapshotId * 10,
            epoch: snapshotId,
        },
        dataset: artifactDataset,
        objectiveKey: artifactObjectiveKey,
        basis,
    };
}

function makeStrictSnapshotMessage(
    snapshotId: number,
    overrides: Partial<WorkerSnapshotMessage> = {},
): WorkerSnapshotMessage {
    const { historyPoint: _legacyHistoryPoint, ...legacy } = makeSnapshotMessage(snapshotId);
    const artifacts: WorkerArtifactProvenanceV2 = {
        decisionBoundary: artifactProvenance(snapshotId, {
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        }),
        neuronGrids: artifactProvenance(snapshotId, {
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        }),
        activationHistogram: artifactProvenance(snapshotId, {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, artifactDataset.trainCount),
            populationCount: artifactDataset.trainCount,
        }),
    };
    const histogramSampleCount = Math.min(128, artifactDataset.trainCount);
    return {
        ...legacy,
        protocolVersion: 2,
        model: {
            generationId: 1,
            revision: snapshotId,
            step: snapshotId * 10,
            epoch: snapshotId,
        },
        activationHistogramBins: Float32Array.from([
            histogramSampleCount,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]),
        activationHistogramLayout: {
            binCount: 12,
            layers: [{
                layerIndex: 0,
                binCount: 12,
                binStart: 0,
                binWidth: 0.5,
                minActivation: 0,
                maxActivation: 1,
                totalCount: histogramSampleCount,
                nearZeroCount: 1,
                saturatedCount: 0,
            }],
        },
        artifacts,
        ...overrides,
    } as WorkerSnapshotMessage;
}

function withTestMetricsStale(
    msg: WorkerSnapshotMessage,
    testMetricsStale: boolean,
): WorkerSnapshotMessage {
    return {
        ...msg,
        scalars: {
            ...msg.scalars,
            testMetricsStale,
        },
    };
}

describe('workerBridge streamed snapshots', () => {
    let receivedMessages: Array<{ msg: WorkerToMainMessage; frameVersion: number }>;
    let unsubscribe: () => void;
    let rafCallback: FrameRequestCallback | null;
    let originalRequestAnimationFrame: typeof globalThis.requestAnimationFrame;
    let originalCancelAnimationFrame: typeof globalThis.cancelAnimationFrame;

    beforeEach(async () => {
        receivedMessages = [];
        terminateWorker();
        resetFrameBuffer();
        fakeWorkerInstance = null;
        rafCallback = null;

        originalRequestAnimationFrame = globalThis.requestAnimationFrame;
        originalCancelAnimationFrame = globalThis.cancelAnimationFrame;
        vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
            rafCallback = callback;
            return 42;
        }));
        vi.stubGlobal('cancelAnimationFrame', vi.fn());

        unsubscribe = onSnapshot((msg) => {
            receivedMessages.push({
                msg: msg as WorkerToMainMessage,
                frameVersion: getFrameBuffer().version,
            });
        });

        getWorkerApi();
        await setupStreamChannel();
        newRunTo(1);
    });

    afterEach(() => {
        unsubscribe();
        terminateWorker();
        resetFrameBuffer();
        vi.stubGlobal('requestAnimationFrame', originalRequestAnimationFrame);
        vi.stubGlobal('cancelAnimationFrame', originalCancelAnimationFrame);
    });

    function runNextAnimationFrame(): void {
        const callback = rafCallback;
        expect(callback).toBeTruthy();
        if (!callback) {
            throw new Error('Expected requestAnimationFrame callback');
        }
        rafCallback = null;
        callback(performance.now());
    }

    it('applies the latest streamed snapshot on rAF, updates frame versions, and acks the frame', () => {
        const listener = getRegisteredStreamListener();
        const startVersion = getFrameBuffer().version;

        startRenderLoop();
        listener({ data: makeSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.version).toBeGreaterThan(startVersion);
        expect(frame.outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(frame.neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));
        expect(frame.weights).toEqual(new Float32Array([0.5, -0.25]));
        expect(frame.biases).toEqual(new Float32Array([0.1]));
        expect(frame.activationHistogramBins).toEqual(new Float32Array([1, 2]));
        expect(frame.activationHistogramLayout?.layers).toHaveLength(1);
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].msg.type).toBe('snapshot');
        expect(receivedMessages[0].frameVersion).toBe(frame.version);
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck' });
    });

    it('atomically stores strict artifact arrays with each artifact provenance', () => {
        const listener = getRegisteredStreamListener();
        const base = makeStrictSnapshotMessage(1);
        const activationStatistics = artifactProvenance(1, {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, artifactDataset.trainCount),
            populationCount: artifactDataset.trainCount,
        });
        const message = {
            ...base,
            layerStats: [{
                meanActivation: 0.2,
                activationStd: 0.1,
                meanAbsWeight: 0.3,
                meanAbsGradient: 0.05,
            }],
            layerStatsGradientRevision: 1,
            artifacts: {
                ...base.artifacts,
                activationStatistics,
            },
        } satisfies WorkerSnapshotMessage;

        startRenderLoop();
        listener({ data: message } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toBe(message.outputGrid);
        expect(frame.decisionBoundaryProvenance).toEqual(message.artifacts?.decisionBoundary);
        expect(frame.neuronGrids).toBe(message.neuronGrids);
        expect(frame.neuronGridsProvenance).toEqual(message.artifacts?.neuronGrids);
        expect(frame.activationHistogramBins).toBe(message.activationHistogramBins);
        expect(frame.activationHistogramProvenance).toEqual(
            message.artifacts?.activationHistogram,
        );
        expect(frame.layerStatsProvenance).toEqual(activationStatistics);
        expect(frame.layerStatsGradientRevision).toBe(1);
    });

    it('rejects an entire strict frame when any artifact payload lacks provenance', () => {
        const listener = getRegisteredStreamListener();
        const before = getFrameBuffer();
        const message = makeStrictSnapshotMessage(1, {
            artifacts: {
                decisionBoundary: artifactProvenance(1, {
                    kind: 'prediction-grid',
                    pointCount: 4,
                    domain: [-1, 1, -1, 1],
                }),
            },
        });

        startRenderLoop();
        listener({ data: message } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer()).toBe(before);
    });

    it('rejects strict provenance without the corresponding artifact payload', () => {
        const listener = getRegisteredStreamListener();
        const before = getFrameBuffer();
        const message = makeStrictSnapshotMessage(1, {
            outputGrid: undefined,
            neuronGrids: undefined,
            neuronGridLayout: undefined,
            activationHistogramBins: undefined,
            activationHistogramLayout: undefined,
            activationHistogramVersion: undefined,
            artifacts: {
                decisionBoundary: artifactProvenance(1, {
                    kind: 'prediction-grid',
                    pointCount: 4,
                    domain: [-1, 1, -1, 1],
                }),
            },
        });

        startRenderLoop();
        listener({ data: message } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer()).toBe(before);
    });

    it.each([
        ['foreign generation', (message: WorkerSnapshotMessage) => ({
            ...message,
            artifacts: {
                ...message.artifacts,
                decisionBoundary: {
                    ...message.artifacts!.decisionBoundary!,
                    model: {
                        ...message.artifacts!.decisionBoundary!.model,
                        generationId: 2,
                    },
                },
            },
        })],
        ['mixed dataset', (message: WorkerSnapshotMessage) => ({
            ...message,
            artifacts: {
                ...message.artifacts,
                neuronGrids: {
                    ...message.artifacts!.neuronGrids!,
                    dataset: {
                        ...message.artifacts!.neuronGrids!.dataset,
                        datasetKey: `d2.1.${'A'.repeat(43)}`,
                    },
                },
            },
        })],
        ['mixed objective', (message: WorkerSnapshotMessage) => ({
            ...message,
            artifacts: {
                ...message.artifacts,
                activationHistogram: {
                    ...message.artifacts!.activationHistogram!,
                    objectiveKey: `o2.1.${'A'.repeat(43)}`,
                },
            },
        })],
    ] as const)('rejects strict artifact provenance with %s identity', (_label, forge) => {
        const listener = getRegisteredStreamListener();
        const before = getFrameBuffer();

        startRenderLoop();
        listener({ data: forge(makeStrictSnapshotMessage(1)) } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer()).toBe(before);
    });

    it('rejects a layer gradient revision newer than its activation model revision', () => {
        const listener = getRegisteredStreamListener();
        const before = getFrameBuffer();
        const base = makeStrictSnapshotMessage(1);
        const message = {
            ...base,
            layerStats: [{
                meanActivation: 0.2,
                activationStd: 0.1,
                meanAbsWeight: 0.3,
                meanAbsGradient: 0.05,
            }],
            layerStatsGradientRevision: 2,
            artifacts: {
                ...base.artifacts,
                activationStatistics: artifactProvenance(1, {
                    kind: 'bounded-sample',
                    split: 'train',
                    sampleCount: Math.min(128, artifactDataset.trainCount),
                    populationCount: artifactDataset.trainCount,
                }),
            },
        } satisfies WorkerSnapshotMessage;

        startRenderLoop();
        listener({ data: message } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer()).toBe(before);
    });

    it('preserves strict artifact bytes and provenance on a reuse frame', () => {
        const listener = getRegisteredStreamListener();
        const first = makeStrictSnapshotMessage(1);

        startRenderLoop();
        listener({ data: first } as MessageEvent);
        runNextAnimationFrame();
        const grid = getFrameBuffer().outputGrid;
        const gridProvenance = getFrameBuffer().decisionBoundaryProvenance;

        listener({
            data: makeStrictSnapshotMessage(2, {
                outputGrid: undefined,
                neuronGrids: undefined,
                neuronGridLayout: undefined,
                activationHistogramBins: undefined,
                activationHistogramLayout: undefined,
                activationHistogramVersion: undefined,
                artifacts: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().outputGrid).toBe(grid);
        expect(getFrameBuffer().decisionBoundaryProvenance).toBe(gridProvenance);
    });

    it('stores a frozen provenance snapshot isolated from later message mutation', () => {
        const listener = getRegisteredStreamListener();
        const message = makeStrictSnapshotMessage(1);

        startRenderLoop();
        listener({ data: message } as MessageEvent);
        const mutable = message as unknown as {
            artifacts: { decisionBoundary: { model: { revision: number } } };
        };
        mutable.artifacts.decisionBoundary.model.revision = 999;
        runNextAnimationFrame();

        const stored = getFrameBuffer().decisionBoundaryProvenance!;
        expect(stored.model.revision).toBe(1);
        expect(Object.isFrozen(stored)).toBe(true);
        expect(Object.isFrozen(stored.model)).toBe(true);
        expect(Object.isFrozen(stored.basis)).toBe(true);
    });

    it('applies strict SAB artifact provenance only with the consistent shared payload', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const sharedSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([0.8, 0.7, 0.6, 0.5]),
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: sharedViews.controlSAB,
                outputGrid: sharedViews.outputGridSAB,
                neuronGrids: sharedViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        const decisionBoundary = artifactProvenance(1, {
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });
        const neuronGrids = artifactProvenance(1, {
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });

        startRenderLoop();
        listener({
            data: makeStrictSnapshotMessage(1, {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
                activationHistogramBins: undefined,
                activationHistogramLayout: undefined,
                activationHistogramVersion: undefined,
                artifacts: { decisionBoundary, neuronGrids },
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().outputGrid).toEqual(
            new Float32Array([0.8, 0.7, 0.6, 0.5]),
        );
        expect(getFrameBuffer().decisionBoundaryProvenance).toEqual(decisionBoundary);
        expect(getFrameBuffer().neuronGridsProvenance).toEqual(neuronGrids);
    });

    it('keeps accepted SAB bytes and provenance unchanged after a torn read', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const firstSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([0.8, 0.7, 0.6, 0.5]),
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: sharedViews.controlSAB,
                outputGrid: sharedViews.outputGridSAB,
                neuronGrids: sharedViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        const strictShared = (snapshotId: number, sharedSeq: number) => {
            const decisionBoundary = artifactProvenance(snapshotId, {
                kind: 'prediction-grid',
                pointCount: 4,
                domain: [-1, 1, -1, 1],
            });
            const neuronGrids = artifactProvenance(snapshotId, {
                kind: 'prediction-grid',
                pointCount: 4,
                domain: [-1, 1, -1, 1],
            });
            return makeStrictSnapshotMessage(snapshotId, {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
                activationHistogramBins: undefined,
                activationHistogramLayout: undefined,
                activationHistogramVersion: undefined,
                artifacts: { decisionBoundary, neuronGrids },
            });
        };

        startRenderLoop();
        listener({ data: strictShared(1, firstSeq) } as MessageEvent);
        runNextAnimationFrame();
        const acceptedGrid = getFrameBuffer().outputGrid;
        const acceptedNeurons = getFrameBuffer().neuronGrids;
        const acceptedProvenance = getFrameBuffer().decisionBoundaryProvenance;

        sharedViews.outputGrid.set([9, 9, 9, 9]);
        sharedViews.neuronGrids.set([8, 8, 8, 8]);
        Atomics.store(sharedViews.control, CTL_FLAGS, FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS);
        Atomics.store(sharedViews.control, CTL_SEQ_END, firstSeq + 1);
        Atomics.store(sharedViews.control, CTL_SEQ_START, firstSeq + 2);
        listener({ data: strictShared(2, firstSeq + 1) } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().outputGrid).toBe(acceptedGrid);
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.8, 0.7, 0.6, 0.5]));
        expect(getFrameBuffer().neuronGrids).toBe(acceptedNeurons);
        expect(getFrameBuffer().neuronGrids).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(getFrameBuffer().decisionBoundaryProvenance).toBe(acceptedProvenance);
    });

    it('does not swap SAB staging bytes when the observed sequence is newer than the envelope', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const firstSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([1, 2, 3, 4]),
            new Float32Array([4, 3, 2, 1]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: sharedViews.controlSAB,
                outputGrid: sharedViews.outputGridSAB,
                neuronGrids: sharedViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        const message = (snapshotId: number, sharedSeq: number) => makeStrictSnapshotMessage(
            snapshotId,
            {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
                activationHistogramBins: undefined,
                activationHistogramLayout: undefined,
                activationHistogramVersion: undefined,
                artifacts: {
                    decisionBoundary: artifactProvenance(snapshotId, {
                        kind: 'prediction-grid',
                        pointCount: 4,
                        domain: [-1, 1, -1, 1],
                    }),
                    neuronGrids: artifactProvenance(snapshotId, {
                        kind: 'prediction-grid',
                        pointCount: 4,
                        domain: [-1, 1, -1, 1],
                    }),
                },
            },
        );

        startRenderLoop();
        listener({ data: message(1, firstSeq) } as MessageEvent);
        runNextAnimationFrame();
        const accepted = Array.from(getFrameBuffer().outputGrid!);
        const secondSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([9, 9, 9, 9]),
            new Float32Array([8, 8, 8, 8]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        expect(secondSeq).toBeGreaterThan(firstSeq);

        listener({ data: message(2, firstSeq) } as MessageEvent);
        runNextAnimationFrame();

        expect(Array.from(getFrameBuffer().outputGrid!)).toEqual(accepted);
        expect(getFrameBuffer().decisionBoundaryProvenance?.model.revision).toBe(1);
    });

    it('does not swap SAB staging bytes when flags disagree with claimed artifacts', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const firstSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([1, 2, 3, 4]),
            new Float32Array([4, 3, 2, 1]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: sharedViews.controlSAB,
                outputGrid: sharedViews.outputGridSAB,
                neuronGrids: sharedViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        const message = (snapshotId: number, sharedSeq: number) => makeStrictSnapshotMessage(
            snapshotId,
            {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
                activationHistogramBins: undefined,
                activationHistogramLayout: undefined,
                activationHistogramVersion: undefined,
                artifacts: {
                    decisionBoundary: artifactProvenance(snapshotId, {
                        kind: 'prediction-grid', pointCount: 4, domain: [-1, 1, -1, 1],
                    }),
                    neuronGrids: artifactProvenance(snapshotId, {
                        kind: 'prediction-grid', pointCount: 4, domain: [-1, 1, -1, 1],
                    }),
                },
            },
        );

        startRenderLoop();
        listener({ data: message(1, firstSeq) } as MessageEvent);
        runNextAnimationFrame();
        const acceptedGrid = Array.from(getFrameBuffer().outputGrid!);
        const acceptedNeurons = Array.from(getFrameBuffer().neuronGrids!);
        const outputOnlySeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([9, 9, 9, 9]),
            null,
            FLAG_OUTPUT_GRID,
        );

        listener({ data: message(2, outputOnlySeq) } as MessageEvent);
        runNextAnimationFrame();

        expect(Array.from(getFrameBuffer().outputGrid!)).toEqual(acceptedGrid);
        expect(Array.from(getFrameBuffer().neuronGrids!)).toEqual(acceptedNeurons);
        expect(getFrameBuffer().decisionBoundaryProvenance?.model.revision).toBe(1);
    });

    it('clears cached scalar grids when a streamed snapshot sends explicit empty grid payloads', () => {
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({ data: makeSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(getFrameBuffer().neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));

        listener({
            data: makeSnapshotMessage(2, {
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
                neuronGridLayout: undefined,
                confusionMatrix: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toBeNull();
        expect(frame.neuronGrids).toBeNull();
        expect(frame.neuronGridLayout).toBeNull();
    });

    it('stores multiclass boundary payloads and clears cached scalar grids', () => {
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({ data: makeSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));

        listener({
            data: makeSnapshotMessage(2, {
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
                neuronGridLayout: undefined,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
                multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                },
                multiclassBoundaryVersion: 1,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toBeNull();
        expect(frame.neuronGrids).toBeNull();
        expect(frame.multiclassClassGrid).toEqual(new Uint8Array([0, 1, 2, 1]));
        expect(frame.multiclassConfidenceGrid).toEqual(new Float32Array([0.7, 0.6, 0.9, 0.5]));
        expect(frame.multiclassBoundaryLayout).toEqual({
            gridSize: 2,
            classCount: 3,
            classLabels: [0, 1, 2],
        });
    });

    it('clears cached scalar grids whenever a multiclass boundary payload arrives', () => {
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({ data: makeSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(getFrameBuffer().neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));

        listener({
            data: makeSnapshotMessage(2, {
                outputGrid: undefined,
                neuronGrids: undefined,
                neuronGridLayout: undefined,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
                multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                },
                multiclassBoundaryVersion: 1,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toBeNull();
        expect(frame.neuronGrids).toBeNull();
        expect(frame.neuronGridLayout).toBeNull();
        expect(frame.multiclassClassGrid).toEqual(new Uint8Array([0, 1, 2, 1]));
    });

    it('retains cached multiclass boundary payloads on cadence omissions and clears them on fresh scalar grids', () => {
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({
            data: makeSnapshotMessage(1, {
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
                neuronGridLayout: undefined,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
                multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                },
                multiclassBoundaryVersion: 1,
            }),
        } as MessageEvent);
        runNextAnimationFrame();
        const initialMulticlassVersion = getFrameBuffer().multiclassBoundaryVersion;

        listener({
            data: makeSnapshotMessage(2, {
                outputGrid: new Float32Array(0),
                neuronGrids: new Float32Array(0),
                neuronGridLayout: undefined,
                multiclassClassGrid: undefined,
                multiclassConfidenceGrid: undefined,
                multiclassBoundaryLayout: undefined,
                multiclassBoundaryVersion: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().multiclassClassGrid).toEqual(new Uint8Array([0, 1, 2, 1]));
        expect(getFrameBuffer().multiclassBoundaryVersion).toBe(initialMulticlassVersion);

        listener({
            data: makeSnapshotMessage(3, {
                outputGrid: new Float32Array([0.2, 0.3, 0.4, 0.5]),
                neuronGrids: undefined,
                multiclassClassGrid: undefined,
                multiclassConfidenceGrid: undefined,
                multiclassBoundaryLayout: undefined,
                multiclassBoundaryVersion: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toEqual(new Float32Array([0.2, 0.3, 0.4, 0.5]));
        expect(frame.multiclassClassGrid).toBeNull();
        expect(frame.multiclassConfidenceGrid).toBeNull();
        expect(frame.multiclassBoundaryLayout).toBeNull();
        expect(frame.multiclassBoundaryVersion).toBe(initialMulticlassVersion + 1);
    });

    it('clears cached confusion matrix only when fresh streamed metrics omit it', () => {
        const listener = getRegisteredStreamListener();
        const confusionMatrix = {
            tp: 8,
            tn: 7,
            fp: 2,
            fn: 1,
        };

        startRenderLoop();
        listener({
            data: withTestMetricsStale(makeSnapshotMessage(1, { confusionMatrix }), false),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterInitial = getFrameBuffer();
        const initialConfusionVersion = afterInitial.confusionMatrixVersion;
        expect(afterInitial.confusionMatrix).toBe(confusionMatrix);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(2, {
                confusionMatrix: undefined,
            }), true),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterStale = getFrameBuffer();
        expect(afterStale.confusionMatrix).toBe(confusionMatrix);
        expect(afterStale.confusionMatrixVersion).toBe(initialConfusionVersion);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(3, {
                confusionMatrix: undefined,
            }), false),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterFresh = getFrameBuffer();
        expect(afterFresh.confusionMatrix).toBeNull();
        expect(afterFresh.confusionMatrixVersion).toBe(initialConfusionVersion + 1);
    });

    it('stores fresh multiclass confusion matrices, clears binary confusion, and respects stale omissions', () => {
        const listener = getRegisteredStreamListener();
        const binaryConfusionMatrix = {
            tp: 8,
            tn: 7,
            fp: 2,
            fn: 1,
        };
        const multiclassConfusionMatrix = {
            classCount: 3 as const,
            classLabels: [0, 1, 2] as const,
            counts: [3, 1, 0, 0, 4, 1, 1, 0, 5] as const,
        };

        startRenderLoop();
        listener({
            data: withTestMetricsStale(makeSnapshotMessage(1, { confusionMatrix: binaryConfusionMatrix }), false),
        } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().confusionMatrix).toBe(binaryConfusionMatrix);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(2, {
                confusionMatrix: undefined,
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }), false),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterMulticlass = getFrameBuffer();
        const initialMulticlassConfusionVersion = afterMulticlass.multiclassConfusionMatrixVersion;
        expect(afterMulticlass.confusionMatrix).toBeNull();
        expect(afterMulticlass.multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);
        expect(initialMulticlassConfusionVersion).toBeGreaterThan(0);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(3, {
                multiclassConfusionMatrix: undefined,
                multiclassConfusionMatrixVersion: undefined,
            }), true),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterStale = getFrameBuffer();
        expect(afterStale.multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);
        expect(afterStale.multiclassConfusionMatrixVersion).toBe(initialMulticlassConfusionVersion);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(4, {
                confusionMatrix: undefined,
                multiclassConfusionMatrix: undefined,
                multiclassConfusionMatrixVersion: undefined,
            }), false),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterFreshOmission = getFrameBuffer();
        expect(afterFreshOmission.multiclassConfusionMatrix).toBeNull();
        expect(afterFreshOmission.multiclassConfusionMatrixVersion).toBe(
            initialMulticlassConfusionVersion + 1,
        );
    });

    it('fresh binary confusion snapshots clear cached multiclass confusion matrices', () => {
        const listener = getRegisteredStreamListener();
        const multiclassConfusionMatrix = {
            classCount: 3 as const,
            classLabels: [0, 1, 2] as const,
            counts: [3, 1, 0, 0, 4, 1, 1, 0, 5] as const,
        };
        const binaryConfusionMatrix = {
            tp: 6,
            tn: 5,
            fp: 1,
            fn: 2,
        };

        startRenderLoop();
        listener({
            data: withTestMetricsStale(makeSnapshotMessage(1, {
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }), false),
        } as MessageEvent);
        runNextAnimationFrame();
        const initialMulticlassConfusionVersion = getFrameBuffer().multiclassConfusionMatrixVersion;
        expect(getFrameBuffer().multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);

        listener({
            data: withTestMetricsStale(makeSnapshotMessage(2, { confusionMatrix: binaryConfusionMatrix }), false),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.confusionMatrix).toBe(binaryConfusionMatrix);
        expect(frame.multiclassConfusionMatrix).toBeNull();
        expect(frame.multiclassConfusionMatrixVersion).toBe(initialMulticlassConfusionVersion + 1);
    });

    it('rejects binary-plus-multiclass confusion snapshots without mutating the frame buffer', () => {
        const listener = getRegisteredStreamListener();
        const multiclassConfusionMatrix = {
            classCount: 3 as const,
            classLabels: [0, 1, 2] as const,
            counts: [3, 1, 0, 0, 4, 1, 1, 0, 5] as const,
        };
        const initialFrame = getFrameBuffer();

        startRenderLoop();
        listener({
            data: withTestMetricsStale(makeSnapshotMessage(1, {
                confusionMatrix: {
                    tp: 1,
                    tn: 1,
                    fp: 0,
                    fn: 0,
                },
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }), false),
        } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer().version).toBe(initialFrame.version);
        expect(getFrameBuffer().confusionMatrix).toBeNull();
        expect(getFrameBuffer().multiclassConfusionMatrix).toBeNull();
    });

    it('closes the stream port on termination and drops later stream commands', () => {
        postStreamCommand({ type: 'stopTraining' });
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'stopTraining' });

        terminateWorker();
        expect(fakePort1.close).toHaveBeenCalledTimes(1);
        fakePort1.postMessage.mockClear();

        postStreamCommand({ type: 'stopTraining' });
        expect(fakePort1.postMessage).not.toHaveBeenCalled();
    });

    it('installs shared buffers and reads snapshot payloads from the SAB handshake', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const sharedSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([0.8, 0.7, 0.6, 0.5]),
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );

        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: sharedViews.controlSAB,
                outputGrid: sharedViews.outputGridSAB,
                neuronGrids: sharedViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);

        startRenderLoop();
        listener({
            data: makeSnapshotMessage(1, {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toEqual(new Float32Array([0.8, 0.7, 0.6, 0.5]));
        expect(frame.neuronGrids).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(frame.neuronGridLayout).toEqual({ count: 1, gridSize: 2 });
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck' });
    });

    it('does not let stale shared buffers install or satisfy a later snapshot', () => {
        const listener = getRegisteredStreamListener();
        const staleViews = allocSharedSnapshotViews(2, 1);
        const staleSeq = publishSharedSnapshot(
            staleViews,
            new Float32Array([9, 9, 9, 9]),
            new Float32Array([8, 8, 8, 8]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );

        newRunTo(2);
        resetFrameBuffer();
        listener({
            data: {
                type: 'sharedBuffers',
                runId: 1,
                control: staleViews.controlSAB,
                outputGrid: staleViews.outputGridSAB,
                neuronGrids: staleViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);

        startRenderLoop();
        listener({
            data: makeSnapshotMessage(1, {
                runId: 2,
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq: staleSeq,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        let frame = getFrameBuffer();
        expect(frame.outputGrid).toBeNull();
        expect(frame.neuronGrids).toBeNull();

        const currentViews = allocSharedSnapshotViews(2, 1);
        const currentSeq = publishSharedSnapshot(
            currentViews,
            new Float32Array([0.2, 0.4, 0.6, 0.8]),
            new Float32Array([0.8, 0.6, 0.4, 0.2]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );

        listener({
            data: {
                type: 'sharedBuffers',
                runId: 2,
                control: currentViews.controlSAB,
                outputGrid: currentViews.outputGridSAB,
                neuronGrids: currentViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        listener({
            data: makeSnapshotMessage(2, {
                runId: 2,
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq: currentSeq,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        frame = getFrameBuffer();
        expect(frame.outputGrid).toEqual(new Float32Array([0.2, 0.4, 0.6, 0.8]));
        expect(frame.neuronGrids).toEqual(new Float32Array([0.8, 0.6, 0.4, 0.2]));
    });

    it('drops stale out-of-order snapshots before they reach the frame buffer', () => {
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({
            data: makeSnapshotMessage(2, {
                outputGrid: new Float32Array([2, 2, 2, 2]),
            }),
        } as MessageEvent);
        listener({
            data: makeSnapshotMessage(1, {
                outputGrid: new Float32Array([1, 1, 1, 1]),
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([2, 2, 2, 2]));
        expect(receivedMessages).toHaveLength(1);
        expect((receivedMessages[0].msg as WorkerSnapshotMessage).snapshotId).toBe(2);
        expect(fakePort1.postMessage).toHaveBeenCalledTimes(1);
    });

    it('flushes a queued final snapshot when automatic paused status stops the render loop', () => {
        unsubscribe();
        receivedMessages = [];
        unsubscribe = onSnapshot((msg) => {
            receivedMessages.push({
                msg: msg as WorkerToMainMessage,
                frameVersion: getFrameBuffer().version,
            });
            if (msg.type === 'status' && msg.status === 'paused' && msg.pauseReason) {
                stopRenderLoop();
            }
        });
        const listener = getRegisteredStreamListener();

        startRenderLoop();
        listener({
            data: makeSnapshotMessage(3, {
                outputGrid: new Float32Array([3, 3, 3, 3]),
            }),
        } as MessageEvent);
        listener({
            data: {
                type: 'status',
                runId: 1,
                status: 'paused',
                pauseReason: 'diverged',
            },
        } as MessageEvent);

        expect(receivedMessages.map(({ msg }) => msg.type)).toEqual(['status', 'snapshot']);
        expect((receivedMessages[0].msg as { type: 'status'; pauseReason?: string }).pauseReason).toBe('diverged');
        expect((receivedMessages[1].msg as WorkerSnapshotMessage).snapshotId).toBe(3);
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([3, 3, 3, 3]));
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck' });
    });
});
