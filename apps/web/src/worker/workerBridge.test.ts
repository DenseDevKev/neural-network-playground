// ── workerBridge Error-Path Tests ──
// Exercises onerror / onmessageerror handlers and stale-run error passthrough.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Stub out Comlink before importing workerBridge ──
const comlinkStub = vi.hoisted(() => ({
    api: {
        setStreamPort: vi.fn().mockResolvedValue(undefined),
        initialize: vi.fn(),
        updateConfig: vi.fn(),
        updateDemand: vi.fn().mockResolvedValue(undefined),
    },
    wrap: vi.fn(),
    transfer: vi.fn((value: unknown) => value),
}));
comlinkStub.wrap.mockImplementation(() => comlinkStub.api);

vi.mock('comlink', () => ({
    wrap: comlinkStub.wrap,
    transfer: comlinkStub.transfer,
}));

function createComlinkApi() {
    return {
        setStreamPort: vi.fn().mockResolvedValue(undefined),
        initialize: vi.fn(),
        updateConfig: vi.fn(),
        updateDemand: vi.fn().mockResolvedValue(undefined),
    };
}

// ── Stub global Worker ──
const VALID_WORKER_READY_MESSAGE = Object.freeze({
    type: 'nn-playground:worker-ready',
    protocolVersion: 1,
});
let autoAnnounceWorkerReady = true;
let fakeWorkerInstance: {
    onerror: ((e: ErrorEvent) => void) | null;
    onmessageerror: (() => void) | null;
    terminate: ReturnType<typeof vi.fn>;
    postMessage: ReturnType<typeof vi.fn>;
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
    dispatchMessage(data: unknown): void;
} | null = null;

vi.stubGlobal('Worker', class FakeWorker {
    onerror: ((e: ErrorEvent) => void) | null = null;
    onmessageerror: (() => void) | null = null;
    terminate = vi.fn();
    postMessage = vi.fn();
    private readonly messageListeners = new Set<(event: MessageEvent<unknown>) => void>();
    addEventListener = vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
        if (type !== 'message') return;
        const callback = typeof listener === 'function'
            ? listener as (event: MessageEvent<unknown>) => void
            : listener.handleEvent.bind(listener);
        this.messageListeners.add(callback);
    });
    removeEventListener = vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
        if (type !== 'message' || typeof listener !== 'function') return;
        this.messageListeners.delete(listener as (event: MessageEvent<unknown>) => void);
    });
    dispatchMessage(data: unknown): void {
        for (const listener of this.messageListeners) {
            listener({ data } as MessageEvent<unknown>);
        }
    }
    constructor() {
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        fakeWorkerInstance = this;
        if (autoAnnounceWorkerReady) {
            queueMicrotask(() => {
                if (fakeWorkerInstance === this) {
                    this.dispatchMessage(VALID_WORKER_READY_MESSAGE);
                }
            });
        }
    }
});

// ── Stub MessageChannel ──
let messageChannelConstructCount = 0;
const fakeMessageChannels: Array<{
    port1: typeof fakePort1;
    port2: typeof fakePort2;
}> = [];
let fakePort1: {
    addEventListener: ReturnType<typeof vi.fn>;
    onmessageerror: (() => void) | null;
    postMessage: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
};
let fakePort2: { close: ReturnType<typeof vi.fn> };

vi.stubGlobal('MessageChannel', class FakeMessageChannel {
    port1: typeof fakePort1;
    port2: typeof fakePort2;
    constructor() {
        messageChannelConstructCount++;
        fakePort1 = {
            addEventListener: vi.fn(),
            onmessageerror: null,
            postMessage: vi.fn(),
            start: vi.fn(),
            close: vi.fn(),
        };
        fakePort2 = { close: vi.fn() };
        this.port1 = fakePort1;
        this.port2 = fakePort2;
        fakeMessageChannels.push({ port1: fakePort1, port2: fakePort2 });
    }
});

// Import after stubs are set up
import {
    getWorkerApi,
    setupStreamChannel,
    onSnapshot,
    newRunTo,
    discardPendingSnapshot,
    postStreamCommand,
    startRenderLoop,
    stopRenderLoop,
    terminateWorker,
    emitE2EWorkerError,
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
    RecipeFingerprint,
    WorkerArtifactProvenanceV2,
    WorkerSnapshotMessage,
    WorkerToMainMessage,
} from '@nn-playground/shared';
import { DEFAULT_DEMAND, WORKER_PROTOCOL_VERSION } from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';

let artifactDataset: DatasetRevision;
let artifactObjectiveKey: string;
let artifactRecipeFingerprint: RecipeFingerprint;

beforeAll(async () => {
    const fixtures = await createScientificTrustFixtures();
    artifactDataset = fixtures.evaluation.dataset;
    artifactObjectiveKey = fixtures.evaluation.objectiveKey;
    artifactRecipeFingerprint = fixtures.prepared.identities.recipeFingerprint;
});

describe('workerBridge readiness gating', () => {
    beforeEach(() => {
        vi.useRealTimers();
        terminateWorker();
        fakeWorkerInstance = null;
        autoAnnounceWorkerReady = false;
        messageChannelConstructCount = 0;
        fakeMessageChannels.length = 0;
        comlinkStub.api.setStreamPort.mockReset().mockResolvedValue(undefined);
        comlinkStub.api.updateDemand.mockReset().mockResolvedValue(undefined);
        comlinkStub.wrap.mockReset().mockImplementation(() => comlinkStub.api);
        comlinkStub.transfer.mockClear();
    });

    afterEach(() => {
        terminateWorker();
        autoAnnounceWorkerReady = true;
        vi.useRealTimers();
    });

    it('does not expose a non-stream RPC proxy before valid READY', async () => {
        const access = getWorkerApi();
        const accessPromise = Promise.resolve(
            access as unknown as typeof comlinkStub.api,
        );
        void accessPromise.catch(() => undefined);

        expect(access).toBeInstanceOf(Promise);
        let resolvedApi: typeof comlinkStub.api | null = null;
        void accessPromise.then((api) => {
            resolvedApi = api;
        });
        await Promise.resolve();

        expect(resolvedApi).toBeNull();
        expect(comlinkStub.api.updateDemand).not.toHaveBeenCalled();

        fakeWorkerInstance!.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        const api = await accessPromise;
        await api.updateDemand(DEFAULT_DEMAND);

        expect(comlinkStub.api.updateDemand).toHaveBeenCalledWith(DEFAULT_DEMAND);
    });

    it('does not create or transfer the stream channel before a valid READY message', async () => {
        const setup = setupStreamChannel();
        await Promise.resolve();

        expect(messageChannelConstructCount).toBe(0);
        expect(comlinkStub.api.setStreamPort).not.toHaveBeenCalled();

        fakeWorkerInstance!.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await setup;

        expect(messageChannelConstructCount).toBe(1);
        expect(comlinkStub.transfer).toHaveBeenCalledWith(fakePort2, [fakePort2]);
        expect(comlinkStub.api.setStreamPort).toHaveBeenCalledTimes(1);
    });

    it('cleans up a rejected channel and shares one retry across concurrent callers', async () => {
        const transferError = new Error('transfer failed');
        let resolveSecondAttempt!: () => void;
        comlinkStub.api.setStreamPort
            .mockRejectedValueOnce(transferError)
            .mockImplementationOnce(() => new Promise<void>((resolve) => {
                resolveSecondAttempt = resolve;
            }));

        const firstAttempt = setupStreamChannel();
        fakeWorkerInstance!.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        const firstError = await firstAttempt.then(
            () => null,
            (error: unknown) => error,
        );
        expect(firstError).toBe(transferError);

        const rejectedChannel = fakeMessageChannels[0];
        expect(rejectedChannel.port1.close).toHaveBeenCalledTimes(1);
        expect(rejectedChannel.port2.close).toHaveBeenCalledTimes(1);
        expect(rejectedChannel.port1.addEventListener).not.toHaveBeenCalled();

        const secondAttempt = setupStreamChannel();
        const concurrentAttempt = setupStreamChannel();
        await vi.waitFor(() => {
            expect(comlinkStub.api.setStreamPort).toHaveBeenCalledTimes(2);
        });
        expect(fakeMessageChannels).toHaveLength(2);

        resolveSecondAttempt();
        await expect(secondAttempt).resolves.toBeUndefined();
        await expect(concurrentAttempt).resolves.toBeUndefined();

        const installedChannel = fakeMessageChannels[1];
        expect(installedChannel.port1.close).not.toHaveBeenCalled();
        expect(installedChannel.port2.close).not.toHaveBeenCalled();
        expect(installedChannel.port1.addEventListener).toHaveBeenCalledWith(
            'message',
            expect.any(Function),
        );
        expect(installedChannel.port1.addEventListener).toHaveBeenCalledTimes(1);
        expect(installedChannel.port1.start).toHaveBeenCalledTimes(1);

        const stopCommand = { type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION } as const;
        postStreamCommand(stopCommand);
        expect(installedChannel.port1.postMessage).toHaveBeenCalledWith(stopCommand);
    });

    it('preserves worker B pending setup when worker A readiness rejects after replacement', async () => {
        const apiA = createComlinkApi();
        const apiB = createComlinkApi();
        let resolveWorkerB!: () => void;
        apiB.setStreamPort.mockImplementationOnce(() => new Promise<void>((resolve) => {
            resolveWorkerB = resolve;
        }));
        comlinkStub.wrap
            .mockImplementationOnce(() => apiA)
            .mockImplementationOnce(() => apiB);

        const setupA = setupStreamChannel();
        const outcomeA = setupA.then(
            () => null,
            (error: unknown) => error,
        );
        const workerA = fakeWorkerInstance!;

        terminateWorker();
        const setupB = setupStreamChannel();
        const workerB = fakeWorkerInstance!;
        expect(workerB).not.toBe(workerA);
        workerB.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await vi.waitFor(() => {
            expect(apiB.setStreamPort).toHaveBeenCalledTimes(1);
        });

        const errorA = await outcomeA;
        expect(errorA).toBeInstanceOf(Error);
        expect((errorA as Error).message).toBe(
            'Training worker terminated before becoming ready',
        );

        const concurrentB = setupStreamChannel();
        await Promise.resolve();
        expect(fakeMessageChannels).toHaveLength(1);
        expect(comlinkStub.transfer).toHaveBeenCalledTimes(1);
        expect(apiB.setStreamPort).toHaveBeenCalledTimes(1);

        resolveWorkerB();
        await expect(setupB).resolves.toBeUndefined();
        await expect(concurrentB).resolves.toBeUndefined();

        const installedChannel = fakeMessageChannels[0];
        expect(installedChannel.port1.addEventListener).toHaveBeenCalledTimes(1);
        expect(installedChannel.port1.start).toHaveBeenCalledTimes(1);
    });

    it('rejects worker A stale transfer without clearing or replacing worker B setup', async () => {
        const apiA = createComlinkApi();
        const apiB = createComlinkApi();
        let resolveWorkerA!: () => void;
        let resolveWorkerB!: () => void;
        apiA.setStreamPort.mockImplementationOnce(() => new Promise<void>((resolve) => {
            resolveWorkerA = resolve;
        }));
        apiB.setStreamPort.mockImplementationOnce(() => new Promise<void>((resolve) => {
            resolveWorkerB = resolve;
        }));
        comlinkStub.wrap
            .mockImplementationOnce(() => apiA)
            .mockImplementationOnce(() => apiB);

        const setupA = setupStreamChannel();
        fakeWorkerInstance!.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await vi.waitFor(() => {
            expect(apiA.setStreamPort).toHaveBeenCalledTimes(1);
        });
        const channelA = fakeMessageChannels[0];
        const outcomeA = setupA.then(
            () => null,
            (error: unknown) => error,
        );

        terminateWorker();
        const setupB = setupStreamChannel();
        fakeWorkerInstance!.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await vi.waitFor(() => {
            expect(apiB.setStreamPort).toHaveBeenCalledTimes(1);
        });
        const channelB = fakeMessageChannels[1];

        resolveWorkerA();
        const errorA = await outcomeA;
        expect(errorA).toBeInstanceOf(Error);
        expect((errorA as Error).message).toBe(
            'Training worker terminated during stream setup',
        );
        expect(channelA.port1.close).toHaveBeenCalledTimes(1);
        expect(channelA.port2.close).toHaveBeenCalledTimes(1);
        expect(channelA.port1.addEventListener).not.toHaveBeenCalled();
        expect(channelA.port1.start).not.toHaveBeenCalled();

        const concurrentB = setupStreamChannel();
        await Promise.resolve();
        expect(fakeMessageChannels).toHaveLength(2);
        expect(comlinkStub.transfer).toHaveBeenCalledTimes(2);
        expect(apiB.setStreamPort).toHaveBeenCalledTimes(1);

        resolveWorkerB();
        await expect(setupB).resolves.toBeUndefined();
        await expect(concurrentB).resolves.toBeUndefined();
        expect(channelB.port1.addEventListener).toHaveBeenCalledTimes(1);
        expect(channelB.port1.start).toHaveBeenCalledTimes(1);
        expect(channelB.port1.close).not.toHaveBeenCalled();

        const stopCommand = { type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION } as const;
        postStreamCommand(stopCommand);
        expect(channelB.port1.postMessage).toHaveBeenCalledWith(stopCommand);
        expect(channelA.port1.postMessage).not.toHaveBeenCalled();
    });

    it('ignores malformed and unrelated worker messages until readiness times out', async () => {
        vi.useFakeTimers();
        const setup = setupStreamChannel();
        const outcome = setup.then(
            () => ({ error: null }),
            (error: unknown) => ({ error }),
        );

        fakeWorkerInstance!.dispatchMessage({ ...VALID_WORKER_READY_MESSAGE, extra: true });
        fakeWorkerInstance!.dispatchMessage({ type: 'APPLY', path: [] });
        await vi.runAllTimersAsync();

        const { error } = await outcome;
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toBe(
            'Training worker did not become ready within 5000ms',
        );
        expect(messageChannelConstructCount).toBe(0);
        expect(comlinkStub.api.setStreamPort).not.toHaveBeenCalled();
        expect(fakeWorkerInstance!.removeEventListener).toHaveBeenCalled();
    });

    it('rejects pending readiness on termination and lets a fresh worker retry', async () => {
        const firstSetup = setupStreamChannel();
        const firstWorker = fakeWorkerInstance!;
        const firstRejection = expect(firstSetup).rejects.toThrow(
            'Training worker terminated before becoming ready',
        );

        terminateWorker();
        await firstRejection;
        expect(firstWorker.removeEventListener).toHaveBeenCalled();
        expect(comlinkStub.api.setStreamPort).not.toHaveBeenCalled();

        const secondSetup = setupStreamChannel();
        const secondWorker = fakeWorkerInstance!;
        expect(secondWorker).not.toBe(firstWorker);
        secondWorker.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await secondSetup;

        expect(comlinkStub.api.setStreamPort).toHaveBeenCalledTimes(1);
    });

    it('rejects and cleans pending readiness when the worker errors', async () => {
        const setup = setupStreamChannel();
        const rejection = expect(setup).rejects.toThrow(
            'Training worker failed before readiness: boot failed',
        );

        fakeWorkerInstance!.onerror!({ message: 'boot failed' } as ErrorEvent);

        await rejection;
        expect(fakeWorkerInstance!.removeEventListener).toHaveBeenCalled();
        expect(messageChannelConstructCount).toBe(0);
    });

    it('rejects and cleans pending readiness on message deserialization errors', async () => {
        const setup = setupStreamChannel();
        const rejection = expect(setup).rejects.toThrow(
            'Training worker message deserialization failed before readiness',
        );

        fakeWorkerInstance!.onmessageerror!();

        await rejection;
        expect(fakeWorkerInstance!.removeEventListener).toHaveBeenCalled();
        expect(messageChannelConstructCount).toBe(0);
    });

    it('ignores late error events from a terminated worker after a replacement is created', async () => {
        const firstAccess = Promise.resolve(getWorkerApi()).catch(() => undefined);
        const firstWorker = fakeWorkerInstance!;
        const lateError = firstWorker.onerror!;
        const lateMessageError = firstWorker.onmessageerror!;

        terminateWorker();
        await firstAccess;

        const secondAccess = Promise.resolve(getWorkerApi());
        void secondAccess.catch(() => undefined);
        const secondWorker = fakeWorkerInstance!;
        const received: WorkerToMainMessage[] = [];
        const unsubscribe = onSnapshot((message) => received.push(message));
        const setup = setupStreamChannel();
        void setup.catch(() => undefined);

        lateError({ message: 'late worker A error' } as ErrorEvent);
        lateMessageError();

        expect(received).toEqual([]);
        expect(messageChannelConstructCount).toBe(0);

        secondWorker.dispatchMessage(VALID_WORKER_READY_MESSAGE);
        await secondAccess;
        await setup;
        expect(comlinkStub.api.setStreamPort).toHaveBeenCalledTimes(1);
        unsubscribe();
    });
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
        void getWorkerApi().catch(() => undefined);
    });

    afterEach(() => {
        unsub();
        terminateWorker();
    });

    it('delivers a protocol-v2 E2E error only after subscription', () => {
        unsub();
        terminateWorker();
        expect(emitE2EWorkerError('injected startup failure')).toBe(false);

        const messages: WorkerToMainMessage[] = [];
        unsub = onSnapshot((message) => messages.push(message));
        expect(emitE2EWorkerError('injected startup failure')).toBe(true);
        expect(messages).toEqual([{
            type: 'error',
            protocolVersion: 2,
            runId: 0,
            message: 'injected startup failure',
        }]);
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
            data: {
                type: 'error',
                protocolVersion: WORKER_PROTOCOL_VERSION,
                runId: 5,
                message: 'stale error',
            },
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

    it('does not throw when a malformed handshake cannot be stringified', async () => {
        await setupStreamChannel();
        const listener = getRegisteredStreamListener();

        expect(() => listener({
            data: {
                type: 'sharedBuffers',
                protocolVersion: WORKER_PROTOCOL_VERSION,
                runId: 1,
                control: {},
                outputGrid: {},
                neuronGrids: {},
                gridSize: 2n,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent)).not.toThrow();
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0]).toMatchObject({ type: 'error' });
    });

    it('rejects a future-version shared-buffer handshake without throwing', async () => {
        await setupStreamChannel();
        const listener = getRegisteredStreamListener();

        expect(() => listener({
            data: {
                type: 'sharedBuffers',
                protocolVersion: WORKER_PROTOCOL_VERSION + 1,
                runId: 1,
                control: new SharedArrayBuffer(32),
                outputGrid: new SharedArrayBuffer(16),
                neuronGrids: new SharedArrayBuffer(16),
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent)).not.toThrow();
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0]).toMatchObject({ type: 'error' });
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

function makeSnapshotPayload(
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
        ...overrides,
    } as WorkerSnapshotMessage;
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
    const base = makeSnapshotPayload(snapshotId);
    const histogramSampleCount = Math.min(128, artifactDataset.trainCount);
    const message: WorkerSnapshotMessage = {
        ...base,
        protocolVersion: WORKER_PROTOCOL_VERSION,
        model: {
            generationId: 1,
            revision: snapshotId,
            step: snapshotId * 10,
            epoch: snapshotId,
        },
        recipeFingerprint: artifactRecipeFingerprint,
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
        checkpointTimeline: {
            checkpoints: [],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
        },
        ...overrides,
    };
    if (Object.prototype.hasOwnProperty.call(overrides, 'runId')
        && !Object.prototype.hasOwnProperty.call(overrides, 'model')) {
        message.model = { ...message.model, generationId: message.runId };
    }
    const currentProvenance = (basis: ArtifactBasis): ArtifactProvenance => {
        const provenance = artifactProvenance(snapshotId, basis);
        return {
            ...provenance,
            model: { ...provenance.model, generationId: message.runId },
        };
    };
    if (!Object.prototype.hasOwnProperty.call(overrides, 'artifacts')) {
        const artifacts: {
            -readonly [Key in keyof WorkerArtifactProvenanceV2]?: WorkerArtifactProvenanceV2[Key];
        } = {};
        if ((message.outputGrid?.length ?? 0) > 0
            || message.multiclassClassGrid !== undefined
            || message.sharedSeq !== undefined) {
            artifacts.decisionBoundary = currentProvenance({
                kind: 'prediction-grid',
                pointCount: 4,
                domain: [-1, 1, -1, 1],
            });
        }
        if ((message.neuronGrids?.length ?? 0) > 0
            || (message.sharedSeq !== undefined && message.neuronGridLayout !== undefined)) {
            artifacts.neuronGrids = currentProvenance({
                kind: 'prediction-grid',
                pointCount: 4,
                domain: [-1, 1, -1, 1],
            });
        }
        if (message.activationHistogramBins !== undefined) {
            artifacts.activationHistogram = currentProvenance({
                kind: 'bounded-sample',
                split: 'train',
                sampleCount: histogramSampleCount,
                populationCount: artifactDataset.trainCount,
            });
        }
        if (message.confusionMatrix !== undefined
            || message.multiclassConfusionMatrix !== undefined) {
            artifacts.confusionMatrix = currentProvenance({
                kind: 'full-split',
                split: 'test',
                sampleCount: artifactDataset.testCount,
                populationCount: artifactDataset.testCount,
            });
            if (!Object.prototype.hasOwnProperty.call(overrides, 'confusionMatrixEvaluationId')) {
                message.confusionMatrixEvaluationId = snapshotId;
            }
            if (message.confusionMatrix !== undefined
                && !Object.prototype.hasOwnProperty.call(overrides, 'confusionMatrixVersion')) {
                message.confusionMatrixVersion = snapshotId;
            }
        }
        message.artifacts = artifacts;
    }
    return message;
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
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.version).toBeGreaterThan(startVersion);
        expect(frame.outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(frame.neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));
        expect(frame.weights).toEqual(new Float32Array([0.5, -0.25]));
        expect(frame.biases).toEqual(new Float32Array([0.1]));
        expect(frame.parameterProvenance).toEqual({
            model: makeStrictSnapshotMessage(1).model,
            recipeFingerprint: makeStrictSnapshotMessage(1).recipeFingerprint,
        });
        expect(frame.activationHistogramBins).toEqual(Float32Array.from([
            128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        expect(frame.activationHistogramLayout?.layers).toHaveLength(1);
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].msg.type).toBe('snapshot');
        expect(receivedMessages[0].frameVersion).toBe(frame.version);
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck', protocolVersion: WORKER_PROTOCOL_VERSION });
    });

    it('discards and acknowledges a queued same-run snapshot before restore publication', () => {
        const listener = getRegisteredStreamListener();
        const startVersion = getFrameBuffer().version;

        startRenderLoop();
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        expect(discardPendingSnapshot(1)).toBe(true);
        runNextAnimationFrame();

        expect(getFrameBuffer().version).toBe(startVersion);
        expect(receivedMessages).toEqual([]);
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck', protocolVersion: WORKER_PROTOCOL_VERSION });

        // Discard is not a run reset: the consumed snapshot ID remains fenced.
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(receivedMessages).toEqual([]);
    });

    it('drops and acknowledges a delayed same-run frame below the restored revision fence', () => {
        const listener = getRegisteredStreamListener();
        const startVersion = getFrameBuffer().version;

        expect(discardPendingSnapshot(1, 5)).toBe(false);
        startRenderLoop();
        listener({ data: makeStrictSnapshotMessage(4) } as MessageEvent);
        runNextAnimationFrame();

        expect(getFrameBuffer().version).toBe(startVersion);
        expect(receivedMessages).toEqual([]);
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck', protocolVersion: WORKER_PROTOCOL_VERSION });

        listener({ data: makeStrictSnapshotMessage(5) } as MessageEvent);
        runNextAnimationFrame();
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].msg).toMatchObject({
            type: 'snapshot',
            model: { generationId: 1, revision: 5 },
        });
        expect(fakePort1.postMessage).toHaveBeenCalledTimes(2);
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

    it('stores immutable parameter provenance isolated from later message mutation', () => {
        const listener = getRegisteredStreamListener();
        const message = makeStrictSnapshotMessage(1);

        startRenderLoop();
        listener({ data: message } as MessageEvent);
        runNextAnimationFrame();

        const stored = getFrameBuffer().parameterProvenance!;
        (message.model as { revision: number }).revision = 99;
        expect(stored.model.revision).toBe(1);
        expect(Object.isFrozen(stored)).toBe(true);
        expect(Object.isFrozen(stored.model)).toBe(true);
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

    it.each(['missing', 'malformed'] as const)(
        'rejects a strict frame with %s checkpoint metadata before frame mutation',
        (kind) => {
            const listener = getRegisteredStreamListener();
            const before = getFrameBuffer();
            const valid = makeStrictSnapshotMessage(1);
            const message: unknown = kind === 'missing'
                ? (({ checkpointTimeline: _timeline, ...rest }) => rest)(valid)
                : {
                    ...valid,
                    checkpointTimeline: {
                        ...valid.checkpointTimeline!,
                        maxCheckpoints: 7,
                    },
                };

            startRenderLoop();
            listener({ data: message } as MessageEvent);
            runNextAnimationFrame();

            expect(getFrameBuffer()).toBe(before);
            expect(receivedMessages).toHaveLength(1);
            expect(receivedMessages[0].msg).toMatchObject({ type: 'error' });
        },
    );

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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(getFrameBuffer().neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));

        listener({
            data: makeStrictSnapshotMessage(2, {
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
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));

        listener({
            data: makeStrictSnapshotMessage(2, {
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
        listener({ data: makeStrictSnapshotMessage(1) } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(getFrameBuffer().neuronGrids).toEqual(new Float32Array([0.4, 0.3, 0.2, 0.1]));

        listener({
            data: makeStrictSnapshotMessage(2, {
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
            data: makeStrictSnapshotMessage(1, {
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
            data: makeStrictSnapshotMessage(2, {
                outputGrid: undefined,
                neuronGrids: undefined,
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
            data: makeStrictSnapshotMessage(3, {
                outputGrid: new Float32Array([0.2, 0.3, 0.4, 0.5]),
                neuronGrids: undefined,
                neuronGridLayout: undefined,
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

    it('retains the last paired confusion artifact when a cadence frame omits it', () => {
        const listener = getRegisteredStreamListener();
        const confusionMatrix = {
            tp: 80,
            tn: 60,
            fp: 5,
            fn: 5,
        };

        startRenderLoop();
        listener({
            data: makeStrictSnapshotMessage(1, { confusionMatrix }),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterInitial = getFrameBuffer();
        const initialConfusionVersion = afterInitial.confusionMatrixVersion;
        expect(afterInitial.confusionMatrix).toBe(confusionMatrix);

        listener({
            data: makeStrictSnapshotMessage(2, {
                confusionMatrix: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterOmission = getFrameBuffer();
        expect(afterOmission.confusionMatrix).toBe(confusionMatrix);
        expect(afterOmission.confusionMatrixVersion).toBe(initialConfusionVersion);

    });

    it('stores paired multiclass confusion, clears binary confusion, and retains cadence omissions', () => {
        const listener = getRegisteredStreamListener();
        const binaryConfusionMatrix = {
            tp: 80,
            tn: 60,
            fp: 5,
            fn: 5,
        };
        const multiclassConfusionMatrix = {
            classCount: 3 as const,
            classLabels: [0, 1, 2] as const,
            counts: [40, 5, 5, 5, 40, 5, 5, 5, 40] as const,
        };

        startRenderLoop();
        listener({
            data: makeStrictSnapshotMessage(1, { confusionMatrix: binaryConfusionMatrix }),
        } as MessageEvent);
        runNextAnimationFrame();
        expect(getFrameBuffer().confusionMatrix).toBe(binaryConfusionMatrix);

        listener({
            data: makeStrictSnapshotMessage(2, {
                confusionMatrix: undefined,
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterMulticlass = getFrameBuffer();
        const initialMulticlassConfusionVersion = afterMulticlass.multiclassConfusionMatrixVersion;
        expect(afterMulticlass.confusionMatrix).toBeNull();
        expect(afterMulticlass.multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);
        expect(initialMulticlassConfusionVersion).toBeGreaterThan(0);

        listener({
            data: makeStrictSnapshotMessage(3, {
                multiclassConfusionMatrix: undefined,
                multiclassConfusionMatrixVersion: undefined,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const afterOmission = getFrameBuffer();
        expect(afterOmission.multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);
        expect(afterOmission.multiclassConfusionMatrixVersion).toBe(initialMulticlassConfusionVersion);

    });

    it('fresh binary confusion snapshots clear cached multiclass confusion matrices', () => {
        const listener = getRegisteredStreamListener();
        const multiclassConfusionMatrix = {
            classCount: 3 as const,
            classLabels: [0, 1, 2] as const,
            counts: [40, 5, 5, 5, 40, 5, 5, 5, 40] as const,
        };
        const binaryConfusionMatrix = {
            tp: 70,
            tn: 70,
            fp: 5,
            fn: 5,
        };

        startRenderLoop();
        listener({
            data: makeStrictSnapshotMessage(1, {
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }),
        } as MessageEvent);
        runNextAnimationFrame();
        const initialMulticlassConfusionVersion = getFrameBuffer().multiclassConfusionMatrixVersion;
        expect(getFrameBuffer().multiclassConfusionMatrix).toBe(multiclassConfusionMatrix);

        listener({
            data: makeStrictSnapshotMessage(2, { confusionMatrix: binaryConfusionMatrix }),
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
            data: makeStrictSnapshotMessage(1, {
                confusionMatrix: {
                    tp: 1,
                    tn: 1,
                    fp: 0,
                    fn: 0,
                },
                multiclassConfusionMatrix,
                multiclassConfusionMatrixVersion: 1,
            }),
        } as MessageEvent);

        expect(receivedMessages.at(-1)?.msg.type).toBe('error');
        expect(getFrameBuffer().version).toBe(initialFrame.version);
        expect(getFrameBuffer().confusionMatrix).toBeNull();
        expect(getFrameBuffer().multiclassConfusionMatrix).toBeNull();
    });

    it('closes the stream port on termination and drops later stream commands', () => {
        postStreamCommand({ type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION });
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION });

        terminateWorker();
        expect(fakePort1.close).toHaveBeenCalledTimes(1);
        fakePort1.postMessage.mockClear();

        postStreamCommand({ type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION });
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
            data: makeStrictSnapshotMessage(1, {
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
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck', protocolVersion: WORKER_PROTOCOL_VERSION });
    });

    it('rejects malformed shared-buffer lengths without replacing the installed transport', () => {
        const listener = getRegisteredStreamListener();
        const sharedViews = allocSharedSnapshotViews(2, 1);
        const sharedSeq = publishSharedSnapshot(
            sharedViews,
            new Float32Array([0.8, 0.7, 0.6, 0.5]),
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );
        const validHandshake = {
            type: 'sharedBuffers' as const,
            protocolVersion: WORKER_PROTOCOL_VERSION,
            runId: 1,
            control: sharedViews.controlSAB,
            outputGrid: sharedViews.outputGridSAB,
            neuronGrids: sharedViews.neuronGridsSAB,
            gridSize: 2,
            neuronGridLayout: { count: 1, gridSize: 2 },
        };
        listener({ data: validHandshake } as MessageEvent);

        expect(() => listener({
            data: {
                ...validHandshake,
                outputGrid: new SharedArrayBuffer(1),
            },
        } as MessageEvent)).not.toThrow();
        expect(receivedMessages.at(-1)?.msg).toMatchObject({ type: 'error' });

        startRenderLoop();
        listener({
            data: makeStrictSnapshotMessage(1, {
                outputGrid: undefined,
                neuronGrids: undefined,
                sharedSeq,
            }),
        } as MessageEvent);
        runNextAnimationFrame();

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toEqual(new Float32Array([0.8, 0.7, 0.6, 0.5]));
        expect(frame.neuronGrids).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
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
            data: makeStrictSnapshotMessage(1, {
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
                type: 'sharedBuffers', protocolVersion: WORKER_PROTOCOL_VERSION,
                runId: 2,
                control: currentViews.controlSAB,
                outputGrid: currentViews.outputGridSAB,
                neuronGrids: currentViews.neuronGridsSAB,
                gridSize: 2,
                neuronGridLayout: { count: 1, gridSize: 2 },
            },
        } as MessageEvent);
        listener({
            data: makeStrictSnapshotMessage(2, {
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
            data: makeStrictSnapshotMessage(2, {
                outputGrid: new Float32Array([2, 2, 2, 2]),
            }),
        } as MessageEvent);
        listener({
            data: makeStrictSnapshotMessage(1, {
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
            data: makeStrictSnapshotMessage(3, {
                outputGrid: new Float32Array([3, 3, 3, 3]),
            }),
        } as MessageEvent);
        listener({
            data: {
                type: 'status', protocolVersion: WORKER_PROTOCOL_VERSION,
                runId: 1,
                status: 'paused',
                pauseReason: 'diverged',
            },
        } as MessageEvent);

        expect(receivedMessages.map(({ msg }) => msg.type)).toEqual(['status', 'snapshot']);
        expect((receivedMessages[0].msg as { type: 'status'; pauseReason?: string }).pauseReason).toBe('diverged');
        expect((receivedMessages[1].msg as WorkerSnapshotMessage).snapshotId).toBe(3);
        expect(getFrameBuffer().outputGrid).toEqual(new Float32Array([3, 3, 3, 3]));
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck', protocolVersion: WORKER_PROTOCOL_VERSION });
    });
});
