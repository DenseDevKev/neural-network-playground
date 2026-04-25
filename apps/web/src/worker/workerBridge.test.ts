// ── workerBridge Error-Path Tests ──
// Exercises onerror / onmessageerror handlers and stale-run error passthrough.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
    startRenderLoop,
    terminateWorker,
} from './workerBridge';
import { getFrameBuffer, resetFrameBuffer } from './frameBuffer.ts';
import {
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    allocSharedSnapshotViews,
    publishSharedSnapshot,
} from './sharedSnapshot.ts';
import type { WorkerSnapshotMessage, WorkerToMainMessage } from '@nn-playground/shared';

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
        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].msg.type).toBe('snapshot');
        expect(receivedMessages[0].frameVersion).toBe(frame.version);
        expect(fakePort1.postMessage).toHaveBeenCalledWith({ type: 'frameAck' });
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
});
