// ── workerBridge Error-Path Tests ──
// Exercises onerror / onmessageerror handlers and stale-run error passthrough.

import { beforeEach, describe, expect, it, vi } from 'vitest';

// ── Stub out Comlink before importing workerBridge ──
vi.mock('comlink', () => ({
    wrap: vi.fn(() => ({
        setStreamPort: vi.fn().mockResolvedValue(undefined),
        initialize: vi.fn(),
        updateConfig: vi.fn(),
    })),
    transfer: vi.fn((_val: unknown, _transfers: Transferable[]) => _val),
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
    terminateWorker,
} from './workerBridge';
import type { WorkerToMainMessage } from '@nn-playground/shared';

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
        const listener = listenerCall[1] as (event: MessageEvent) => void;

        listener({
            data: { foo: 'bar' }, // not a valid WorkerToMainMessage
        } as MessageEvent);

        expect(receivedMessages).toHaveLength(1);
        expect(receivedMessages[0].type).toBe('error');
    });
});
