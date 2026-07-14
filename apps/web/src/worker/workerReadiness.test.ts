import { describe, expect, it, vi } from 'vitest';
import {
    WORKER_READY_MESSAGE,
    exposeWorkerApiAndAnnounceReady,
    isWorkerReadyMessage,
} from './workerReadiness.ts';

describe('worker readiness protocol', () => {
    it('accepts only the exact READY discriminator, version, and shape', () => {
        expect(isWorkerReadyMessage({
            type: 'nn-playground:worker-ready',
            protocolVersion: 1,
        })).toBe(true);

        for (const malformed of [
            null,
            [],
            { type: 'nn-playground:worker-ready' },
            { type: 'nn-playground:worker-ready', protocolVersion: 2 },
            { type: 'worker-ready', protocolVersion: 1 },
            { type: 'nn-playground:worker-ready', protocolVersion: 1, extra: true },
        ]) {
            expect(isWorkerReadyMessage(malformed)).toBe(false);
        }
    });

    it('exposes the RPC handler before announcing READY', () => {
        const order: string[] = [];
        const expose = vi.fn(() => order.push('expose'));
        const target = {
            postMessage: vi.fn((message: unknown) => {
                expect(message).toEqual(WORKER_READY_MESSAGE);
                order.push('ready');
            }),
        };

        exposeWorkerApiAndAnnounceReady(expose, target);

        expect(order).toEqual(['expose', 'ready']);
        expect(expose).toHaveBeenCalledTimes(1);
        expect(target.postMessage).toHaveBeenCalledTimes(1);
    });
});
