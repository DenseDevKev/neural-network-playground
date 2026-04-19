import { afterEach, describe, expect, it } from 'vitest';
import { detectWebGPU, resetWebGPUDetectionCache } from '../detect.js';

describe('detectWebGPU', () => {
    afterEach(() => {
        resetWebGPUDetectionCache();
    });

    it('returns null in environments without navigator.gpu', async () => {
        // Node + Vitest's default 'node' environment expose neither
        // `navigator` nor `navigator.gpu`. Detection must collapse to
        // null so callers can fall back without exception.
        const device = await detectWebGPU();
        expect(device).toBeNull();
    });

    it('caches the null result so subsequent calls do not re-attempt', async () => {
        const a = await detectWebGPU();
        const b = await detectWebGPU();
        expect(a).toBeNull();
        expect(b).toBeNull();
    });

    it('returns null without throwing when adapter request rejects', async () => {
        // Inject a fake navigator.gpu that throws on requestAdapter. Node's
        // built-in `navigator` property is read-only, so we have to splice
        // the stub on with defineProperty (configurable:true) and tear it
        // down in the finally block.
        const desc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
        Object.defineProperty(globalThis, 'navigator', {
            configurable: true,
            value: {
                gpu: {
                    requestAdapter: () => Promise.reject(new Error('mock failure')),
                },
            },
        });
        try {
            const device = await detectWebGPU();
            expect(device).toBeNull();
        } finally {
            if (desc) {
                Object.defineProperty(globalThis, 'navigator', desc);
            } else {
                // No prior descriptor — remove the stub entirely.
                delete (globalThis as unknown as { navigator?: unknown }).navigator;
            }
        }
    });
});
