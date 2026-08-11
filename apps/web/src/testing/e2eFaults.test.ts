import { describe, expect, it, vi } from 'vitest';
import {
    E2E_WORKER_FAULT_MESSAGE,
    consumeE2EWorkerFault,
} from './e2eFaults.ts';

class MemoryStorage implements Pick<Storage, 'getItem' | 'setItem'> {
    private readonly values = new Map<string, string>();

    getItem(key: string): string | null {
        return this.values.get(key) ?? null;
    }

    setItem(key: string, value: string): void {
        this.values.set(key, value);
    }
}

describe('consumeE2EWorkerFault', () => {
    it('consumes the enabled startup fault once per session', () => {
        const storage = new MemoryStorage();
        const url = new URL('https://example.test/?e2eWorkerFault=startup-once');

        expect(consumeE2EWorkerFault(url, storage, true)).toBe('startup-once');
        expect(consumeE2EWorkerFault(url, storage, true)).toBeNull();
        expect(consumeE2EWorkerFault(url, storage, false)).toBeNull();
        expect(E2E_WORKER_FAULT_MESSAGE).toMatch(/startup/i);
    });

    it('ignores disabled and unknown fault names without consuming the session', () => {
        const storage = {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
        };
        expect(consumeE2EWorkerFault(
            new URL('https://example.test/?e2eWorkerFault=startup-once'),
            storage,
            false,
        )).toBeNull();
        expect(storage.getItem).not.toHaveBeenCalled();
        expect(storage.setItem).not.toHaveBeenCalled();
        expect(consumeE2EWorkerFault(
            new URL('https://example.test/?e2eWorkerFault=unknown'),
            storage,
            true,
        )).toBeNull();
        expect(storage.getItem).not.toHaveBeenCalled();
        expect(storage.setItem).not.toHaveBeenCalled();
        expect(consumeE2EWorkerFault(
            new URL('https://example.test/?e2eWorkerFault=startup-once'),
            storage,
            true,
        )).toBe('startup-once');
        expect(storage.setItem).toHaveBeenCalledTimes(1);
    });

    it.each(['getItem', 'setItem'] as const)('fails closed when storage.%s throws', (method) => {
        const storage = {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
        };
        storage[method].mockImplementation(() => {
            throw new DOMException('storage unavailable', 'SecurityError');
        });

        expect(consumeE2EWorkerFault(
            new URL('https://example.test/?e2eWorkerFault=startup-once'),
            storage,
            true,
        )).toBeNull();
    });
});
