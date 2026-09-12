import { StrictMode, type ReactNode } from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
    useExperimentMemoryStore,
} from '../store/experimentMemoryStore.ts';
import { useExperimentMemoryStorageSync } from './useExperimentMemoryStorageSync.ts';

function StrictWrapper({ children }: { children: ReactNode }) {
    return <StrictMode>{children}</StrictMode>;
}

function dispatchStorage(key: string | null, storageArea: Storage | null): void {
    act(() => window.dispatchEvent(new StorageEvent('storage', { key, storageArea })));
}

describe('useExperimentMemoryStorageSync', () => {
    const originalHydrate = useExperimentMemoryStore.getState().hydrate;
    let hydrate: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        hydrate = vi.fn().mockResolvedValue(undefined);
        useExperimentMemoryStore.setState({ hydrate });
    });

    afterEach(() => {
        useExperimentMemoryStore.setState({ hydrate: originalHydrate });
        vi.restoreAllMocks();
    });

    it('keeps one StrictMode listener and hydrates once for every accepted memory event', () => {
        const localStorageArea = window.localStorage;
        renderHook(() => useExperimentMemoryStorageSync(), { wrapper: StrictWrapper });

        dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, localStorageArea);
        dispatchStorage(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY, localStorageArea);
        dispatchStorage(null, localStorageArea);

        expect(hydrate).toHaveBeenCalledTimes(3);
    });

    it('ignores panel, session-storage, and null-storage-area events', () => {
        renderHook(() => useExperimentMemoryStorageSync(), { wrapper: StrictWrapper });

        dispatchStorage('panel-v2-network', window.localStorage);
        dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, window.sessionStorage);
        dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, null);

        expect(hydrate).not.toHaveBeenCalled();
    });

    it('removes the storage listener on unmount', () => {
        const localStorageArea = window.localStorage;
        const { unmount } = renderHook(() => useExperimentMemoryStorageSync(), {
            wrapper: StrictWrapper,
        });

        unmount();
        dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, localStorageArea);

        expect(hydrate).not.toHaveBeenCalled();
    });

    it('installs no listener when reading localStorage throws a SecurityError', () => {
        const localStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage');
        if (!localStorageDescriptor) throw new Error('localStorage descriptor is required');
        const addEventListener = vi.spyOn(window, 'addEventListener');
        let unmount: (() => void) | undefined;

        try {
            Object.defineProperty(window, 'localStorage', {
                configurable: true,
                get() {
                    throw new DOMException('Storage access denied.', 'SecurityError');
                },
            });

            expect(() => {
                ({ unmount } = renderHook(() => useExperimentMemoryStorageSync(), {
                    wrapper: StrictWrapper,
                }));
            }).not.toThrow();
            expect(addEventListener.mock.calls.filter(([type]) => type === 'storage'))
                .toHaveLength(0);
            expect(hydrate).not.toHaveBeenCalled();
        } finally {
            Object.defineProperty(window, 'localStorage', localStorageDescriptor);
            unmount?.();
        }
    });
    it('releases each subscription across repeated StrictMode mount cycles', () => {
        const area = window.localStorage;
        for (let cycle = 0; cycle < 3; cycle++) {
            const { unmount } = renderHook(() => useExperimentMemoryStorageSync(), { wrapper: StrictWrapper });
            dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, area);
            expect(hydrate).toHaveBeenCalledTimes(cycle + 1);
            unmount();
            dispatchStorage(EXPERIMENT_MEMORY_STORAGE_KEY, area);
            expect(hydrate).toHaveBeenCalledTimes(cycle + 1);
        }
    });

});
