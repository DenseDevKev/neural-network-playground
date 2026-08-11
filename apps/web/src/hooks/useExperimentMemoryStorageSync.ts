import { useEffect } from 'react';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
    useExperimentMemoryStore,
} from '../store/experimentMemoryStore.ts';

const MEMORY_KEYS = new Set([
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
]);

export function useExperimentMemoryStorageSync(): void {
    useEffect(() => {
        let localStorageArea: Storage;
        try {
            localStorageArea = window.localStorage;
        } catch {
            return;
        }

        const onStorage = (event: StorageEvent) => {
            if (event.storageArea !== localStorageArea) return;
            if (event.key !== null && !MEMORY_KEYS.has(event.key)) return;
            void useExperimentMemoryStore.getState().hydrate();
        };

        window.addEventListener('storage', onStorage);
        return () => window.removeEventListener('storage', onStorage);
    }, []);
}
