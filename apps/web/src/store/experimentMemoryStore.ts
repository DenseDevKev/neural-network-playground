import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import {
    createExperimentMemoryEnvelope,
    normalizeExperimentMemoryEnvelope,
} from '@nn-playground/shared';

export const EXPERIMENT_MEMORY_STORAGE_KEY = 'nn-playground-experiment-memory';

interface ExperimentMemoryStore {
    records: ExperimentRunRecordV1[];
    saveRecord: (record: ExperimentRunRecordV1) => void;
    removeRecord: (id: string) => void;
    clearRecords: () => void;
}

function loadRecords(): ExperimentRunRecordV1[] {
    try {
        const raw = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);
        if (!raw) return [];
        return normalizeExperimentMemoryEnvelope(JSON.parse(raw)).records;
    } catch {
        return [];
    }
}

function persistRecords(
    records: ExperimentRunRecordV1[],
    fallback: ExperimentRunRecordV1[],
): ExperimentRunRecordV1[] {
    const envelope = createExperimentMemoryEnvelope(records);
    try {
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify(envelope));
        return envelope.records;
    } catch {
        return fallback;
    }
}

export function createExperimentMemoryStore() {
    return createStore<ExperimentMemoryStore>((set) => ({
        records: loadRecords(),
        saveRecord: (record) => set((state) => {
            const next = [record, ...state.records.filter((existing) => existing.id !== record.id)];
            return { records: persistRecords(next, state.records) };
        }),
        removeRecord: (id) => set((state) => ({
            records: persistRecords(state.records.filter((record) => record.id !== id), state.records),
        })),
        clearRecords: () => set((state) => ({ records: persistRecords([], state.records) })),
    }));
}

const experimentMemoryStore = createExperimentMemoryStore();

type Selector<T> = (state: ExperimentMemoryStore) => T;
type ExperimentMemoryStoreHook = {
    (): ExperimentMemoryStore;
    <T>(selector: Selector<T>): T;
} & typeof experimentMemoryStore;

const boundUseExperimentMemoryStore = ((selector?: Selector<unknown>) => (
    selector ? useStore(experimentMemoryStore, selector) : useStore(experimentMemoryStore)
)) as ExperimentMemoryStoreHook;

export const useExperimentMemoryStore = Object.assign(boundUseExperimentMemoryStore, experimentMemoryStore);
