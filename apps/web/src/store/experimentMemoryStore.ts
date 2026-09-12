import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import {
    parseExperimentMemoryEnvelopeV2,
    serializeExperimentMemoryEnvelopeV2,
    validateExperimentRunRecordV2,
} from '@nn-playground/shared';
import type {
    IncompatibleExperimentMemoryEnvelopeV2,
    ExperimentMemoryIssue,
    ExperimentRunRecordV2,
    RejectedExperimentRunRecordV2,
} from '@nn-playground/shared';

export const EXPERIMENT_MEMORY_STORAGE_KEY = 'nn-playground-experiment-memory-v2';
export const LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY = 'nn-playground-experiment-memory';

export interface ExperimentMemoryStore {
    hydrationStatus: 'loading' | 'ready';
    records: readonly ExperimentRunRecordV2[];
    rejectedRecords: readonly RejectedExperimentRunRecordV2[];
    incompatibleEnvelope: IncompatibleExperimentMemoryEnvelopeV2 | null;
    legacyRaw: string | null;
    legacyNoticeDismissed: boolean;
    persistenceError: ExperimentMemoryIssue | null;
    /** Exact worker-authored artifact retained for retry; never recaptured. */
    pendingSave: ExperimentRunRecordV2 | null;
    hydrate: () => Promise<void>;
    saveRecord: (record: ExperimentRunRecordV2) => Promise<boolean>;
    retryPersistence: () => Promise<boolean>;
    dismissPersistenceError: () => void;
    discardPendingSave: () => Promise<void>;
    renameRecord: (id: string, title: string, now?: () => Date) => Promise<boolean>;
    removeRecord: (id: string) => Promise<boolean>;
    clearRecords: () => Promise<boolean>;
    deleteRejectedRecord: (sourceIndex: number) => Promise<boolean>;
    deleteIncompatibleEnvelope: () => Promise<boolean>;
    dismissLegacyNotice: () => void;
    deleteLegacyStorage: () => Promise<boolean>;
}

function storageIssue(error: unknown): ExperimentMemoryIssue {
    const quota = error instanceof DOMException && error.name === 'QuotaExceededError';
    return Object.freeze({
        code: quota ? 'resource-limit' : 'invalid-field',
        path: '$',
        message: quota
            ? 'Storage quota was exceeded. Delete saved records or free browser storage, then retry.'
            : `Saving experiment memory failed: ${error instanceof Error ? error.message : String(error)}`,
    });
}

function firstIssue(
    issues: readonly ExperimentMemoryIssue[],
    fallback: string,
): ExperimentMemoryIssue {
    return issues[0] ?? Object.freeze({
        code: 'invalid-field' as const,
        path: '$',
        message: fallback,
    });
}

function incompatibleEnvelopeWriteIssue(): ExperimentMemoryIssue {
    return Object.freeze({
        code: 'unsupported-version',
        path: '$',
        message: 'Saving is blocked by incompatible saved-run data. Download and delete the incompatible saved-run file, then retry saving.',
    });
}

export function createExperimentMemoryStore() {
    let queue: Promise<void> = Promise.resolve();

    function enqueue<T>(operation: () => Promise<T>): Promise<T> {
        const pending = queue.then(operation, operation);
        queue = pending.then(() => undefined, () => undefined);
        return pending;
    }

    const store = createStore<ExperimentMemoryStore>((set, get) => {
        let persistedRaw: string | null = null;
        const readStorage = async (): Promise<boolean> => {
            const previous = get();
            let legacyRaw: string | null = null;
            let raw: string | null = null;
            try {
                legacyRaw = window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY);
                raw = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);
                persistedRaw = raw;
            } catch (error) {
                set({
                    hydrationStatus: 'ready',
                    records: previous.records,
                    rejectedRecords: previous.rejectedRecords,
                    incompatibleEnvelope: previous.incompatibleEnvelope,
                    legacyRaw,
                    legacyNoticeDismissed: previous.legacyNoticeDismissed
                        && previous.legacyRaw === legacyRaw,
                    persistenceError: previous.persistenceError ?? storageIssue(error),
                    pendingSave: previous.pendingSave,
                });
                return false;
            }

            if (raw === null) {
                set({
                    hydrationStatus: 'ready',
                    records: Object.freeze([]),
                    rejectedRecords: Object.freeze([]),
                    incompatibleEnvelope: null,
                    legacyRaw,
                    legacyNoticeDismissed: previous.legacyNoticeDismissed
                        && previous.legacyRaw === legacyRaw,
                    persistenceError: previous.persistenceError,
                    pendingSave: previous.pendingSave,
                });
                return true;
            }

            const parsed = await parseExperimentMemoryEnvelopeV2(raw);
            set({
                hydrationStatus: 'ready',
                records: parsed.records,
                rejectedRecords: parsed.rejectedRecords,
                incompatibleEnvelope: parsed.incompatibleEnvelope,
                legacyRaw,
                legacyNoticeDismissed: previous.legacyNoticeDismissed
                    && previous.legacyRaw === legacyRaw,
                persistenceError: previous.persistenceError ?? (
                    parsed.envelopeIssues.length > 0
                        ? firstIssue(parsed.envelopeIssues, 'Saved experiment memory is incompatible.')
                        : null
                ),
                pendingSave: previous.pendingSave,
            });
            return true;
        };

        // All tabs use the same origin-scoped lock. Re-read after acquiring it:
        // storage events refresh views, but cannot coordinate read/modify/write.
        const mutate = async (
            operation: () => Promise<boolean>,
            pendingSaveOnFailure: ExperimentRunRecordV2 | null = get().pendingSave,
        ): Promise<boolean> => {
            try {
                if (!navigator.locks) {
                    throw new Error('Safe saving requires Web Locks support. Open this app over HTTPS or localhost in a current browser.');
                }
                return await navigator.locks.request(EXPERIMENT_MEMORY_STORAGE_KEY, async () => {
                    if (!await readStorage()) {
                        set({ pendingSave: pendingSaveOnFailure });
                        return false;
                    }
                    return operation();
                });
            } catch (error) {
                set({ persistenceError: storageIssue(error), pendingSave: pendingSaveOnFailure });
                return false;
            }
        };

        const persistCandidate = async (
            records: readonly ExperimentRunRecordV2[],
            rejectedRecords: readonly RejectedExperimentRunRecordV2[],
            pendingSaveOnFailure: ExperimentRunRecordV2 | null,
            pendingSaveOnSuccess: ExperimentRunRecordV2 | null,
        ): Promise<boolean> => {
            if (get().incompatibleEnvelope !== null) {
                set({
                    persistenceError: incompatibleEnvelopeWriteIssue(),
                    pendingSave: pendingSaveOnFailure,
                });
                return false;
            }
            const serialized = await serializeExperimentMemoryEnvelopeV2(
                records,
                rejectedRecords.map((entry) => entry.rawJson),
            );
            if (!serialized.ok) {
                set({
                    persistenceError: firstIssue(serialized.issues, 'Experiment memory validation failed.'),
                    pendingSave: pendingSaveOnFailure,
                });
                return false;
            }

            const verified = await parseExperimentMemoryEnvelopeV2(serialized.value);
            if (verified.records.length !== records.length || verified.rejectedRecords.length !== rejectedRecords.length) {
                set({
                    persistenceError: firstIssue([], 'Delete the affected rejected records first. This change would otherwise restore rejected saved-run data.'),
                    pendingSave: pendingSaveOnFailure,
                });
                return false;
            }
            if (verified.envelopeIssues.length > 0) {
                set({
                    persistenceError: firstIssue(
                        verified.envelopeIssues,
                        'Serialized experiment memory could not be verified.',
                    ),
                    pendingSave: pendingSaveOnFailure,
                });
                return false;
            }

            try {
                // Detect an older/uncoordinated client changing bytes while validation awaited.
                if (window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) !== persistedRaw) {
                    throw new Error('Saved-run storage changed during saving. Retry with the latest saved runs.');
                }
                window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, serialized.value);
            } catch (error) {
                set({ persistenceError: storageIssue(error), pendingSave: pendingSaveOnFailure });
                return false;
            }

            set((previous) => ({
                records: verified.records,
                rejectedRecords: verified.rejectedRecords,
                incompatibleEnvelope: null,
                persistenceError: pendingSaveOnSuccess === null
                    ? null
                    : previous.persistenceError,
                pendingSave: pendingSaveOnSuccess,
            }));
            return true;
        };

        return {
            hydrationStatus: 'loading',
            records: Object.freeze([]),
            rejectedRecords: Object.freeze([]),
            incompatibleEnvelope: null,
            legacyRaw: null,
            legacyNoticeDismissed: false,
            persistenceError: null,
            pendingSave: null,

            hydrate: () => enqueue(async () => {
                try {
                    if (navigator.locks) await navigator.locks.request(EXPERIMENT_MEMORY_STORAGE_KEY, readStorage);
                    else await readStorage();
                } catch (error) {
                    set({ hydrationStatus: 'ready', persistenceError: storageIssue(error) });
                }
            }),

            saveRecord: (record) => {
                if (get().pendingSave !== null) {
                    return Promise.resolve(false);
                }
                // Validation takes its defensive JSON snapshot synchronously,
                // before this call returns control to caller-owned code.
                const validation = validateExperimentRunRecordV2(record);
                return enqueue(async () => {
                    const validated = await validation;
                    if (!validated.ok) {
                        set({
                            persistenceError: firstIssue(
                                validated.issues,
                                'Experiment memory validation failed.',
                            ),
                        });
                        return false;
                    }
                    const artifact = validated.value;
                    if (get().pendingSave !== null) return false;
                    return mutate(async () => {
                        const state = get();
                        if (state.pendingSave !== null) return false;
                        const candidate = [
                            artifact,
                            ...state.records.filter((existing) => existing.id !== artifact.id),
                        ];
                        return persistCandidate(
                            candidate,
                            state.rejectedRecords,
                            artifact,
                            null,
                        );
                    }, artifact);
                });
            },

            retryPersistence: () => enqueue(() => mutate(async () => {
                const state = get();
                if (!state.pendingSave) return false;
                const candidate = [
                    state.pendingSave,
                    ...state.records.filter((existing) => existing.id !== state.pendingSave?.id),
                ];
                return persistCandidate(
                    candidate,
                    state.rejectedRecords,
                    state.pendingSave,
                    null,
                );
            })),

            dismissPersistenceError: () => set({ persistenceError: null }),

            discardPendingSave: () => enqueue(async () => {
                set({ pendingSave: null, persistenceError: null });
            }),

            renameRecord: (id, title, now = () => new Date()) => enqueue(() => mutate(async () => {
                const state = get();
                const trimmed = title.trim();
                const record = state.records.find((entry) => entry.id === id);
                if (!record) return false;
                const renamed: ExperimentRunRecordV2 = {
                    ...record,
                    updatedAt: now().toISOString(),
                    ...(trimmed ? { title: trimmed } : {}),
                };
                if (!trimmed) delete (renamed as { title?: string }).title;
                const candidate = state.records.map((entry) => entry.id === id ? renamed : entry);
                return persistCandidate(
                    candidate,
                    state.rejectedRecords,
                    state.pendingSave,
                    state.pendingSave,
                );
            })),

            removeRecord: (id) => enqueue(() => mutate(async () => {
                const state = get();
                return persistCandidate(
                    state.records.filter((record) => record.id !== id),
                    state.rejectedRecords,
                    state.pendingSave,
                    state.pendingSave,
                );
            })),

            clearRecords: () => enqueue(() => mutate(async () => {
                const state = get();
                return persistCandidate(
                    [],
                    state.rejectedRecords,
                    state.pendingSave,
                    state.pendingSave,
                );
            })),

            deleteRejectedRecord: (sourceIndex) => {
                const target = get().rejectedRecords.find((record) => record.sourceIndex === sourceIndex);
                return enqueue(() => mutate(async () => {
                    const state = get();
                    const index = state.rejectedRecords.findIndex((record) => record.rawJson === target?.rawJson);
                    const candidate = state.rejectedRecords.filter((_, position) => position !== index);
                    if (candidate.length === state.rejectedRecords.length) return false;
                    return persistCandidate(
                        state.records,
                        candidate,
                        state.pendingSave,
                        state.pendingSave,
                    );
                }));
            },

            deleteIncompatibleEnvelope: () => {
                const target = get().incompatibleEnvelope?.rawJson;
                return enqueue(() => mutate(async () => {
                    const state = get();
                    if (state.incompatibleEnvelope === null) return false;
                    try {
                        const current = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);
                        if (current !== target) {
                            set({
                                persistenceError: Object.freeze({
                                    code: 'invalid-field',
                                    path: '$',
                                    message: 'Saved-run storage changed before deletion. Reload before trying again.',
                                }),
                            });
                            return false;
                        }
                        window.localStorage.removeItem(EXPERIMENT_MEMORY_STORAGE_KEY);
                    } catch (error) {
                        set({ persistenceError: storageIssue(error) });
                        return false;
                    }
                    set({
                        records: Object.freeze([]),
                        rejectedRecords: Object.freeze([]),
                        incompatibleEnvelope: null,
                        persistenceError: null,
                    });
                    return true;
                }));
            },

            dismissLegacyNotice: () => set({ legacyNoticeDismissed: true }),

            deleteLegacyStorage: () => {
                const target = get().legacyRaw;
                return enqueue(() => mutate(async () => {
                    if (target === null) return false;
                    if (window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY) !== target) {
                        throw new Error('Earlier saved-run storage changed before deletion. Review the latest file before trying again.');
                    }
                    window.localStorage.removeItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY);
                    set({ legacyRaw: null, legacyNoticeDismissed: false });
                    return true;
                }));
            },
        };
    });

    void store.getState().hydrate();
    return store;
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
