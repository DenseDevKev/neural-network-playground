import { useCallback, useEffect, useRef, useState } from 'react';
import { EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../store/experimentMemoryStore.ts';
import { captureCurrentRun } from './captureCurrentRun.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';

/** Shared by the transport and History; App owns the production instance. */
export interface SaveCurrentRunController {
    readonly busy: boolean;
    readonly error: string | null;
    readonly disabledReason: string | null;
    readonly pending: boolean;
    readonly commands: {
        readonly save: (title?: string) => Promise<boolean>;
        readonly retry: () => Promise<boolean>;
        readonly discard: () => Promise<void>;
        readonly dismiss: () => void;
    };
}

function saveDisabledReason(busy: boolean): string | null {
    const memory = useExperimentMemoryStore.getState();
    if (busy) return 'A save operation is in progress.';
    if (memory.pendingSave) return 'Retry or discard the pending artifact before saving another run.';
    if (memory.hydrationStatus !== 'ready') return 'Saved runs are loading.';
    if (usePlaygroundStore.getState().access.status !== 'ready') return 'A compatible experiment is required.';
    return null;
}

export function useSaveCurrentRun(): SaveCurrentRunController {
    // Subscribe to display changes; command guards read current stores synchronously
    // so two surfaces cannot race between an event and the next React render.
    const hydrationStatus = useExperimentMemoryStore((state) => state.hydrationStatus);
    const pendingSave = useExperimentMemoryStore((state) => state.pendingSave);
    const persistenceError = useExperimentMemoryStore((state) => state.persistenceError);
    const accessStatus = usePlaygroundStore((state) => state.access.status);
    const [busy, setBusy] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);
    const busyRef = useRef(false);
    const mounted = useRef(true);
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; };
    }, []);

    const operate = useCallback(async (action: () => Promise<boolean>): Promise<boolean> => {
        if (busyRef.current) return false;
        busyRef.current = true;
        if (mounted.current) { setBusy(true); setActionError(null); }
        try {
            return await action();
        } catch (error) {
            if (mounted.current) setActionError(error instanceof Error ? error.message : 'Saving the current run failed.');
            return false;
        } finally {
            busyRef.current = false;
            if (mounted.current) setBusy(false);
        }
    }, []);

    const save = useCallback(async (title = '') => {
        const trimmed = title.trim();
        if (saveDisabledReason(busyRef.current)
            || Array.from(trimmed).length > EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS) return false;
        const timestamp = new Date().toISOString();
        return operate(async () => {
            const artifact = await captureCurrentRun(trimmed, timestamp);
            return useExperimentMemoryStore.getState().saveRecord(artifact);
        });
    }, [operate]);
    const retry = useCallback(() => operate(() => (
        // Never contact the worker, mint an ID, or read the current recipe here.
        useExperimentMemoryStore.getState().retryPersistence()
    )), [operate]);
    const discard = useCallback(async () => {
        await operate(async () => {
            await useExperimentMemoryStore.getState().discardPendingSave();
            return true;
        });
    }, [operate]);
    const dismiss = useCallback(() => {
        setActionError(null);
        useExperimentMemoryStore.getState().dismissPersistenceError();
    }, []);

    void hydrationStatus;
    void accessStatus;
    return {
        busy, pending: pendingSave !== null,
        error: actionError ?? persistenceError?.message ?? null,
        disabledReason: saveDisabledReason(busy),
        commands: { save, retry, discard, dismiss },
    };
}
