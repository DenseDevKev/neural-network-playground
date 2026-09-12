import { makeSavedRunRecord } from '../test/savedRunFixtures.ts';
import { StrictMode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import type { ExperimentRunRecordV2 } from '@nn-playground/shared';
import { useSaveCurrentRun } from './useSaveCurrentRun.ts';
import { useExperimentMemoryStore, EXPERIMENT_MEMORY_STORAGE_KEY } from '../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { currentPreparedForTest } from '../test/playgroundStoreTestUtils.ts';

const workerApi = vi.hoisted(() => ({ captureRunArtifact: vi.fn() }));
vi.mock('../worker/workerBridge.ts', () => ({ getWorkerApi: async () => workerApi }));



describe('one shared current-run capture controller', () => {
    afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

    beforeEach(async () => {
        vi.restoreAllMocks();
        workerApi.captureRunArtifact.mockReset();
        window.localStorage.clear();
        await act(async () => {
            await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
            await useExperimentMemoryStore.getState().hydrate();
        });
    });

    it('synchronously serializes transport and history commands and survives StrictMode', async () => {
        let resolve!: (record: ExperimentRunRecordV2) => void;
        workerApi.captureRunArtifact.mockImplementation(() => new Promise<ExperimentRunRecordV2>((done) => { resolve = done; }));
        const { result } = renderHook(() => useSaveCurrentRun(), { wrapper: StrictMode });
        let first!: Promise<boolean>;
        let second!: Promise<boolean>;
        act(() => {
            first = result.current.commands.save('  Shared capture  ');
            second = result.current.commands.save('Must not capture twice');
        });
        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        expect(result.current.busy).toBe(true);
        expect(result.current.disabledReason).toBe('A save operation is in progress.');
        expect(await second).toBe(false);
        const metadata = workerApi.captureRunArtifact.mock.calls[0][0];
        expect(Object.keys(metadata).sort()).toEqual(['createdAt', 'id', 'title', 'updatedAt']);
        expect(metadata.title).toBe('Shared capture');
        await act(async () => {
            resolve(makeSavedRunRecord(currentPreparedForTest()!, metadata.id, metadata.title));
            expect(await first).toBe(true);
        });
        expect(result.current.busy).toBe(false);
        expect(result.current.disabledReason).toBeNull();
    });

    it('retries the byte-equivalent pending artifact after a recipe change without recapture', async () => {
        workerApi.captureRunArtifact.mockImplementation(async ({ id }: { id: string }) => makeSavedRunRecord(currentPreparedForTest()!, id));
        const write = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
        const { result } = renderHook(() => useSaveCurrentRun());
        await act(async () => { expect(await result.current.commands.save()).toBe(false); });
        const pending = useExperimentMemoryStore.getState().pendingSave;
        expect(pending).not.toBeNull();
        const bytes = JSON.stringify(pending);
        expect(result.current.pending).toBe(true);
        await act(async () => {
            await usePlaygroundStore.getState().replaceDocument(PREPARED_PRESETS.at(-1)!.prepared.document);
            expect(await result.current.commands.save('New live state must not replace pending')).toBe(false);
            expect(await result.current.commands.retry()).toBe(false);
        });
        expect(useExperimentMemoryStore.getState().pendingSave).toBe(pending);
        expect(JSON.stringify(useExperimentMemoryStore.getState().pendingSave)).toBe(bytes);
        write.mockRestore();
        await act(async () => { expect(await result.current.commands.retry()).toBe(true); });
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(JSON.stringify(useExperimentMemoryStore.getState().records[0])).toBe(bytes);
        expect(localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toContain(bytes);
        expect(result.current.pending).toBe(false);
    });

    it('downloads byte-identical pending evidence after live recipe changes and retains it for retry', async () => {
        const pending = makeSavedRunRecord(currentPreparedForTest()!);
        const bytes = JSON.stringify(pending);
        const create = vi.fn((_blob: Blob) => 'blob:pending-evidence');
        const revoke = vi.fn();
        vi.stubGlobal('URL', class extends URL { static createObjectURL = create; static revokeObjectURL = revoke; });
        const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
        const { result } = renderHook(() => useSaveCurrentRun());
        await act(async () => {
            useExperimentMemoryStore.setState({ pendingSave: pending });
            await usePlaygroundStore.getState().replaceDocument(PREPARED_PRESETS.at(-1)!.prepared.document);
            expect(await result.current.commands.downloadPending()).toBe(true);
        });
        const blob = create.mock.calls[0][0] as Blob;
        const downloaded = await new Promise<string>((resolve) => {
            const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob);
        });
        expect(downloaded).toBe(bytes);
        expect(click).toHaveBeenCalledOnce();
        expect(revoke).toHaveBeenCalledWith('blob:pending-evidence');
        expect(useExperimentMemoryStore.getState().pendingSave).toBe(pending);
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
        expect(result.current.disabledReason).toContain('pending artifact');
        vi.unstubAllGlobals();
    });

    it('checks pending storage synchronously even before React delivers the next render', async () => {
        const { result } = renderHook(() => useSaveCurrentRun());
        await act(async () => {
            useExperimentMemoryStore.setState({ pendingSave: makeSavedRunRecord(currentPreparedForTest()!) });
            expect(await result.current.commands.save()).toBe(false);
        });
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
        await act(async () => { await result.current.commands.discard(); });
        expect(result.current.pending).toBe(false);
    });

    it('blocks loading and oversized metadata without contacting the worker', async () => {
        const { result } = renderHook(() => useSaveCurrentRun());
        await act(async () => {
            useExperimentMemoryStore.setState({ hydrationStatus: 'loading' });
            expect(await result.current.commands.save()).toBe(false);
            useExperimentMemoryStore.setState({ hydrationStatus: 'ready' });
            expect(await result.current.commands.save('a'.repeat(121))).toBe(false);
        });
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
    });

    it('exposes rejected captures without locking out a fresh save', async () => {
        workerApi.captureRunArtifact.mockRejectedValue(new Error('capture refused'));
        const { result } = renderHook(() => useSaveCurrentRun(), { wrapper: StrictMode });
        await act(async () => { expect(await result.current.commands.save()).toBe(false); });
        expect(result.current.error).toBe('capture refused');
        expect(result.current.busy).toBe(false);
        expect(result.current.disabledReason).toBeNull();
        act(() => result.current.commands.dismiss());
        expect(result.current.error).toBeNull();
    });
    it.each([
        [true, 'loading', 'Retry or discard the pending artifact before saving another run.'],
        [false, 'loading', 'Saved runs are loading.'],
        [false, 'ready', 'A compatible experiment is required.'],
    ] as const)('prioritizes pending=%s hydration=%s above incompatible access', (pending, hydrationStatus, expected) => {
        const prepared = currentPreparedForTest()!;
        const { result } = renderHook(() => useSaveCurrentRun());
        act(() => {
            useExperimentMemoryStore.setState({ pendingSave: pending ? makeSavedRunRecord(prepared) : null, hydrationStatus });
            usePlaygroundStore.setState({ access: { status: 'incompatible', prepared: null, source: { kind: 'url', rawHash: '#invalid' }, issues: [] } });
        });
        expect(result.current.disabledReason).toBe(expected);
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
    });

    it('recomputes the disabled reason on access-only and hydration-only transitions', () => {
        const access = usePlaygroundStore.getState().access;
        const { result } = renderHook(() => useSaveCurrentRun());
        expect(result.current.disabledReason).toBeNull();
        act(() => usePlaygroundStore.setState({ access: { status: 'incompatible', prepared: null, source: { kind: 'url', rawHash: '#invalid' }, issues: [] } }));
        expect(result.current.disabledReason).toBe('A compatible experiment is required.');
        act(() => usePlaygroundStore.setState({ access }));
        expect(result.current.disabledReason).toBeNull();
        act(() => useExperimentMemoryStore.setState({ hydrationStatus: 'loading' }));
        expect(result.current.disabledReason).toBe('Saved runs are loading.');
        act(() => useExperimentMemoryStore.setState({ hydrationStatus: 'ready' }));
        expect(result.current.disabledReason).toBeNull();
    });

    it('finishes the originally requested artifact after unmount without a duplicate capture', async () => {
        const prepared = currentPreparedForTest()!;
        let complete!: (record: ExperimentRunRecordV2) => void;
        workerApi.captureRunArtifact.mockImplementation(() => new Promise<ExperimentRunRecordV2>((resolve) => { complete = resolve; }));
        const { result, unmount } = renderHook(() => useSaveCurrentRun(), { wrapper: StrictMode });
        const commands = result.current.commands;
        let operation!: Promise<boolean>;
        act(() => { operation = commands.save('Original request'); });
        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        const metadata = workerApi.captureRunArtifact.mock.calls[0][0];
        unmount();
        expect(await commands.save('Competing request')).toBe(false);
        const artifact = makeSavedRunRecord(prepared, metadata.id, metadata.title);
        await act(async () => { complete(artifact); expect(await operation).toBe(true); });
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(useExperimentMemoryStore.getState().records).toHaveLength(1);
        expect(useExperimentMemoryStore.getState().records[0]).toEqual(artifact);
    });

});
