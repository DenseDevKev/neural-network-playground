import { StrictMode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import type { ExperimentRunRecordV2, PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { useSaveCurrentRun } from './useSaveCurrentRun.ts';
import { useExperimentMemoryStore, EXPERIMENT_MEMORY_STORAGE_KEY } from '../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { currentPreparedForTest } from '../test/playgroundStoreTestUtils.ts';

const workerApi = vi.hoisted(() => ({ captureRunArtifact: vi.fn() }));
vi.mock('../worker/workerBridge.ts', () => ({ getWorkerApi: async () => workerApi }));

function makeRecord(
    prepared: PreparedExperimentDocumentV2,
    id: string = '00000000-0000-0000-0000-000000000001',
    title = 'Saved evidence',
    testDataLoss = 0.5,
): ExperimentRunRecordV2 {
    const sampleCount = prepared.document.recipe.data.sampleCount;
    const trainCount = Math.floor(sampleCount * prepared.document.recipe.data.trainFraction);
    const testCount = sampleCount - trainCount;
    const model = { generationId: Number(id.at(-1)) || 1, revision: 4, step: 4, epoch: 0 };
    const dataset = {
        generatorVersion: 2,
        datasetKey: prepared.identities.datasetKey,
        trainCount,
        testCount,
    };
    const evaluation = {
        evaluationId: 2,
        trigger: 'save' as const,
        model,
        dataset,
        objectiveKey: prepared.identities.objectiveKey,
        train: {
            basis: { kind: 'full-split' as const, split: 'train' as const, sampleCount: trainCount, populationCount: trainCount },
            values: { dataLoss: 0.4 },
        },
        test: {
            basis: { kind: 'full-split' as const, split: 'test' as const, sampleCount: testCount, populationCount: testCount },
            values: { dataLoss: testDataLoss },
        },
        objective: { regularizationPenalty: 0, trainTotalObjective: 0.4 },
    };
    return {
        kind: 'nn-playground-run',
        schemaVersion: 2,
        id,
        createdAt: '2026-07-11T12:00:00.000Z',
        updatedAt: '2026-07-11T12:00:00.000Z',
        title,
        recipe: prepared.document.recipe,
        recipeFingerprint: prepared.identities.recipeFingerprint,
        snapshot: {
            model,
            evaluation,
            trendHistory: [],
            evaluationHistory: [evaluation],
        },
    };
}


describe('one shared current-run capture controller', () => {
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
        expect(await second).toBe(false);
        const metadata = workerApi.captureRunArtifact.mock.calls[0][0];
        expect(Object.keys(metadata).sort()).toEqual(['createdAt', 'id', 'title', 'updatedAt']);
        expect(metadata.title).toBe('Shared capture');
        await act(async () => {
            resolve(makeRecord(currentPreparedForTest()!, metadata.id, metadata.title));
            expect(await first).toBe(true);
        });
        expect(result.current.busy).toBe(false);
        expect(result.current.disabledReason).toBeNull();
    });

    it('retries the byte-equivalent pending artifact after a recipe change without recapture', async () => {
        workerApi.captureRunArtifact.mockImplementation(async ({ id }: { id: string }) => makeRecord(currentPreparedForTest()!, id));
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

    it('checks pending storage synchronously even before React delivers the next render', async () => {
        const { result } = renderHook(() => useSaveCurrentRun());
        await act(async () => {
            useExperimentMemoryStore.setState({ pendingSave: makeRecord(currentPreparedForTest()!) });
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
});
