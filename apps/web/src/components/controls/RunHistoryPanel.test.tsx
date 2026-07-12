import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
    EXPERIMENT_MEMORY_ENVELOPE_KIND,
    PREPARED_PRESETS,
} from '@nn-playground/shared';
import type {
    ExperimentRunRecordV2,
    PreparedExperimentDocumentV2,
} from '@nn-playground/shared';
import { RunHistoryPanel } from './RunHistoryPanel.tsx';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
    useExperimentMemoryStore,
} from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';

const workerApi = vi.hoisted(() => ({
    captureRunArtifact: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: () => workerApi,
}));

const IDS = [
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000003',
] as const;

function preset(id: (typeof PREPARED_PRESETS)[number]['id']) {
    const entry = PREPARED_PRESETS.find((candidate) => candidate.id === id);
    if (!entry) throw new Error(`Missing preset ${id}`);
    return entry.prepared;
}

function makeRecord(
    prepared: PreparedExperimentDocumentV2,
    id = IDS[0],
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

async function hydrateSingleton(): Promise<void> {
    await act(async () => {
        await useExperimentMemoryStore.getState().hydrate();
    });
}

describe('RunHistoryPanel V2 evidence memory', () => {
    beforeEach(async () => {
        window.localStorage.clear();
        workerApi.captureRunArtifact.mockReset();
        vi.restoreAllMocks();
        await hydrateSingleton();
    });

    it('asks the worker to author the record from metadata only and persists its artifact', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: {
            id: string;
            createdAt: string;
            updatedAt: string;
        }) => makeRecord(prepared, metadata.id, 'Captured run'));

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        const request = workerApi.captureRunArtifact.mock.calls[0][0];
        expect(request).toEqual({
            id: expect.stringMatching(/^[0-9a-f-]{36}$/u),
            createdAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/u),
            updatedAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/u),
        });
        expect(Object.keys(request).sort()).toEqual(['createdAt', 'id', 'updatedAt']);
        await waitFor(() => expect(screen.getByText('Captured run')).toBeInTheDocument());
        const persisted = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '';
        expect(persisted).toContain('"kind":"nn-playground-run"');
        expect(persisted).not.toMatch(/weights|biases|parameters/i);
    });

    it('retries persistence with the captured record instead of recapturing scientific state', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeRecord(prepared, metadata.id, 'Retry me')
        ));
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await screen.findByText(/quota/i);
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);

        setItem.mockRestore();
        await userEvent.click(screen.getByRole('button', { name: 'Retry saving' }));

        await waitFor(() => expect(screen.getByText('Retry me')).toBeInTheDocument());
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(useExperimentMemoryStore.getState().persistenceError).toBeNull();
    });

    it('blocks recapture and keeps retry or discard available after error dismissal', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeRecord(prepared, metadata.id, 'Pending exact artifact')
        ));
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await screen.findByText(/quota/i);
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeDisabled();

        await userEvent.click(screen.getByRole('button', { name: 'Dismiss error' }));
        expect(screen.queryByText(/quota/i)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Retry saving' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Discard pending save' })).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);

        setItem.mockRestore();
        await userEvent.click(screen.getByRole('button', { name: 'Discard pending save' }));
        expect(screen.queryByRole('button', { name: 'Retry saving' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeEnabled();
    });

    it('applies a saved recipe through a fresh version-2 document', async () => {
        const prepared = preset('xor-hidden');
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeRecord(prepared));
        });
        const replaceDocument = vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument');

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Apply saved recipe' }));

        await waitFor(() => expect(replaceDocument).toHaveBeenCalledWith(expect.objectContaining({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: prepared.document.recipe,
        })));
        expect(screen.queryByRole('button', { name: /restore/i })).not.toBeInTheDocument();
    });

    it('shows no numeric winner when dataset or objective identity differs', async () => {
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeRecord(
                preset('circle-one-layer'),
                IDS[0],
                'Circle',
                0.4,
            ));
            await useExperimentMemoryStore.getState().saveRecord(makeRecord(
                preset('regression-plane'),
                IDS[1],
                'Regression',
                0.2,
            ));
        });

        render(<RunHistoryPanel />);

        const comparison = screen.getByRole('group', { name: 'Saved run comparison' });
        expect(comparison).toHaveTextContent('Not directly comparable');
        expect(comparison).not.toHaveTextContent(/winner|lower by|better/i);
    });

    it('computes a numeric winner only for equal dataset and objective identities', async () => {
        const prepared = preset('circle-one-layer');
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeRecord(prepared, IDS[0], 'Baseline', 0.6));
            await useExperimentMemoryStore.getState().saveRecord(makeRecord(prepared, IDS[1], 'Tuned', 0.4));
        });

        render(<RunHistoryPanel />);

        expect(screen.getByRole('group', { name: 'Saved run comparison' }))
            .toHaveTextContent('Tuned has lower test data loss by 0.2000');
    });

    it('preserves legacy bytes through dismissal and deletes only explicitly', async () => {
        const raw = '{"schemaVersion":1,"records":[{"id":"old"}]}';
        window.localStorage.setItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY, raw);
        await hydrateSingleton();

        render(<RunHistoryPanel />);
        expect(screen.getByRole('note', { name: 'Earlier saved runs' })).toHaveTextContent(/incompatible/i);
        expect(screen.getByRole('button', { name: 'Download earlier runs' })).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Dismiss earlier runs notice' }));
        expect(screen.queryByRole('note', { name: 'Earlier saved runs' })).not.toBeInTheDocument();
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(raw);

        await act(async () => {
            await useExperimentMemoryStore.getState().deleteLegacyStorage();
        });
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBeNull();
    });

    it('offers raw download and explicit deletion for a rejected record', async () => {
        const valid = makeRecord(preset('circle-one-layer'));
        const rejected = { schemaVersion: 1, id: 'rejected' };
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [valid, rejected],
        }));
        await hydrateSingleton();

        render(<RunHistoryPanel />);
        expect(screen.getByText('Rejected saved record')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Download rejected record' })).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Delete rejected record' }));
        await waitFor(() => expect(screen.queryByText('Rejected saved record')).not.toBeInTheDocument());
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toContain('"schemaVersion":1');
    });
});
