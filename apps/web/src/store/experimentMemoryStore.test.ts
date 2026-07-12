import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
    EXPERIMENT_MEMORY_ENVELOPE_KIND,
    EXPERIMENT_MEMORY_MAX_RECORDS,
} from '@nn-playground/shared';
import type { ExperimentRunRecordV2 } from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
    createExperimentMemoryStore,
} from './experimentMemoryStore.ts';

const IDS = Array.from({ length: 22 }, (_, index) => (
    `00000000-0000-0000-0000-${(index + 1).toString(16).padStart(12, '0')}`
));

async function makeRecord(
    index = 0,
    overrides: Partial<ExperimentRunRecordV2> = {},
): Promise<ExperimentRunRecordV2> {
    const fixtures = await createScientificTrustFixtures();
    const evaluation = {
        ...fixtures.evaluation,
        trigger: 'save' as const,
    };
    return {
        kind: 'nn-playground-run',
        schemaVersion: 2,
        id: IDS[index],
        createdAt: '2026-07-11T12:00:00.000Z',
        updatedAt: '2026-07-11T12:00:00.000Z',
        title: `Run ${index + 1}`,
        recipe: fixtures.prepared.document.recipe,
        recipeFingerprint: fixtures.prepared.identities.recipeFingerprint,
        snapshot: {
            model: evaluation.model,
            evaluation,
            trendHistory: [],
            evaluationHistory: [evaluation],
        },
        ...overrides,
    };
}

describe('version-2 experiment memory store', () => {
    beforeEach(() => {
        window.localStorage.clear();
        vi.restoreAllMocks();
    });

    it('hydrates valid siblings and rejected raw records without touching legacy bytes', async () => {
        const valid = await makeRecord();
        const rejected = { schemaVersion: 1, id: 'legacy-record' };
        const legacyRaw = '{"schemaVersion":1,"records":[{"id":"old"}]}';
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [valid, rejected],
        }));
        window.localStorage.setItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY, legacyRaw);

        const store = createExperimentMemoryStore();
        await store.getState().hydrate();

        expect(store.getState().records.map((record) => record.id)).toEqual([valid.id]);
        expect(store.getState().rejectedRecords).toHaveLength(1);
        expect(store.getState().rejectedRecords[0]?.rawJson).toBe(JSON.stringify(rejected));
        expect(store.getState().legacyRaw).toBe(legacyRaw);
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(legacyRaw);
    });

    it('serializes overlapping async saves without losing a record', async () => {
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();
        const first = await makeRecord(0);
        const second = await makeRecord(1);

        const [firstSaved, secondSaved] = await Promise.all([
            store.getState().saveRecord(first),
            store.getState().saveRecord(second),
        ]);

        expect(firstSaved).toBe(true);
        expect(secondSaved).toBe(true);
        expect(store.getState().records.map((record) => record.id)).toEqual([second.id, first.id]);
        const persisted = JSON.parse(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '{}');
        expect(persisted.records.map((record: { id: string }) => record.id)).toEqual([second.id, first.id]);
    });

    it('rejects a 21st record without silent eviction or storage mutation', async () => {
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();
        for (let index = 0; index < EXPERIMENT_MEMORY_MAX_RECORDS; index++) {
            expect(await store.getState().saveRecord(await makeRecord(index))).toBe(true);
        }
        const beforeIds = store.getState().records.map((record) => record.id);
        const beforeBytes = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);

        expect(await store.getState().saveRecord(await makeRecord(EXPERIMENT_MEMORY_MAX_RECORDS))).toBe(false);

        expect(store.getState().records.map((record) => record.id)).toEqual(beforeIds);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(beforeBytes);
        expect(store.getState().persistenceError).toMatchObject({ code: 'resource-limit' });
        expect(store.getState().pendingSave?.id).toBe(IDS[EXPERIMENT_MEMORY_MAX_RECORDS]);
    });

    it('keeps quota errors until retry and reuses the exact captured record', async () => {
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();
        const record = await makeRecord();
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        expect(await store.getState().saveRecord(record)).toBe(false);
        expect(store.getState().records).toEqual([]);
        expect(store.getState().persistenceError?.message).toMatch(/quota/i);
        expect(store.getState().pendingSave).toBe(record);

        setItem.mockRestore();
        expect(await store.getState().retryPersistence()).toBe(true);
        expect(store.getState().records[0]).toEqual(record);
        expect(store.getState().pendingSave).toBeNull();
        expect(store.getState().persistenceError).toBeNull();
    });

    it('does not clear a failed save or its error during later hydration', async () => {
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();
        const record = await makeRecord();
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });
        expect(await store.getState().saveRecord(record)).toBe(false);
        setItem.mockRestore();
        const error = store.getState().persistenceError;

        await store.getState().hydrate();

        expect(store.getState().pendingSave).toBe(record);
        expect(store.getState().persistenceError).toBe(error);
    });

    it('preserves rejected siblings until explicit deletion', async () => {
        const valid = await makeRecord();
        const rejected = { kind: 'nn-playground-run', schemaVersion: 1, id: 'old' };
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [valid, rejected],
        }));
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();
        await store.getState().saveRecord(await makeRecord(1));
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toContain('"schemaVersion":1');
        expect(store.getState().rejectedRecords).toHaveLength(1);

        const sourceIndex = store.getState().rejectedRecords[0]!.sourceIndex;
        expect(await store.getState().deleteRejectedRecord(sourceIndex)).toBe(true);
        expect(store.getState().rejectedRecords).toEqual([]);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toContain('"schemaVersion":1');
        expect(store.getState().records).toHaveLength(2);
    });

    it('dismisses the legacy notice without deleting bytes and deletes only explicitly', async () => {
        const legacyRaw = '{"schemaVersion":1,"records":[]}';
        window.localStorage.setItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY, legacyRaw);
        const store = createExperimentMemoryStore();
        await store.getState().hydrate();

        store.getState().dismissLegacyNotice();
        expect(store.getState().legacyNoticeDismissed).toBe(true);
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(legacyRaw);
        await store.getState().hydrate();
        expect(store.getState().legacyNoticeDismissed).toBe(true);

        expect(await store.getState().deleteLegacyStorage()).toBe(true);
        expect(store.getState().legacyRaw).toBeNull();
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBeNull();
    });
});
