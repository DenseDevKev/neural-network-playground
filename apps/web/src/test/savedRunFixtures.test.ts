import { describe, expect, it } from 'vitest';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { makeSavedRunRecord } from './savedRunFixtures.ts';

const prepared = PREPARED_PRESETS[0].prepared;

describe('saved-run test fixtures', () => {
    it('keeps deterministic metadata and explicit scientific evidence defaults', () => {
        const record = makeSavedRunRecord(prepared);
        expect(record.id).toBe('00000000-0000-0000-0000-000000000001');
        expect(record.createdAt).toBe('2026-07-11T12:00:00.000Z');
        expect(record.updatedAt).toBe(record.createdAt);
        expect(record.title).toBe('Saved evidence');
        expect(record.snapshot.model).toEqual({ generationId: 1, revision: 4, step: 4, epoch: 0 });
        expect(record.snapshot.evaluation?.train.values.dataLoss).toBe(0.4);
        expect(record.snapshot.evaluation?.test.values.dataLoss).toBe(0.5);
        expect(record.snapshot.evaluation?.trigger).toBe('save');
        expect(record.recipe).toEqual(prepared.document.recipe);
        expect(JSON.stringify(record)).toBe(JSON.stringify(makeSavedRunRecord(prepared)));
    });
    it('supports caller IDs, titles and test losses without changing train evidence', () => {
        const record = makeSavedRunRecord(prepared, '00000000-0000-0000-0000-000000000003', 'Comparison', 0.125);
        expect(record.title).toBe('Comparison');
        expect(record.snapshot.model.generationId).toBe(3);
        expect(record.snapshot.evaluation?.test.values.dataLoss).toBe(0.125);
        expect(record.snapshot.evaluation?.train.values.dataLoss).toBe(0.4);
    });
    it('does not share mutable nested fixtures between calls or with the prepared catalog', () => {
        const first = makeSavedRunRecord(prepared);
        const second = makeSavedRunRecord(prepared);
        expect(first.recipe).not.toBe(second.recipe);
        expect(first.recipe).not.toBe(prepared.document.recipe);
        expect(first.snapshot.model).not.toBe(second.snapshot.model);
        expect(first.snapshot.evaluationHistory).not.toBe(second.snapshot.evaluationHistory);
        expect(first.snapshot.evaluation).not.toBe(second.snapshot.evaluation);
        Reflect.set(first.recipe.data, 'sampleCount', 999);
        Reflect.set(first.snapshot.model, 'step', 999);
        expect(second.recipe.data.sampleCount).toBe(prepared.document.recipe.data.sampleCount);
        expect(second.snapshot.model.step).toBe(4);
    });
});
