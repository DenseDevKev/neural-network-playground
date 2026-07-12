import { describe, expect, it } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    EXPERIMENT_MEMORY_ENVELOPE_KIND,
    EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES,
    EXPERIMENT_MEMORY_MAX_EVALUATIONS,
    EXPERIMENT_MEMORY_MAX_RECORD_BYTES,
    EXPERIMENT_MEMORY_MAX_RECORDS,
    EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS,
    EXPERIMENT_MEMORY_MAX_TRENDS,
    EXPERIMENT_MEMORY_RECORD_KIND,
    compactEvenly,
    parseExperimentMemoryEnvelopeV2,
    prepareExperimentDocument,
    serializeExperimentMemoryEnvelopeV2,
    validateExperimentRunRecordV2,
} from '../index.js';
import type {
    DatasetRevision,
    EvaluationPoint,
    ExperimentRunRecordV2,
    ModelRevision,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    TrainingTrendPoint,
} from '../index.js';

const IDS = Array.from({ length: 24 }, (_, index) => (
    `00000000-0000-0000-0000-${(index + 1).toString(16).padStart(12, '0')}`
));

async function preparedDefault(): Promise<PreparedExperimentDocumentV2> {
    const result = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
    if (!result.ok) throw new Error(result.issues.map((issue) => issue.message).join(', '));
    return result.value;
}

function modelAt(revision: number, generationId = 7): ModelRevision {
    return { generationId, revision, step: revision, epoch: Math.floor(revision / 10) };
}

function datasetFor(prepared: PreparedExperimentDocumentV2): DatasetRevision {
    return {
        generatorVersion: 2,
        datasetKey: prepared.identities.datasetKey,
        trainCount: 100,
        testCount: 100,
    };
}

function evaluationAt(
    prepared: PreparedExperimentDocumentV2,
    evaluationId: number,
    revision: number,
    trigger: PairedEvaluation['trigger'] = 'cadence',
): EvaluationPoint {
    const dataset = datasetFor(prepared);
    return {
        evaluationId,
        trigger,
        model: modelAt(revision),
        dataset,
        objectiveKey: prepared.identities.objectiveKey,
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 100, populationCount: 100 },
            values: { dataLoss: 0.4 },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 100, populationCount: 100 },
            values: { dataLoss: 0.5 },
        },
        objective: { regularizationPenalty: 0.1, trainTotalObjective: 0.5 },
    };
}

function trendAt(
    prepared: PreparedExperimentDocumentV2,
    revision: number,
): TrainingTrendPoint {
    return {
        model: modelAt(revision),
        dataset: datasetFor(prepared),
        objectiveKey: prepared.identities.objectiveKey,
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 10,
            throughStep: revision,
        },
        dataLoss: 1 / (revision + 1),
    };
}

async function makeRecord(
    overrides: Partial<ExperimentRunRecordV2> = {},
): Promise<ExperimentRunRecordV2> {
    const prepared = await preparedDefault();
    const evaluation = evaluationAt(prepared, 2, 20, 'save');
    return {
        kind: EXPERIMENT_MEMORY_RECORD_KIND,
        schemaVersion: 2,
        id: IDS[0],
        createdAt: '2026-07-11T12:00:00.000Z',
        updatedAt: '2026-07-11T12:00:00.000Z',
        title: 'Circle baseline',
        recipe: prepared.document.recipe,
        recipeFingerprint: prepared.identities.recipeFingerprint,
        snapshot: {
            model: evaluation.model,
            evaluation,
            trendHistory: [trendAt(prepared, 1), trendAt(prepared, 20)],
            evaluationHistory: [evaluationAt(prepared, 1, 0, 'initial'), evaluation],
        },
        ...overrides,
    };
}

describe('version-2 experiment memory contracts', () => {
    it('accepts only a canonical lowercase 36-character UUID and 120 title code points', async () => {
        const record = await makeRecord({ title: '🧠'.repeat(EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS) });
        await expect(validateExperimentRunRecordV2(record)).resolves.toMatchObject({ ok: true });

        for (const id of [
            '00000000-0000-0000-0000-00000000001',
            '00000000-0000-0000-0000-0000000000010',
            '00000000-0000-0000-0000-00000000000G',
            '00000000-0000-0000-0000-00000000000A',
        ]) {
            const result = await validateExperimentRunRecordV2({ ...record, id });
            expect(result.ok).toBe(false);
            if (!result.ok) expect(result.issues.some((issue) => issue.path === 'id')).toBe(true);
        }

        const tooLong = await validateExperimentRunRecordV2({
            ...record,
            title: '🧠'.repeat(EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS + 1),
        });
        expect(tooLong.ok).toBe(false);
        if (!tooLong.ok) expect(tooLong.issues[0]?.path).toBe('title');
    });

    it('takes a stable record snapshot before asynchronous fingerprinting', async () => {
        const record = await makeRecord();
        const pending = validateExperimentRunRecordV2(record);
        (record as { title?: string }).title = 'mutated after validation began';

        const result = await pending;
        expect(result.ok).toBe(true);
        if (result.ok) expect(result.value.title).toBe('Circle baseline');
    });

    it('compacts histories by exact endpoint-preserving rounded indices', () => {
        expect(compactEvenly(
            Array.from({ length: 1025 }, (_, index) => index),
            EXPERIMENT_MEMORY_MAX_TRENDS,
        )).toEqual(
            Array.from(
                { length: EXPERIMENT_MEMORY_MAX_TRENDS },
                (_, index) => Math.round(index * 1024 / 511),
            ),
        );
        expect(compactEvenly([0, 1, 2], 3)).toEqual([0, 1, 2]);
    });

    it('enforces exact 512 trend and 256 evaluation history caps', async () => {
        const prepared = await preparedDefault();
        const final = evaluationAt(prepared, 257, 1024, 'save');
        const result = await validateExperimentRunRecordV2(await makeRecord({
            snapshot: {
                model: final.model,
                evaluation: final,
                trendHistory: Array.from({ length: EXPERIMENT_MEMORY_MAX_TRENDS + 1 }, (_, index) => (
                    trendAt(prepared, index)
                )),
                evaluationHistory: Array.from(
                    { length: EXPERIMENT_MEMORY_MAX_EVALUATIONS + 1 },
                    (_, index) => evaluationAt(
                        prepared,
                        index + 1,
                        index * 4,
                        index === EXPERIMENT_MEMORY_MAX_EVALUATIONS ? 'save' : 'cadence',
                    ),
                ),
            },
        }));

        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.issues.map((issue) => issue.path)).toEqual(expect.arrayContaining([
                'snapshot.trendHistory',
                'snapshot.evaluationHistory',
            ]));
        }
    });

    it('rejects duplicate evaluation IDs and requires the final save pair', async () => {
        const record = await makeRecord();
        const duplicate = await validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                evaluationHistory: [record.snapshot.evaluation, record.snapshot.evaluation],
            },
        });
        expect(duplicate.ok).toBe(false);
        if (!duplicate.ok) expect(duplicate.issues.some((issue) => /duplicate/i.test(issue.message))).toBe(true);

        const notSave = {
            ...record.snapshot.evaluation,
            trigger: 'pause' as const,
        };
        const wrongFinal = await validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                model: notSave.model,
                evaluation: notSave,
                evaluationHistory: [notSave],
            },
        });
        expect(wrongFinal.ok).toBe(false);
        if (!wrongFinal.ok) expect(wrongFinal.issues.some((issue) => issue.path === 'snapshot.evaluation.trigger')).toBe(true);
    });

    it('rejects fingerprint, dataset, objective, generation, and final-pair identity mismatches', async () => {
        const record = await makeRecord();
        const cases: Array<[string, ExperimentRunRecordV2]> = [
            ['recipeFingerprint', { ...record, recipeFingerprint: `r2.1.${'A'.repeat(43)}` }],
            ['datasetKey', {
                ...record,
                snapshot: {
                    ...record.snapshot,
                    trendHistory: [{
                        ...record.snapshot.trendHistory[0],
                        dataset: { ...record.snapshot.trendHistory[0].dataset, datasetKey: 'wrong' },
                    }],
                },
            }],
            ['objectiveKey', {
                ...record,
                snapshot: {
                    ...record.snapshot,
                    trendHistory: [{ ...record.snapshot.trendHistory[0], objectiveKey: 'wrong' }],
                },
            }],
            ['generationId', {
                ...record,
                snapshot: {
                    ...record.snapshot,
                    trendHistory: [{
                        ...record.snapshot.trendHistory[0],
                        model: { ...record.snapshot.trendHistory[0].model, generationId: 8 },
                    }],
                },
            }],
            ['snapshot.evaluationHistory', {
                ...record,
                snapshot: {
                    ...record.snapshot,
                    evaluationHistory: [record.snapshot.evaluationHistory[0]],
                },
            }],
            ['snapshot.model', {
                ...record,
                snapshot: {
                    ...record.snapshot,
                    model: { ...record.snapshot.model, silentlyIgnored: true },
                },
            } as ExperimentRunRecordV2],
        ];

        for (const [expectedPath, candidate] of cases) {
            const result = await validateExperimentRunRecordV2(candidate);
            expect(result.ok).toBe(false);
            if (!result.ok) {
                expect(result.issues.some((issue) => issue.path.includes(expectedPath))).toBe(true);
            }
        }
    });

    it('matches the final evaluation structurally instead of relying on JSON key order', async () => {
        const record = await makeRecord();
        const evaluation = record.snapshot.evaluation;
        const reorderedEvaluation: PairedEvaluation = {
            objective: evaluation.objective,
            test: evaluation.test,
            train: evaluation.train,
            objectiveKey: evaluation.objectiveKey,
            dataset: evaluation.dataset,
            model: evaluation.model,
            trigger: evaluation.trigger,
            evaluationId: evaluation.evaluationId,
        };

        await expect(validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                evaluation: reorderedEvaluation,
            },
        })).resolves.toMatchObject({ ok: true });
    });

    it('enforces the 512 KiB record and 4 MiB envelope UTF-8 byte budgets', async () => {
        const record = await makeRecord();
        const oversizedRecord = {
            ...record,
            padding: 'x'.repeat(EXPERIMENT_MEMORY_MAX_RECORD_BYTES),
        };
        const parsedRecord = await parseExperimentMemoryEnvelopeV2(JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [oversizedRecord],
        }));
        expect(parsedRecord.records).toEqual([]);
        expect(parsedRecord.rejectedRecords[0]?.issues[0]?.code).toBe('resource-limit');

        const parsedEnvelope = await parseExperimentMemoryEnvelopeV2(JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [{ padding: '🧠'.repeat(EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES / 2) }],
        }));
        expect(parsedEnvelope.records).toEqual([]);
        expect(parsedEnvelope.envelopeIssues[0]?.code).toBe('resource-limit');
    });

    it('returns a valid sibling beside a structured rejected raw record', async () => {
        const valid = await makeRecord();
        const rejectedRaw = { kind: 'nn-playground-run', schemaVersion: 1, id: 'legacy' };
        const result = await parseExperimentMemoryEnvelopeV2(JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [valid, rejectedRaw],
        }));

        expect(result.records.map((record) => record.id)).toEqual([valid.id]);
        expect(result.rejectedRecords).toHaveLength(1);
        expect(result.rejectedRecords[0]).toMatchObject({ sourceIndex: 1 });
        expect(result.rejectedRecords[0]?.rawJson).toBe(JSON.stringify(rejectedRaw));
        expect(result.rejectedRecords[0]?.issues.length).toBeGreaterThan(0);
    });

    it('does not hide a valid sibling that follows many rejected raw entries', async () => {
        const valid = await makeRecord();
        const rejected = Array.from({ length: EXPERIMENT_MEMORY_MAX_RECORDS }, (_, index) => ({
            schemaVersion: 1,
            id: `rejected-${index}`,
        }));
        const result = await parseExperimentMemoryEnvelopeV2(JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [...rejected, valid],
        }));

        expect(result.records.map((record) => record.id)).toEqual([valid.id]);
        expect(result.rejectedRecords).toHaveLength(EXPERIMENT_MEMORY_MAX_RECORDS);
    });

    it('rejects a 21st record without evicting or rewriting the first 20', async () => {
        const base = await makeRecord();
        const records = Array.from({ length: EXPERIMENT_MEMORY_MAX_RECORDS }, (_, index) => ({
            ...base,
            id: IDS[index],
        }));
        const serialized = await serializeExperimentMemoryEnvelopeV2(records);
        expect(serialized.ok).toBe(true);
        if (!serialized.ok) return;

        const overLimit = await serializeExperimentMemoryEnvelopeV2([
            { ...base, id: IDS[EXPERIMENT_MEMORY_MAX_RECORDS] },
            ...records,
        ]);
        expect(overLimit.ok).toBe(false);
        if (!overLimit.ok) expect(overLimit.issues[0]?.code).toBe('resource-limit');

        const original = await parseExperimentMemoryEnvelopeV2(serialized.value);
        expect(original.records.map((record) => record.id)).toEqual(records.map((record) => record.id));
    });
});
