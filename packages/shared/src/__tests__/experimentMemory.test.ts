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
    const sampleCount = prepared.document.recipe.data.sampleCount;
    const trainCount = Math.floor(
        sampleCount * prepared.document.recipe.data.trainFraction,
    );
    return {
        generatorVersion: 2,
        datasetKey: prepared.identities.datasetKey,
        trainCount,
        testCount: sampleCount - trainCount,
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
            basis: {
                kind: 'full-split',
                split: 'train',
                sampleCount: dataset.trainCount,
                populationCount: dataset.trainCount,
            },
            values: { dataLoss: 0.4 },
        },
        test: {
            basis: {
                kind: 'full-split',
                split: 'test',
                sampleCount: dataset.testCount,
                populationCount: dataset.testCount,
            },
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

    it('accepts exactly 512 trends and 256 evaluations and rejects one over either cap', async () => {
        const prepared = await preparedDefault();
        const final = evaluationAt(prepared, EXPERIMENT_MEMORY_MAX_EVALUATIONS, 1024, 'save');
        const exactSnapshot = {
            model: final.model,
            evaluation: final,
            trendHistory: Array.from({ length: EXPERIMENT_MEMORY_MAX_TRENDS }, (_, index) => (
                trendAt(prepared, index * 2)
            )),
            evaluationHistory: Array.from(
                { length: EXPERIMENT_MEMORY_MAX_EVALUATIONS },
                (_, index) => index === EXPERIMENT_MEMORY_MAX_EVALUATIONS - 1
                    ? final
                    : evaluationAt(prepared, index + 1, index * 4),
            ),
        };
        await expect(validateExperimentRunRecordV2(await makeRecord({
            snapshot: exactSnapshot,
        }))).resolves.toMatchObject({ ok: true });

        const result = await validateExperimentRunRecordV2(await makeRecord({
            snapshot: {
                ...exactSnapshot,
                trendHistory: [...exactSnapshot.trendHistory, trendAt(prepared, 1023)],
                evaluationHistory: [
                    ...exactSnapshot.evaluationHistory.slice(0, -1),
                    evaluationAt(prepared, EXPERIMENT_MEMORY_MAX_EVALUATIONS, 1020),
                    final,
                ],
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

    it('binds final, history, and full-split population counts to the recipe split', async () => {
        const record = await makeRecord();
        const wrongFinal = {
            ...record.snapshot.evaluation,
            dataset: {
                ...record.snapshot.evaluation.dataset,
                trainCount: record.snapshot.evaluation.dataset.trainCount - 1,
                testCount: record.snapshot.evaluation.dataset.testCount + 1,
            },
            train: {
                ...record.snapshot.evaluation.train,
                basis: {
                    ...record.snapshot.evaluation.train.basis,
                    sampleCount: record.snapshot.evaluation.dataset.trainCount - 1,
                    populationCount: record.snapshot.evaluation.dataset.trainCount - 1,
                },
            },
            test: {
                ...record.snapshot.evaluation.test,
                basis: {
                    ...record.snapshot.evaluation.test.basis,
                    sampleCount: record.snapshot.evaluation.dataset.testCount + 1,
                    populationCount: record.snapshot.evaluation.dataset.testCount + 1,
                },
            },
        };
        const finalCount = await validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                evaluation: wrongFinal,
                model: wrongFinal.model,
                evaluationHistory: [record.snapshot.evaluationHistory[0], wrongFinal],
            },
        });
        expect(finalCount.ok).toBe(false);
        if (!finalCount.ok) expect(finalCount.issues.some(
            (entry) => entry.path === 'snapshot.evaluation.dataset.trainCount',
        )).toBe(true);

        const historical = record.snapshot.evaluationHistory[0];
        const wrongHistory = {
            ...historical,
            dataset: {
                ...historical.dataset,
                trainCount: historical.dataset.trainCount - 1,
                testCount: historical.dataset.testCount + 1,
            },
            train: {
                ...historical.train,
                basis: {
                    ...historical.train.basis,
                    sampleCount: historical.dataset.trainCount - 1,
                    populationCount: historical.dataset.trainCount - 1,
                },
            },
            test: {
                ...historical.test,
                basis: {
                    ...historical.test.basis,
                    sampleCount: historical.dataset.testCount + 1,
                    populationCount: historical.dataset.testCount + 1,
                },
            },
        };
        const historyCount = await validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                evaluationHistory: [wrongHistory, record.snapshot.evaluation],
            },
        });
        expect(historyCount.ok).toBe(false);
        if (!historyCount.ok) expect(historyCount.issues.some(
            (entry) => entry.path === 'snapshot.evaluationHistory[0].dataset.trainCount',
        )).toBe(true);

        const wrongPopulation = await validateExperimentRunRecordV2({
            ...record,
            snapshot: {
                ...record.snapshot,
                evaluation: {
                    ...record.snapshot.evaluation,
                    train: {
                        ...record.snapshot.evaluation.train,
                        basis: {
                            ...record.snapshot.evaluation.train.basis,
                            populationCount: record.snapshot.evaluation.dataset.trainCount - 1,
                        },
                    },
                },
            },
        });
        expect(wrongPopulation.ok).toBe(false);
        if (!wrongPopulation.ok) expect(wrongPopulation.issues.some(
            (entry) => entry.path.includes('snapshot.evaluation'),
        )).toBe(true);
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

    it('accepts the largest closed-schema record and rejects a raw 512 KiB + 1 record', async () => {
        const prepared = await preparedDefault();
        const dataset = datasetFor(prepared);
        const maxInteger = Number.MAX_SAFE_INTEGER;
        const largeModelAt = (offset: number): ModelRevision => ({
            generationId: maxInteger,
            revision: maxInteger - offset,
            step: maxInteger - offset,
            epoch: maxInteger - offset,
        });
        const largeValues = (count: number) => ({
            dataLoss: Number.MAX_VALUE,
            accuracy: 1,
            confusionMatrix: {
                classCount: 3 as const,
                classLabels: [0, 1, 2] as const,
                counts: [count, 0, 0, 0, 0, 0, 0, 0, 0] as const,
            },
        });
        const largeEvaluationAt = (index: number): EvaluationPoint => {
            const trainValues = largeValues(dataset.trainCount);
            return {
                evaluationId: maxInteger - (EXPERIMENT_MEMORY_MAX_EVALUATIONS - 1 - index),
                trigger: index === EXPERIMENT_MEMORY_MAX_EVALUATIONS - 1 ? 'save' : 'cadence',
                model: largeModelAt(EXPERIMENT_MEMORY_MAX_EVALUATIONS - 1 - index),
                dataset,
                objectiveKey: prepared.identities.objectiveKey,
                train: {
                    basis: {
                        kind: 'full-split',
                        split: 'train',
                        sampleCount: dataset.trainCount,
                        populationCount: dataset.trainCount,
                    },
                    values: trainValues,
                },
                test: {
                    basis: {
                        kind: 'full-split',
                        split: 'test',
                        sampleCount: dataset.testCount,
                        populationCount: dataset.testCount,
                    },
                    values: largeValues(dataset.testCount),
                },
                objective: {
                    regularizationPenalty: 0,
                    trainTotalObjective: trainValues.dataLoss,
                },
            };
        };
        const final = largeEvaluationAt(EXPERIMENT_MEMORY_MAX_EVALUATIONS - 1);
        const largestLegal = await makeRecord({
            title: '🧠'.repeat(EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS),
            snapshot: {
                model: final.model,
                evaluation: final,
                trendHistory: Array.from({ length: EXPERIMENT_MEMORY_MAX_TRENDS }, (_, index) => (
                    {
                        model: largeModelAt(EXPERIMENT_MEMORY_MAX_TRENDS - 1 - index),
                        dataset,
                        objectiveKey: prepared.identities.objectiveKey,
                        basis: {
                            kind: 'mini-batch-ema' as const,
                            alpha: Number.MIN_VALUE,
                            latestBatchSize: dataset.trainCount,
                            throughStep: maxInteger - (EXPERIMENT_MEMORY_MAX_TRENDS - 1 - index),
                        },
                        dataLoss: Number.MAX_VALUE,
                    }
                )),
                evaluationHistory: Array.from(
                    { length: EXPERIMENT_MEMORY_MAX_EVALUATIONS },
                    (_, index) => largeEvaluationAt(index),
                ),
            },
        });
        const legalJson = JSON.stringify(largestLegal);
        const legalBytes = new TextEncoder().encode(legalJson).byteLength;
        // The closed schema cannot reach 512 KiB without an unknown padding
        // field; this maximum-cardinality fixture is still over 400 KiB.
        expect(legalBytes).toBeGreaterThan(400_000);
        expect(legalBytes).toBeLessThan(EXPERIMENT_MEMORY_MAX_RECORD_BYTES);
        await expect(validateExperimentRunRecordV2(largestLegal)).resolves.toMatchObject({ ok: true });

        const emptyPaddingBytes = new TextEncoder().encode(JSON.stringify({ padding: '' })).byteLength;
        const oversizedRecord = {
            padding: 'x'.repeat(EXPERIMENT_MEMORY_MAX_RECORD_BYTES + 1 - emptyPaddingBytes),
        };
        expect(new TextEncoder().encode(JSON.stringify(oversizedRecord)).byteLength)
            .toBe(EXPERIMENT_MEMORY_MAX_RECORD_BYTES + 1);
        const parsedRecord = await parseExperimentMemoryEnvelopeV2(JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [oversizedRecord],
        }));
        expect(parsedRecord.records).toEqual([]);
        expect(parsedRecord.rejectedRecords[0]?.issues[0]?.code).toBe('resource-limit');
    });

    it('accepts an exact 4 MiB raw envelope and rejects one additional UTF-8 byte', async () => {
        const envelopeWithPadding = (paddingLength: number) => JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: ['x'.repeat(paddingLength)],
        });
        const emptyBytes = new TextEncoder().encode(envelopeWithPadding(0)).byteLength;
        const exact = envelopeWithPadding(EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES - emptyBytes);
        expect(new TextEncoder().encode(exact).byteLength).toBe(EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES);
        const accepted = await parseExperimentMemoryEnvelopeV2(exact);
        expect(accepted.envelopeIssues).toEqual([]);
        expect(accepted.rejectedRecords).toHaveLength(1);

        const over = envelopeWithPadding(EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES + 1 - emptyBytes);
        expect(new TextEncoder().encode(over).byteLength).toBe(EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES + 1);
        const rejected = await parseExperimentMemoryEnvelopeV2(over);
        expect(rejected.records).toEqual([]);
        expect(rejected.envelopeIssues[0]?.code).toBe('resource-limit');
    });

    it('preserves invalid whole-envelope bytes separately from rejected record siblings', async () => {
        for (const rawJson of [
            ' {not valid JSON',
            '{ "kind": "nn-playground-experiment-memory", "schemaVersion": 3, "records": [] }',
        ]) {
            const result = await parseExperimentMemoryEnvelopeV2(rawJson);

            expect(result.records).toEqual([]);
            expect(result.rejectedRecords).toEqual([]);
            expect(result.incompatibleEnvelope).toEqual({
                rawJson,
                issues: result.envelopeIssues,
            });
        }
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
