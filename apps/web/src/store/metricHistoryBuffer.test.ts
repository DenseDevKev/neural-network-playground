import { describe, expect, it } from 'vitest';
import type {
    DatasetRevision,
    LiveTrainingSignal,
    PairedEvaluation,
} from '@nn-playground/shared';
import { MetricHistoryBuffer } from './metricHistoryBuffer.ts';

const DATASET: DatasetRevision = {
    generatorVersion: 3,
    datasetKey: 'packed-dataset',
    trainCount: 4,
    testCount: 2,
};

function signalAt(step: number): LiveTrainingSignal {
    return {
        model: { generationId: 3, revision: step + 10, step, epoch: Math.floor(step / 4) },
        dataset: DATASET,
        objectiveKey: 'packed-objective',
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.2,
            latestBatchSize: step % 2 === 0 ? 2 : 3,
            throughStep: step,
        },
        dataLoss: step / 10,
    };
}

function evaluationAt(
    evaluationId: number,
    overrides: Partial<PairedEvaluation> = {},
): PairedEvaluation {
    const step = evaluationId * 50;
    return {
        evaluationId,
        trigger: evaluationId === 1 ? 'initial' : 'cadence',
        model: { generationId: 3, revision: step + 10, step, epoch: evaluationId },
        dataset: DATASET,
        objectiveKey: 'packed-objective',
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 4, populationCount: 4 },
            values: {
                dataLoss: 0.4 / evaluationId,
                accuracy: 0.75,
                confusionMatrix: { tp: 2, tn: 1, fp: 1, fn: 0 },
            },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 2, populationCount: 2 },
            values: {
                dataLoss: 0.6 / evaluationId,
                accuracy: 0.5,
                confusionMatrix: { tp: 1, tn: 0, fp: 1, fn: 0 },
            },
        },
        objective: {
            regularizationPenalty: 0.1,
            trainTotalObjective: 0.4 / evaluationId + 0.1,
        },
        ...overrides,
    };
}

describe('MetricHistoryBuffer', () => {
    it('keeps trend and evaluation storage and versions independent', () => {
        const buffer = new MetricHistoryBuffer({ trendCapacity: 4, evaluationCapacity: 4 });
        expect(buffer.versions).toEqual({ trendVersion: 0, evaluationVersion: 0 });

        expect(buffer.appendTrend(signalAt(1))).toBe(1);
        expect(buffer.versions).toEqual({ trendVersion: 1, evaluationVersion: 0 });
        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: true, version: 1 });
        expect(buffer.versions).toEqual({ trendVersion: 1, evaluationVersion: 1 });

        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: false, version: 1 });
        expect(buffer.versions).toEqual({ trendVersion: 1, evaluationVersion: 1 });
    });

    it('drops unseen lower IDs and rejects conflicts only while an ID is retained', () => {
        const buffer = new MetricHistoryBuffer();
        expect(buffer.appendEvaluation(evaluationAt(2))).toEqual({ appended: true, version: 1 });
        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: false, version: 1 });
        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: false, version: 1 });
        expect(() => buffer.appendEvaluation(evaluationAt(2, {
            model: { ...evaluationAt(2).model, step: 125 },
        }))).toThrow('conflicting evaluationId 2');

        expect(buffer.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([2]);
        expect(buffer.versions.evaluationVersion).toBe(1);
    });

    it('bounds each packed series independently while preserving chronological order', () => {
        const buffer = new MetricHistoryBuffer({ trendCapacity: 2, evaluationCapacity: 2 });
        buffer.appendTrend(signalAt(1));
        buffer.appendTrend(signalAt(2));
        buffer.appendTrend(signalAt(3));
        buffer.appendEvaluation(evaluationAt(1));
        buffer.appendEvaluation(evaluationAt(2));
        buffer.appendEvaluation(evaluationAt(3));

        const snapshot = buffer.read();
        expect(snapshot.trendHistory.map(({ model }) => model.step)).toEqual([2, 3]);
        expect(snapshot.evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([2, 3]);
        expect(snapshot.trendVersion).toBe(3);
        expect(snapshot.evaluationVersion).toBe(3);
        expect(buffer.getRetainedEvaluationFingerprintCountForTests()).toBe(2);
    });

    it('drops evicted stale IDs without re-entry and bounds replay metadata', () => {
        const buffer = new MetricHistoryBuffer({ evaluationCapacity: 2 });
        buffer.appendEvaluation(evaluationAt(1));
        buffer.appendEvaluation(evaluationAt(2));
        buffer.appendEvaluation(evaluationAt(3));
        expect(buffer.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([
            2,
            3,
        ]);

        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: false, version: 3 });
        expect(buffer.appendEvaluation(evaluationAt(1, {
            model: { ...evaluationAt(1).model, step: 75 },
        }))).toEqual({ appended: false, version: 3 });
        expect(buffer.versions.evaluationVersion).toBe(3);
        expect(buffer.getRetainedEvaluationFingerprintCountForTests()).toBe(2);
        expect(buffer.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([
            2,
            3,
        ]);
    });

    it('reconstructs every provenance field without creating a synthetic scalar point', () => {
        const buffer = new MetricHistoryBuffer();
        const signal = signalAt(7);
        const evaluation = evaluationAt(2);

        buffer.appendTrend(signal);
        buffer.appendEvaluation(evaluation);
        const snapshot = buffer.read();

        expect(snapshot.trendHistory[0]).toEqual(signal);
        expect(snapshot.evaluationHistory[0]).toEqual(evaluation);
        expect(Object.keys(snapshot.trendHistory[0] ?? {})).toEqual([
            'model',
            'dataset',
            'objectiveKey',
            'basis',
            'dataLoss',
        ]);
        expect(snapshot.trendHistory[0]).not.toHaveProperty('trainLoss');
        expect(snapshot.trendHistory[0]).not.toHaveProperty('testLoss');
    });

    it('caches immutable snapshots until an actual append changes the packed storage', () => {
        const buffer = new MetricHistoryBuffer();
        buffer.appendTrend(signalAt(1));
        buffer.appendEvaluation(evaluationAt(1));

        const first = buffer.read();
        const second = buffer.read();
        expect(first).toEqual(second);
        expect(first).toBe(second);
        expect(first.trendHistory).toBe(second.trendHistory);
        expect(Object.isFrozen(first)).toBe(true);
        expect(Object.isFrozen(first.trendHistory)).toBe(true);
        expect(Object.isFrozen(first.trendHistory[0]?.basis)).toBe(true);
        expect(Object.isFrozen(first.evaluationHistory[0]?.train.values.confusionMatrix)).toBe(true);

        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: false, version: 1 });
        expect(buffer.read()).toBe(first);
        buffer.appendTrend(signalAt(2));
        buffer.appendEvaluation(evaluationAt(2));
        const changed = buffer.read();
        expect(changed).not.toBe(first);
        expect(first.trendHistory.map(({ model }) => model.step)).toEqual([1]);
        expect(changed.trendHistory.map(({ model }) => model.step)).toEqual([1, 2]);
        expect(changed.trendHistory[0]).toBe(first.trendHistory[0]);
        expect(changed.evaluationHistory[0]).toBe(first.evaluationHistory[0]);
        expect(changed.evaluationHistory[0]?.train.values.confusionMatrix).toBe(
            first.evaluationHistory[0]?.train.values.confusionMatrix,
        );
    });

    it('resets both packed series, their replay guards, and exact versions', () => {
        const buffer = new MetricHistoryBuffer();
        buffer.appendTrend(signalAt(1));
        buffer.appendEvaluation(evaluationAt(1));

        expect(buffer.reset()).toEqual({ trendVersion: 2, evaluationVersion: 2 });
        expect(buffer.read()).toEqual({
            trendHistory: [],
            evaluationHistory: [],
            trendVersion: 2,
            evaluationVersion: 2,
        });
        expect(buffer.getRetainedEvaluationFingerprintCountForTests()).toBe(0);
        expect(buffer.appendEvaluation(evaluationAt(1))).toEqual({ appended: true, version: 3 });
    });

    it('rejects invalid capacities and non-finite evidence before changing versions', () => {
        expect(() => new MetricHistoryBuffer({ trendCapacity: 0 })).toThrow('trendCapacity');
        const buffer = new MetricHistoryBuffer();
        const invalid = { ...signalAt(1), dataLoss: Number.NaN };

        expect(() => buffer.appendTrend(invalid)).toThrow('dataLoss');
        expect(buffer.versions).toEqual({ trendVersion: 0, evaluationVersion: 0 });
    });

    it('prepares replacement validation without mutating accepted history', () => {
        const buffer = new MetricHistoryBuffer();
        buffer.appendTrend(signalAt(1));
        buffer.appendEvaluation(evaluationAt(1));
        const before = buffer.read();

        expect(() => buffer.prepareReplacement({
            ...signalAt(2),
            dataLoss: Number.NaN,
        }, evaluationAt(2))).toThrow('dataLoss');

        expect(buffer.read()).toBe(before);
    });

    it('commits prepared appends idempotently', () => {
        const buffer = new MetricHistoryBuffer();
        const prepared = buffer.prepareAppend(signalAt(1), evaluationAt(1));

        prepared.commit();
        const committed = buffer.read();
        prepared.commit();

        expect(buffer.read()).toBe(committed);
        expect(committed.trendHistory).toHaveLength(1);
        expect(committed.evaluationHistory).toHaveLength(1);
        expect(committed.trendVersion).toBe(1);
        expect(committed.evaluationVersion).toBe(1);
    });

    it('rejects stale prepared append and replacement plans without overwriting newer history', () => {
        const appendBuffer = new MetricHistoryBuffer();
        const appendPlan = appendBuffer.prepareAppend(signalAt(1), evaluationAt(1));
        appendBuffer.appendTrend(signalAt(2));
        expect(() => appendPlan.commit()).toThrow(/stale/i);
        expect(appendBuffer.read().trendHistory.map(({ model }) => model.step)).toEqual([2]);
        expect(appendBuffer.read().evaluationHistory).toEqual([]);

        const replacementBuffer = new MetricHistoryBuffer();
        const replacementPlan = replacementBuffer.prepareReplacement(
            signalAt(1),
            evaluationAt(1),
        );
        replacementBuffer.appendEvaluation(evaluationAt(2));
        expect(() => replacementPlan.commit()).toThrow(/stale/i);
        expect(replacementBuffer.read().evaluationHistory.map(({ evaluationId }) => evaluationId))
            .toEqual([2]);
    });

    it('preflights safe-integer version exhaustion before mutating packed history', () => {
        const trendBuffer = new MetricHistoryBuffer();
        const trendInternals = trendBuffer as unknown as {
            trendVersion: number;
            cachedSnapshot: unknown;
        };
        trendInternals.trendVersion = Number.MAX_SAFE_INTEGER;
        trendInternals.cachedSnapshot = undefined;
        expect(() => trendBuffer.prepareAppend(signalAt(1))).toThrow(/trendVersion exhausted/);
        expect(trendBuffer.read().trendHistory).toEqual([]);

        const evaluationBuffer = new MetricHistoryBuffer();
        const evaluationInternals = evaluationBuffer as unknown as {
            evaluationVersion: number;
            cachedSnapshot: unknown;
        };
        evaluationInternals.evaluationVersion = Number.MAX_SAFE_INTEGER;
        evaluationInternals.cachedSnapshot = undefined;
        expect(() => evaluationBuffer.appendEvaluation(evaluationAt(1)))
            .toThrow(/evaluationVersion exhausted/);
        expect(evaluationBuffer.read().evaluationHistory).toEqual([]);
    });
});
