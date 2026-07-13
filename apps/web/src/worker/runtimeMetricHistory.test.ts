import { describe, expect, it } from 'vitest';
import type {
    DatasetRevision,
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
} from '@nn-playground/shared';
import { RuntimeMetricHistory } from './runtimeMetricHistory.ts';

const DATASET: DatasetRevision = {
    generatorVersion: 1,
    datasetKey: 'dataset-11',
    trainCount: 4,
    testCount: 2,
};

function modelAt(step: number, generationId = 11): ModelRevision {
    return { generationId, revision: step, step, epoch: 0 };
}

function signalAt(step: number, overrides: Partial<LiveTrainingSignal> = {}): LiveTrainingSignal {
    return {
        model: modelAt(step),
        dataset: DATASET,
        objectiveKey: 'objective-11',
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 2,
            throughStep: step,
        },
        dataLoss: 1 / (step + 1),
        ...overrides,
    };
}

function evaluationAt(
    evaluationId: number,
    step = evaluationId * 50,
    overrides: Partial<PairedEvaluation> = {},
): PairedEvaluation {
    const trainDataLoss = 0.4;
    return {
        evaluationId,
        trigger: 'cadence',
        model: modelAt(step),
        dataset: DATASET,
        objectiveKey: 'objective-11',
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 4, populationCount: 4 },
            values: { dataLoss: trainDataLoss },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 2, populationCount: 2 },
            values: { dataLoss: 0.6 },
        },
        objective: { regularizationPenalty: 0.1, trainTotalObjective: 0.5 },
        ...overrides,
    };
}

function createHistory(capacities: {
    trendCapacity?: number;
    evaluationCapacity?: number;
} = {}): RuntimeMetricHistory {
    return new RuntimeMetricHistory({
        generationId: 11,
        dataset: DATASET,
        objectiveKey: 'objective-11',
        ...capacities,
    });
}

describe('RuntimeMetricHistory', () => {
    it('appends every live signal while deduplicating paired evaluation IDs', () => {
        const history = createHistory();

        history.appendLiveSignal(signalAt(1));
        history.appendLiveSignal(signalAt(2));
        expect(history.appendEvaluation(evaluationAt(1))).toBe(true);
        expect(history.appendEvaluation(evaluationAt(1))).toBe(false);

        const snapshot = history.read();
        expect(snapshot.trendHistory.map(({ model }) => model.step)).toEqual([1, 2]);
        expect(snapshot.evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([1]);
        expect(snapshot.evaluationHistory[0]?.model.step).toBe(50);
    });

    it('drops unseen lower IDs and rejects conflicts only while an ID is retained', () => {
        const history = createHistory();
        expect(history.appendEvaluation(evaluationAt(2))).toBe(true);
        expect(history.appendEvaluation(evaluationAt(1))).toBe(false);
        expect(history.appendEvaluation(evaluationAt(1))).toBe(false);
        expect(() => history.appendEvaluation(evaluationAt(2, 75))).toThrow(
            'conflicting evaluationId 2',
        );

        expect(history.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([2]);
    });

    it('preserves the first and latest points at capacity two after overflow', () => {
        const history = createHistory({ trendCapacity: 2, evaluationCapacity: 2 });
        history.appendLiveSignal(signalAt(1));
        history.appendLiveSignal(signalAt(2));
        history.appendLiveSignal(signalAt(3));
        history.appendEvaluation(evaluationAt(2));
        history.appendEvaluation(evaluationAt(1));
        history.appendEvaluation(evaluationAt(3));

        const snapshot = history.read();
        expect(snapshot.trendHistory.map(({ model }) => model.step)).toEqual([1, 3]);
        expect(snapshot.evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([2, 3]);
    });

    it('preserves the first and latest points at capacity three after repeated overflow', () => {
        const history = createHistory({ trendCapacity: 3, evaluationCapacity: 3 });
        for (let index = 1; index <= 5; index++) {
            history.appendLiveSignal(signalAt(index));
            history.appendEvaluation(evaluationAt(index));
        }

        const snapshot = history.read();
        expect(snapshot.trendHistory.map(({ model }) => model.step)).toEqual([1, 4, 5]);
        expect(snapshot.evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([1, 4, 5]);
        expect(history.getRetainedEvaluationFingerprintCountForTests()).toBe(3);
    });

    it('drops evicted stale IDs without re-entry and bounds replay metadata', () => {
        const history = createHistory({ evaluationCapacity: 2 });
        history.appendEvaluation(evaluationAt(1));
        history.appendEvaluation(evaluationAt(2));
        history.appendEvaluation(evaluationAt(3));
        expect(history.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([
            1,
            3,
        ]);

        expect(history.appendEvaluation(evaluationAt(2))).toBe(false);
        expect(history.appendEvaluation(evaluationAt(2, 125))).toBe(false);
        expect(history.getRetainedEvaluationFingerprintCountForTests()).toBe(2);
        expect(history.read().evaluationHistory.map(({ evaluationId }) => evaluationId)).toEqual([
            1,
            3,
        ]);
    });

    it.each([
        ['generation', signalAt(1, { model: modelAt(1, 12) }), 'generationId'],
        ['dataset key', signalAt(1, {
            dataset: { ...DATASET, datasetKey: 'other-dataset' },
        }), 'dataset'],
        ['dataset population', signalAt(1, {
            dataset: { ...DATASET, trainCount: 3, testCount: 3 },
        }), 'dataset'],
        ['objective', signalAt(1, { objectiveKey: 'other-objective' }), 'objectiveKey'],
    ])('rejects live-signal %s mixing', (_label, signal, message) => {
        expect(() => createHistory().appendLiveSignal(signal)).toThrow(message);
    });

    it.each([
        ['generation', evaluationAt(1, 50, { model: modelAt(50, 12) }), 'generationId'],
        ['dataset', evaluationAt(1, 50, {
            dataset: { ...DATASET, datasetKey: 'other-dataset' },
        }), 'dataset'],
        ['objective', evaluationAt(1, 50, { objectiveKey: 'other-objective' }), 'objectiveKey'],
    ])('rejects evaluation %s mixing before duplicate handling', (_label, evaluation, message) => {
        const history = createHistory();
        history.appendEvaluation(evaluationAt(1));

        expect(() => history.appendEvaluation(evaluation)).toThrow(message);
    });

    it('caches immutable snapshots until append changes and clears both series on reset', () => {
        const history = createHistory();
        history.appendLiveSignal(signalAt(1));
        history.appendEvaluation(evaluationAt(1));

        const first = history.read();
        const second = history.read();
        expect(first).toEqual(second);
        expect(first).toBe(second);
        expect(first.trendHistory).toBe(second.trendHistory);
        expect(Object.isFrozen(first)).toBe(true);
        expect(Object.isFrozen(first.trendHistory)).toBe(true);
        expect(Object.isFrozen(first.trendHistory[0]?.model)).toBe(true);
        expect(Object.isFrozen(first.evaluationHistory[0]?.train.values)).toBe(true);

        history.appendLiveSignal(signalAt(2));
        const changed = history.read();
        expect(changed).not.toBe(first);
        expect(first.trendHistory.map(({ model }) => model.step)).toEqual([1]);
        expect(changed.trendHistory.map(({ model }) => model.step)).toEqual([1, 2]);

        history.reset();
        expect(history.read()).toEqual({ trendHistory: [], evaluationHistory: [] });
        expect(history.getRetainedEvaluationFingerprintCountForTests()).toBe(0);
        expect(history.appendEvaluation(evaluationAt(1))).toBe(true);
    });

    it('rejects invalid bounded capacities', () => {
        expect(() => createHistory({ trendCapacity: 0 })).toThrow('trendCapacity');
        expect(() => createHistory({ trendCapacity: 1 })).toThrow('trendCapacity');
        expect(() => createHistory({ evaluationCapacity: 0 })).toThrow('evaluationCapacity');
        expect(() => createHistory({ evaluationCapacity: 1 })).toThrow('evaluationCapacity');
    });

    it('rejects non-finite evidence before publication', () => {
        const history = createHistory();

        expect(() => history.appendLiveSignal(signalAt(1, { dataLoss: Number.NaN }))).toThrow(
            'dataLoss',
        );
        expect(() => history.appendEvaluation(evaluationAt(1, 50, {
            test: {
                basis: { kind: 'full-split', split: 'test', sampleCount: 2, populationCount: 2 },
                values: { dataLoss: Number.POSITIVE_INFINITY },
            },
        }))).toThrow('test.values.dataLoss');
        expect(history.read()).toEqual({ trendHistory: [], evaluationHistory: [] });
    });
});
