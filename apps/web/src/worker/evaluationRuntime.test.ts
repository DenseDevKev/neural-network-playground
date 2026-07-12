import { describe, expect, it } from 'vitest';
import type {
    DatasetRevision,
    EvaluationTrigger,
    EvaluationValues,
    ModelRevision,
} from '@nn-playground/shared';
import {
    EvaluationRuntime,
    TerminalDivergenceError,
} from './evaluationRuntime.ts';

const DATASET: DatasetRevision = {
    generatorVersion: 1,
    datasetKey: 'dataset-7',
    trainCount: 4,
    testCount: 2,
};

function modelAt(step: number, revision = step): ModelRevision {
    return { generationId: 7, revision, step, epoch: Math.floor(step / 4) };
}

function createRuntime(overrides: Partial<{
    train: EvaluationValues;
    test: EvaluationValues;
    penalty: number;
    alpha: number;
    modelAfterTrain: ModelRevision;
}> = {}) {
    const calls: Array<{ split: 'train' | 'test' | 'penalty'; model: ModelRevision }> = [];
    const modelState = { current: modelAt(0) };
    const state: {
        train: EvaluationValues;
        test: EvaluationValues;
        penalty: number;
        modelAfterTrain?: ModelRevision;
    } = {
        train: overrides.train ?? { dataLoss: 0.4 },
        test: overrides.test ?? { dataLoss: 0.6 },
        penalty: overrides.penalty ?? 0.1,
        modelAfterTrain: overrides.modelAfterTrain,
    };
    const runtime = new EvaluationRuntime({
        generationId: 7,
        dataset: DATASET,
        objectiveKey: 'objective-7',
        emaAlpha: overrides.alpha ?? 0.25,
        getCurrentModel: () => modelState.current,
        evaluateTrain(model) {
            calls.push({ split: 'train', model });
            if (state.modelAfterTrain !== undefined) {
                modelState.current = state.modelAfterTrain;
            }
            return state.train;
        },
        evaluateTest(model) {
            calls.push({ split: 'test', model });
            return state.test;
        },
        evaluateRegularizationPenalty(model) {
            calls.push({ split: 'penalty', model });
            return state.penalty;
        },
    });
    return { runtime, calls, state, modelState };
}

function recordAt(
    fixture: ReturnType<typeof createRuntime>,
    step: number,
    batchSize: number,
    dataLoss: number,
    revision = step,
) {
    const model = modelAt(step, revision);
    fixture.modelState.current = model;
    return fixture.runtime.recordBatch({ model, batchSize, dataLoss });
}

describe('EvaluationRuntime', () => {
    it('publishes batch signals at steps 1-49 and one frame-independent cadence pair at step 50', () => {
        const fixture = createRuntime();
        const { runtime } = fixture;

        for (let step = 1; step < 50; step++) {
            const signal = recordAt(fixture, step, 4, 1 / step);
            expect(signal.basis.throughStep).toBe(step);
            expect(runtime.takeCadenceEvaluation()).toBeUndefined();
        }

        recordAt(fixture, 50, 2, 0.2);
        const evaluation = runtime.takeCadenceEvaluation();
        expect(evaluation?.trigger).toBe('cadence');
        expect(evaluation?.model).toEqual(modelAt(50));
        expect(runtime.takeCadenceEvaluation()).toBeUndefined();
    });

    it('computes the EMA while carrying the exact latest batch size and through-step', () => {
        const fixture = createRuntime({ alpha: 0.25 });
        const { runtime } = fixture;

        const first = recordAt(fixture, 1, 4, 1);
        const second = recordAt(fixture, 2, 2, 0.2);

        expect(first.dataLoss).toBe(1);
        expect(second.dataLoss).toBeCloseTo(0.8, 15);
        expect(second.basis).toEqual({
            kind: 'mini-batch-ema',
            alpha: 0.25,
            latestBatchSize: 2,
            throughStep: 2,
        });
        expect(runtime.latestLiveSignal).toBe(second);
        expect(Object.isFrozen(second)).toBe(true);
        expect(Object.isFrozen(second.model)).toBe(true);
        expect(Object.isFrozen(second.basis)).toBe(true);
    });

    it('evaluates train then test against one frozen revision and computes one penalty', () => {
        const { runtime, calls } = createRuntime();

        const evaluation = runtime.forceEvaluation('initial');

        expect(calls.map(({ split }) => split)).toEqual(['train', 'test', 'penalty']);
        expect(calls[0]?.model).toBe(calls[1]?.model);
        expect(calls[1]?.model).toBe(calls[2]?.model);
        expect(Object.isFrozen(calls[0]?.model)).toBe(true);
        expect(evaluation).toMatchObject({
            evaluationId: 1,
            trigger: 'initial',
            model: modelAt(0),
            train: {
                basis: { kind: 'full-split', split: 'train', sampleCount: 4, populationCount: 4 },
                values: { dataLoss: 0.4 },
            },
            test: {
                basis: { kind: 'full-split', split: 'test', sampleCount: 2, populationCount: 2 },
                values: { dataLoss: 0.6 },
            },
            objective: { regularizationPenalty: 0.1, trainTotalObjective: 0.5 },
        });
        expect(Object.isFrozen(evaluation)).toBe(true);
        expect(Object.isFrozen(evaluation.train)).toBe(true);
        expect(Object.isFrozen(evaluation.train.values)).toBe(true);
    });

    it('supports every forced trigger and assigns monotonically increasing evaluation IDs', () => {
        const fixture = createRuntime();
        const { runtime, modelState } = fixture;
        const evaluations = [runtime.forceEvaluation('initial')];
        recordAt(fixture, 1, 4, 0.5);
        evaluations.push(runtime.forceEvaluation('manual-step'));
        evaluations.push(runtime.forceEvaluation('pause'));
        evaluations.push(runtime.forceEvaluation('checkpoint'));
        evaluations.push(runtime.forceEvaluation('save'));
        evaluations.push(runtime.forceEvaluation('stop-condition'));
        modelState.current = modelAt(0, 2);
        evaluations.push(runtime.forceEvaluation('restore'));
        const triggers: readonly Exclude<EvaluationTrigger, 'cadence'>[] = [
            'initial', 'manual-step', 'pause', 'checkpoint', 'save', 'stop-condition', 'restore',
        ];

        expect(evaluations.map(({ trigger }) => trigger)).toEqual(triggers);
        expect(evaluations.map(({ evaluationId }) => evaluationId)).toEqual([1, 2, 3, 4, 5, 6, 7]);
        expect(runtime.latestEvaluation).toBe(evaluations[6]);
    });

    it('rejects non-finite batch data before replacing the latest published signal', () => {
        const fixture = createRuntime();
        const { runtime, modelState } = fixture;
        const valid = recordAt(fixture, 1, 4, 0.5);
        modelState.current = modelAt(2);

        expect(() => runtime.recordBatch({
            model: modelAt(2),
            batchSize: 4,
            dataLoss: Number.NaN,
        })).toThrow(TerminalDivergenceError);
        try {
            runtime.recordBatch({
                model: modelAt(2),
                batchSize: 4,
                dataLoss: Number.NaN,
            });
        } catch (error) {
            expect(error).toMatchObject({
                name: 'TerminalDivergenceError',
                path: '$.objective.dataLoss',
                value: Number.NaN,
            });
        }
        expect(runtime.latestLiveSignal).toBe(valid);
    });

    it('excludes invalid evaluations without consuming the ID or cadence opportunity', () => {
        const fixture = createRuntime();
        const { runtime, state } = fixture;
        recordAt(fixture, 50, 4, 0.5);
        state.test = { dataLoss: Number.POSITIVE_INFINITY };

        expect(() => runtime.takeCadenceEvaluation()).toThrow(TerminalDivergenceError);
        try {
            runtime.takeCadenceEvaluation();
        } catch (error) {
            expect(error).toMatchObject({
                name: 'TerminalDivergenceError',
                path: '$.evaluation.test.values.dataLoss',
                value: Number.POSITIVE_INFINITY,
            });
        }
        expect(runtime.latestEvaluation).toBeUndefined();

        state.test = { dataLoss: 0.6 };
        const recovered = runtime.takeCadenceEvaluation();
        expect(recovered?.evaluationId).toBe(1);
        expect(runtime.takeCadenceEvaluation()).toBeUndefined();
    });

    it.each([
        ['train data loss', { train: { dataLoss: Number.NaN } }, '$.evaluation.train.values.dataLoss'],
        ['test accuracy', { test: { dataLoss: 0.6, accuracy: Number.POSITIVE_INFINITY } }, '$.evaluation.test.values.accuracy'],
        ['regularization penalty', { penalty: Number.NEGATIVE_INFINITY }, '$.evaluation.objective.regularizationPenalty'],
    ] as const)(
        'classifies non-finite %s as terminal divergence before consuming publication state',
        (_label, overrides, path) => {
            const fixture = createRuntime(overrides);

            expect(() => fixture.runtime.forceEvaluation('initial')).toThrow(TerminalDivergenceError);
            try {
                fixture.runtime.forceEvaluation('initial');
            } catch (error) {
                expect(error).toMatchObject({
                    name: 'TerminalDivergenceError',
                    path,
                });
            }
            expect(fixture.runtime.latestEvaluation).toBeUndefined();

            fixture.state.train = { dataLoss: 0.4 };
            fixture.state.test = { dataLoss: 0.6 };
            fixture.state.penalty = 0.1;
            expect(fixture.runtime.forceEvaluation('initial').evaluationId).toBe(1);
        },
    );

    it('rejects a model from another generation and non-monotonic batch identities', () => {
        const fixture = createRuntime();
        const { runtime, modelState } = fixture;
        recordAt(fixture, 1, 4, 0.5);

        modelState.current = { ...modelAt(2), generationId: 8 };
        expect(() => runtime.recordBatch({
            model: { ...modelAt(2), generationId: 8 },
            batchSize: 4,
            dataLoss: 0.4,
        })).toThrow('generationId');
        modelState.current = modelAt(1, 2);
        expect(() => runtime.recordBatch({
            model: modelAt(1, 2),
            batchSize: 4,
            dataLoss: 0.4,
        })).toThrow('step');
        modelState.current = modelAt(2, 1);
        expect(() => runtime.recordBatch({
            model: modelAt(2, 1),
            batchSize: 4,
            dataLoss: 0.4,
        })).toThrow('revision');
    });

    it.each([
        ['stale', modelAt(0)],
        ['future', modelAt(2)],
    ])('rejects a %s current identity before a forced evaluation', (_label, candidate) => {
        const fixture = createRuntime();
        recordAt(fixture, 1, 4, 0.5);
        fixture.modelState.current = candidate;

        expect(() => fixture.runtime.forceEvaluation('pause')).toThrow('last observed model');
        expect(fixture.runtime.latestEvaluation).toBeUndefined();

        fixture.modelState.current = modelAt(1);
        expect(fixture.runtime.forceEvaluation('pause').evaluationId).toBe(1);
    });

    it.each([
        ['stale', modelAt(0)],
        ['future', modelAt(2)],
    ])('rejects publication if evaluation changes the model to a %s identity', (_label, changed) => {
        const fixture = createRuntime();
        recordAt(fixture, 1, 4, 0.5);
        fixture.state.modelAfterTrain = changed;

        expect(() => fixture.runtime.forceEvaluation('pause')).toThrow('changed during evaluation');
        expect(fixture.calls.map(({ split }) => split)).toEqual(['train']);
        expect(fixture.runtime.latestEvaluation).toBeUndefined();

        fixture.modelState.current = modelAt(1);
        fixture.state.modelAfterTrain = undefined;
        expect(fixture.runtime.forceEvaluation('pause').evaluationId).toBe(1);
    });

    it('blocks step 51 until the pending step-50 cadence evaluation is consumed', () => {
        const fixture = createRuntime();
        recordAt(fixture, 50, 4, 0.5);
        fixture.modelState.current = modelAt(51);

        expect(() => fixture.runtime.recordBatch({
            model: modelAt(51),
            batchSize: 4,
            dataLoss: 0.4,
        })).toThrow('pending cadence');
        expect(fixture.runtime.latestLiveSignal?.model.step).toBe(50);

        fixture.modelState.current = modelAt(50);
        expect(fixture.runtime.takeCadenceEvaluation()?.model.step).toBe(50);
        expect(recordAt(fixture, 51, 4, 0.4).model.step).toBe(51);
    });

    it('lets a forced restore pair supersede an older pending cadence revision', () => {
        const fixture = createRuntime();
        recordAt(fixture, 50, 4, 0.5);
        fixture.modelState.current = modelAt(40, 51);

        expect(fixture.runtime.forceEvaluation('restore')).toMatchObject({
            trigger: 'restore',
            model: modelAt(40, 51),
        });
        expect(recordAt(fixture, 41, 4, 0.4, 52).model).toEqual(modelAt(41, 52));
    });

    it('clears pre-restore live EMA so the first restored batch is exact', () => {
        const fixture = createRuntime({ alpha: 0.25 });
        expect(recordAt(fixture, 1, 4, 1).dataLoss).toBe(1);
        fixture.modelState.current = modelAt(0, 2);

        fixture.runtime.forceEvaluation('restore');
        expect(fixture.runtime.latestLiveSignal).toBeUndefined();

        const restored = recordAt(fixture, 1, 2, 0.2, 3);
        expect(restored.dataLoss).toBe(0.2);
        expect(restored.basis.latestBatchSize).toBe(2);
        expect(restored.basis.throughStep).toBe(1);
    });

    it('requires a valid fixed policy and EMA alpha at construction', () => {
        const base = {
            generationId: 7,
            dataset: DATASET,
            objectiveKey: 'objective-7',
            getCurrentModel: () => modelAt(0),
            evaluateTrain: () => ({ dataLoss: 0.4 }),
            evaluateTest: () => ({ dataLoss: 0.6 }),
            evaluateRegularizationPenalty: () => 0.1,
        };

        expect(() => new EvaluationRuntime({ ...base, emaAlpha: 0 })).toThrow('emaAlpha');
        expect(() => new EvaluationRuntime({
            ...base,
            emaAlpha: 0.1,
            policy: {
                everySteps: 0,
                forceOnPause: true,
                forceOnManualStep: true,
                forceOnCheckpoint: true,
                forceOnSave: true,
            },
        })).toThrow('everySteps');
    });
});
