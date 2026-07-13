import type {
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
} from '@nn-playground/shared';
import { beforeAll, describe, expect, it } from 'vitest';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';
import {
    createInitialStopConditionState,
    DEFAULT_RUNTIME_STOP_CONDITIONS,
    evaluateStopConditions,
    PAUSE_REASON_PRIORITY,
    stopConditionsRequireCurrentEvaluation,
    type StopCondition,
    type StopConditionContext,
    type StopConditionMetric,
    type StopConditionState,
} from './stopConditions.ts';

let fixtureEvaluation: PairedEvaluation;
let fixtureLiveSignal: LiveTrainingSignal;

beforeAll(async () => {
    const fixtures = await createScientificTrustFixtures();
    fixtureEvaluation = fixtures.evaluation;
    fixtureLiveSignal = fixtures.liveSignal;
});

interface MetricOverrides {
    trainDataLoss?: number;
    testDataLoss?: number;
    regularizationPenalty?: number;
    trainObjective?: number;
    trainAccuracy?: number;
    testAccuracy?: number;
}

function modelAt(step: number): ModelRevision {
    return {
        generationId: 7,
        revision: 100 + step,
        step,
        epoch: step / 10,
    };
}

function evaluationAt(step: number, overrides: MetricOverrides = {}): PairedEvaluation {
    const model = modelAt(step);
    const trainDataLoss = overrides.trainDataLoss ?? 0.4;
    const regularizationPenalty = overrides.regularizationPenalty ?? 0.05;
    return {
        ...fixtureEvaluation,
        evaluationId: step + 1,
        trigger: 'stop-condition',
        model,
        train: {
            ...fixtureEvaluation.train,
            values: {
                dataLoss: trainDataLoss,
                accuracy: overrides.trainAccuracy ?? 0.8,
            },
        },
        test: {
            ...fixtureEvaluation.test,
            values: {
                dataLoss: overrides.testDataLoss ?? 0.35,
                accuracy: overrides.testAccuracy ?? 0.75,
            },
        },
        objective: {
            regularizationPenalty,
            trainTotalObjective:
                overrides.trainObjective ?? trainDataLoss + regularizationPenalty,
        },
    };
}

function contextAt(step: number, overrides: MetricOverrides = {}): StopConditionContext {
    const model = modelAt(step);
    return {
        model,
        liveSignal: {
            ...fixtureLiveSignal,
            model,
            basis: {
                ...fixtureLiveSignal.basis,
                throughStep: step,
            },
            dataLoss: overrides.trainDataLoss ?? 0.4,
        },
        currentEvaluation: evaluationAt(step, overrides),
    };
}

function evaluateOnce(
    conditions: readonly StopCondition[],
    context: StopConditionContext,
    state: StopConditionState = createInitialStopConditionState(),
) {
    return evaluateStopConditions(conditions, context, state);
}

describe('basis-explicit stop condition evaluator', () => {
    it.each<{
        metric: StopConditionMetric;
        overrides: MetricOverrides;
        threshold: number;
        reason: 'target-loss-reached' | 'target-accuracy-reached';
    }>([
        {
            metric: 'trainDataLoss',
            overrides: { trainDataLoss: 0.19 },
            threshold: 0.2,
            reason: 'target-loss-reached',
        },
        {
            metric: 'testDataLoss',
            overrides: { testDataLoss: 0.19 },
            threshold: 0.2,
            reason: 'target-loss-reached',
        },
        {
            metric: 'trainObjective',
            overrides: { trainDataLoss: 0.12, regularizationPenalty: 0.02 },
            threshold: 0.15,
            reason: 'target-loss-reached',
        },
        {
            metric: 'accuracy',
            overrides: { testAccuracy: 0.91 },
            threshold: 0.9,
            reason: 'target-accuracy-reached',
        },
    ])('evaluates target $metric from the current paired evaluation', ({
        metric,
        overrides,
        threshold,
        reason,
    }) => {
        const result = evaluateOnce([
            { kind: 'target', metric, threshold },
        ], contextAt(10, overrides));

        expect(result.pauseReason).toBe(reason);
    });

    it('defines accuracy as paired test-split task accuracy', () => {
        const result = evaluateOnce([
            { kind: 'target', metric: 'accuracy', threshold: 0.9 },
        ], contextAt(10, { trainAccuracy: 0.99, testAccuracy: 0.6 }));

        expect(result.pauseReason).toBeNull();
    });

    it('requires an exact current model pair for every comparison-sensitive condition', () => {
        const current = contextAt(10);
        const stale = {
            ...current,
            currentEvaluation: evaluationAt(9),
        };
        const comparisons: readonly StopCondition[][] = [
            [{ kind: 'target', metric: 'testDataLoss', threshold: 0.2 }],
            [{ kind: 'plateau', metric: 'accuracy', minDelta: 0.01, patienceSteps: 3 }],
            [{ kind: 'divergence', metric: 'trainObjective', lossMultiplier: 2 }],
        ];

        for (const conditions of comparisons) {
            expect(() => evaluateOnce(conditions, {
                ...current,
                currentEvaluation: null,
            })).toThrow('current PairedEvaluation');
            expect(() => evaluateOnce(conditions, stale)).toThrow('current PairedEvaluation');
        }
    });

    it('advertises exactly which condition sets require a forced current evaluation', () => {
        expect(stopConditionsRequireCurrentEvaluation([
            { kind: 'maxSteps', steps: 50 },
            { kind: 'divergence' },
        ])).toBe(false);
        expect(stopConditionsRequireCurrentEvaluation([
            { kind: 'target', metric: 'trainObjective', threshold: 0.2 },
        ])).toBe(true);
        expect(stopConditionsRequireCurrentEvaluation([
            { kind: 'plateau', metric: 'accuracy', minDelta: 0.01, patienceSteps: 3 },
        ])).toBe(true);
        expect(stopConditionsRequireCurrentEvaluation([
            { kind: 'divergence', lossMultiplier: 2 },
        ])).toBe(true);
    });

    it('tracks plateau windows against the selected paired metric', () => {
        const condition: StopCondition = {
            kind: 'plateau',
            metric: 'trainObjective',
            minDelta: 0.01,
            patienceSteps: 3,
        };
        const first = evaluateOnce([condition], contextAt(1, { trainObjective: 0.5 }));
        const second = evaluateOnce(
            [condition],
            contextAt(2, { trainObjective: 0.495 }),
            first.nextState,
        );
        const third = evaluateOnce(
            [condition],
            contextAt(5, { trainObjective: 0.494 }),
            second.nextState,
        );

        expect(first.pauseReason).toBeNull();
        expect(second.pauseReason).toBeNull();
        expect(third.pauseReason).toBe('plateau');
    });

    it('treats zero as a valid divergence baseline', () => {
        const condition: StopCondition = {
            kind: 'divergence',
            metric: 'testDataLoss',
            lossMultiplier: 2,
        };
        const baseline = evaluateOnce(
            [condition],
            contextAt(1, { testDataLoss: 0 }),
        );
        const increased = evaluateOnce(
            [condition],
            contextAt(2, { testDataLoss: 0.01 }),
            baseline.nextState,
        );

        expect(baseline.nextState.bestDivergenceMetric).toBe(0);
        expect(increased.pauseReason).toBe('diverged');
    });

    it('treats every non-finite live or evaluated value as unconditional terminal divergence', () => {
        const cases: Array<readonly [string, StopConditionContext]> = [
            [
                'liveSignal.dataLoss',
                {
                    ...contextAt(10),
                    liveSignal: {
                        ...contextAt(10).liveSignal!,
                        dataLoss: Number.NaN,
                    },
                },
            ],
            [
                'currentEvaluation.objective.trainTotalObjective',
                {
                    ...contextAt(11),
                    currentEvaluation: {
                        ...contextAt(11).currentEvaluation!,
                        objective: {
                            ...contextAt(11).currentEvaluation!.objective,
                            trainTotalObjective: Number.POSITIVE_INFINITY,
                        },
                    },
                },
            ],
            [
                'currentEvaluation.test.values.accuracy',
                {
                    ...contextAt(12),
                    currentEvaluation: {
                        ...contextAt(12).currentEvaluation!,
                        test: {
                            ...contextAt(12).currentEvaluation!.test,
                            values: {
                                ...contextAt(12).currentEvaluation!.test.values,
                                accuracy: Number.NEGATIVE_INFINITY,
                            },
                        },
                    },
                },
            ],
        ];

        for (const [path, context] of cases) {
            const result = evaluateOnce([], context);
            expect(result.pauseReason).toBe('diverged');
            expect(result.terminalDivergence).toEqual({
                path,
                value: expect.any(Number),
            });
        }
    });

    it('preserves maxSteps as an inclusive model-step condition without requiring a pair', () => {
        const result = evaluateOnce([
            { kind: 'maxSteps', steps: 50 },
        ], {
            model: modelAt(50),
            liveSignal: null,
            currentEvaluation: null,
        });

        expect(result.pauseReason).toBe('max-steps');
        expect(result.terminalDivergence).toBeNull();
    });

    it('uses deterministic priority when multiple conditions trigger', () => {
        expect(PAUSE_REASON_PRIORITY).toEqual([
            'diverged',
            'max-steps',
            'target-loss-reached',
            'target-accuracy-reached',
            'plateau',
        ]);
        const context = contextAt(20, { testDataLoss: 0.1, testAccuracy: 0.95 });
        const divergent = {
            ...context,
            liveSignal: {
                ...context.liveSignal!,
                dataLoss: Number.NaN,
            },
        };
        const result = evaluateOnce([
            { kind: 'target', metric: 'testDataLoss', threshold: 0.2 },
            { kind: 'target', metric: 'accuracy', threshold: 0.9 },
            { kind: 'maxSteps', steps: 20 },
        ], divergent);

        expect(result.pauseReason).toBe('diverged');
    });

    it('rejects duplicate stateful condition windows', () => {
        expect(() => evaluateOnce([
            { kind: 'plateau', metric: 'testDataLoss', minDelta: 0.01, patienceSteps: 3 },
            { kind: 'plateau', metric: 'accuracy', minDelta: 0.01, patienceSteps: 3 },
        ], contextAt(10))).toThrow('Only one plateau');
        expect(() => evaluateOnce([
            { kind: 'divergence', lossMultiplier: 2 },
            { kind: 'divergence', lossMultiplier: 3 },
        ], contextAt(10))).toThrow('Only one divergence');
    });

    it('creates fresh comparison state at reset boundaries', () => {
        expect(createInitialStopConditionState()).toEqual({
            bestMetric: null,
            bestDivergenceMetric: null,
            plateauStartStep: null,
            divergenceStartStep: null,
        });
    });

    it('keeps the default condition declarative while non-finite values stay unconditional', () => {
        expect(DEFAULT_RUNTIME_STOP_CONDITIONS).toEqual([
            { kind: 'divergence' },
        ]);
        expect(evaluateOnce(DEFAULT_RUNTIME_STOP_CONDITIONS, contextAt(1)).pauseReason).toBeNull();
    });
});
