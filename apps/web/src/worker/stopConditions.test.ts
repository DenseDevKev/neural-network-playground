import { describe, expect, it } from 'vitest';
import {
    createInitialStopConditionState,
    DEFAULT_RUNTIME_STOP_CONDITIONS,
    evaluateStopConditions,
    PAUSE_REASON_PRIORITY,
    type StopCondition,
    type StopConditionContext,
    type StopConditionState,
} from './stopConditions.ts';

const baseContext: StopConditionContext = {
    step: 10,
    trainLoss: 0.4,
    testLoss: 0.35,
    trainAccuracy: 0.8,
    testAccuracy: 0.75,
};

function evaluateOnce(
    conditions: StopCondition[],
    context: Partial<StopConditionContext>,
    state: StopConditionState = createInitialStopConditionState(),
) {
    return evaluateStopConditions(conditions, { ...baseContext, ...context }, state);
}

describe('stop condition evaluator', () => {
    it('pauses when test loss reaches the target threshold', () => {
        const result = evaluateOnce([
            { kind: 'targetLoss', threshold: 0.2 },
        ], { testLoss: 0.19 });

        expect(result.pauseReason).toBe('target-loss-reached');
    });

    it('pauses when test accuracy reaches the target threshold', () => {
        const result = evaluateOnce([
            { kind: 'targetAccuracy', threshold: 0.9 },
        ], { testAccuracy: 0.91 });

        expect(result.pauseReason).toBe('target-accuracy-reached');
    });

    it('ignores target accuracy when accuracy is unavailable', () => {
        const result = evaluateOnce([
            { kind: 'targetAccuracy', threshold: 0.9 },
        ], { testAccuracy: undefined });

        expect(result.pauseReason).toBeNull();
    });

    it('waits for enough non-improving loss steps before pausing for plateau', () => {
        const condition: StopCondition = {
            kind: 'plateau',
            metric: 'loss',
            minDelta: 0.01,
            patienceSteps: 3,
        };

        const first = evaluateOnce([condition], { step: 1, testLoss: 0.5 });
        const second = evaluateOnce([condition], { step: 2, testLoss: 0.495 }, first.nextState);
        const third = evaluateOnce([condition], { step: 4, testLoss: 0.494 }, second.nextState);
        const fourth = evaluateOnce([condition], { step: 5, testLoss: 0.493 }, third.nextState);

        expect(first.pauseReason).toBeNull();
        expect(second.pauseReason).toBeNull();
        expect(third.pauseReason).toBeNull();
        expect(fourth.pauseReason).toBe('plateau');
    });

    it('rejects multiple plateau conditions because plateau state is single-window', () => {
        expect(() => evaluateOnce([
            { kind: 'plateau', metric: 'loss', minDelta: 0.01, patienceSteps: 3 },
            { kind: 'plateau', metric: 'accuracy', minDelta: 0.01, patienceSteps: 3 },
        ], {})).toThrow('Only one plateau stop condition is supported.');
    });

    it('pauses immediately when divergence sees NaN loss', () => {
        const result = evaluateOnce([
            { kind: 'divergence', nanOrInfinity: true, patienceSteps: 5 },
        ], { testLoss: Number.NaN });

        expect(result.pauseReason).toBe('diverged');
    });

    it('pauses immediately when divergence sees infinite loss', () => {
        const result = evaluateOnce([
            { kind: 'divergence', nanOrInfinity: true, patienceSteps: 5 },
        ], { trainLoss: Number.POSITIVE_INFINITY });

        expect(result.pauseReason).toBe('diverged');
    });

    it('pauses when loss rises beyond the configured divergence multiplier', () => {
        const condition: StopCondition = {
            kind: 'divergence',
            lossMultiplier: 2,
        };

        const first = evaluateOnce([condition], { step: 1, testLoss: 0.4 });
        const second = evaluateOnce([condition], { step: 2, testLoss: 0.81 }, first.nextState);

        expect(first.pauseReason).toBeNull();
        expect(second.pauseReason).toBe('diverged');
    });

    it('rejects multiple divergence conditions because divergence state is single-window', () => {
        expect(() => evaluateOnce([
            { kind: 'divergence', lossMultiplier: 2 },
            { kind: 'divergence', nanOrInfinity: true },
        ], {})).toThrow('Only one divergence stop condition is supported.');
    });

    it('keeps divergence tracking separate from loss plateau tracking', () => {
        const conditions: StopCondition[] = [
            { kind: 'plateau', metric: 'loss', minDelta: 0.1, patienceSteps: 2 },
            { kind: 'divergence', lossMultiplier: 2 },
        ];

        const first = evaluateOnce(conditions, { step: 1, testLoss: 1 });
        const second = evaluateOnce(conditions, { step: 2, testLoss: 0.95 }, first.nextState);
        const third = evaluateOnce(conditions, { step: 3, testLoss: 0.85 }, second.nextState);

        expect(second.nextState.bestLoss).toBe(1);
        expect(second.nextState.bestDivergenceLoss).toBe(0.95);
        expect(third.pauseReason).toBeNull();
        expect(third.nextState.bestLoss).toBe(0.85);
    });

    it('ignores divergence multipliers that are not greater than 1', () => {
        const condition: StopCondition = {
            kind: 'divergence',
            lossMultiplier: 0.5,
        };

        const first = evaluateOnce([condition], { step: 1, testLoss: 1 });
        const second = evaluateOnce([condition], { step: 2, testLoss: 0.8 }, first.nextState);

        expect(second.pauseReason).toBeNull();
    });

    it('pauses when the maximum step is reached', () => {
        const result = evaluateOnce([
            { kind: 'maxSteps', steps: 50 },
        ], { step: 50 });

        expect(result.pauseReason).toBe('max-steps');
    });

    it('uses deterministic priority when multiple conditions trigger', () => {
        expect(PAUSE_REASON_PRIORITY).toEqual([
            'diverged',
            'max-steps',
            'target-loss-reached',
            'target-accuracy-reached',
            'plateau',
        ]);

        const plateauState: StopConditionState = {
            bestLoss: 0.4,
            bestAccuracy: null,
            bestDivergenceLoss: null,
            plateauStartStep: 1,
            divergenceStartStep: null,
        };

        const allReasons = evaluateOnce([
            { kind: 'targetLoss', threshold: 0.5 },
            { kind: 'targetAccuracy', threshold: 0.7 },
            { kind: 'maxSteps', steps: 10 },
            { kind: 'divergence', nanOrInfinity: true },
            { kind: 'plateau', metric: 'loss', minDelta: 0.01, patienceSteps: 1 },
        ], { testLoss: Number.NaN, testAccuracy: 0.95, step: 20 }, plateauState);
        const withoutDivergence = evaluateOnce([
            { kind: 'targetLoss', threshold: 0.5 },
            { kind: 'targetAccuracy', threshold: 0.7 },
            { kind: 'maxSteps', steps: 10 },
            { kind: 'plateau', metric: 'loss', minDelta: 0.01, patienceSteps: 1 },
        ], { testLoss: 0.3, testAccuracy: 0.95, step: 20 }, plateauState);
        const withoutMaxSteps = evaluateOnce([
            { kind: 'targetLoss', threshold: 0.5 },
            { kind: 'targetAccuracy', threshold: 0.7 },
            { kind: 'plateau', metric: 'loss', minDelta: 0.01, patienceSteps: 1 },
        ], { testLoss: 0.3, testAccuracy: 0.95, step: 20 }, plateauState);
        const withoutTargetLoss = evaluateOnce([
            { kind: 'targetAccuracy', threshold: 0.7 },
            { kind: 'plateau', metric: 'loss', minDelta: 0.01, patienceSteps: 1 },
        ], { testLoss: 0.4, testAccuracy: 0.95, step: 20 }, plateauState);

        expect(allReasons.pauseReason).toBe('diverged');
        expect(withoutDivergence.pauseReason).toBe('max-steps');
        expect(withoutMaxSteps.pauseReason).toBe('target-loss-reached');
        expect(withoutTargetLoss.pauseReason).toBe('target-accuracy-reached');
    });

    it('creates a fresh empty state for reset boundaries', () => {
        expect(createInitialStopConditionState()).toEqual({
            bestLoss: null,
            bestAccuracy: null,
            bestDivergenceLoss: null,
            plateauStartStep: null,
            divergenceStartStep: null,
        });
    });

    it('keeps P0B default runtime stop conditions limited to non-finite divergence', () => {
        expect(DEFAULT_RUNTIME_STOP_CONDITIONS).toEqual([
            { kind: 'divergence', nanOrInfinity: true },
        ]);

        const finite = evaluateOnce(DEFAULT_RUNTIME_STOP_CONDITIONS, {
            trainLoss: 0.4,
            testLoss: 0.5,
            step: 1,
        });
        const nonFiniteTrain = evaluateOnce(DEFAULT_RUNTIME_STOP_CONDITIONS, {
            trainLoss: Number.NaN,
            testLoss: 0.5,
            step: 2,
        });
        const nonFiniteTest = evaluateOnce(DEFAULT_RUNTIME_STOP_CONDITIONS, {
            trainLoss: 0.4,
            testLoss: Number.POSITIVE_INFINITY,
            step: 3,
        });

        expect(finite.pauseReason).toBeNull();
        expect(nonFiniteTrain.pauseReason).toBe('diverged');
        expect(nonFiniteTest.pauseReason).toBe('diverged');
    });
});
