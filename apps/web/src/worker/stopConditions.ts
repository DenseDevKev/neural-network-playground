import type { PauseReason } from '@nn-playground/shared';

export type { PauseReason };

export type AutomaticPauseReason = Exclude<PauseReason, 'manual' | 'error'>;

export type StopCondition =
    | { kind: 'targetLoss'; threshold: number }
    | { kind: 'targetAccuracy'; threshold: number }
    | { kind: 'plateau'; metric: 'loss' | 'accuracy'; minDelta: number; patienceSteps: number }
    | { kind: 'divergence'; lossMultiplier?: number; nanOrInfinity?: boolean; patienceSteps?: number }
    | { kind: 'maxSteps'; steps: number };

export const DEFAULT_RUNTIME_STOP_CONDITIONS: readonly StopCondition[] = [
    { kind: 'divergence', nanOrInfinity: true },
];

export interface StopConditionContext {
    step: number;
    trainLoss: number;
    testLoss: number;
    trainAccuracy?: number;
    testAccuracy?: number;
}

export interface StopConditionState {
    bestLoss: number | null;
    bestAccuracy: number | null;
    bestDivergenceLoss: number | null;
    plateauStartStep: number | null;
    divergenceStartStep: number | null;
}

export interface StopConditionEvaluation {
    pauseReason: AutomaticPauseReason | null;
    nextState: StopConditionState;
}

export const PAUSE_REASON_PRIORITY: AutomaticPauseReason[] = [
    'diverged',
    'max-steps',
    'target-loss-reached',
    'target-accuracy-reached',
    'plateau',
];

export function createInitialStopConditionState(): StopConditionState {
    return {
        bestLoss: null,
        bestAccuracy: null,
        bestDivergenceLoss: null,
        plateauStartStep: null,
        divergenceStartStep: null,
    };
}

function hasFiniteNumber(value: number | undefined): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function addReason(reasons: Set<AutomaticPauseReason>, reason: AutomaticPauseReason): void {
    reasons.add(reason);
}

function pickPauseReason(reasons: Set<AutomaticPauseReason>): AutomaticPauseReason | null {
    for (const reason of PAUSE_REASON_PRIORITY) {
        if (reasons.has(reason)) return reason;
    }
    return null;
}

function assertSupportedConditionMix(conditions: readonly StopCondition[]): void {
    let plateauCount = 0;
    let divergenceCount = 0;
    for (const condition of conditions) {
        if (condition.kind === 'plateau') plateauCount++;
        if (condition.kind === 'divergence') divergenceCount++;
    }
    if (plateauCount > 1) {
        throw new RangeError('Only one plateau stop condition is supported.');
    }
    if (divergenceCount > 1) {
        throw new RangeError('Only one divergence stop condition is supported.');
    }
}

function hasNonFiniteLoss(context: StopConditionContext): boolean {
    return !Number.isFinite(context.trainLoss) || !Number.isFinite(context.testLoss);
}

function lossImproved(
    bestLoss: number | null,
    currentLoss: number,
    minDelta: number,
): boolean {
    return bestLoss === null || currentLoss < bestLoss - minDelta;
}

function accuracyImproved(
    bestAccuracy: number | null,
    currentAccuracy: number,
    minDelta: number,
): boolean {
    return bestAccuracy === null || currentAccuracy > bestAccuracy + minDelta;
}

function applyPlateauCondition(
    condition: Extract<StopCondition, { kind: 'plateau' }>,
    context: StopConditionContext,
    state: StopConditionState,
    nextState: StopConditionState,
    reasons: Set<AutomaticPauseReason>,
): void {
    const value = condition.metric === 'loss' ? context.testLoss : context.testAccuracy;
    if (!hasFiniteNumber(value)) return;

    const patienceSteps = Math.max(0, Math.trunc(condition.patienceSteps));
    const minDelta = Math.max(0, condition.minDelta);
    const improved = condition.metric === 'loss'
        ? lossImproved(state.bestLoss, value, minDelta)
        : accuracyImproved(state.bestAccuracy, value, minDelta);

    if (improved) {
        if (condition.metric === 'loss') nextState.bestLoss = value;
        else nextState.bestAccuracy = value;
        nextState.plateauStartStep = null;
        return;
    }

    const startStep = state.plateauStartStep ?? context.step;
    nextState.plateauStartStep = startStep;
    if (context.step - startStep >= patienceSteps) {
        addReason(reasons, 'plateau');
    }
}

function applyDivergenceCondition(
    condition: Extract<StopCondition, { kind: 'divergence' }>,
    context: StopConditionContext,
    state: StopConditionState,
    nextState: StopConditionState,
    reasons: Set<AutomaticPauseReason>,
): void {
    if (condition.nanOrInfinity !== false && hasNonFiniteLoss(context)) {
        addReason(reasons, 'diverged');
        return;
    }

    const multiplier = condition.lossMultiplier;
    if (Number.isFinite(context.testLoss)) {
        if (state.bestDivergenceLoss === null || context.testLoss < state.bestDivergenceLoss) {
            nextState.bestDivergenceLoss = context.testLoss;
        }
    }

    if (
        !hasFiniteNumber(multiplier) ||
        multiplier <= 1 ||
        state.bestDivergenceLoss === null ||
        state.bestDivergenceLoss <= 0 ||
        !Number.isFinite(context.testLoss) ||
        context.testLoss <= state.bestDivergenceLoss * multiplier
    ) {
        nextState.divergenceStartStep = null;
        return;
    }

    const patienceSteps = Math.max(0, Math.trunc(condition.patienceSteps ?? 0));
    const startStep = state.divergenceStartStep ?? context.step;
    nextState.divergenceStartStep = startStep;
    if (context.step - startStep >= patienceSteps) {
        addReason(reasons, 'diverged');
    }
}

export function evaluateStopConditions(
    conditions: readonly StopCondition[],
    context: StopConditionContext,
    state: StopConditionState = createInitialStopConditionState(),
): StopConditionEvaluation {
    assertSupportedConditionMix(conditions);

    const reasons = new Set<AutomaticPauseReason>();
    const nextState: StopConditionState = { ...state };

    for (const condition of conditions) {
        switch (condition.kind) {
            case 'targetLoss':
                if (Number.isFinite(context.testLoss) && context.testLoss <= condition.threshold) {
                    addReason(reasons, 'target-loss-reached');
                }
                break;

            case 'targetAccuracy':
                if (
                    hasFiniteNumber(context.testAccuracy) &&
                    context.testAccuracy >= condition.threshold
                ) {
                    addReason(reasons, 'target-accuracy-reached');
                }
                break;

            case 'plateau':
                applyPlateauCondition(condition, context, state, nextState, reasons);
                break;

            case 'divergence':
                applyDivergenceCondition(condition, context, state, nextState, reasons);
                break;

            case 'maxSteps':
                if (context.step >= condition.steps) {
                    addReason(reasons, 'max-steps');
                }
                break;
        }
    }

    return {
        pauseReason: pickPauseReason(reasons),
        nextState,
    };
}
