import type {
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
    PauseReason,
} from '@nn-playground/shared';

export type { PauseReason };

export type AutomaticPauseReason = Exclude<PauseReason, 'manual' | 'error'>;

/** `accuracy` always means the current paired test-split task accuracy. */
export type StopConditionMetric =
    | 'trainDataLoss'
    | 'testDataLoss'
    | 'trainObjective'
    | 'accuracy';

type LossStopConditionMetric = Exclude<StopConditionMetric, 'accuracy'>;

export type StopCondition =
    | { kind: 'target'; metric: StopConditionMetric; threshold: number }
    | {
        kind: 'plateau';
        metric: StopConditionMetric;
        minDelta: number;
        patienceSteps: number;
    }
    | {
        kind: 'divergence';
        metric?: LossStopConditionMetric;
        lossMultiplier?: number;
        patienceSteps?: number;
    }
    | { kind: 'maxSteps'; steps: number };

export const DEFAULT_RUNTIME_STOP_CONDITIONS: readonly StopCondition[] = [
    { kind: 'divergence' },
];

export interface StopConditionContext {
    readonly model: ModelRevision;
    readonly liveSignal?: LiveTrainingSignal | null;
    readonly currentEvaluation?: PairedEvaluation | null;
}

export interface StopConditionState {
    bestMetric: number | null;
    bestDivergenceMetric: number | null;
    plateauStartStep: number | null;
    divergenceStartStep: number | null;
}

export interface TerminalDivergence {
    readonly path: string;
    readonly value: number;
}

export interface StopConditionEvaluation {
    pauseReason: AutomaticPauseReason | null;
    nextState: StopConditionState;
    terminalDivergence: TerminalDivergence | null;
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
        bestMetric: null,
        bestDivergenceMetric: null,
        plateauStartStep: null,
        divergenceStartStep: null,
    };
}

function hasFiniteNumber(value: number | undefined): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function sameModel(left: ModelRevision, right: ModelRevision): boolean {
    return left.generationId === right.generationId
        && left.revision === right.revision
        && left.step === right.step
        && left.epoch === right.epoch;
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

export function stopConditionsRequireCurrentEvaluation(
    conditions: readonly StopCondition[],
): boolean {
    return conditions.some((condition) => (
        condition.kind === 'target'
        || condition.kind === 'plateau'
        || (condition.kind === 'divergence' && condition.lossMultiplier !== undefined)
    ));
}

function requireCurrentEvaluation(context: StopConditionContext): PairedEvaluation {
    const evaluation = context.currentEvaluation;
    if (!evaluation || !sameModel(evaluation.model, context.model)) {
        throw new Error('comparison stop conditions require a current PairedEvaluation');
    }
    return evaluation;
}

function metricValue(
    metric: StopConditionMetric,
    context: StopConditionContext,
): number | undefined {
    const evaluation = requireCurrentEvaluation(context);
    switch (metric) {
        case 'trainDataLoss':
            return evaluation.train.values.dataLoss;
        case 'testDataLoss':
            return evaluation.test.values.dataLoss;
        case 'trainObjective':
            return evaluation.objective.trainTotalObjective;
        case 'accuracy':
            return evaluation.test.values.accuracy;
    }
}

function firstNonFinite(context: StopConditionContext): TerminalDivergence | null {
    const candidates: Array<readonly [string, number | undefined]> = [
        ['liveSignal.dataLoss', context.liveSignal?.dataLoss],
        ['currentEvaluation.train.values.dataLoss', context.currentEvaluation?.train.values.dataLoss],
        ['currentEvaluation.test.values.dataLoss', context.currentEvaluation?.test.values.dataLoss],
        [
            'currentEvaluation.objective.regularizationPenalty',
            context.currentEvaluation?.objective.regularizationPenalty,
        ],
        [
            'currentEvaluation.objective.trainTotalObjective',
            context.currentEvaluation?.objective.trainTotalObjective,
        ],
        ['currentEvaluation.train.values.accuracy', context.currentEvaluation?.train.values.accuracy],
        ['currentEvaluation.test.values.accuracy', context.currentEvaluation?.test.values.accuracy],
    ];
    for (const [path, value] of candidates) {
        if (value !== undefined && !Number.isFinite(value)) return { path, value };
    }
    return null;
}

function improved(
    metric: StopConditionMetric,
    best: number | null,
    current: number,
    minDelta: number,
): boolean {
    if (best === null) return true;
    return metric === 'accuracy'
        ? current > best + minDelta
        : current < best - minDelta;
}

function applyPlateauCondition(
    condition: Extract<StopCondition, { kind: 'plateau' }>,
    context: StopConditionContext,
    state: StopConditionState,
    nextState: StopConditionState,
    reasons: Set<AutomaticPauseReason>,
): void {
    const value = metricValue(condition.metric, context);
    if (!hasFiniteNumber(value)) return;

    const minDelta = Math.max(0, condition.minDelta);
    if (improved(condition.metric, state.bestMetric, value, minDelta)) {
        nextState.bestMetric = value;
        nextState.plateauStartStep = null;
        return;
    }

    const patienceSteps = Math.max(0, Math.trunc(condition.patienceSteps));
    const startStep = state.plateauStartStep ?? context.model.step;
    nextState.plateauStartStep = startStep;
    if (context.model.step - startStep >= patienceSteps) {
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
    const multiplier = condition.lossMultiplier;
    if (!hasFiniteNumber(multiplier) || multiplier <= 1) {
        nextState.divergenceStartStep = null;
        return;
    }

    const metric = condition.metric ?? 'testDataLoss';
    const value = metricValue(metric, context);
    if (!hasFiniteNumber(value)) return;

    if (state.bestDivergenceMetric === null || value < state.bestDivergenceMetric) {
        nextState.bestDivergenceMetric = value;
    }
    if (
        state.bestDivergenceMetric === null
        || value <= state.bestDivergenceMetric * multiplier
    ) {
        nextState.divergenceStartStep = null;
        return;
    }

    const patienceSteps = Math.max(0, Math.trunc(condition.patienceSteps ?? 0));
    const startStep = state.divergenceStartStep ?? context.model.step;
    nextState.divergenceStartStep = startStep;
    if (context.model.step - startStep >= patienceSteps) {
        addReason(reasons, 'diverged');
    }
}

export function evaluateStopConditions(
    conditions: readonly StopCondition[],
    context: StopConditionContext,
    state: StopConditionState = createInitialStopConditionState(),
): StopConditionEvaluation {
    assertSupportedConditionMix(conditions);
    if (stopConditionsRequireCurrentEvaluation(conditions)) requireCurrentEvaluation(context);

    const reasons = new Set<AutomaticPauseReason>();
    const nextState: StopConditionState = { ...state };
    const terminalDivergence = firstNonFinite(context);
    if (terminalDivergence) addReason(reasons, 'diverged');

    for (const condition of conditions) {
        switch (condition.kind) {
            case 'target': {
                const value = metricValue(condition.metric, context);
                if (!hasFiniteNumber(value)) break;
                const reached = condition.metric === 'accuracy'
                    ? value >= condition.threshold
                    : value <= condition.threshold;
                if (reached) {
                    addReason(
                        reasons,
                        condition.metric === 'accuracy'
                            ? 'target-accuracy-reached'
                            : 'target-loss-reached',
                    );
                }
                break;
            }
            case 'plateau':
                applyPlateauCondition(condition, context, state, nextState, reasons);
                break;
            case 'divergence':
                applyDivergenceCondition(condition, context, state, nextState, reasons);
                break;
            case 'maxSteps':
                if (context.model.step >= condition.steps) {
                    addReason(reasons, 'max-steps');
                }
                break;
        }
    }

    return {
        pauseReason: pickPauseReason(reasons),
        nextState,
        terminalDivergence,
    };
}
