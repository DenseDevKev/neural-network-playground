// ── Learning-rate schedules ──
// Small, engine-local helpers to shape the learning rate over training steps.
// Consumed by Network.applyGradients to derive the effective LR each step.

export type LRScheduleType = 'constant' | 'step' | 'cosine';

export type LRSchedule =
    | { type: 'constant' }
    | {
        /** Multiply LR by `gamma` every `stepSize` steps. */
        type: 'step';
        stepSize: number;
        gamma: number;
    }
    | {
        /**
         * Cosine-anneal from the base LR down to `minLr` over `totalSteps`.
         * After `totalSteps`, the LR stays at `minLr`.
         */
        type: 'cosine';
        totalSteps: number;
        minLr: number;
    };

export interface LRScheduleSanitizeOptions {
    baseLearningRate?: number;
    fallbackStepSize?: number;
    fallbackGamma?: number;
    fallbackTotalSteps?: number;
    fallbackMinLr?: number;
}

function finiteOrFallback(value: unknown, fallback: number): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function positiveInteger(value: unknown, fallback: number): number {
    return Math.max(1, Math.trunc(finiteOrFallback(value, fallback)));
}

function unitIntervalExclusive(value: unknown, fallback: number): number {
    const n = finiteOrFallback(value, fallback);
    if (n <= 0) return fallback;
    if (n >= 1) return fallback;
    return n;
}

export function sanitizeLRSchedule(
    schedule: LRSchedule | undefined,
    options: LRScheduleSanitizeOptions = {},
): LRSchedule | undefined {
    if (!schedule || schedule.type === 'constant') return undefined;

    if (schedule.type === 'step') {
        return {
            type: 'step',
            stepSize: positiveInteger(schedule.stepSize, options.fallbackStepSize ?? 100),
            gamma: unitIntervalExclusive(schedule.gamma, options.fallbackGamma ?? 0.5),
        };
    }

    const totalSteps = positiveInteger(schedule.totalSteps, options.fallbackTotalSteps ?? 1000);
    const fallbackMinLr = Math.max(0, options.fallbackMinLr ?? 0);
    const rawMinLr = Math.max(0, finiteOrFallback(schedule.minLr, fallbackMinLr));
    const minLr = typeof options.baseLearningRate === 'number' && Number.isFinite(options.baseLearningRate)
        ? Math.min(rawMinLr, Math.max(0, options.baseLearningRate))
        : rawMinLr;

    return {
        type: 'cosine',
        totalSteps,
        minLr,
    };
}

export function validateLRSchedule(schedule: LRSchedule | undefined): void {
    if (!schedule || schedule.type === 'constant') return;
    if (schedule.type === 'step') {
        if (
            !Number.isFinite(schedule.stepSize) ||
            !Number.isInteger(schedule.stepSize) ||
            schedule.stepSize < 1 ||
            !Number.isFinite(schedule.gamma) ||
            schedule.gamma <= 0 ||
            schedule.gamma >= 1
        ) {
            throw new RangeError('lrSchedule step requires a positive integer stepSize and gamma between 0 and 1');
        }
        return;
    }

    if (
        !Number.isFinite(schedule.totalSteps) ||
        !Number.isInteger(schedule.totalSteps) ||
        schedule.totalSteps < 1 ||
        !Number.isFinite(schedule.minLr) ||
        schedule.minLr < 0
    ) {
        throw new RangeError('lrSchedule cosine requires positive integer totalSteps and non-negative minLr');
    }
}

/** Resolve the effective learning rate at a given step. */
export function computeLearningRate(baseLr: number, step: number, schedule?: LRSchedule): number {
    if (!schedule || schedule.type === 'constant') return baseLr;
    if (schedule.type === 'step') {
        if (schedule.stepSize <= 0) return baseLr;
        const decays = Math.floor(step / schedule.stepSize);
        return baseLr * Math.pow(schedule.gamma, decays);
    }
    // cosine
    if (schedule.totalSteps <= 0) return baseLr;
    const t = Math.min(step, schedule.totalSteps);
    const progress = t / schedule.totalSteps;
    const cos = 0.5 * (1 + Math.cos(Math.PI * progress));
    return schedule.minLr + (baseLr - schedule.minLr) * cos;
}
