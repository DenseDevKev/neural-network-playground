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
