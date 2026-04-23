export const MIN_TRAINING_STEPS_PER_TICK = 1;
export const MAX_TRAINING_STEPS_PER_TICK = 100;

export function normalizeTrainingSpeed(value: number): number {
    if (!Number.isFinite(value)) {
        return MIN_TRAINING_STEPS_PER_TICK;
    }
    return Math.max(
        MIN_TRAINING_STEPS_PER_TICK,
        Math.min(MAX_TRAINING_STEPS_PER_TICK, Math.trunc(value)),
    );
}

export function getTrainingStepsForTick(stepsPerFrame: number): number {
    return normalizeTrainingSpeed(stepsPerFrame);
}
