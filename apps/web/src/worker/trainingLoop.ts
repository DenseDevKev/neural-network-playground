export const MIN_TRAINING_STEPS_PER_TICK = 1;
export const MAX_TRAINING_STEPS_PER_TICK = 100;

export interface MiniBatchScratch {
    inputs: number[][];
    targets: number[][];
}

export function createMiniBatchScratch(capacity: number): MiniBatchScratch {
    return {
        inputs: new Array<number[]>(capacity),
        targets: new Array<number[]>(capacity),
    };
}

export function fillMiniBatchScratch(
    scratch: MiniBatchScratch,
    inputs: number[][],
    targets: number[][],
    shuffledIndices: number[],
    startIdx: number,
    endIdx: number,
): MiniBatchScratch {
    const len = endIdx - startIdx;
    scratch.inputs.length = len;
    scratch.targets.length = len;
    for (let offset = 0; offset < len; offset++) {
        const sampleIdx = shuffledIndices[startIdx + offset];
        scratch.inputs[offset] = inputs[sampleIdx];
        scratch.targets[offset] = targets[sampleIdx];
    }
    return scratch;
}

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
