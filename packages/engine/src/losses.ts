// ── Loss functions with output-layer gradients ──
import type { LossType } from './types.js';

export interface LossFn {
    /** Compute loss for a single sample. */
    loss: (predicted: number, target: number) => number;
    /** d(loss)/d(predicted) for backprop. */
    dloss: (predicted: number, target: number) => number;
}

const EPSILON = 1e-7;

const mse: LossFn = {
    loss: (p, t) => 0.5 * (p - t) ** 2,
    dloss: (p, t) => p - t,
};

const crossEntropy: LossFn = {
    loss: (p, t) => {
        const clamped = Math.max(EPSILON, Math.min(1 - EPSILON, p));
        return -(t * Math.log(clamped) + (1 - t) * Math.log(1 - clamped));
    },
    dloss: (p, t) => {
        const clamped = Math.max(EPSILON, Math.min(1 - EPSILON, p));
        return -(t / clamped) + (1 - t) / (1 - clamped);
    },
};

const HUBER_DELTA = 1.0;

const huber: LossFn = {
    loss: (p, t) => {
        const a = Math.abs(p - t);
        return a <= HUBER_DELTA
            ? 0.5 * a * a
            : HUBER_DELTA * (a - 0.5 * HUBER_DELTA);
    },
    dloss: (p, t) => {
        const diff = p - t;
        const a = Math.abs(diff);
        return a <= HUBER_DELTA ? diff : HUBER_DELTA * Math.sign(diff);
    },
};

const LOSSES: Record<LossType, LossFn> = { mse, crossEntropy, huber };

export function getLoss(type: LossType): LossFn {
    return LOSSES[type];
}

/** Compute mean loss over a batch. */
export function batchLoss(
    fn: LossFn,
    predictions: number[],
    targets: number[],
): number {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        sum += fn.loss(predictions[i], targets[i]);
    }
    return sum / predictions.length;
}

export const LOSS_LABELS: Record<LossType, string> = {
    mse: 'MSE (Squared)',
    crossEntropy: 'Cross-Entropy',
    huber: 'Huber',
};
