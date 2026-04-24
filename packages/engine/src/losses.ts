// ── Loss functions with output-layer gradients ──
import type { LossType, ActivationType } from './types.js';

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

export const DEFAULT_HUBER_DELTA = 1.0;

/** Build a Huber loss with a configurable transition point δ. */
function makeHuber(delta: number): LossFn {
    if (!Number.isFinite(delta) || delta <= 0) {
        throw new RangeError('huberDelta must be finite and greater than 0');
    }
    return {
        loss: (p, t) => {
            const a = Math.abs(p - t);
            return a <= delta
                ? 0.5 * a * a
                : delta * (a - 0.5 * delta);
        },
        dloss: (p, t) => {
            const diff = p - t;
            const a = Math.abs(diff);
            return a <= delta ? diff : delta * Math.sign(diff);
        },
    };
}

const DEFAULT_HUBER = makeHuber(DEFAULT_HUBER_DELTA);
const LOSSES: Record<LossType, LossFn> = { mse, crossEntropy, huber: DEFAULT_HUBER };

/**
 * Resolve a loss type to its function. For Huber, an optional `huberDelta`
 * produces a delta-configured LossFn; if omitted, the module default is used.
 */
export function getLoss(type: LossType, opts?: { huberDelta?: number }): LossFn {
    if (type === 'huber' && opts?.huberDelta != null) {
        if (!Number.isFinite(opts.huberDelta) || opts.huberDelta <= 0) {
            throw new RangeError('huberDelta must be finite and greater than 0');
        }
        if (opts.huberDelta !== DEFAULT_HUBER_DELTA) {
            return makeHuber(opts.huberDelta);
        }
    }
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

/**
 * Output activations that are compatible with each loss. The engine computes
 * cross-entropy gradients via d/dp = -(t/p) + (1-t)/(1-p), which is only safe
 * when the predictions are bounded in [0, 1]. Huber and MSE are regression
 * losses and should be paired with outputs that can represent the target
 * range (i.e. not sigmoid).
 */
const LOSS_COMPATIBLE_ACTIVATIONS: Record<LossType, ReadonlyArray<ActivationType>> = {
    crossEntropy: ['sigmoid'],
    mse: ['linear', 'tanh', 'relu', 'leakyRelu', 'elu', 'swish', 'softplus'],
    huber: ['linear', 'tanh', 'relu', 'leakyRelu', 'elu', 'swish', 'softplus'],
};

/** Returns true iff the given loss type can be safely combined with the output activation. */
export function isLossCompatible(lossType: LossType, outputActivation: ActivationType): boolean {
    return LOSS_COMPATIBLE_ACTIVATIONS[lossType].includes(outputActivation);
}

/** Human-readable explanation for an incompatible loss/activation pair. */
export function describeLossIncompatibility(
    lossType: LossType,
    outputActivation: ActivationType,
): string {
    const allowed = LOSS_COMPATIBLE_ACTIVATIONS[lossType].join(', ');
    return (
        `Loss "${LOSS_LABELS[lossType]}" is not compatible with output activation ` +
        `"${outputActivation}". Compatible activations: ${allowed}.`
    );
}
