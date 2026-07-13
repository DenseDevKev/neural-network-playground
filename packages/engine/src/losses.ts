// ── Loss functions with output-layer gradients ──
import type { LossType, ScalarLossType, ActivationType } from './types.js';
import { meanSquaredErrorScalar, meanSquaredErrorScalarDelta } from './objective.js';
import { NonFiniteNumericalError } from './numericalError.js';

export interface LossFn {
    /** Compute loss for a single sample. */
    loss: (predicted: number, target: number) => number;
    /** d(loss)/d(predicted) for backprop. */
    dloss: (predicted: number, target: number) => number;
}

const EPSILON = 1e-7;
const DISTRIBUTION_SUM_TOLERANCE = 1e-5;

function assertFinite(value: number, path: string): void {
    if (!Number.isFinite(value)) {
        throw new NonFiniteNumericalError(path, value);
    }
}

const mse: LossFn = {
    loss: meanSquaredErrorScalar,
    dloss: meanSquaredErrorScalarDelta,
};

const crossEntropy: LossFn = {
    loss: (p, t) => {
        assertFinite(p, 'crossEntropy.prediction');
        assertFinite(t, 'crossEntropy.target');
        const clamped = Math.max(EPSILON, Math.min(1 - EPSILON, p));
        const result = -(t * Math.log(clamped) + (1 - t) * Math.log(1 - clamped));
        assertFinite(result, 'crossEntropy.loss');
        return result;
    },
    dloss: (p, t) => {
        assertFinite(p, 'crossEntropy.prediction');
        assertFinite(t, 'crossEntropy.target');
        const clamped = Math.max(EPSILON, Math.min(1 - EPSILON, p));
        const result = -(t / clamped) + (1 - t) / (1 - clamped);
        assertFinite(result, 'crossEntropy.gradient');
        return result;
    },
};

function assertCategoricalVectorPair(probabilities: ArrayLike<number>, target: ArrayLike<number>): void {
    if (probabilities.length === 0) {
        throw new RangeError('categorical cross-entropy vectors must not be empty');
    }
    if (probabilities.length !== target.length) {
        throw new RangeError('probabilities and target must have the same length');
    }

    let probabilitySum = 0;
    let targetSum = 0;
    for (let i = 0; i < probabilities.length; i++) {
        const probability = probabilities[i];
        const targetValue = target[i];
        if (!Number.isFinite(probability)) {
            throw new NonFiniteNumericalError(`probabilities[${i}]`, probability);
        }
        if (probability < 0 || probability > 1) {
            throw new RangeError('probabilities must be finite values in [0, 1]');
        }
        if (!Number.isFinite(targetValue)) {
            throw new NonFiniteNumericalError(`target[${i}]`, targetValue);
        }
        if (targetValue < 0) {
            throw new RangeError('target values must be finite and non-negative');
        }
        probabilitySum += probability;
        targetSum += targetValue;
        assertFinite(probabilitySum, 'probabilities.sum');
        assertFinite(targetSum, 'target.sum');
    }

    if (Math.abs(probabilitySum - 1) > DISTRIBUTION_SUM_TOLERANCE) {
        throw new RangeError('probabilities must sum to 1');
    }
    if (Math.abs(targetSum - 1) > DISTRIBUTION_SUM_TOLERANCE) {
        throw new RangeError('target values must sum to 1');
    }
}

/** Categorical cross-entropy for one-hot or soft target distributions. */
export function categoricalCrossEntropy(
    probabilities: ArrayLike<number>,
    target: ArrayLike<number>,
): number {
    assertCategoricalVectorPair(probabilities, target);

    let sum = 0;
    for (let i = 0; i < probabilities.length; i++) {
        if (target[i] === 0) continue;
        const clamped = Math.max(EPSILON, Math.min(1 - EPSILON, probabilities[i]));
        sum -= target[i] * Math.log(clamped);
        assertFinite(sum, 'categoricalCrossEntropy.loss');
    }
    return sum;
}

/** Gradient of softmax + categorical cross-entropy with respect to logits. */
export function categoricalCrossEntropyLogitGradient(
    probabilities: ArrayLike<number>,
    target: ArrayLike<number>,
): number[] {
    assertCategoricalVectorPair(probabilities, target);

    const gradient = new Array<number>(probabilities.length);
    for (let i = 0; i < probabilities.length; i++) {
        gradient[i] = probabilities[i] - target[i];
        assertFinite(gradient[i], `categoricalCrossEntropy.gradient[${i}]`);
    }
    return gradient;
}

export const DEFAULT_HUBER_DELTA = 1.0;

/** Build a Huber loss with a configurable transition point δ. */
function makeHuber(delta: number): LossFn {
    if (!Number.isFinite(delta)) {
        throw new NonFiniteNumericalError('huberDelta', delta);
    }
    if (delta <= 0) {
        throw new RangeError('huberDelta must be finite and greater than 0');
    }
    return {
        loss: (p, t) => {
            assertFinite(p, 'huber.prediction');
            assertFinite(t, 'huber.target');
            const a = Math.abs(p - t);
            const result = a <= delta
                ? 0.5 * a * a
                : delta * (a - 0.5 * delta);
            assertFinite(result, 'huber.loss');
            return result;
        },
        dloss: (p, t) => {
            assertFinite(p, 'huber.prediction');
            assertFinite(t, 'huber.target');
            const diff = p - t;
            const a = Math.abs(diff);
            const result = a <= delta ? diff : delta * Math.sign(diff);
            assertFinite(result, 'huber.gradient');
            return result;
        },
    };
}

const DEFAULT_HUBER = makeHuber(DEFAULT_HUBER_DELTA);
const LOSSES: Record<ScalarLossType, LossFn> = { mse, crossEntropy, huber: DEFAULT_HUBER };

/**
 * Resolve a loss type to its function. For Huber, an optional `huberDelta`
 * produces a delta-configured LossFn; if omitted, the module default is used.
 */
export function getLoss(type: LossType, opts?: { huberDelta?: number }): LossFn {
    if (type === 'categoricalCrossEntropy') {
        throw new RangeError('categoricalCrossEntropy is a vector loss; use categoricalCrossEntropy()');
    }
    if (type === 'huber' && opts?.huberDelta != null) {
        if (!Number.isFinite(opts.huberDelta)) {
            throw new NonFiniteNumericalError('huberDelta', opts.huberDelta);
        }
        if (opts.huberDelta <= 0) {
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
    if (predictions.length !== targets.length) {
        throw new RangeError('predictions and targets must have the same batch length');
    }
    if (predictions.length === 0) return 0;

    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        const sampleLoss = fn.loss(predictions[i], targets[i]);
        assertFinite(sampleLoss, `batchLoss[${i}]`);
        sum += sampleLoss;
        assertFinite(sum, 'batchLoss.sum');
    }
    const result = sum / predictions.length;
    assertFinite(result, 'batchLoss');
    return result;
}

export const LOSS_LABELS: Record<ScalarLossType, string> = {
    mse: 'MSE (Squared)',
    crossEntropy: 'Cross-Entropy',
    huber: 'Huber',
};

const LOSS_DISPLAY_LABELS: Record<LossType, string> = {
    ...LOSS_LABELS,
    categoricalCrossEntropy: 'Categorical Cross-Entropy',
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
    categoricalCrossEntropy: ['softmax'],
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
        `Loss "${LOSS_DISPLAY_LABELS[lossType]}" is not compatible with output activation ` +
        `"${outputActivation}". Compatible activations: ${allowed}.`
    );
}
