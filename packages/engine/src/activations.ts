// ── Activation functions with derivatives ──
import type { ActivationType, ScalarActivationType } from './types.js';
import { NonFiniteNumericalError } from './numericalError.js';

export interface ActivationFn {
    f: (x: number) => number;
    df: (x: number, output: number) => number;
}

const relu: ActivationFn = {
    f: (x) => Math.max(0, x),
    df: (_x, output) => (output > 0 ? 1 : 0),
};

const tanh_: ActivationFn = {
    f: (x) => Math.tanh(x),
    df: (_x, output) => 1 - output * output,
};

const sigmoid: ActivationFn = {
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (_x, output) => output * (1 - output),
};

const linear: ActivationFn = {
    f: (x) => x,
    df: () => 1,
};

const leakyRelu: ActivationFn = {
    f: (x) => (x > 0 ? x : 0.01 * x),
    df: (_x, output) => (output > 0 ? 1 : 0.01),
};

const elu: ActivationFn = {
    f: (x) => (x >= 0 ? x : Math.exp(x) - 1),
    df: (x, output) => (x >= 0 ? 1 : output + 1),
};

const swish: ActivationFn = {
    f: (x) => x / (1 + Math.exp(-x)),
    df: (x, output) => {
        const sig = 1 / (1 + Math.exp(-x));
        return output + sig * (1 - output);
    },
};

function stableSoftplus(x: number): number {
    return x > 0
        ? x + Math.log1p(Math.exp(-x))
        : Math.log1p(Math.exp(x));
}

function stableSigmoid(x: number): number {
    if (x >= 0) {
        return 1 / (1 + Math.exp(-x));
    }
    const ex = Math.exp(x);
    return ex / (1 + ex);
}

/** Stable vector softmax for multiclass probability outputs. */
export function softmax(logits: ArrayLike<number>): number[] {
    if (logits.length === 0) {
        throw new RangeError('softmax logits must not be empty');
    }

    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        const value = logits[i];
        if (!Number.isFinite(value)) {
            throw new NonFiniteNumericalError(`logits[${i}]`, value);
        }
        if (value > maxLogit) {
            maxLogit = value;
        }
    }

    const exps = new Array<number>(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        const expValue = Math.exp(logits[i] - maxLogit);
        exps[i] = expValue;
        sum += expValue;
    }

    if (!Number.isFinite(sum)) {
        throw new NonFiniteNumericalError('softmax.normalizer', sum);
    }
    if (sum <= 0) {
        throw new RangeError('softmax logits produced an invalid normalizer');
    }

    const invSum = 1 / sum;
    for (let i = 0; i < exps.length; i++) {
        exps[i] *= invSum;
        if (!Number.isFinite(exps[i])) {
            throw new NonFiniteNumericalError(`softmax.output[${i}]`, exps[i]);
        }
    }
    return exps;
}

const softplus: ActivationFn = {
    f: stableSoftplus,
    df: (x) => stableSigmoid(x),
};

const ACTIVATIONS: Record<ScalarActivationType, ActivationFn> = {
    relu,
    tanh: tanh_,
    sigmoid,
    linear,
    leakyRelu,
    elu,
    swish,
    softplus,
};

export function getActivation(type: ScalarActivationType): ActivationFn {
    const activation = ACTIVATIONS[type];
    if (activation == null) {
        throw new RangeError('softmax is a vector activation; use softmax() for logits');
    }
    return activation;
}

/** Human-readable labels for the UI. */
export const ACTIVATION_LABELS: Record<ActivationType, string> = {
    relu: 'ReLU',
    tanh: 'Tanh',
    sigmoid: 'Sigmoid',
    linear: 'Linear',
    leakyRelu: 'Leaky ReLU',
    elu: 'ELU',
    swish: 'Swish',
    softplus: 'Softplus',
    softmax: 'Softmax',
};
