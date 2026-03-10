// ── Activation functions with derivatives ──
import type { ActivationType } from './types.js';

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

const softplus: ActivationFn = {
    f: (x) => Math.log(1 + Math.exp(x)),
    df: (x) => 1 / (1 + Math.exp(-x)),
};

const ACTIVATIONS: Record<ActivationType, ActivationFn> = {
    relu,
    tanh: tanh_,
    sigmoid,
    linear,
    leakyRelu,
    elu,
    swish,
    softplus,
};

export function getActivation(type: ActivationType): ActivationFn {
    return ACTIVATIONS[type];
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
};
