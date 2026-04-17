// ── Optimizers for weight updates ──
// Each optimizer maintains per-parameter state as needed.

import type { OptimizerType } from './types.js';

/**
 * OptimizerState holds per-parameter momentum/velocity values.
 * Shape mirrors network weights: optState[layer][stateVar][neuron][weightIdx]
 * where `weightIdx` ranges over [0..fanIn-1] for weights plus index `fanIn`
 * for the neuron's bias.
 */
export type OptimizerState = number[][][][]; // [layer][stateVar][neuron][weightOrBiasIdx]

/** Optional optimizer hyperparameters — omitted fields fall back to canonical defaults. */
export interface OptimizerHyperparams {
    /** Momentum coefficient for sgdMomentum. Default 0.9. */
    momentum?: number;
    /** Adam first-moment decay rate. Default 0.9. */
    beta1?: number;
    /** Adam second-moment decay rate. Default 0.999. */
    beta2?: number;
    /** Adam numerical-stability epsilon. Default 1e-8. */
    eps?: number;
}

export interface Optimizer {
    type: OptimizerType;
    /** Update a single weight given its gradient. Returns the new weight. */
    update(
        weight: number,
        gradient: number,
        lr: number,
        state: number[],   // per-parameter state array (e.g. [m] or [m, v])
        step: number,
        hyper?: OptimizerHyperparams,
    ): number;
    /** Number of state vars per parameter (0 for SGD, 1 for momentum, 2 for Adam). */
    stateSize: number;
}

const sgd: Optimizer = {
    type: 'sgd',
    stateSize: 0,
    update: (w, g, lr) => w - lr * g,
};

const sgdMomentum: Optimizer = {
    type: 'sgdMomentum',
    stateSize: 1,
    update: (w, g, lr, state, _step, hyper) => {
        const momentum = hyper?.momentum ?? 0.9;
        state[0] = momentum * state[0] + g;
        return w - lr * state[0];
    },
};

const adam: Optimizer = {
    type: 'adam',
    stateSize: 2,
    update: (w, g, lr, state, step, hyper) => {
        const beta1 = hyper?.beta1 ?? 0.9;
        const beta2 = hyper?.beta2 ?? 0.999;
        const eps = hyper?.eps ?? 1e-8;
        state[0] = beta1 * state[0] + (1 - beta1) * g;
        state[1] = beta2 * state[1] + (1 - beta2) * g * g;
        const mHat = state[0] / (1 - beta1 ** (step + 1));
        const vHat = state[1] / (1 - beta2 ** (step + 1));
        return w - lr * mHat / (Math.sqrt(vHat) + eps);
    },
};

const OPTIMIZERS: Record<OptimizerType, Optimizer> = { sgd, sgdMomentum, adam };

export function getOptimizer(type: OptimizerType): Optimizer {
    return OPTIMIZERS[type];
}

/**
 * Create optimizer state for the entire network.
 * Returns state[layer][stateVar][neuron][weightOrBiasIdx] initialized to 0.
 *
 * For each neuron, the array has `prevSizes[l] + 1` slots: the first
 * `prevSizes[l]` track the incoming weights, and the last slot tracks
 * the neuron's bias.
 */
export function createOptimizerState(
    layerSizes: number[],
    prevSizes: number[],
    stateSize: number,
): OptimizerState {
    const state: OptimizerState = [];
    for (let l = 0; l < layerSizes.length; l++) {
        const layerState: number[][][] = [];
        for (let s = 0; s < stateSize; s++) {
            const vars: number[][] = [];
            for (let n = 0; n < layerSizes[l]; n++) {
                // +1 slot for the bias (stored at index prevSizes[l])
                vars.push(new Array(prevSizes[l] + 1).fill(0));
            }
            layerState.push(vars);
        }
        state.push(layerState);
    }
    return state;
}
