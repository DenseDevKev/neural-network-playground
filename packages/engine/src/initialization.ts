// ── Weight initialization strategies ──
import type { WeightInitType } from './types.js';
import { PRNG } from './prng.js';

/**
 * Initialize a weight matrix for a layer.
 * Returns weights[neuron][prevNeuron].
 *
 * Kept for legacy callers that still want a nested number[][]. The engine's
 * hot path uses `initWeightsInto` to fill a preallocated Float32Array.
 */
export function initWeights(
    fanIn: number,
    fanOut: number,
    init: WeightInitType,
    rng: PRNG,
): number[][] {
    const weights: number[][] = [];
    for (let i = 0; i < fanOut; i++) {
        const row: number[] = [];
        for (let j = 0; j < fanIn; j++) {
            row.push(initSingleWeight(fanIn, fanOut, init, rng));
        }
        weights.push(row);
    }
    return weights;
}

/**
 * Fill a packed (fanOut × fanIn) typed-array buffer in place with initialized
 * weights. Row-major layout: `target[n * fanIn + w]` is the weight from
 * the w-th previous-layer neuron into the n-th neuron of this layer.
 *
 * Accepts either Float32Array (used for wire-format buffers) or
 * Float64Array (used inside the engine for numerical accuracy).
 */
export function initWeightsInto(
    target: Float32Array | Float64Array,
    fanIn: number,
    fanOut: number,
    init: WeightInitType,
    rng: PRNG,
): void {
    for (let n = 0; n < fanOut; n++) {
        const rowStart = n * fanIn;
        for (let w = 0; w < fanIn; w++) {
            target[rowStart + w] = initSingleWeight(fanIn, fanOut, init, rng);
        }
    }
}

/** Initialize a bias vector (zeros or small random). */
export function initBiases(size: number): number[] {
    return new Array(size).fill(0);
}

/** Zero-fill a packed bias buffer (Xavier/He recipes leave biases at 0). */
export function initBiasesInto(target: Float32Array | Float64Array): void {
    target.fill(0);
}

function initSingleWeight(
    fanIn: number,
    fanOut: number,
    init: WeightInitType,
    rng: PRNG,
): number {
    switch (init) {
        case 'xavier':
            return rng.gaussian(0, Math.sqrt(2 / (fanIn + fanOut)));
        case 'he':
            return rng.gaussian(0, Math.sqrt(2 / fanIn));
        case 'uniform': {
            const limit = Math.sqrt(6 / (fanIn + fanOut));
            return rng.range(-limit, limit);
        }
        case 'zeros':
            return 0;
    }
}
