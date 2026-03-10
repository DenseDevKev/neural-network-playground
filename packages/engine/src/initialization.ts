// ── Weight initialization strategies ──
import type { WeightInitType } from './types.js';
import { PRNG } from './prng.js';

/**
 * Initialize a weight matrix for a layer.
 * Returns weights[neuron][prevNeuron].
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

/** Initialize a bias vector (zeros or small random). */
export function initBiases(size: number): number[] {
    return new Array(size).fill(0);
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
