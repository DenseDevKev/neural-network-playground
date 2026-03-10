import { describe, it, expect } from 'vitest';
import { getOptimizer, createOptimizerState } from '../optimizers.js';

describe('SGD optimizer', () => {
    const sgd = getOptimizer('sgd');

    it('has stateSize 0', () => {
        expect(sgd.stateSize).toBe(0);
    });

    it('updates weight by -lr * gradient', () => {
        const w = 1.0;
        const g = 0.5;
        const lr = 0.1;
        expect(sgd.update(w, g, lr, [], 0)).toBeCloseTo(w - lr * g, 8);
    });

    it('with zero gradient, weight unchanged', () => {
        expect(sgd.update(5, 0, 0.1, [], 0)).toBe(5);
    });
});

describe('SGD with Momentum', () => {
    const sgdM = getOptimizer('sgdMomentum');

    it('has stateSize 1', () => {
        expect(sgdM.stateSize).toBe(1);
    });

    it('accumulates momentum over steps', () => {
        const state = [0];
        const w0 = 1.0;
        const g = 0.5;
        const lr = 0.1;

        // Step 1: state[0] = 0.9 * 0 + 0.5 = 0.5, w = 1 - 0.1 * 0.5 = 0.95
        const w1 = sgdM.update(w0, g, lr, state, 0);
        expect(w1).toBeCloseTo(0.95, 8);
        expect(state[0]).toBeCloseTo(0.5, 8);

        // Step 2: state[0] = 0.9 * 0.5 + 0.5 = 0.95, w = 0.95 - 0.1 * 0.95 = 0.855
        const w2 = sgdM.update(w1, g, lr, state, 1);
        expect(w2).toBeCloseTo(0.855, 8);
        expect(state[0]).toBeCloseTo(0.95, 8);
    });
});

describe('Adam optimizer', () => {
    const adam = getOptimizer('adam');

    it('has stateSize 2', () => {
        expect(adam.stateSize).toBe(2);
    });

    it('updates weight toward zero for positive gradient', () => {
        const state = [0, 0];
        const w = 1.0;
        const newW = adam.update(w, 0.5, 0.01, state, 0);
        expect(newW).toBeLessThan(w);
    });

    it('state is updated after call', () => {
        const state = [0, 0];
        adam.update(1.0, 0.5, 0.01, state, 0);
        expect(state[0]).not.toBe(0); // m updated
        expect(state[1]).not.toBe(0); // v updated
    });
});

describe('createOptimizerState', () => {
    it('creates zero-initialized state of correct shape', () => {
        const layerSizes = [4, 3]; // two layers with 4, 3 neurons
        const prevSizes = [2, 4];  // previous layers with 2, 4 neurons
        const stateSize = 2;       // Adam needs 2

        const state = createOptimizerState(layerSizes, prevSizes, stateSize);
        expect(state).toHaveLength(2); // 2 layers
        expect(state[0]).toHaveLength(2); // 2 state vars (m, v)
        expect(state[0][0]).toHaveLength(4); // 4 neurons
        expect(state[0][0][0]).toHaveLength(3); // 2 prev weights + 1 bias

        // All zeros
        for (const layer of state) {
            for (const sv of layer) {
                for (const neuron of sv) {
                    for (const val of neuron) {
                        expect(val).toBe(0);
                    }
                }
            }
        }
    });

    it('returns empty for stateSize 0', () => {
        const state = createOptimizerState([4], [2], 0);
        expect(state[0]).toHaveLength(0);
    });
});
