import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { initWeights, initBiases } from '../initialization.js';
import { PRNG } from '../prng.js';

describe('initBiases', () => {
    it('creates an array of the specified size filled with zeros', () => {
        const biases = initBiases(5);
        expect(biases.length).toBe(5);
        expect(biases.every((b) => b === 0)).toBe(true);
    });

    it('handles size 0', () => {
        const biases = initBiases(0);
        expect(biases.length).toBe(0);
    });
});

describe('initWeights', () => {
    let rng: PRNG;

    beforeEach(() => {
        rng = new PRNG(42);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('returns a matrix of shape [fanOut][fanIn]', () => {
        const fanIn = 3;
        const fanOut = 2;
        const weights = initWeights(fanIn, fanOut, 'zeros', rng);

        expect(weights.length).toBe(fanOut);
        for (const row of weights) {
            expect(row.length).toBe(fanIn);
        }
    });

    it('initializes with zeros when init is "zeros"', () => {
        const gaussianSpy = vi.spyOn(rng, 'gaussian');
        const rangeSpy = vi.spyOn(rng, 'range');

        const weights = initWeights(3, 2, 'zeros', rng);

        expect(weights.flat().every(w => w === 0)).toBe(true);
        expect(gaussianSpy).not.toHaveBeenCalled();
        expect(rangeSpy).not.toHaveBeenCalled();
    });

    it('initializes with xavier when init is "xavier"', () => {
        const fanIn = 3;
        const fanOut = 2;
        const gaussianSpy = vi.spyOn(rng, 'gaussian').mockReturnValue(0.5);

        const weights = initWeights(fanIn, fanOut, 'xavier', rng);

        // fanOut * fanIn = 6 calls
        expect(gaussianSpy).toHaveBeenCalledTimes(6);
        const expectedStd = Math.sqrt(2 / (fanIn + fanOut));
        expect(gaussianSpy).toHaveBeenCalledWith(0, expectedStd);

        // Ensure values are populated from mock
        expect(weights.flat().every(w => w === 0.5)).toBe(true);
    });

    it('initializes with he when init is "he"', () => {
        const fanIn = 3;
        const fanOut = 2;
        const gaussianSpy = vi.spyOn(rng, 'gaussian').mockReturnValue(0.7);

        const weights = initWeights(fanIn, fanOut, 'he', rng);

        // fanOut * fanIn = 6 calls
        expect(gaussianSpy).toHaveBeenCalledTimes(6);
        const expectedStd = Math.sqrt(2 / fanIn);
        expect(gaussianSpy).toHaveBeenCalledWith(0, expectedStd);

        // Ensure values are populated from mock
        expect(weights.flat().every(w => w === 0.7)).toBe(true);
    });

    it('initializes with uniform when init is "uniform"', () => {
        const fanIn = 3;
        const fanOut = 2;
        const rangeSpy = vi.spyOn(rng, 'range').mockReturnValue(0.1);

        const weights = initWeights(fanIn, fanOut, 'uniform', rng);

        // fanOut * fanIn = 6 calls
        expect(rangeSpy).toHaveBeenCalledTimes(6);
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        expect(rangeSpy).toHaveBeenCalledWith(-limit, limit);

        // Ensure values are populated from mock
        expect(weights.flat().every(w => w === 0.1)).toBe(true);
    });
});
