import { describe, it, expect } from 'vitest';
import { getLoss, batchLoss, isLossCompatible, describeLossIncompatibility } from '../losses.js';

describe('MSE loss', () => {
    const mse = getLoss('mse');

    it('loss is 0 when predicted equals target', () => {
        expect(mse.loss(0.5, 0.5)).toBe(0);
    });

    it('loss is 0.5 * (p - t)²', () => {
        expect(mse.loss(1, 0)).toBeCloseTo(0.5, 8);
        expect(mse.loss(0.3, 0.7)).toBeCloseTo(0.5 * 0.16, 8);
    });

    it('gradient is (p - t)', () => {
        expect(mse.dloss(1, 0)).toBe(1);
        expect(mse.dloss(0.3, 0.7)).toBeCloseTo(-0.4, 8);
    });

    it('gradient at equal values is 0', () => {
        expect(mse.dloss(0.5, 0.5)).toBe(0);
    });
});

describe('Cross-Entropy loss', () => {
    const ce = getLoss('crossEntropy');

    it('loss is small for correct confident prediction', () => {
        const l = ce.loss(0.99, 1);
        expect(l).toBeLessThan(0.02);
    });

    it('loss is large for incorrect confident prediction', () => {
        const l = ce.loss(0.01, 1);
        expect(l).toBeGreaterThan(4);
    });

    it('loss for target=0 penalizes high prediction', () => {
        expect(ce.loss(0.99, 0)).toBeGreaterThan(4);
    });

    it('gradient sign is correct', () => {
        // When predicted > target, gradient should be positive
        expect(ce.dloss(0.8, 0.2)).toBeGreaterThan(0);
        // When predicted < target, gradient should be negative
        expect(ce.dloss(0.2, 0.8)).toBeLessThan(0);
    });

    it('numerical gradient matches analytical', () => {
        const h = 1e-6;
        const p = 0.7;
        const t = 0.3;
        const numGrad = (ce.loss(p + h, t) - ce.loss(p - h, t)) / (2 * h);
        expect(ce.dloss(p, t)).toBeCloseTo(numGrad, 3);
    });
});

describe('Huber loss', () => {
    const huber = getLoss('huber');
    const delta = 1.0;

    it('behaves like MSE for small errors', () => {
        expect(huber.loss(0.5, 0.3)).toBeCloseTo(0.5 * 0.04, 8);
    });

    it('behaves like MAE (linear) for large errors', () => {
        const error = 5;
        expect(huber.loss(5, 0)).toBeCloseTo(delta * (error - 0.5 * delta), 8);
    });

    it('gradient is clamped for large errors', () => {
        expect(huber.dloss(10, 0)).toBe(delta);
        expect(huber.dloss(-10, 0)).toBe(-delta);
    });

    it('gradient equals diff for small errors', () => {
        expect(huber.dloss(0.3, 0)).toBeCloseTo(0.3, 8);
    });

    it('honours custom huberDelta', () => {
        // With δ=0.5: an error of 1 lies in the linear regime.
        const custom = getLoss('huber', { huberDelta: 0.5 });
        expect(custom.dloss(1, 0)).toBeCloseTo(0.5, 8);
        expect(custom.dloss(-1, 0)).toBeCloseTo(-0.5, 8);
        // An error of 0.4 lies in the quadratic regime.
        expect(custom.dloss(0.4, 0)).toBeCloseTo(0.4, 8);
        // Loss at large error: δ·(a − 0.5·δ) = 0.5·(1 − 0.25) = 0.375
        expect(custom.loss(1, 0)).toBeCloseTo(0.375, 8);
    });

    it('rejects invalid huberDelta values', () => {
        for (const huberDelta of [0, -1, Number.POSITIVE_INFINITY, Number.NaN]) {
            expect(() => getLoss('huber', { huberDelta })).toThrow(RangeError);
        }
    });
});

describe('loss/activation compatibility', () => {
    it('cross-entropy is only compatible with sigmoid', () => {
        expect(isLossCompatible('crossEntropy', 'sigmoid')).toBe(true);
        expect(isLossCompatible('crossEntropy', 'linear')).toBe(false);
        expect(isLossCompatible('crossEntropy', 'tanh')).toBe(false);
    });

    it('MSE and Huber accept unbounded activations', () => {
        expect(isLossCompatible('mse', 'linear')).toBe(true);
        expect(isLossCompatible('mse', 'tanh')).toBe(true);
        expect(isLossCompatible('huber', 'linear')).toBe(true);
    });

    it('MSE and Huber reject sigmoid', () => {
        expect(isLossCompatible('mse', 'sigmoid')).toBe(false);
        expect(isLossCompatible('huber', 'sigmoid')).toBe(false);
    });

    it('describeLossIncompatibility names both loss and activation', () => {
        const msg = describeLossIncompatibility('crossEntropy', 'tanh');
        expect(msg).toContain('Cross-Entropy');
        expect(msg).toContain('tanh');
        expect(msg).toContain('sigmoid');
    });
});

describe('batchLoss', () => {
    it('computes mean loss over a batch', () => {
        const mse = getLoss('mse');
        const preds = [0, 1];
        const targets = [1, 0];
        // Each sample: 0.5 * 1² = 0.5
        expect(batchLoss(mse, preds, targets)).toBeCloseTo(0.5, 8);
    });

    it('returns 0 for perfect predictions', () => {
        const mse = getLoss('mse');
        expect(batchLoss(mse, [0.5, 0.5], [0.5, 0.5])).toBe(0);
    });
});
