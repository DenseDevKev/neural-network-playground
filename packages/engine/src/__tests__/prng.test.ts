import { describe, it, expect } from 'vitest';
import { PRNG } from '../prng.js';

describe('PRNG determinism', () => {
    it('same seed produces same sequence', () => {
        const a = new PRNG(42);
        const b = new PRNG(42);
        for (let i = 0; i < 100; i++) {
            expect(a.next()).toBe(b.next());
        }
    });

    it('different seeds produce different sequences', () => {
        const a = new PRNG(1);
        const b = new PRNG(2);
        // At least some values should differ
        let allEqual = true;
        for (let i = 0; i < 10; i++) {
            if (a.next() !== b.next()) allEqual = false;
        }
        expect(allEqual).toBe(false);
    });
});

describe('PRNG.next', () => {
    it('returns values in [0, 1)', () => {
        const rng = new PRNG(123);
        for (let i = 0; i < 1000; i++) {
            const v = rng.next();
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThan(1);
        }
    });

    it('produces a reasonable distribution (not all the same)', () => {
        const rng = new PRNG(99);
        const values = Array.from({ length: 1000 }, () => rng.next());
        const min = Math.min(...values);
        const max = Math.max(...values);
        expect(max - min).toBeGreaterThan(0.9);
    });
});

describe('PRNG.range', () => {
    it('returns values in [min, max)', () => {
        const rng = new PRNG(42);
        for (let i = 0; i < 500; i++) {
            const v = rng.range(-5, 5);
            expect(v).toBeGreaterThanOrEqual(-5);
            expect(v).toBeLessThan(5);
        }
    });
});

describe('PRNG.gaussian', () => {
    it('mean is approximately correct over many samples', () => {
        const rng = new PRNG(42);
        let sum = 0;
        const n = 10000;
        for (let i = 0; i < n; i++) sum += rng.gaussian(3, 1);
        const mean = sum / n;
        expect(mean).toBeCloseTo(3, 0); // within ~1
    });

    it('std is approximately correct over many samples', () => {
        const rng = new PRNG(42);
        const n = 10000;
        const values = Array.from({ length: n }, () => rng.gaussian(0, 2));
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / n;
        const std = Math.sqrt(variance);
        expect(std).toBeCloseTo(2, 0); // roughly within 0.5
    });
});

describe('PRNG.shuffle', () => {
    it('preserves all elements', () => {
        const rng = new PRNG(42);
        const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const shuffled = rng.shuffle([...arr]);
        expect(shuffled.sort((a, b) => a - b)).toEqual(arr);
    });

    it('shuffles deterministically with same seed', () => {
        const a = new PRNG(42);
        const b = new PRNG(42);
        const arr1 = a.shuffle([1, 2, 3, 4, 5]);
        const arr2 = b.shuffle([1, 2, 3, 4, 5]);
        expect(arr1).toEqual(arr2);
    });

    it('actually changes the order', () => {
        const rng = new PRNG(42);
        const arr = Array.from({ length: 20 }, (_, i) => i);
        const orig = [...arr];
        rng.shuffle(arr);
        // With 20 elements, probability of same order is ~1/20!
        expect(arr).not.toEqual(orig);
    });
});

describe('PRNG.fork', () => {
    it('produces independent but deterministic PRNGs', () => {
        const a = new PRNG(42);
        const forked = a.fork();
        // Forked PRNG should produce different values from parent
        const parentVal = a.next();
        const forkVal = forked.next();
        // They should be different sequences
        expect(parentVal).not.toBe(forkVal);
    });

    it('forking from same state produces same result', () => {
        const a = new PRNG(42);
        const b = new PRNG(42);
        const fa = a.fork();
        const fb = b.fork();
        for (let i = 0; i < 10; i++) {
            expect(fa.next()).toBe(fb.next());
        }
    });
});
