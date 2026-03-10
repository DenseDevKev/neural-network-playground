import { describe, it, expect } from 'vitest';
import { generateDataset, getDefaultProblemType } from '../datasets.js';
import type { DatasetType } from '../types.js';

describe('generateDataset', () => {
    const DATASETS: DatasetType[] = ['circle', 'xor', 'gauss', 'spiral', 'moons'];
    const N = 200;

    for (const ds of DATASETS) {
        describe(ds, () => {
            const split = generateDataset(ds, N, 0, 0.5, 42);

            it('produces non-empty train and test sets', () => {
                expect(split.train.length).toBeGreaterThan(0);
                expect(split.test.length).toBeGreaterThan(0);
            });

            it('respects the train/test ratio approximately', () => {
                const total = split.train.length + split.test.length;
                const ratio = split.train.length / total;
                expect(ratio).toBeCloseTo(0.5, 1);
            });

            it('all points have x, y, label fields', () => {
                for (const p of [...split.train, ...split.test]) {
                    expect(typeof p.x).toBe('number');
                    expect(typeof p.y).toBe('number');
                    expect(typeof p.label).toBe('number');
                }
            });

            it('classification labels are 0 or 1', () => {
                for (const p of [...split.train, ...split.test]) {
                    expect(p.label === 0 || p.label === 1).toBe(true);
                }
            });

            it('has both classes', () => {
                const allPoints = [...split.train, ...split.test];
                const labels = new Set(allPoints.map((p) => p.label));
                expect(labels.has(0)).toBe(true);
                expect(labels.has(1)).toBe(true);
            });
        });
    }

    it('is deterministic (same seed = same data)', () => {
        const a = generateDataset('circle', 100, 0, 0.5, 42);
        const b = generateDataset('circle', 100, 0, 0.5, 42);
        expect(a.train).toEqual(b.train);
        expect(a.test).toEqual(b.test);
    });

    it('different seeds produce different data', () => {
        const a = generateDataset('circle', 100, 0, 0.5, 1);
        const b = generateDataset('circle', 100, 0, 0.5, 2);
        expect(a.train).not.toEqual(b.train);
    });

    it('respects different train/test ratios', () => {
        const split80 = generateDataset('circle', 200, 0, 0.8, 42);
        const total = split80.train.length + split80.test.length;
        const ratio = split80.train.length / total;
        expect(ratio).toBeCloseTo(0.8, 1);
    });
});

describe('regression datasets', () => {
    for (const ds of ['reg-plane', 'reg-gauss'] as DatasetType[]) {
        describe(ds, () => {
            const split = generateDataset(ds, 100, 0, 0.5, 42);

            it('produces numeric labels (regression targets)', () => {
                for (const p of [...split.train, ...split.test]) {
                    expect(typeof p.label).toBe('number');
                    // Regression labels can be any real number
                    expect(isNaN(p.label)).toBe(false);
                }
            });

            it('produces data with both train and test', () => {
                expect(split.train.length).toBeGreaterThan(0);
                expect(split.test.length).toBeGreaterThan(0);
            });
        });
    }
});

describe('getDefaultProblemType', () => {
    it('returns classification for classification datasets', () => {
        expect(getDefaultProblemType('circle')).toBe('classification');
        expect(getDefaultProblemType('xor')).toBe('classification');
        expect(getDefaultProblemType('spiral')).toBe('classification');
        expect(getDefaultProblemType('gauss')).toBe('classification');
        expect(getDefaultProblemType('moons')).toBe('classification');
    });

    it('returns regression for regression datasets', () => {
        expect(getDefaultProblemType('reg-plane')).toBe('regression');
        expect(getDefaultProblemType('reg-gauss')).toBe('regression');
    });
});
