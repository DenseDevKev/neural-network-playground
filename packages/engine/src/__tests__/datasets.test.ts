import { describe, it, expect } from 'vitest';
import { generateDataset, getDefaultProblemType } from '../datasets.js';
import type { DatasetType } from '../types.js';

describe('generateDataset', () => {
    const DATASETS: DatasetType[] = [
        'circle',
        'xor',
        'gauss',
        'spiral',
        'moons',
        'checkerboard',
        'rings',
        'heart',
        'reg-plane',
        'reg-gauss',
    ];
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
                if (ds.startsWith('reg-')) return;

                for (const p of [...split.train, ...split.test]) {
                    expect(p.label === 0 || p.label === 1).toBe(true);
                }
            });

            it('has both classes', () => {
                if (ds.startsWith('reg-')) return;

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

    describe('tiny sample counts', () => {
        for (const ds of DATASETS) {
            it(`${ds} returns no points for zero requested samples`, () => {
                const split = generateDataset(ds, 0, 0, 0.5, 42);

                expect(split.train).toHaveLength(0);
                expect(split.test).toHaveLength(0);
            });

            it(`${ds} keeps a single requested sample in the train split`, () => {
                const split = generateDataset(ds, 1, 0, 0.5, 42);

                expect(split.train).toHaveLength(1);
                expect(split.test).toHaveLength(0);
            });

            it(`${ds} keeps train and test non-empty for two requested samples`, () => {
                const split = generateDataset(ds, 2, 0, 0.5, 42);

                expect(split.train).toHaveLength(1);
                expect(split.test).toHaveLength(1);
            });
        }
    });

    describe('defensive train ratio handling', () => {
        for (const trainRatio of [0, 1, NaN, Infinity]) {
            it(`keeps train and test non-empty for trainRatio ${String(trainRatio)}`, () => {
                for (const ds of DATASETS) {
                    const split = generateDataset(ds, 10, 0, trainRatio, 42);

                    expect(split.train.length, `${ds} train`).toBeGreaterThan(0);
                    expect(split.test.length, `${ds} test`).toBeGreaterThan(0);
                }
            });
        }
    });

    describe('total sample counts', () => {
        for (const count of [2, 10, 100, 200]) {
            it(`xor returns exactly ${count} total samples`, () => {
                const split = generateDataset('xor', count, 0, 0.5, 42);

                expect(split.train.length + split.test.length).toBe(count);
            });
        }

        it('rings does not round tiny non-zero sample requests down to zero', () => {
            const one = generateDataset('rings', 1, 0, 0.5, 42);
            const two = generateDataset('rings', 2, 0, 0.5, 42);

            expect(one.train.length + one.test.length).toBe(1);
            expect(two.train.length + two.test.length).toBe(2);
        });
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
