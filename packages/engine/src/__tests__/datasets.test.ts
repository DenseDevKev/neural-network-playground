import { describe, it, expect } from 'vitest';
import {
    DatasetGenerationError,
    generateDataset,
    generateDatasetV2,
    generateThreeClassClusters,
    getDefaultProblemType,
    THREE_CLASS_CLUSTER_DATASET_CONTRACT,
} from '../datasets.js';
import {
    DATASET_IDS,
    getDatasetContract,
    REGRESSION_DATASET_IDS,
} from '../datasetContracts.js';
import * as Engine from '../index.js';
import type { DataPoint, DatasetContract, DatasetType } from '../types.js';

function allPoints(split: { train: DataPoint[]; test: DataPoint[] }): DataPoint[] {
    return [...split.train, ...split.test];
}

function expectPointsToSatisfyContract(
    points: readonly DataPoint[],
    contract: DatasetContract,
    noise: number,
): void {
    expect(points).toHaveLength(1_000);

    for (const point of points) {
        expect(Number.isFinite(point.x)).toBe(true);
        expect(Number.isFinite(point.y)).toBe(true);
        expect(Number.isFinite(point.label)).toBe(true);
        expect(point.x).toBeGreaterThanOrEqual(contract.inputDomain.x[0]);
        expect(point.x).toBeLessThanOrEqual(contract.inputDomain.x[1]);
        expect(point.y).toBeGreaterThanOrEqual(contract.inputDomain.y[0]);
        expect(point.y).toBeLessThanOrEqual(contract.inputDomain.y[1]);

        switch (contract.targetDomain.kind) {
            case 'binary':
                expect(contract.targetDomain.values).toContain(point.label);
                break;
            case 'classes':
                expect(Number.isInteger(point.label)).toBe(true);
                expect(point.label).toBeGreaterThanOrEqual(0);
                expect(point.label).toBeLessThan(contract.targetDomain.classCount);
                break;
            case 'continuous': {
                const [minimum, maximum] = contract.targetDomain.boundsForNoise(noise);
                expect(point.label).toBeGreaterThanOrEqual(minimum);
                expect(point.label).toBeLessThanOrEqual(maximum);
                break;
            }
        }
    }
}

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

describe('generateDatasetV2', () => {
    it.each(DATASET_IDS)('%s stays inside its declared contract at noise extremes', (id) => {
        const contract = getDatasetContract(id);

        for (const noise of [contract.noise.minimum, contract.noise.maximum]) {
            const split = generateDatasetV2({ dataset: id, sampleCount: 1_000, noise, seed: 42 });
            const points = allPoints(split);

            expectPointsToSatisfyContract(points, contract, noise);
            if (noise === contract.noise.maximum && contract.noise.meaning === 'coordinate-perturbation') {
                expect(points.some((point) => (
                    point.x === contract.inputDomain.x[0]
                    || point.x === contract.inputDomain.x[1]
                    || point.y === contract.inputDomain.y[0]
                    || point.y === contract.inputDomain.y[1]
                ))).toBe(false);
            }
        }
    });

    it.each(DATASET_IDS)('%s is byte-for-byte deterministic for identical settings', (dataset) => {
        const request = { dataset, sampleCount: 200, noise: 37, seed: 4_294_967_295 } as const;

        expect(JSON.stringify(generateDatasetV2(request))).toBe(JSON.stringify(generateDatasetV2(request)));
    });

    it.each(REGRESSION_DATASET_IDS)(
        '%s keeps ordered coordinates stable when only target noise changes',
        (dataset) => {
            const request = { dataset, sampleCount: 1_000, seed: 42, trainFraction: 0.5 } as const;
            const clean = generateDatasetV2({ ...request, noise: 0 });
            const noisy = generateDatasetV2({ ...request, noise: 100 });
            const orderedCoordinates = (split: typeof clean) => ({
                train: split.train.map(({ x, y }) => [x, y]),
                test: split.test.map(({ x, y }) => [x, y]),
            });
            const orderedTargets = (split: typeof clean) => ({
                train: split.train.map(({ label }) => label),
                test: split.test.map(({ label }) => label),
            });

            expect(orderedCoordinates(noisy)).toEqual(orderedCoordinates(clean));
            expect(orderedTargets(noisy)).not.toEqual(orderedTargets(clean));
        },
    );

    it('rejects invalid settings with a structured failure', () => {
        const requests = [
            { dataset: 'circle', sampleCount: 1, noise: 0, seed: 42 },
            { dataset: 'circle', sampleCount: 1_001, noise: 0, seed: 42 },
            { dataset: 'circle', sampleCount: 100, noise: -1, seed: 42 },
            { dataset: 'circle', sampleCount: 100, noise: 101, seed: 42 },
            { dataset: 'circle', sampleCount: 100, noise: 0, seed: 42.9 },
            { dataset: 'circle', sampleCount: 100, noise: 0, seed: 42, trainFraction: 0.09 },
            { dataset: 'circle', sampleCount: 100, noise: 0, seed: 42, trainFraction: 0.91 },
        ] as const;

        for (const request of requests) {
            try {
                generateDatasetV2(request);
                throw new Error('Expected dataset generation to fail');
            } catch (error) {
                expect(error).toBeInstanceOf(DatasetGenerationError);
                expect(error).toMatchObject({
                    name: 'DatasetGenerationError',
                    code: 'invalid-settings',
                });
            }
        }
    });

    it('computes Checkerboard labels before coordinate perturbation', () => {
        const points = allPoints(generateDatasetV2({
            dataset: 'checkerboard',
            sampleCount: 1_000,
            noise: 100,
            seed: 42,
        }));
        const labelAtFinalCoordinate = ({ x, y }: DataPoint) => (
            ((x >= 0 ? 1 : 0) + (y >= 0 ? 1 : 0)) % 2
        );

        expect(points.some((point) => point.label !== labelAtFinalCoordinate(point))).toBe(true);
    });

    it('computes Heart labels before coordinate perturbation', () => {
        const points = allPoints(generateDatasetV2({
            dataset: 'heart',
            sampleCount: 1_000,
            noise: 100,
            seed: 42,
        }));
        const labelAtFinalCoordinate = ({ x, y }: DataPoint) => {
            const x2 = x * x;
            const y2 = y * y;
            const inner = x2 + y2 - 0.6;
            return inner * inner * inner - x2 * y2 * y < 0 ? 1 : 0;
        };

        expect(points.some((point) => point.label !== labelAtFinalCoordinate(point))).toBe(true);
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

describe('generateThreeClassClusters', () => {
    it('has an explicit bounded multiclass dataset contract', () => {
        expect(THREE_CLASS_CLUSTER_DATASET_CONTRACT).toMatchObject({
            id: 'three-class-clusters',
            problemType: 'classification',
            classCount: 3,
            outputSize: 3,
            outputActivation: 'softmax',
            lossType: 'categoricalCrossEntropy',
            defaultNumSamples: 300,
            defaultNoise: 8,
            defaultTrainRatio: 0.5,
        });
        expect(THREE_CLASS_CLUSTER_DATASET_CONTRACT.classLabels).toEqual([0, 1, 2]);
        expect(Object.isFrozen(THREE_CLASS_CLUSTER_DATASET_CONTRACT.classLabels)).toBe(true);

        const defaultSplit = THREE_CLASS_CLUSTER_DATASET_CONTRACT.generate();
        expect(defaultSplit.train.length + defaultSplit.test.length).toBe(300);
        expect(defaultSplit).toEqual(generateThreeClassClusters(300, 8, 0.5, 42));
    });

    it('generates a deterministic bounded dataset with three class labels', () => {
        const split = generateThreeClassClusters(90, 12, 0.6, 123);
        const same = generateThreeClassClusters(90, 12, 0.6, 123);
        const different = generateThreeClassClusters(90, 12, 0.6, 124);
        const allPoints = [...split.train, ...split.test];

        expect(split).toEqual(same);
        expect(split.train).not.toEqual(different.train);
        expect(split.train).toHaveLength(54);
        expect(split.test).toHaveLength(36);
        expect(new Set(allPoints.map((point) => point.label))).toEqual(new Set([0, 1, 2]));

        for (const point of allPoints) {
            expect(point.x).toBeGreaterThanOrEqual(-1);
            expect(point.x).toBeLessThanOrEqual(1);
            expect(point.y).toBeGreaterThanOrEqual(-1);
            expect(point.y).toBeLessThanOrEqual(1);
            expect(Number.isInteger(point.label)).toBe(true);
        }
    });

    it('keeps class counts balanced within one sample', () => {
        const split = generateThreeClassClusters(302, 0, 0.5, 42);
        const counts = [0, 0, 0];

        for (const point of [...split.train, ...split.test]) {
            counts[point.label]++;
        }

        expect(counts).toEqual([101, 101, 100]);
        expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1);
    });

    it('gives both splits all three classes for the intended preset-sized case', () => {
        const split = generateThreeClassClusters(300, 8, 0.5, 42);

        expect(new Set(split.train.map((point) => point.label))).toEqual(new Set([0, 1, 2]));
        expect(new Set(split.test.map((point) => point.label))).toEqual(new Set([0, 1, 2]));
    });

    it('is available through the public dataset generator once public wiring starts', () => {
        const split = generateDataset('three-class-clusters' as DatasetType, 90, 8, 0.5, 42);
        const labels = new Set([...split.train, ...split.test].map((point) => point.label));

        expect(labels).toEqual(new Set([0, 1, 2]));
        expect(split.train.length + split.test.length).toBe(90);
        expect(getDefaultProblemType('three-class-clusters' as DatasetType)).toBe('classification');
    });

    it('stays out of the package-level engine API until public config wiring is approved', () => {
        expect('generateThreeClassClusters' in Engine).toBe(false);
        expect('THREE_CLASS_CLUSTER_DATASET_CONTRACT' in Engine).toBe(false);
    });

    it('handles tiny and invalid sample requests like the public dataset generator', () => {
        expect(generateThreeClassClusters(0, 0, 0.5, 42).train).toHaveLength(0);
        expect(generateThreeClassClusters(0, 0, 0.5, 42).test).toHaveLength(0);

        const one = generateThreeClassClusters(1, 0, 0.5, 42);
        expect(one.train).toHaveLength(1);
        expect(one.test).toHaveLength(0);

        const defensive = generateThreeClassClusters(10, 0, Number.NaN, 42);
        expect(defensive.train).toHaveLength(5);
        expect(defensive.test).toHaveLength(5);
    });
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
