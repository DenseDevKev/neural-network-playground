import { describe, expect, it } from 'vitest';
import {
    BINARY_DATASET_IDS,
    DATASET_IDS,
    getDatasetContract,
    MULTICLASS_DATASET_IDS,
    REGRESSION_DATASET_IDS,
} from '../datasetContracts.js';

describe('dataset contracts', () => {
    it('registers every dataset under its task-specific ID group', () => {
        expect(BINARY_DATASET_IDS).toEqual([
            'circle',
            'xor',
            'gauss',
            'spiral',
            'moons',
            'checkerboard',
            'rings',
            'heart',
        ]);
        expect(MULTICLASS_DATASET_IDS).toEqual(['three-class-clusters']);
        expect(REGRESSION_DATASET_IDS).toEqual(['reg-plane', 'reg-gauss']);
        expect(DATASET_IDS).toEqual([
            ...BINARY_DATASET_IDS,
            ...MULTICLASS_DATASET_IDS,
            ...REGRESSION_DATASET_IDS,
        ]);
        expect(new Set(DATASET_IDS).size).toBe(11);
    });

    it.each(DATASET_IDS)('%s declares a version-2 bounded input and noise contract', (id) => {
        const contract = getDatasetContract(id);

        expect(contract.id).toBe(id);
        expect(contract.inputDomain).toEqual({ x: [-1, 1], y: [-1, 1] });
        expect(contract.noise.minimum).toBe(0);
        expect(contract.noise.maximum).toBe(100);
        expect(contract.generatorVersion).toBe(2);
    });

    it.each(BINARY_DATASET_IDS)('%s declares binary targets and coordinate noise', (id) => {
        expect(getDatasetContract(id)).toMatchObject({
            taskKind: 'binary-classification',
            targetDomain: { kind: 'binary', values: [0, 1] },
            noise: { meaning: 'coordinate-perturbation' },
        });
    });

    it('declares the three-class target contract', () => {
        expect(getDatasetContract('three-class-clusters')).toMatchObject({
            taskKind: 'multiclass-classification',
            targetDomain: { kind: 'classes', classCount: 3 },
            noise: { meaning: 'coordinate-perturbation' },
        });
    });

    it('declares conservative noise-dependent regression target bounds', () => {
        const plane = getDatasetContract('reg-plane');
        const gauss = getDatasetContract('reg-gauss');

        expect(plane).toMatchObject({
            taskKind: 'regression',
            noise: { meaning: 'target-perturbation' },
        });
        expect(gauss).toMatchObject({
            taskKind: 'regression',
            noise: { meaning: 'target-perturbation' },
        });

        if (plane.targetDomain.kind !== 'continuous' || gauss.targetDomain.kind !== 'continuous') {
            throw new Error('Regression datasets must declare continuous targets');
        }

        expect(plane.targetDomain.boundsForNoise(0)).toEqual([-2, 2]);
        expect(plane.targetDomain.boundsForNoise(100)).toEqual([-8, 8]);
        expect(gauss.targetDomain.boundsForNoise(0)).toEqual([-0.06 * 0, 2 + 0.06 * 0]);
        expect(gauss.targetDomain.boundsForNoise(100)).toEqual([-6, 8]);
    });
});
