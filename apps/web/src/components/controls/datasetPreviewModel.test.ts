import { describe, expect, it } from 'vitest';
import { DATASET_IDS, generateDatasetV2, getDatasetContract } from '@nn-playground/engine';
import { deriveDatasetPreviewModel } from './datasetPreviewModel.ts';

describe('production dataset previews', () => {
    it.each(DATASET_IDS)('uses the exact deterministic production split for %s', (datasetId) => {
        const model = deriveDatasetPreviewModel({ datasetId, seed: 42, noise: 5 });
        const split = generateDatasetV2({ dataset: datasetId, seed: 42, noise: 5, sampleCount: 72, trainFraction: 0.5 });
        expect(model.points).toEqual([...split.train, ...split.test]);
        expect(model.points).toHaveLength(72);
        expect(model.taskKind).toBe(getDatasetContract(datasetId).taskKind);
        expect(model.points.every(({ x, y, label }) => [x, y, label].every(Number.isFinite))).toBe(true);
        expect(deriveDatasetPreviewModel({ datasetId, seed: 42, noise: 5 })).toEqual(model);
        expect(Object.isFrozen(model)).toBe(true);
        expect(Object.isFrozen(model.points)).toBe(true);
    });
    it('changes with the data seed or dataset without mutating its inputs', () => {
        const input = Object.freeze({ datasetId: 'xor' as const, seed: 42, noise: 0 });
        const model = deriveDatasetPreviewModel(input);
        expect(deriveDatasetPreviewModel({ ...input, seed: 43 }).points).not.toEqual(model.points);
        expect(deriveDatasetPreviewModel({ ...input, datasetId: 'gauss' }).points).not.toEqual(model.points);
        expect(input).toEqual({ datasetId: 'xor', seed: 42, noise: 0 });
    });
    it('honors sample count and the production input/target domains', () => {
        const model = deriveDatasetPreviewModel({ datasetId: 'reg-plane', seed: 1, noise: 10, sampleCount: 30 });
        expect(model.points).toHaveLength(30);
        expect(model.inputDomain).toEqual(getDatasetContract('reg-plane').inputDomain);
        expect(model.valueDomain).toEqual([-2.6, 2.6]);
    });
    it('retains strict generator rejection instead of drawing invented data', () => {
        expect(() => deriveDatasetPreviewModel({ datasetId: 'xor', seed: 42, noise: -1 })).toThrow();
        expect(() => deriveDatasetPreviewModel({ datasetId: 'xor', seed: 42, noise: 0, sampleCount: 0 })).toThrow();
    });
});
