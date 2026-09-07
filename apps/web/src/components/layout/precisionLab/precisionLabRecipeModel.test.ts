import { describe, expect, it } from 'vitest';
import { derivePrecisionLabRecipeModel } from './precisionLabRecipeModel.ts';

const base = {
    dataset: 'three-class-clusters',
    architecture: '2 -> 6 -> 6 -> 3',
    hiddenActivation: 'tanh',
    output: 'softmax',
    seed: 42,
    hasRecipeDrift: false,
    pendingConfiguration: false,
    evaluationAgeSteps: null,
} as const;

describe('derivePrecisionLabRecipeModel', () => {
    it('returns an unavailable display model when no prepared recipe exists', () => {
        expect(derivePrecisionLabRecipeModel(null)).toEqual({
            dataset: 'Unavailable',
            architecture: 'Unavailable',
            hiddenActivation: 'Unavailable',
            output: 'Unavailable',
            seed: 'Unavailable',
            evaluationLabel: 'No evaluation',
            tone: 'unavailable',
        });
    });

    it('gives pending configuration precedence over drift and evaluation age', () => {
        expect(derivePrecisionLabRecipeModel({
            ...base,
            pendingConfiguration: true,
            hasRecipeDrift: true,
            evaluationAgeSteps: 12,
        })).toMatchObject({ tone: 'updating', evaluationLabel: 'Updating' });
    });

    it('reports trained-recipe drift before evaluation age', () => {
        expect(derivePrecisionLabRecipeModel({
            ...base,
            hasRecipeDrift: true,
            evaluationAgeSteps: 12,
        })).toMatchObject({
            tone: 'drift',
            evaluationLabel: 'Recipe differs from trained model',
        });
    });

    it('reports exact paired-evaluation age without changing recipe identity fields', () => {
        expect(derivePrecisionLabRecipeModel({
            ...base,
            evaluationAgeSteps: 12,
        })).toEqual({
            dataset: 'three-class-clusters',
            architecture: '2 -> 6 -> 6 -> 3',
            hiddenActivation: 'tanh',
            output: 'softmax',
            seed: '42',
            evaluationLabel: 'Evaluation 12 steps behind',
            tone: 'stale',
        });
        const fresh = derivePrecisionLabRecipeModel({ ...base, evaluationAgeSteps: 0 });
        const stale = derivePrecisionLabRecipeModel({ ...base, evaluationAgeSteps: 99 });
        expect({ ...stale, evaluationLabel: fresh.evaluationLabel, tone: fresh.tone })
            .toEqual(fresh);
    });

    it.each([0])('reports fresh evaluation only at measured age %s', (evaluationAgeSteps) => {
        expect(derivePrecisionLabRecipeModel({ ...base, evaluationAgeSteps })).toMatchObject({
            tone: 'ready',
            evaluationLabel: 'Evaluation fresh',
        });
    });
    it.each([null, -1, Number.NaN, Number.POSITIVE_INFINITY])('never calls absent or invalid evaluation age %s fresh', (evaluationAgeSteps) => {
        expect(derivePrecisionLabRecipeModel({ ...base, evaluationAgeSteps })).toMatchObject({
            dataset: base.dataset,
            tone: 'unavailable',
            evaluationLabel: 'No current evaluation',
        });
    });

});
