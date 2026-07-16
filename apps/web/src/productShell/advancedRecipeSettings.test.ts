import { describe, expect, it } from 'vitest';
import {
    PREPARED_PRESETS,
    type ValidatedStandardExperimentRecipeV2,
} from '@nn-playground/shared';
import { deriveAdvancedRecipeSettings } from './advancedRecipeSettings.ts';

function preparedRecipe(id: string): ValidatedStandardExperimentRecipeV2 {
    const match = PREPARED_PRESETS.find((entry) => entry.id === id)?.prepared;
    if (!match) throw new Error(`missing prepared preset ${id}`);
    return match.document.recipe;
}

function advancedRegressionRecipe(
    training: ValidatedStandardExperimentRecipeV2['training'],
): ValidatedStandardExperimentRecipeV2 {
    const base = preparedRecipe('regression-plane');
    return {
        ...base,
        training,
        objective: {
            ...base.objective,
            dataLoss: { kind: 'huber', delta: 1.75 },
            penalty: { kind: 'l2', coefficient: 0.012, applyTo: 'weights' },
        },
    } as ValidatedStandardExperimentRecipeV2;
}

describe('deriveAdvancedRecipeSettings', () => {
    it('omits the default constant schedule, SGD, no penalty, and no clipping', () => {
        expect(deriveAdvancedRecipeSettings(preparedRecipe('xor-hidden'))).toEqual([]);
    });

    it('returns exact ordered step, momentum, Huber, penalty, and clipping settings', () => {
        const base = preparedRecipe('regression-plane');
        const recipe = advancedRegressionRecipe({
            ...base.training,
            schedule: { kind: 'step', interval: 17, gamma: 0.63 },
            optimizer: { kind: 'sgd-momentum', momentum: 0.81 },
            gradientClipping: {
                kind: 'global-norm',
                maximumNorm: 2.5,
                scope: 'total-objective-gradient',
            },
        });

        expect(deriveAdvancedRecipeSettings(recipe)).toEqual([
            { id: 'schedule-step-interval', label: 'Step interval', value: '17' },
            { id: 'schedule-step-gamma', label: 'Step gamma', value: '0.63' },
            { id: 'optimizer-momentum', label: 'Momentum', value: '0.81' },
            { id: 'huber-delta', label: 'Huber delta', value: '1.75' },
            { id: 'penalty-l2', label: 'L2 coefficient (weights)', value: '0.012' },
            {
                id: 'gradient-clip-maximum',
                label: 'Global norm clip (total objective gradient)',
                value: '2.5',
            },
        ]);
    });

    it('returns exact ordered cosine and Adam settings', () => {
        const base = preparedRecipe('regression-plane');
        const recipe = advancedRegressionRecipe({
            ...base.training,
            schedule: { kind: 'cosine', totalSteps: 2500, minimumRate: 0.004 },
            optimizer: { kind: 'adam', beta1: 0.81, beta2: 0.97, epsilon: 1e-8 },
            gradientClipping: { kind: 'none' },
        });

        expect(deriveAdvancedRecipeSettings(recipe).slice(0, 5)).toEqual([
            { id: 'schedule-cosine-total-steps', label: 'Cosine total steps', value: '2500' },
            { id: 'schedule-cosine-minimum-rate', label: 'Cosine minimum rate', value: '0.004' },
            { id: 'optimizer-adam-beta1', label: 'Adam beta1', value: '0.81' },
            { id: 'optimizer-adam-beta2', label: 'Adam beta2', value: '0.97' },
            { id: 'optimizer-adam-epsilon', label: 'Adam epsilon', value: '1e-8' },
        ]);
    });
});
