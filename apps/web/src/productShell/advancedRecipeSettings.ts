import type { ValidatedStandardExperimentRecipeV2 } from '@nn-playground/shared';

export interface AdvancedRecipeSetting {
    readonly id: string;
    readonly label: string;
    readonly value: string;
}

export function deriveAdvancedRecipeSettings(
    recipe: ValidatedStandardExperimentRecipeV2,
): readonly AdvancedRecipeSetting[] {
    const settings: AdvancedRecipeSetting[] = [];
    const add = (id: string, label: string, value: number) => {
        settings.push({ id, label, value: String(value) });
    };
    const schedule = recipe.training.schedule;
    if (schedule.kind === 'step') {
        add('schedule-step-interval', 'Step interval', schedule.interval);
        add('schedule-step-gamma', 'Step gamma', schedule.gamma);
    } else if (schedule.kind === 'cosine') {
        add('schedule-cosine-total-steps', 'Cosine total steps', schedule.totalSteps);
        add(
            'schedule-cosine-minimum-rate',
            'Cosine minimum rate',
            schedule.minimumRate,
        );
    }

    const optimizer = recipe.training.optimizer;
    if (optimizer.kind === 'sgd-momentum') {
        add('optimizer-momentum', 'Momentum', optimizer.momentum);
    } else if (optimizer.kind === 'adam') {
        add('optimizer-adam-beta1', 'Adam beta1', optimizer.beta1);
        add('optimizer-adam-beta2', 'Adam beta2', optimizer.beta2);
        add('optimizer-adam-epsilon', 'Adam epsilon', optimizer.epsilon);
    }

    if (recipe.objective.dataLoss.kind === 'huber') {
        add('huber-delta', 'Huber delta', recipe.objective.dataLoss.delta);
    }

    const penalty = recipe.objective.penalty;
    if (penalty.kind !== 'none') {
        add(
            `penalty-${penalty.kind}`,
            `${penalty.kind.toUpperCase()} coefficient (weights)`,
            penalty.coefficient,
        );
    }

    const clipping = recipe.training.gradientClipping;
    if (clipping.kind === 'global-norm') {
        add(
            'gradient-clip-maximum',
            'Global norm clip (total objective gradient)',
            clipping.maximumNorm,
        );
    }

    return settings;
}
