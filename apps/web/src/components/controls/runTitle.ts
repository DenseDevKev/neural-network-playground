import type {
    ExperimentRunRecordV2,
    StandardExperimentRecipeV2,
} from '@nn-playground/shared';

const DATASET_LABELS: Readonly<Record<StandardExperimentRecipeV2['task']['dataset'], string>> = {
    circle: 'Circle',
    xor: 'XOR',
    gauss: 'Gaussian',
    spiral: 'Spiral',
    moons: 'Moons',
    checkerboard: 'Checkerboard',
    rings: 'Rings',
    heart: 'Heart',
    'three-class-clusters': 'Three-class clusters',
    'reg-plane': 'Regression plane',
    'reg-gauss': 'Regression gaussian',
};

export function createDefaultRunTitle(
    recipe: StandardExperimentRecipeV2,
    snapshot: ExperimentRunRecordV2['snapshot'],
): string {
    const outputWidth = recipe.task.kind === 'multiclass-classification' ? 3 : 1;
    const architecture = [
        recipe.inputs.featureIds.length,
        ...recipe.model.hiddenLayers,
        outputWidth,
    ].join('-');
    return `${DATASET_LABELS[recipe.task.dataset]} · ${architecture} · step ${snapshot.model.step}`;
}
