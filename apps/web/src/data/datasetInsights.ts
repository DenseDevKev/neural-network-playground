import type { DatasetType } from '@nn-playground/engine';

export const DATASET_TOOLTIPS: Record<DatasetType, string> = {
    circle: 'Cause: circle data wraps one class around another. Effect: hidden layers or squared features help make a curved boundary.',
    xor: 'Cause: XOR alternates labels by quadrant. Effect: a straight boundary fails, so hidden layers have something meaningful to learn.',
    gauss: 'Cause: Gaussian blobs are mostly separable clusters. Effect: simple models learn quickly unless noise overlaps the classes.',
    spiral: 'Cause: spiral arms twist around each other. Effect: deeper networks usually need more training steps to untangle the boundary.',
    moons: 'Cause: moon shapes curve past each other. Effect: extra neurons help bend the decision boundary between the arcs.',
    checkerboard: 'Cause: checkerboard labels alternate in many small regions. Effect: the model needs more local bends and may train slowly.',
    rings: 'Cause: rings stack circular bands. Effect: curved features or hidden layers make the class transitions easier to fit.',
    heart: 'Cause: the heart outline has tight curves and a notch. Effect: low-capacity networks underfit the shape.',
    'reg-plane': 'Cause: plane regression is almost linear. Effect: a simple network can fit it without hidden layers.',
    'reg-gauss': 'Cause: multi-Gauss regression has several smooth bumps. Effect: hidden layers help approximate the changing surface.',
};

export function getDatasetTopologyHint(dataset: DatasetType, hiddenLayers: readonly number[]): string | null {
    if (dataset === 'xor' && hiddenLayers.length === 0) {
        return 'XOR is not linearly separable, so add a hidden layer before training.';
    }
    if (dataset === 'gauss' && hiddenLayers.length >= 2) {
        return 'Gaussian blobs are usually simple enough that this much depth can obscure the linear story.';
    }
    if (
        ['spiral', 'moons', 'checkerboard', 'rings', 'heart', 'reg-gauss'].includes(dataset) &&
        hiddenLayers.length === 0
    ) {
        return 'This dataset usually needs hidden layers to bend the model beyond a straight boundary.';
    }
    return null;
}
