import { PRESETS } from '@nn-playground/shared';
import type { Preset } from '@nn-playground/shared';
import type { LessonDefinition, LessonTarget } from './types.ts';

export const VALID_LESSON_TARGETS = [
    'data',
    'features',
    'network',
    'hyperparams',
    'transport',
] as const satisfies readonly LessonTarget[];

export const DEFAULT_LESSON_ID = 'lesson-xor-hidden-layers';

export const LESSON_DEFINITIONS = [
    {
        id: 'lesson-xor-hidden-layers',
        title: 'XOR Needs Hidden Layers',
        summary: 'See why a straight boundary cannot solve XOR and how hidden layers create bends.',
        presetId: 'xor-hidden',
        estimatedMinutes: 4,
        steps: [
            {
                id: 'read-xor-pattern',
                title: 'Read the XOR pattern',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'The XOR preset alternates labels by quadrant, so no single straight line can separate every point.',
            },
            {
                id: 'give-model-capacity',
                title: 'Give the model capacity',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'Two hidden layers let the network combine simple bends into the corners needed for XOR.',
            },
            {
                id: 'use-steady-updates',
                title: 'Use steady updates',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'A moderate learning rate and small batches make the loss react without bouncing wildly.',
            },
            {
                id: 'train-in-small-moves',
                title: 'Train in small moves',
                target: 'transport',
                phase: 'run',
                body: 'Step or play from the transport controls and watch the boundary change as weights update.',
            },
        ],
    },
    {
        id: 'lesson-single-neuron-linear-separator',
        title: 'Single Neuron Linear Separator',
        summary: 'Start with the smallest classifier and see why one neuron draws one straight boundary.',
        presetId: 'single-neuron',
        estimatedMinutes: 3,
        steps: [
            {
                id: 'inspect-gaussian-data',
                title: 'Inspect the two clusters',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'The Gaussian dataset places two clouds where a straight separator is enough to split the labels.',
            },
            {
                id: 'inspect-single-neuron',
                title: 'Use one neuron',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'With no hidden layers, the model can only learn a weighted sum of the input features.',
            },
            {
                id: 'train-linear-boundary',
                title: 'Train the straight boundary',
                target: 'transport',
                phase: 'run',
                body: 'Run a few steps and watch the boundary rotate toward the gap between the two clusters.',
            },
        ],
    },
    {
        id: 'lesson-regression-plane-baseline',
        title: 'Regression Plane Baseline',
        summary: 'Switch from class labels to continuous values and fit a simple plane.',
        presetId: 'regression-plane',
        estimatedMinutes: 3,
        steps: [
            {
                id: 'switch-to-regression',
                title: 'Switch to regression data',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'Regression predicts a continuous surface instead of choosing between classes.',
            },
            {
                id: 'inspect-regression-settings',
                title: 'Use regression settings',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Mean squared error and a linear output match a continuous target better than classification settings.',
            },
            {
                id: 'confirm-linear-model',
                title: 'Keep the model linear',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'No hidden layer is needed when the target is already shaped like a plane.',
            },
            {
                id: 'train-plane-fit',
                title: 'Train the plane fit',
                target: 'transport',
                phase: 'run',
                body: 'Run training and watch the loss drop as the plane aligns with the generated surface.',
            },
        ],
    },
    {
        id: 'lesson-circle-hidden-layer',
        title: 'Circle With One Hidden Layer',
        summary: 'Watch a small hidden layer bend a straight model into a circular boundary.',
        presetId: 'circle-one-layer',
        estimatedMinutes: 4,
        steps: [
            {
                id: 'read-circle-shape',
                title: 'Read the circle shape',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'The circle dataset puts one class near the center and the other around it, so a straight line cannot separate both rings.',
            },
            {
                id: 'use-one-hidden-layer',
                title: 'Use one hidden layer',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'A single hidden layer gives the model several small bends that can combine into a rounded boundary.',
            },
            {
                id: 'keep-updates-smooth',
                title: 'Keep updates smooth',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Moderate learning settings make it easier to see the boundary curve inward instead of jumping around.',
            },
            {
                id: 'train-circle-boundary',
                title: 'Train the curved boundary',
                target: 'transport',
                phase: 'run',
                body: 'Run training and compare the center region against the outer ring as the hidden layer learns the curve.',
            },
        ],
    },
    {
        id: 'lesson-feature-engineering-circle',
        title: 'Feature Engineering Helps',
        summary: 'Use squared inputs to make a circular problem easier before adding hidden layers.',
        presetId: 'feature-engineering',
        estimatedMinutes: 4,
        steps: [
            {
                id: 'inspect-engineered-features',
                title: 'Inspect engineered features',
                target: 'features',
                tab: 'features',
                phase: 'build',
                body: 'Squared coordinate features expose distance-from-center information that the raw x and y inputs hide from a linear model.',
            },
            {
                id: 'compare-with-linear-model',
                title: 'Keep the network simple',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'With the right features, even a model with no hidden layers can draw a useful circular separator.',
            },
            {
                id: 'use-classification-loss',
                title: 'Use classification loss',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Cross-entropy and a sigmoid output keep the lesson focused on class probability rather than regression error.',
            },
            {
                id: 'train-feature-model',
                title: 'Train the feature model',
                target: 'transport',
                phase: 'run',
                body: 'Run training and notice how the boundary becomes round without adding hidden-layer capacity.',
            },
        ],
    },
    {
        id: 'lesson-spiral-depth',
        title: 'Spiral Needs Depth',
        summary: 'Explore why twisted data needs more capacity and steadier training.',
        presetId: 'spiral-deep',
        estimatedMinutes: 5,
        steps: [
            {
                id: 'read-spiral-twist',
                title: 'Read the spiral twist',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'The spiral arms wrap around each other, so the model needs many local bends to follow the class boundary.',
            },
            {
                id: 'inspect-deeper-network',
                title: 'Inspect the deeper network',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'Multiple hidden layers let early bends combine into more detailed bends later in the network.',
            },
            {
                id: 'slow-spiral-learning',
                title: 'Slow the learning down',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'A smaller learning rate helps the deeper model adjust gradually instead of overshooting the narrow spiral arms.',
            },
            {
                id: 'train-spiral-boundary',
                title: 'Train the spiral boundary',
                target: 'transport',
                phase: 'run',
                body: 'Run training for longer than the simpler lessons and watch the boundary untwist section by section.',
            },
        ],
    },
    {
        id: 'lesson-learning-rate-tuning',
        title: 'Learning Rate Tuning',
        summary: 'Compare update sizes and learn why steady progress beats dramatic jumps.',
        presetId: 'xor-hidden',
        estimatedMinutes: 4,
        steps: [
            {
                id: 'find-learning-rate-controls',
                title: 'Find the update-size controls',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Learning rate controls how far each training update moves the weights after a gradient step.',
            },
            {
                id: 'connect-rate-to-loss',
                title: 'Connect rate to the loss curve',
                target: 'transport',
                phase: 'run',
                body: 'Step through training and watch whether loss falls smoothly, stalls, or jumps around.',
            },
            {
                id: 'keep-capacity-fixed',
                title: 'Keep the model fixed',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'Change one idea at a time: keep the topology stable while experimenting with update size.',
            },
            {
                id: 'retry-with-smaller-steps',
                title: 'Retry with smaller steps',
                target: 'transport',
                phase: 'run',
                body: 'After changing the learning rate or clipping settings, reset and run again to compare the curve.',
            },
        ],
    },
    {
        id: 'lesson-regularization-overfitting',
        title: 'Regularization and Overfitting',
        summary: 'Use regularization and capacity choices to manage the gap between train and test behavior.',
        presetId: 'spiral-deep',
        estimatedMinutes: 5,
        steps: [
            {
                id: 'start-with-flexible-model',
                title: 'Start with a flexible model',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'A deeper network can trace complicated data, but extra flexibility can also memorize noisy details.',
            },
            {
                id: 'open-regularization-controls',
                title: 'Open regularization controls',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Regularization nudges the model toward smaller weights so it must prefer simpler explanations.',
            },
            {
                id: 'compare-training-and-holdout',
                title: 'Compare training and holdout',
                target: 'transport',
                phase: 'run',
                body: 'Run training and watch for a pattern where training improves faster than held-out test behavior.',
            },
            {
                id: 'simplify-or-regularize',
                title: 'Simplify or regularize',
                target: 'transport',
                phase: 'run',
                body: 'After simplifying or regularizing, run another short trial and compare whether the train/test gap narrows.',
            },
        ],
    },
    {
        id: 'lesson-noisy-data-robustness',
        title: 'Noisy Data Robustness',
        summary: 'Learn why noisy points need smoother decisions and more cautious interpretation.',
        presetId: 'circle-one-layer',
        estimatedMinutes: 4,
        steps: [
            {
                id: 'inspect-noise-controls',
                title: 'Inspect the noise controls',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'Noise moves points away from the clean pattern, so a perfect boundary may be the wrong goal.',
            },
            {
                id: 'prefer-smooth-boundaries',
                title: 'Prefer smooth boundaries',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'A modest hidden layer can capture the circle while avoiding tiny bends around individual noisy points.',
            },
            {
                id: 'use-conservative-settings',
                title: 'Use conservative settings',
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                body: 'Regularization and steady update sizes help the model ignore isolated noisy examples.',
            },
            {
                id: 'train-and-read-uncertainty',
                title: 'Train and read uncertainty',
                target: 'transport',
                phase: 'run',
                body: 'Run training, then treat uncertain or isolated mistakes as signals about data quality, not just model quality.',
            },
        ],
    },
    {
        id: 'lesson-three-class-softmax',
        title: 'Three-Class Softmax Lab',
        summary: 'Learn how three output neurons compete through softmax on a compact multiclass dataset.',
        presetId: 'three-class-clusters',
        estimatedMinutes: 5,
        steps: [
            {
                id: 'read-three-clusters',
                title: 'Read the three clusters',
                target: 'data',
                tab: 'data',
                phase: 'build',
                body: 'Each point belongs to Class 0, Class 1, or Class 2, so the model must choose among three labels instead of drawing a binary split.',
            },
            {
                id: 'inspect-three-outputs',
                title: 'Inspect the three outputs',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'The network uses one output neuron per class. Softmax turns those outputs into competing confidence scores.',
            },
            {
                id: 'connect-softmax-loss',
                title: 'Keep the softmax tuple together',
                target: 'network',
                tab: 'network',
                phase: 'build',
                body: 'This preset keeps three outputs, softmax, and categorical cross-entropy paired so the highest class score wins after each update.',
            },
            {
                id: 'train-class-regions',
                title: 'Train three class regions',
                target: 'transport',
                phase: 'run',
                body: 'Start training and watch the boundary divide the plane into three winning-class regions, with uncertainty near class borders.',
            },
        ],
    },
] as const satisfies readonly LessonDefinition[];

export function getLessonDefinition(id = DEFAULT_LESSON_ID): LessonDefinition | null {
    return LESSON_DEFINITIONS.find((lesson) => lesson.id === id) ?? null;
}

export function getLessonPreset(lesson: LessonDefinition): Preset {
    const preset = PRESETS.find((item) => item.id === lesson.presetId);
    if (!preset) {
        throw new Error(`Missing guided lesson preset: ${lesson.presetId}`);
    }
    return preset;
}

export type { LessonDefinition, LessonStep, LessonTarget } from './types.ts';
