import { PRESETS } from '@nn-playground/shared';
import type { Preset } from '@nn-playground/shared';
import type { LessonDefinition, LessonTarget } from './types.ts';

export const VALID_LESSON_TARGETS = [
    'data',
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
