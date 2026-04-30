import type { PauseReason } from '@nn-playground/shared';
import type { LeftTabId, RightTabId } from '../store/useLayoutStore.ts';

export type RelatedPanelId = LeftTabId | RightTabId;

export const VALID_RELATED_PANEL_IDS = [
    'presets',
    'data',
    'features',
    'network',
    'hyperparams',
    'config',
    'boundary',
    'loss',
    'confusion',
    'inspection',
    'code',
] as const satisfies readonly RelatedPanelId[];

export interface ExplanationContext {
    step: number;
    trainLoss: number;
    testLoss: number;
    trainAccuracy?: number;
    testAccuracy?: number;
    pauseReason?: PauseReason | null;
    testMetricsStale?: boolean;
}

export interface ExplanationRuleDescriptor {
    id: string;
    priority: number;
    title: string;
    explanation: string;
    suggestedAction?: string;
    relatedPanelIds?: RelatedPanelId[];
}

export interface RuntimeExplanationRule extends ExplanationRuleDescriptor {
    when: (context: ExplanationContext) => boolean;
}

const GENERALIZATION_GAP_THRESHOLD = 0.15;

function finite(value: number | undefined): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

export const TRAINING_EXPLANATION_RULES: readonly RuntimeExplanationRule[] = [
    {
        id: 'pause-diverged',
        priority: 100,
        title: 'Training diverged',
        explanation: 'A loss value became non-finite, so training paused before the model state became harder to inspect.',
        suggestedAction: 'Try lowering the learning rate, enabling gradient clipping, or resetting the network.',
        relatedPanelIds: ['hyperparams', 'loss'],
        when: (context) => context.pauseReason === 'diverged',
    },
    {
        id: 'pause-error',
        priority: 90,
        title: 'Training stopped after an error',
        explanation: 'The worker reported a runtime error and paused the training loop.',
        suggestedAction: 'Reset the run after checking the current settings.',
        relatedPanelIds: ['hyperparams'],
        when: (context) => context.pauseReason === 'error',
    },
    {
        id: 'pause-plateau',
        priority: 80,
        title: 'Loss plateaued',
        explanation: 'Recent metrics stopped improving enough to satisfy the plateau stop condition.',
        suggestedAction: 'Try a different learning rate, more hidden units, or a fresh initialization.',
        relatedPanelIds: ['loss', 'network'],
        when: (context) => context.pauseReason === 'plateau',
    },
    {
        id: 'test-metrics-stale',
        priority: 60,
        title: 'Test metrics are catching up',
        explanation: 'The test set is evaluated less often than training updates, so the latest test value may be a cached reading.',
        suggestedAction: 'Pause briefly or wait for the next full test evaluation before judging generalization.',
        relatedPanelIds: ['loss'],
        when: (context) => context.testMetricsStale === true,
    },
    {
        id: 'generalization-gap',
        priority: 40,
        title: 'Test loss is higher than train loss',
        explanation: 'The model is fitting the training data better than the held-out test data.',
        suggestedAction: 'Try more regularization, less training time, or a simpler architecture.',
        relatedPanelIds: ['loss', 'hyperparams'],
        when: (context) => (
            context.step > 0 &&
            finite(context.trainLoss) &&
            finite(context.testLoss) &&
            context.testLoss - context.trainLoss >= GENERALIZATION_GAP_THRESHOLD
        ),
    },
];

export function selectTrainingExplanations(
    context: ExplanationContext,
): ExplanationRuleDescriptor[] {
    return TRAINING_EXPLANATION_RULES
        .filter((rule) => rule.when(context))
        .sort((a, b) => (
            b.priority - a.priority ||
            (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
        ))
        .map(({ when: _when, ...descriptor }) => descriptor);
}
