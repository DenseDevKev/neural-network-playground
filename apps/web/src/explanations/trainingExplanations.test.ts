import { describe, expect, it } from 'vitest';
import {
    selectTrainingExplanations,
    TRAINING_EXPLANATION_RULES,
    VALID_RELATED_PANEL_IDS,
    type ExplanationContext,
} from './trainingExplanations.ts';

const baseContext: ExplanationContext = {
    step: 10,
    trainLoss: 0.4,
    testLoss: 0.5,
    trainAccuracy: 0.7,
    testAccuracy: 0.65,
    testMetricsStale: false,
};

describe('selectTrainingExplanations', () => {
    it('returns deterministic explanations ordered by priority', () => {
        const explanations = selectTrainingExplanations({
            ...baseContext,
            pauseReason: 'diverged',
            testMetricsStale: true,
            trainLoss: 0.2,
            testLoss: 0.6,
        });

        expect(explanations.map((rule) => rule.id)).toEqual([
            'pause-diverged',
            'test-metrics-stale',
            'generalization-gap',
        ]);
    });

    it('explains divergence stop reasons with a safe suggested action', () => {
        const [explanation] = selectTrainingExplanations({
            ...baseContext,
            pauseReason: 'diverged',
        });

        expect(explanation).toMatchObject({
            id: 'pause-diverged',
            title: 'Training diverged',
            suggestedAction: expect.stringContaining('learning rate'),
        });
        expect(explanation.relatedPanelIds).toContain('hyperparams');
    });

    it('explains stale test metrics without requiring a pause reason', () => {
        const [explanation] = selectTrainingExplanations({
            ...baseContext,
            testMetricsStale: true,
        });

        expect(explanation).toMatchObject({
            id: 'test-metrics-stale',
            title: 'Test metrics are catching up',
        });
    });

    it('keeps related panel ids constrained to known layout panels', () => {
        const valid = new Set<string>(VALID_RELATED_PANEL_IDS);

        for (const rule of TRAINING_EXPLANATION_RULES) {
            for (const panelId of rule.relatedPanelIds ?? []) {
                expect(valid.has(panelId), `${rule.id} uses ${panelId}`).toBe(true);
            }
        }
    });

    it('exposes deterministic action metadata for each explanation rule', () => {
        const actionsByRule = Object.fromEntries(
            TRAINING_EXPLANATION_RULES.map((rule) => [
                rule.id,
                rule.actions?.map((action) => ({
                    label: action.label,
                    targetPanelId: action.targetPanelId,
                })),
            ]),
        );

        expect(actionsByRule).toMatchObject({
            'pause-diverged': [
                { label: 'Tune learning rate & clipping', targetPanelId: 'hyperparams' },
                { label: 'Read the loss spike', targetPanelId: 'loss' },
            ],
            'pause-error': [
                { label: 'Review run settings', targetPanelId: 'hyperparams' },
            ],
            'pause-plateau': [
                { label: 'Inspect the plateau', targetPanelId: 'loss' },
                { label: 'Adjust model capacity', targetPanelId: 'network' },
            ],
            'test-metrics-stale': [
                { label: 'Open loss & accuracy', targetPanelId: 'loss' },
            ],
            'generalization-gap': [
                { label: 'Tune regularization', targetPanelId: 'hyperparams' },
                { label: 'Compare train vs test', targetPanelId: 'loss' },
            ],
        });
    });

    it('keeps explanation action targets constrained to known layout panels', () => {
        const valid = new Set<string>(VALID_RELATED_PANEL_IDS);

        for (const rule of TRAINING_EXPLANATION_RULES) {
            expect(rule.actions?.length, `${rule.id} has actions`).toBeGreaterThan(0);
            for (const action of rule.actions ?? []) {
                expect(valid.has(action.targetPanelId), `${rule.id} action ${action.label} targets ${action.targetPanelId}`).toBe(true);
            }
        }
    });
});
