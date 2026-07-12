import { describe, expect, it } from 'vitest';
import {
    selectTrainingExplanations,
    TRAINING_EXPLANATION_RULES,
    VALID_RELATED_PANEL_IDS,
    type ExplanationContext,
} from './trainingExplanations.ts';

const baseContext: ExplanationContext = {
    currentStep: 10,
    fullEvaluation: {
        step: 10,
        trainDataLoss: 0.4,
        testDataLoss: 0.5,
        trainAccuracy: 0.7,
        testAccuracy: 0.65,
        trainSampleCount: 70,
        testSampleCount: 30,
    },
};

describe('selectTrainingExplanations', () => {
    it('returns deterministic explanations ordered by priority', () => {
        const explanations = selectTrainingExplanations({
            ...baseContext,
            pauseReason: 'diverged',
            currentStep: 12,
            fullEvaluation: {
                ...baseContext.fullEvaluation!,
                step: 10,
                trainDataLoss: 0.2,
                testDataLoss: 0.6,
            },
        });

        expect(explanations.map((rule) => rule.id)).toEqual([
            'pause-diverged',
            'evaluation-age',
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

    it('explains exact full-evaluation age without a global stale flag', () => {
        const [explanation] = selectTrainingExplanations({
            ...baseContext,
            currentStep: 12,
        });

        expect(explanation).toMatchObject({
            id: 'evaluation-age',
            title: 'Full evaluation trails the batch trend',
        });
    });

    it('never infers generalization without one paired full evaluation', () => {
        const explanations = selectTrainingExplanations({
            currentStep: 50,
            fullEvaluation: null,
        });

        expect(explanations.map((rule) => rule.id)).not.toContain('generalization-gap');
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
            'evaluation-age': [
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
