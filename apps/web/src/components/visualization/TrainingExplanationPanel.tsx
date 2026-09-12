import { memo, type KeyboardEvent } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import { selectTrainingExplanations } from '../../explanations/trainingExplanations.ts';
import { focusExplanationActionTarget } from '../../explanations/explanationActionFocus.ts';

const PANEL_TITLE_ID = 'training-explanation-panel-title';

function stopShortcutPropagation(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.stopPropagation();
    }
}

export const TrainingExplanationPanel = /* @__PURE__ */ memo(function TrainingExplanationPanel() {
    const latestLiveSignal = useTrainingStore((state) => state.latestLiveSignal);
    const latestEvaluation = useTrainingStore((state) => state.latestEvaluation);
    const pauseReason = useTrainingStore((state) => state.pauseReason);
    const evidence = selectScientificEvidence({ latestLiveSignal, latestEvaluation });
    const fullEvaluation = evidence.fullEvaluation;

    const [explanation] = selectTrainingExplanations({
        currentStep: evidence.currentModel?.step ?? 0,
        fullEvaluation: fullEvaluation === null ? null : {
            step: fullEvaluation.step,
            trainDataLoss: fullEvaluation.trainDataLoss,
            testDataLoss: fullEvaluation.testDataLoss,
            trainAccuracy: fullEvaluation.trainAccuracy,
            testAccuracy: fullEvaluation.testAccuracy,
            trainSampleCount: fullEvaluation.trainSampleCount,
            testSampleCount: fullEvaluation.testSampleCount,
        },
        pauseReason,
    });

    if (!explanation) return null;

    return (
        <section
            aria-labelledby={PANEL_TITLE_ID}
            style={{
                marginTop: 8,
                padding: '10px 12px',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 6,
                background: 'rgba(255,255,255,0.04)',
                color: 'var(--text-primary)',
            }}
        >
            <h3 id={PANEL_TITLE_ID} style={{ margin: 0, fontSize: 12, fontWeight: 700 }}>
                Why did this happen?
            </h3>
            <div style={{ marginTop: 6, fontSize: 12, fontWeight: 700 }}>
                {explanation.title}
            </div>
            <p style={{ margin: '4px 0 0', fontSize: 12, lineHeight: 1.45, color: 'var(--text-secondary)' }}>
                {explanation.explanation}
            </p>
            {explanation.suggestedAction && (
                <p style={{ margin: '4px 0 0', fontSize: 12, lineHeight: 1.45, color: 'var(--text-secondary)' }}>
                    {explanation.suggestedAction}
                </p>
            )}
            {explanation.actions && explanation.actions.length > 0 && (
                <div
                    className="training-explanation-actions"
                    role="group"
                    aria-label="Suggested explanation actions"
                >
                    {explanation.actions.map((action, index) => {
                        const descriptionId = `training-explanation-action-${index}-description`;
                        return (
                            <button
                                type="button"
                                key={`${action.targetPanelId}-${action.label}`}
                                className="training-explanation-action"
                                aria-label={action.label}
                                aria-describedby={descriptionId}
                                onClick={() => focusExplanationActionTarget(action.targetPanelId)}
                                onKeyDown={stopShortcutPropagation}
                            >
                                <span className="training-explanation-action__label">
                                    {action.label}
                                </span>
                                <span
                                    id={descriptionId}
                                    className="training-explanation-action__reason"
                                >
                                    {action.learningReason}
                                </span>
                            </button>
                        );
                    })}
                </div>
            )}
        </section>
    );
});
