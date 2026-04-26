import { memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { selectTrainingExplanations } from '../../explanations/trainingExplanations.ts';

const PANEL_TITLE_ID = 'training-explanation-panel-title';

export const TrainingExplanationPanel = memo(function TrainingExplanationPanel() {
    const snapshot = useTrainingStore((state) => state.snapshot);
    const pauseReason = useTrainingStore((state) => state.pauseReason);
    const testMetricsStale = useTrainingStore((state) => state.testMetricsStale);

    if (!snapshot) return null;

    const [explanation] = selectTrainingExplanations({
        step: snapshot.step,
        trainLoss: snapshot.trainLoss,
        testLoss: snapshot.testLoss,
        trainAccuracy: snapshot.trainMetrics.accuracy,
        testAccuracy: snapshot.testMetrics.accuracy,
        pauseReason,
        testMetricsStale,
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
        </section>
    );
});
