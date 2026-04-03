// ── Confusion Matrix Component ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { EmptyState } from '../common/EmptyState.tsx';

function formatPercent(value: number, total: number): string {
    if (total === 0) return '0.0%';
    return `${((value / total) * 100).toFixed(1)}%`;
}

function formatRatio(numerator: number, denominator: number): string {
    if (denominator === 0) return '0.0%';
    return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

export const ConfusionMatrix = memo(function ConfusionMatrix() {
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const cm = useTrainingStore((s) => s.snapshot?.testMetrics.confusionMatrix);

    if (problemType !== 'classification') return null;
    if (testPoints.length === 0 || !cm) {
        return (
            <div className="panel confusion-matrix">
                <div className="panel__title">Confusion Matrix (Test Set)</div>
                <EmptyState
                    icon="📊"
                    title="No test data"
                    description="Train the model to generate test predictions and evaluation metrics."
                />
            </div>
        );
    }

    const total = cm.tp + cm.tn + cm.fp + cm.fn;
    const actual0Total = cm.tn + cm.fp;
    const actual1Total = cm.fn + cm.tp;
    const predicted0Total = cm.tn + cm.fn;
    const predicted1Total = cm.fp + cm.tp;
    const accuracy = formatRatio(cm.tp + cm.tn, total);
    const precision = formatRatio(cm.tp, cm.tp + cm.fp);
    const recall = formatRatio(cm.tp, cm.tp + cm.fn);

    const Cell = ({ value, label, isCorrect }: { value: number; label: string; isCorrect: boolean }) => {
        const intensity = total === 0 ? 0.12 : Math.max(0.12, value / total);
        const backgroundColor = isCorrect
            ? `rgba(34, 197, 94, ${intensity.toFixed(2)})`
            : `rgba(239, 68, 68, ${intensity.toFixed(2)})`;

        return (
            <div
                aria-label={`${label} cell`}
                className={`cm-cell cm-${label.toLowerCase()}`}
                style={{ backgroundColor }}
            >
                <div className="cm-value">{value}</div>
                <div className="cm-percentage">{formatPercent(value, total)}</div>
                <div className="cm-label">{label}</div>
            </div>
        );
    };

    return (
        <div className="panel confusion-matrix">
            <div className="panel__title">Confusion Matrix (Test Set)</div>
            <div className="cm-grid-container">
                <div className="cm-axis-label">Predicted</div>
                <div className="cm-layout">
                    <div className="cm-axis-label cm-axis-label--side">Actual</div>
                    <div className="cm-grid">
                        <div className="cm-header cm-header--empty" />
                        <div className="cm-header">Pred 0</div>
                        <div className="cm-header">Pred 1</div>
                        <div className="cm-header">Total</div>

                        <div className="cm-header cm-header--row">Actual 0</div>
                        <Cell value={cm.tn} label="TN" isCorrect={true} />
                        <Cell value={cm.fp} label="FP" isCorrect={false} />
                        <div className="cm-total">{actual0Total}</div>

                        <div className="cm-header cm-header--row">Actual 1</div>
                        <Cell value={cm.fn} label="FN" isCorrect={false} />
                        <Cell value={cm.tp} label="TP" isCorrect={true} />
                        <div className="cm-total">{actual1Total}</div>

                        <div className="cm-header cm-header--row">Total</div>
                        <div className="cm-total">{predicted0Total}</div>
                        <div className="cm-total">{predicted1Total}</div>
                        <div className="cm-total cm-total--grand">{total}</div>
                    </div>
                </div>
                <div className="cm-metrics">
                    <div className="cm-metric">
                        <span className="cm-metric__label">Accuracy</span>
                        <span className="cm-metric__value">{accuracy}</span>
                    </div>
                    <div className="cm-metric">
                        <span className="cm-metric__label">Precision</span>
                        <span className="cm-metric__value">{precision}</span>
                    </div>
                    <div className="cm-metric">
                        <span className="cm-metric__label">Recall</span>
                        <span className="cm-metric__value">{recall}</span>
                    </div>
                </div>
            </div>
        </div>
    );
});
