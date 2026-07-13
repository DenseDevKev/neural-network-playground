import { Fragment, memo } from 'react';
import type {
    ConfusionMatrixData,
    MulticlassConfusionMatrixData,
} from '@nn-playground/engine';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { EmptyState } from '../common/EmptyState.tsx';

function formatRatio(numerator: number, denominator: number): string {
    return denominator > 0 ? `${((numerator / denominator) * 100).toFixed(1)}%` : '0.0%';
}

function formatPercent(value: number, total: number): string {
    return formatRatio(value, total);
}

function isMulticlass(
    matrix: ConfusionMatrixData | MulticlassConfusionMatrixData,
): matrix is MulticlassConfusionMatrixData {
    return 'classCount' in matrix;
}

function ProvenanceCaption({
    evaluationId,
    step,
    sampleCount,
}: {
    evaluationId: number;
    step: number;
    sampleCount: number;
}) {
    return (
        <p className="cm-note">
            {`Evaluation ${evaluationId} · model step ${step.toLocaleString()} · all ${sampleCount.toLocaleString()} test samples`}
        </p>
    );
}

function BinaryMatrix({
    matrix,
    evaluationId,
    step,
    sampleCount,
}: {
    matrix: ConfusionMatrixData;
    evaluationId: number;
    step: number;
    sampleCount: number;
}) {
    const total = matrix.tp + matrix.tn + matrix.fp + matrix.fn;
    const actual0Total = matrix.tn + matrix.fp;
    const actual1Total = matrix.fn + matrix.tp;
    const predicted0Total = matrix.tn + matrix.fn;
    const predicted1Total = matrix.fp + matrix.tp;

    const Cell = ({ value, label, correct }: {
        value: number;
        label: string;
        correct: boolean;
    }) => {
        const intensity = total === 0 ? 0.12 : Math.max(0.12, value / total);
        return (
            <div
                aria-label={`${label} cell`}
                className={`cm-cell cm-${label.toLowerCase()}`}
                style={{
                    backgroundColor: correct
                        ? `rgba(34, 197, 94, ${intensity.toFixed(2)})`
                        : `rgba(239, 68, 68, ${intensity.toFixed(2)})`,
                }}
            >
                <div className="cm-value">{value}</div>
                <div className="cm-percentage">{formatPercent(value, total)}</div>
                <div className="cm-label">{label}</div>
            </div>
        );
    };

    return (
        <div className="panel confusion-matrix">
            <div className="panel__title">Binary Confusion Matrix (Full Test Split)</div>
            <div className="cm-grid-container">
                <div className="cm-axis-label">Predicted</div>
                <div className="cm-layout">
                    <div className="cm-axis-label cm-axis-label--side">Actual</div>
                    <div className="cm-grid">
                        <div className="cm-header cm-header--empty" />
                        <div className="cm-header">C0</div>
                        <div className="cm-header">C1</div>
                        <div className="cm-header">Total</div>
                        <div className="cm-header cm-header--row">C0</div>
                        <Cell value={matrix.tn} label="TN" correct />
                        <Cell value={matrix.fp} label="FP" correct={false} />
                        <div className="cm-total">{actual0Total}</div>
                        <div className="cm-header cm-header--row">C1</div>
                        <Cell value={matrix.fn} label="FN" correct={false} />
                        <Cell value={matrix.tp} label="TP" correct />
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
                        <span className="cm-metric__value">{formatRatio(matrix.tp + matrix.tn, total)}</span>
                    </div>
                    <div className="cm-metric">
                        <span className="cm-metric__label">Precision</span>
                        <span className="cm-metric__value">{formatRatio(matrix.tp, matrix.tp + matrix.fp)}</span>
                    </div>
                    <div className="cm-metric">
                        <span className="cm-metric__label">Recall</span>
                        <span className="cm-metric__value">{formatRatio(matrix.tp, matrix.tp + matrix.fn)}</span>
                    </div>
                </div>
                <ProvenanceCaption evaluationId={evaluationId} step={step} sampleCount={sampleCount} />
            </div>
        </div>
    );
}

function MulticlassMatrix({
    matrix,
    evaluationId,
    step,
    sampleCount,
}: {
    matrix: MulticlassConfusionMatrixData;
    evaluationId: number;
    step: number;
    sampleCount: number;
}) {
    const labels = matrix.classLabels;
    const counts = matrix.counts;
    const rowTotal = (actual: number) => labels.reduce<number>(
        (sum, _label, predicted) => sum + counts[actual * matrix.classCount + predicted],
        0,
    );
    const columnTotal = (predicted: number) => labels.reduce<number>(
        (sum, _label, actual) => sum + counts[actual * matrix.classCount + predicted],
        0,
    );
    const total = counts.reduce((sum, count) => sum + count, 0);
    const correct = labels.reduce<number>(
        (sum, _label, index) => sum + counts[index * matrix.classCount + index],
        0,
    );

    return (
        <div className="panel confusion-matrix">
            <div className="panel__title">Multiclass Confusion Matrix (Full Test Split)</div>
            <div className="cm-grid-container">
                <div className="cm-axis-label">Predicted</div>
                <div className="cm-layout">
                    <div className="cm-axis-label cm-axis-label--side">Actual</div>
                    <div className="cm-grid cm-grid--multiclass">
                        <div className="cm-header cm-header--empty" />
                        {labels.map((label) => <div className="cm-header" key={`pred-${label}`}>C{label}</div>)}
                        <div className="cm-header">Total</div>
                        {labels.map((actual, actualIndex) => (
                            <Fragment key={`actual-${actual}`}>
                                <div className="cm-header cm-header--row">C{actual}</div>
                                {labels.map((predicted, predictedIndex) => {
                                    const value = counts[actualIndex * matrix.classCount + predictedIndex];
                                    const correctCell = actualIndex === predictedIndex;
                                    return (
                                        <div
                                            key={`${actual}-${predicted}`}
                                            aria-label={`${value} test samples with actual Class ${actual} predicted Class ${predicted}`}
                                            className="cm-cell cm-cell--multiclass"
                                            style={{
                                                backgroundColor: correctCell
                                                    ? `rgba(34, 197, 94, ${Math.max(0.12, value / Math.max(1, total)).toFixed(2)})`
                                                    : `rgba(239, 68, 68, ${Math.max(0.12, value / Math.max(1, total)).toFixed(2)})`,
                                            }}
                                        >
                                            <div className="cm-value">{value}</div>
                                            <div className="cm-percentage">{formatPercent(value, total)}</div>
                                        </div>
                                    );
                                })}
                                <div className="cm-total">{rowTotal(actualIndex)}</div>
                            </Fragment>
                        ))}
                        <div className="cm-header cm-header--row">Total</div>
                        {labels.map((_label, predicted) => (
                            <div className="cm-total" key={`total-${predicted}`}>{columnTotal(predicted)}</div>
                        ))}
                        <div className="cm-total cm-total--grand">{total}</div>
                    </div>
                </div>
                <div className="cm-metrics">
                    <div className="cm-metric">
                        <span className="cm-metric__label">Accuracy</span>
                        <span className="cm-metric__value">{formatRatio(correct, total)}</span>
                    </div>
                    <div className="cm-metric">
                        <span className="cm-metric__label">Classes</span>
                        <span className="cm-metric__value">{matrix.classCount}</span>
                    </div>
                </div>
                <ProvenanceCaption evaluationId={evaluationId} step={step} sampleCount={sampleCount} />
            </div>
        </div>
    );
}

export const ConfusionMatrix = memo(function ConfusionMatrix() {
    const taskKind = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared.document.recipe.task.kind
        : null);
    const evaluation = useTrainingStore((state) => state.latestEvaluation);

    if (taskKind === 'regression' || taskKind === null) return null;
    const matrix = evaluation?.test.values.confusionMatrix;
    if (!evaluation || !matrix) {
        return (
            <div className="panel confusion-matrix">
                <div className="panel__title">Confusion Matrix (Full Test Split)</div>
                <EmptyState
                    icon="📊"
                    title="Confusion matrix unavailable"
                    description="A paired full test evaluation with confusion evidence has not been published yet."
                />
            </div>
        );
    }

    const props = {
        evaluationId: evaluation.evaluationId,
        step: evaluation.model.step,
        sampleCount: evaluation.test.basis.sampleCount,
    };
    return isMulticlass(matrix)
        ? <MulticlassMatrix matrix={matrix} {...props} />
        : <BinaryMatrix matrix={matrix} {...props} />;
});
