// ── Confusion Matrix Component ──
import { Fragment, memo, useMemo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { EmptyState } from '../common/EmptyState.tsx';
import { getActiveFeatures, transformPoint } from '@nn-playground/engine';
import type { DataPoint, FeatureFlags, NetworkConfig } from '@nn-playground/engine';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';

const MULTICLASS_LABELS = [0, 1, 2] as const;

interface MulticlassConfusionReadout {
    matrix: number[][];
    rowTotals: number[];
    columnTotals: number[];
    total: number;
    correct: number;
}

interface FrameNetworkParams {
    weights: number[][][];
    biases: number[][];
}

function formatPercent(value: number, total: number): string {
    if (total === 0) return '0.0%';
    return `${((value / total) * 100).toFixed(1)}%`;
}

function formatRatio(numerator: number, denominator: number): string {
    if (denominator === 0) return '0.0%';
    return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function formatSampleCount(value: number): string {
    return `${value} test sample${value === 1 ? '' : 's'}`;
}

function hasNonBinaryLabels(points: DataPoint[]): boolean {
    return points.some((point) => {
        const { label } = point;
        return typeof label === 'number' && label !== 0 && label !== 1;
    });
}

function argmax(values: ArrayLike<number>): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

function sameLayerSizes(a: readonly number[], b: readonly number[]): boolean {
    return a.length === b.length && a.every((value, index) => value === b[index]);
}

function stableSigmoid(x: number): number {
    if (x >= 0) {
        return 1 / (1 + Math.exp(-x));
    }
    const ex = Math.exp(x);
    return ex / (1 + ex);
}

function stableSoftplus(x: number): number {
    return x > 0
        ? x + Math.log1p(Math.exp(-x))
        : Math.log1p(Math.exp(x));
}

function activateScalar(value: number, activation: NetworkConfig['activation']): number | null {
    switch (activation) {
        case 'relu':
            return Math.max(0, value);
        case 'tanh':
            return Math.tanh(value);
        case 'sigmoid':
            return stableSigmoid(value);
        case 'linear':
            return value;
        case 'leakyRelu':
            return value > 0 ? value : 0.01 * value;
        case 'elu':
            return value >= 0 ? value : Math.exp(value) - 1;
        case 'swish':
            return value * stableSigmoid(value);
        case 'softplus':
            return stableSoftplus(value);
        case 'softmax':
            return null;
    }
}

function applySoftmax(logits: readonly number[]): number[] | null {
    if (logits.length === 0) return null;

    let maxLogit = -Infinity;
    for (const value of logits) {
        if (!Number.isFinite(value)) return null;
        if (value > maxLogit) maxLogit = value;
    }

    const probabilities = new Array<number>(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        const expValue = Math.exp(logits[i] - maxLogit);
        probabilities[i] = expValue;
        sum += expValue;
    }

    if (!Number.isFinite(sum) || sum <= 0) return null;
    for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sum;
    }
    return probabilities;
}

function unpackWeights(weights: Float32Array, layerSizes: readonly number[]): number[][][] | null {
    const unpacked: number[][][] = [];
    let offset = 0;

    for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex++) {
        const fanIn = layerSizes[layerIndex];
        const fanOut = layerSizes[layerIndex + 1];
        const layer: number[][] = [];

        for (let neuronIndex = 0; neuronIndex < fanOut; neuronIndex++) {
            const row: number[] = [];
            for (let inputIndex = 0; inputIndex < fanIn; inputIndex++) {
                const value = weights[offset++];
                if (!Number.isFinite(value)) return null;
                row.push(value);
            }
            layer.push(row);
        }

        unpacked.push(layer);
    }

    return offset === weights.length ? unpacked : null;
}

function unpackBiases(biases: Float32Array, layerSizes: readonly number[]): number[][] | null {
    const unpacked: number[][] = [];
    let offset = 0;

    for (let layerIndex = 1; layerIndex < layerSizes.length; layerIndex++) {
        const layer: number[] = [];
        for (let neuronIndex = 0; neuronIndex < layerSizes[layerIndex]; neuronIndex++) {
            const value = biases[offset++];
            if (!Number.isFinite(value)) return null;
            layer.push(value);
        }
        unpacked.push(layer);
    }

    return offset === biases.length ? unpacked : null;
}

function getNetworkParamsFromFrame(networkConfig: NetworkConfig): FrameNetworkParams | null {
    const frame = getFrameBuffer();
    if (!frame.weights || !frame.biases || !frame.weightLayout) return null;

    const layerSizes = frame.weightLayout.layerSizes;
    const expectedLayerSizes = [
        networkConfig.inputSize,
        ...networkConfig.hiddenLayers,
        networkConfig.outputSize,
    ];
    if (!sameLayerSizes(layerSizes, expectedLayerSizes)) return null;
    if (layerSizes.at(-1) !== 3) return null;

    const weights = unpackWeights(frame.weights, layerSizes);
    const biases = unpackBiases(frame.biases, layerSizes);
    if (!weights || !biases) return null;

    return { weights, biases };
}

function forwardFromParams(
    input: readonly number[],
    params: FrameNetworkParams,
    networkConfig: NetworkConfig,
): number[] | null {
    let activations = [...input];

    for (let layerIndex = 0; layerIndex < params.weights.length; layerIndex++) {
        const isOutputLayer = layerIndex === params.weights.length - 1;
        const logits = params.weights[layerIndex].map((row, neuronIndex) => {
            let sum = params.biases[layerIndex][neuronIndex];
            for (let inputIndex = 0; inputIndex < row.length; inputIndex++) {
                sum += row[inputIndex] * activations[inputIndex];
            }
            return sum;
        });

        if (logits.some((value) => !Number.isFinite(value))) return null;

        if (isOutputLayer) {
            return applySoftmax(logits);
        }

        const nextActivations = logits.map((value) => activateScalar(value, networkConfig.activation));
        if (nextActivations.some((value) => value == null || !Number.isFinite(value))) return null;
        activations = nextActivations as number[];
    }

    return null;
}

export function deriveMulticlassConfusionReadout(
    networkConfig: NetworkConfig,
    features: FeatureFlags,
    testPoints: readonly DataPoint[],
): MulticlassConfusionReadout | null {
    if (
        networkConfig.outputSize !== 3 ||
        networkConfig.outputActivation !== 'softmax' ||
        testPoints.length === 0
    ) {
        return null;
    }

    const activeFeatures = getActiveFeatures(features);
    if (activeFeatures.length !== networkConfig.inputSize) return null;

    const params = getNetworkParamsFromFrame(networkConfig);
    if (!params) return null;

    try {
        const matrix = MULTICLASS_LABELS.map(() => MULTICLASS_LABELS.map(() => 0));
        let correct = 0;

        for (const point of testPoints) {
            if (!Number.isInteger(point.label) || point.label < 0 || point.label > 2) {
                return null;
            }
            const input = transformPoint(point.x, point.y, activeFeatures);
            const probabilities = forwardFromParams(input, params, networkConfig);
            if (!probabilities || probabilities.length !== MULTICLASS_LABELS.length) return null;
            const predicted = argmax(probabilities);
            matrix[point.label][predicted]++;
            if (point.label === predicted) correct++;
        }

        const rowTotals = matrix.map((row) => row.reduce((sum, value) => sum + value, 0));
        const columnTotals = MULTICLASS_LABELS.map((predicted) => (
            matrix.reduce((sum, row) => sum + row[predicted], 0)
        ));
        const total = rowTotals.reduce((sum, value) => sum + value, 0);

        return { matrix, rowTotals, columnTotals, total, correct };
    } catch {
        return null;
    }
}

export const ConfusionMatrix = memo(function ConfusionMatrix() {
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const network = usePlaygroundStore((s) => s.network);
    const features = usePlaygroundStore((s) => s.features);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const paramsVersion = useTrainingStore((s) => s.paramsVersion);
    const cm = useTrainingStore((s) => s.snapshot?.testMetrics.confusionMatrix);
    const status = useTrainingStore((s) => s.status);
    const isConfigPending = useTrainingStore((s) => (
        s.pendingConfigSource !== null ||
        s.dataConfigLoading ||
        s.networkConfigLoading ||
        s.featuresConfigLoading ||
        s.trainingConfigLoading ||
        s.presetConfigLoading
    ));
    const isThreeClassSoftmax =
        problemType === 'classification' &&
        network.outputSize === 3 &&
        network.outputActivation === 'softmax';
    const canDeriveMulticlassReadout =
        isThreeClassSoftmax &&
        status !== 'running' &&
        !isConfigPending;
    const multiclassReadout = useMemo(() => {
        void paramsVersion;
        if (!canDeriveMulticlassReadout) return null;
        return deriveMulticlassConfusionReadout(network, features, testPoints);
    }, [canDeriveMulticlassReadout, features, network, paramsVersion, testPoints]);

    if (problemType !== 'classification') return null;
    if (testPoints.length === 0) {
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

    if (isThreeClassSoftmax) {
        if (!multiclassReadout) {
            const unavailableDescription = status === 'running'
                ? 'Pause training to inspect the derived multiclass readout without recomputing it every training frame.'
                : isConfigPending
                    ? 'The current configuration is still syncing; wait for the worker to finish applying the active settings.'
                    : 'Current network parameters are still loading or do not match the active 3-class configuration.';

            return (
                <div className="panel confusion-matrix">
                    <div className="panel__title">Confusion Matrix (Test Set)</div>
                    <EmptyState
                        icon="📊"
                        title="Multiclass readout unavailable"
                        description={unavailableDescription}
                    />
                </div>
            );
        }

        const accuracy = formatRatio(multiclassReadout.correct, multiclassReadout.total);

        const MulticlassCell = ({
            actual,
            predicted,
            value,
        }: {
            actual: number;
            predicted: number;
            value: number;
        }) => {
            const intensity = multiclassReadout.total === 0 ? 0.12 : Math.max(0.12, value / multiclassReadout.total);
            const backgroundColor = actual === predicted
                ? `rgba(34, 197, 94, ${intensity.toFixed(2)})`
                : `rgba(239, 68, 68, ${intensity.toFixed(2)})`;
            const cellPercent = formatPercent(value, multiclassReadout.total);

            return (
                <div
                    aria-label={`${formatSampleCount(value)} (${cellPercent}) with actual Class ${actual} predicted Class ${predicted}`}
                    className="cm-cell cm-cell--multiclass"
                    style={{ backgroundColor }}
                >
                    <div className="cm-value">{value}</div>
                    <div className="cm-percentage">{cellPercent}</div>
                    <div className="cm-label">{`C${actual} -> C${predicted}`}</div>
                </div>
            );
        };

        return (
            <div className="panel confusion-matrix">
                <div className="panel__title">Multiclass Confusion Readout (Test Set)</div>
                <div className="cm-grid-container">
                    <div className="cm-axis-label">Predicted</div>
                    <div className="cm-layout">
                        <div className="cm-axis-label cm-axis-label--side">Actual</div>
                        <div className="cm-grid cm-grid--multiclass">
                            <div className="cm-header cm-header--empty" />
                            {MULTICLASS_LABELS.map((label) => (
                                <div className="cm-header" key={`pred-${label}`}>Pred Class {label}</div>
                            ))}
                            <div className="cm-header">Total</div>

                            {MULTICLASS_LABELS.map((actual) => (
                                <Fragment key={`actual-${actual}`}>
                                    <div className="cm-header cm-header--row">Actual Class {actual}</div>
                                    {MULTICLASS_LABELS.map((predicted) => (
                                        <MulticlassCell
                                            actual={actual}
                                            key={`cell-${actual}-${predicted}`}
                                            predicted={predicted}
                                            value={multiclassReadout.matrix[actual][predicted]}
                                        />
                                    ))}
                                    <div
                                        aria-label={`Actual Class ${actual} total ${multiclassReadout.rowTotals[actual]}`}
                                        className="cm-total"
                                    >
                                        {multiclassReadout.rowTotals[actual]}
                                    </div>
                                </Fragment>
                            ))}

                            <div className="cm-header cm-header--row">Total</div>
                            {MULTICLASS_LABELS.map((predicted) => (
                                <div
                                    aria-label={`Predicted Class ${predicted} total ${multiclassReadout.columnTotals[predicted]}`}
                                    className="cm-total"
                                    key={`col-total-${predicted}`}
                                >
                                    {multiclassReadout.columnTotals[predicted]}
                                </div>
                            ))}
                            <div
                                aria-label={`Total test samples ${multiclassReadout.total}`}
                                className="cm-total cm-total--grand"
                            >
                                {multiclassReadout.total}
                            </div>
                        </div>
                    </div>
                    <p className="sr-only">
                        {`${multiclassReadout.correct} of ${multiclassReadout.total} test samples land on the diagonal (${accuracy} accuracy). Rows are actual classes and columns are predicted classes.`}
                    </p>
                    <div className="cm-metrics">
                        <div className="cm-metric">
                            <span className="cm-metric__label">Accuracy</span>
                            <span className="cm-metric__value">{accuracy}</span>
                        </div>
                        <div className="cm-metric">
                            <span className="cm-metric__label">Classes</span>
                            <span className="cm-metric__value">3</span>
                        </div>
                        <div className="cm-metric">
                            <span className="cm-metric__label">Samples</span>
                            <span className="cm-metric__value">{multiclassReadout.total}</span>
                        </div>
                    </div>
                    <p className="cm-note">
                        Derived from current frame-buffer parameters and test points; not a worker-persisted metric.
                    </p>
                </div>
            </div>
        );
    }

    if (network.outputSize !== 1 || network.outputActivation === 'softmax' || hasNonBinaryLabels(trainPoints) || hasNonBinaryLabels(testPoints)) {
        return (
            <div className="panel confusion-matrix">
                <div className="panel__title">Confusion Matrix (Test Set)</div>
                <EmptyState
                    icon="📊"
                    title="Confusion matrix unavailable"
                    description="This panel only renders binary classification matrices. Use loss and accuracy while multiclass matrix support is unavailable."
                />
            </div>
        );
    }

    if (!cm) {
        return (
            <div className="panel confusion-matrix">
                <div className="panel__title">Confusion Matrix (Test Set)</div>
                <EmptyState
                    icon="📊"
                    title="Confusion matrix unavailable"
                    description="Metrics are still loading, stale, or unavailable for this snapshot. Use loss and accuracy until a fresh binary matrix arrives."
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
