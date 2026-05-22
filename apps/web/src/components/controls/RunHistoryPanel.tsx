import { memo, useCallback, useState } from 'react';
import type { ArenaModelSummary, ExperimentRunRecordV1 } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { readHistory } from '../../store/historyBuffer.ts';
import {
    captureExperimentRun,
    createSerializedNetworkFromFrameBuffer,
    historyArraysToPoints,
} from '../../store/experimentRunCapture.ts';
import { Tooltip } from '../common/Tooltip.tsx';

interface RunHistoryPanelProps {
    onRestore: () => void;
    onInitializeArena?: (modelA: ExperimentRunRecordV1, modelB: ExperimentRunRecordV1) => void | Promise<void>;
    onStepArena?: () => void | Promise<void>;
}

function formatMetric(value: number): string {
    return Number.isFinite(value) ? value.toFixed(4) : 'n/a';
}

function formatSignedMetric(value: number): string {
    if (!Number.isFinite(value)) return 'n/a';
    const prefix = value > 0 ? '+' : '';
    return `${prefix}${value.toFixed(4)}`;
}

function formatSignedInteger(value: number): string {
    if (!Number.isFinite(value)) return 'n/a';
    const prefix = value > 0 ? '+' : '';
    return `${prefix}${value.toLocaleString()}`;
}

function formatPercent(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function getRecordLabel(record: ExperimentRunRecordV1): string {
    return record.title ?? record.id;
}

function isScalarLiveArenaRecord(record: ExperimentRunRecordV1): boolean {
    return (
        record.config.network.outputSize === 1 &&
        record.config.network.outputActivation !== 'softmax' &&
        record.config.training.lossType !== 'categoricalCrossEntropy' &&
        record.config.data.dataset !== 'three-class-clusters'
    );
}

function generalizationGap(record: ExperimentRunRecordV1): number {
    return record.summary.testLoss - record.summary.trainLoss;
}

function formatArenaLossComparison(a: ExperimentRunRecordV1, b: ExperimentRunRecordV1): string {
    const diff = a.summary.testLoss - b.summary.testLoss;
    if (!Number.isFinite(diff)) return 'Test loss comparison unavailable.';
    if (Math.abs(diff) < 0.00005) return 'Both models have the same test loss.';
    const direction = diff < 0 ? 'lower' : 'higher';
    return `Model A ${direction} test loss by ${Math.abs(diff).toFixed(4)}.`;
}

function formatArenaStepComparison(a: ExperimentRunRecordV1, b: ExperimentRunRecordV1): string {
    const diff = a.summary.step - b.summary.step;
    if (diff === 0) return 'Both models trained for the same number of steps.';
    return `Model A trained ${Math.abs(diff).toLocaleString()} ${diff > 0 ? 'more' : 'fewer'} steps.`;
}

function formatArenaGapComparison(a: ExperimentRunRecordV1, b: ExperimentRunRecordV1): string {
    const diff = generalizationGap(a) - generalizationGap(b);
    if (!Number.isFinite(diff)) return 'Generalization gap comparison unavailable.';
    if (Math.abs(diff) < 0.00005) return 'Both models have the same generalization gap.';
    const direction = diff < 0 ? 'smaller' : 'larger';
    return `Model A ${direction} generalization gap by ${Math.abs(diff).toFixed(4)}.`;
}

function formatNextAdjustment(a: ExperimentRunRecordV1, b: ExperimentRunRecordV1): string {
    const testLossDiff = a.summary.testLoss - b.summary.testLoss;
    const gapDiff = generalizationGap(a) - generalizationGap(b);
    if (!Number.isFinite(testLossDiff) || !Number.isFinite(gapDiff)) {
        return 'Next adjustment: compare architecture and metric differences before changing the recipe.';
    }
    if (testLossDiff < -0.00005 && gapDiff <= 0.00005) {
        return 'Next adjustment: keep the tuned recipe direction; it improved test loss without widening the gap.';
    }
    if (testLossDiff < -0.00005) {
        return 'Next adjustment: keep the stronger test loss, then reduce any wider gap with regularization or simpler layers.';
    }
    if (testLossDiff > 0.00005) {
        return 'Next adjustment: restore or compare against the better saved run before tuning further.';
    }
    return 'Next adjustment: compare architecture differences because the metrics are effectively tied.';
}

function formatFeatureList(record: ExperimentRunRecordV1): string {
    const enabled = Object.entries(record.config.features)
        .filter(([, value]) => value)
        .map(([key]) => key);
    return enabled.length ? enabled.join(', ') : 'none';
}

function formatConfigNumber(value: number): string {
    return Number.isFinite(value) ? String(value) : 'n/a';
}

function formatHiddenLayers(record: ExperimentRunRecordV1): string {
    return `[${record.config.network.hiddenLayers.join(', ')}]`;
}

function totalHiddenUnits(record: ExperimentRunRecordV1): number {
    return record.config.network.hiddenLayers.reduce((sum, units) => sum + units, 0);
}

function formatUnitDelta(a: number, b: number): string {
    return a === b ? 'same' : formatSignedInteger(a - b);
}

function createArchitectureRows(a: ExperimentRunRecordV1, b: ExperimentRunRecordV1) {
    const aUnits = totalHiddenUnits(a);
    const bUnits = totalHiddenUnits(b);
    return [
        {
            label: 'Hidden layers',
            value: `A ${formatHiddenLayers(a)} / B ${formatHiddenLayers(b)}`,
        },
        {
            label: 'Total hidden units',
            value: `A ${aUnits.toLocaleString()} / B ${bUnits.toLocaleString()} (${formatUnitDelta(aUnits, bUnits)})`,
        },
        {
            label: 'Activation',
            value: `A ${a.config.network.activation} / B ${b.config.network.activation}`,
        },
        {
            label: 'Output/loss',
            value: [
                `A ${a.config.network.outputActivation} + ${a.config.training.lossType}`,
                `B ${b.config.network.outputActivation} + ${b.config.training.lossType}`,
            ].join(' / '),
        },
        {
            label: 'Optimizer/lr',
            value: [
                `A ${a.config.training.optimizer} @ ${formatConfigNumber(a.config.training.learningRate)}`,
                `B ${b.config.training.optimizer} @ ${formatConfigNumber(b.config.training.learningRate)}`,
            ].join(' / '),
        },
        {
            label: 'Batch size',
            value: [
                `A ${a.config.training.batchSize.toLocaleString()}`,
                `B ${b.config.training.batchSize.toLocaleString()} (${formatUnitDelta(a.config.training.batchSize, b.config.training.batchSize)})`,
            ].join(' / '),
        },
        {
            label: 'Regularization',
            value: [
                `A ${a.config.training.regularization} ${formatConfigNumber(a.config.training.regularizationRate)}`,
                `B ${b.config.training.regularization} ${formatConfigNumber(b.config.training.regularizationRate)}`,
            ].join(' / '),
        },
        {
            label: 'Data',
            value: [
                `A ${a.config.data.dataset}, ${a.config.data.numSamples.toLocaleString()} samples, noise ${formatConfigNumber(a.config.data.noise)}`,
                `B ${b.config.data.dataset}, ${b.config.data.numSamples.toLocaleString()} samples, noise ${formatConfigNumber(b.config.data.noise)}`,
            ].join(' / '),
        },
        {
            label: 'Features',
            value: `A ${formatFeatureList(a)} / B ${formatFeatureList(b)}`,
        },
    ] as const;
}

function createLossThumbnailLabel(record: ExperimentRunRecordV1, labelPrefix?: string): string {
    const first = record.history[0];
    const last = record.history.at(-1);
    if (!first || !last) return `No loss history thumbnail for ${getRecordLabel(record)}.`;
    const label = [
        `Loss thumbnail for ${getRecordLabel(record)}: ${record.history.length} points`,
        `train loss ${formatMetric(first.trainLoss)} to ${formatMetric(last.trainLoss)}`,
        `test loss ${formatMetric(first.testLoss)} to ${formatMetric(last.testLoss)}.`,
    ].join(', ');
    return labelPrefix ? `${labelPrefix}: ${label}` : label;
}

function createSparklinePath(
    record: ExperimentRunRecordV1,
    key: 'trainLoss' | 'testLoss',
): string {
    const width = 180;
    const height = 42;
    const padding = 4;
    const values = record.history.flatMap((point) => [point.trainLoss, point.testLoss])
        .filter((value) => Number.isFinite(value));
    if (record.history.length < 2 || values.length === 0) return '';
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    return record.history.map((point, index) => {
        const x = padding + (index / (record.history.length - 1)) * (width - padding * 2);
        const normalized = (point[key] - min) / span;
        const y = padding + (1 - normalized) * (height - padding * 2);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
}

function RunHistoryThumbnail({
    record,
    labelPrefix,
}: {
    record: ExperimentRunRecordV1;
    labelPrefix?: string;
}) {
    if (record.history.length < 2) {
        return (
            <div className="inspection__empty" style={{ marginTop: 8 }}>
                {labelPrefix ? `${labelPrefix} has no loss history thumbnail` : 'No loss history thumbnail'}
            </div>
        );
    }
    const trainPath = createSparklinePath(record, 'trainLoss');
    const testPath = createSparklinePath(record, 'testLoss');
    return (
        <div style={{ marginTop: 8 }}>
            <svg
                role="img"
                aria-label={createLossThumbnailLabel(record, labelPrefix)}
                viewBox="0 0 180 56"
                preserveAspectRatio="none"
                style={{ display: 'block', width: '100%', height: 56 }}
            >
                <rect x="0" y="0" width="180" height="56" rx="4" fill="rgba(255,255,255,0.035)" />
                <line x1="4" y1="42" x2="176" y2="42" stroke="rgba(255,255,255,0.14)" strokeWidth="1" />
                <path d={testPath} fill="none" stroke="#f2a65a" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                <path d={trainPath} fill="none" stroke="#7dd3fc" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                <text x="6" y="53" fill="currentColor" fontSize="8">train</text>
                <text x="40" y="53" fill="currentColor" fontSize="8">test</text>
            </svg>
        </div>
    );
}

function LiveArenaSummary({ summary }: { summary: ArenaModelSummary }) {
    const label = summary.side === 'A' ? 'Model A' : 'Model B';
    return (
        <section
            className="run-arena__model"
            aria-label={`${label} live arena summary for ${summary.label}`}
        >
            <div className="run-arena__model-title">
                <span>{label} live: {summary.label}</span>
                <strong>{summary.status}</strong>
            </div>
            <div className="run-arena__metrics">
                <span>step {summary.step.toLocaleString()}</span>
                <span>train {formatMetric(summary.trainLoss)}</span>
                <span>test {formatMetric(summary.testLoss)}</span>
                <span>accuracy {formatPercent(summary.testAccuracy)}</span>
            </div>
        </section>
    );
}

function ArenaModelPane({
    label,
    record,
}: {
    label: 'Model A' | 'Model B';
    record: ExperimentRunRecordV1;
}) {
    return (
        <section className="run-arena__model" aria-label={`${label}: ${getRecordLabel(record)}`}>
            <div className="run-arena__model-title">
                <span>{label}</span>
                <strong>{getRecordLabel(record)}</strong>
            </div>
            <div className="run-arena__metrics">
                <span>step {record.summary.step.toLocaleString()}</span>
                <span>train {formatMetric(record.summary.trainLoss)}</span>
                <span>test {formatMetric(record.summary.testLoss)}</span>
                <span>gap {formatMetric(generalizationGap(record))}</span>
            </div>
            <RunHistoryThumbnail record={record} labelPrefix={label} />
        </section>
    );
}

function ArchitectureComparison({
    modelA,
    modelB,
}: {
    modelA: ExperimentRunRecordV1;
    modelB: ExperimentRunRecordV1;
}) {
    return (
        <div className="run-arena__architecture" role="group" aria-label="Architecture comparison">
            <div className="inspection__layer-name">Architecture comparison</div>
            <div className="run-arena__architecture-rows">
                {createArchitectureRows(modelA, modelB).map((row) => (
                    <div key={row.label} className="run-arena__architecture-row">
                        <span className="inspection__stat-label">{row.label} </span>
                        <span className="inspection__stat-value">{row.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function SideBySideModelArena({
    records,
    onInitializeArena,
    onStepArena,
}: {
    records: ExperimentRunRecordV1[];
    onInitializeArena?: (modelA: ExperimentRunRecordV1, modelB: ExperimentRunRecordV1) => void | Promise<void>;
    onStepArena?: () => void | Promise<void>;
}) {
    const [modelAId, setModelAId] = useState(records[0]?.id ?? '');
    const [modelBId, setModelBId] = useState(records[1]?.id ?? records[0]?.id ?? '');
    const arenaSummaries = useTrainingStore((s) => s.arenaSummaries);
    if (records.length < 2) return null;

    const modelA = records.find((record) => record.id === modelAId) ?? records[0];
    const modelB = records.find((record) => record.id === modelBId) ?? records[1] ?? records[0];
    const selectedRecordsAreScalar = isScalarLiveArenaRecord(modelA) && isScalarLiveArenaRecord(modelB);
    const canStartLiveArena = Boolean(onInitializeArena && selectedRecordsAreScalar);
    const liveArenaNoteId = 'run-arena-live-scalar-note';

    return (
        <section className="run-arena" aria-label="Side-by-side model arena">
            <div className="run-arena__header">
                <div>
                    <div className="inspection__layer-name">Side-by-side model arena</div>
                    <p className="run-arena__summary">
                        Compare two saved runs using existing metrics and loss thumbnails.
                    </p>
                </div>
            </div>
            <div className="run-arena__selectors">
                <label>
                    <span>Model A run</span>
                    <select value={modelA.id} onChange={(event) => setModelAId(event.currentTarget.value)}>
                        {records.map((record) => (
                            <option key={record.id} value={record.id}>{getRecordLabel(record)}</option>
                        ))}
                    </select>
                </label>
                <label>
                    <span>Model B run</span>
                    <select value={modelB.id} onChange={(event) => setModelBId(event.currentTarget.value)}>
                        {records.map((record) => (
                            <option key={record.id} value={record.id}>{getRecordLabel(record)}</option>
                        ))}
                    </select>
                </label>
            </div>
            <div className="run-arena__comparison" role="group" aria-label="Live arena controls">
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => {
                        if (canStartLiveArena) void onInitializeArena?.(modelA, modelB);
                    }}
                    disabled={!canStartLiveArena}
                    aria-label="Start live arena with selected saved runs"
                    aria-describedby={selectedRecordsAreScalar ? undefined : liveArenaNoteId}
                >
                    Start live arena
                </button>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => void onStepArena?.()}
                    disabled={!onStepArena || !arenaSummaries}
                    aria-label="Step live arena once"
                >
                    Step live arena
                </button>
            </div>
            {!selectedRecordsAreScalar && (
                <div id={liveArenaNoteId} className="inspection__empty" role="status" style={{ marginTop: 8 }}>
                    Live arena supports saved scalar runs only; restore this multiclass run to train it.
                </div>
            )}
            {arenaSummaries && (
                <div className="run-arena__models" role="group" aria-label="Live arena scalar summaries">
                    {arenaSummaries.map((summary) => (
                        <LiveArenaSummary key={summary.side} summary={summary} />
                    ))}
                    <span className="sr-only">
                        Live arena accuracy {formatPercent(arenaSummaries[0]?.testAccuracy)} / {formatPercent(arenaSummaries[1]?.testAccuracy)}
                    </span>
                </div>
            )}
            <div className="run-arena__models">
                <ArenaModelPane label="Model A" record={modelA} />
                <ArenaModelPane label="Model B" record={modelB} />
            </div>
            <div className="run-arena__comparison" role="group" aria-label="Arena comparison summary">
                <span>{formatArenaLossComparison(modelA, modelB)}</span>
                <span>{formatArenaGapComparison(modelA, modelB)}</span>
                <span>{formatArenaStepComparison(modelA, modelB)}</span>
                <span>{formatNextAdjustment(modelA, modelB)}</span>
            </div>
            <ArchitectureComparison modelA={modelA} modelB={modelB} />
        </section>
    );
}

function createMarkdownReport(record: ExperimentRunRecordV1): string {
    const gap = generalizationGap(record);
    const lines = [
        `# ${record.title ?? 'Neural Network Playground Run'}`,
        '',
        '## Summary',
        '',
        `- Dataset: ${record.config.data.dataset}`,
        `- Problem type: ${record.config.data.problemType}`,
        `- Step: ${record.summary.step}`,
        `- Epoch: ${record.summary.epoch}`,
        `- Train loss: ${formatMetric(record.summary.trainLoss)}`,
        `- Test loss: ${formatMetric(record.summary.testLoss)}`,
        `- Generalization gap: ${formatMetric(gap)}`,
        `- Train accuracy: ${record.summary.trainMetrics.accuracy === undefined ? 'n/a' : formatMetric(record.summary.trainMetrics.accuracy)}`,
        `- Test accuracy: ${record.summary.testMetrics.accuracy === undefined ? 'n/a' : formatMetric(record.summary.testMetrics.accuracy)}`,
        `- Pause reason: ${record.summary.pauseReason ?? 'none'}`,
        '',
        '## Setup',
        '',
        `- Hidden layers: [${record.config.network.hiddenLayers.join(', ')}]`,
        `- Activation: ${record.config.network.activation}`,
        `- Output activation: ${record.config.network.outputActivation}`,
        `- Weight init: ${record.config.network.weightInit}`,
        `- Learning rate: ${record.config.training.learningRate}`,
        `- Optimizer: ${record.config.training.optimizer}`,
        `- Batch size: ${record.config.training.batchSize}`,
        `- Regularization: ${record.config.training.regularization}`,
        `- Regularization rate: ${record.config.training.regularizationRate}`,
        `- Gradient clip: ${record.config.training.gradientClip ?? 'none'}`,
        `- Active features: ${formatFeatureList(record)}`,
        `- Samples: ${record.config.data.numSamples}`,
        `- Noise: ${record.config.data.noise}`,
        `- Train/test ratio: ${record.config.data.trainTestRatio}`,
        `- Saved parameters: ${record.network ? 'yes' : 'no'}`,
        '',
        '## History',
        '',
        '| Step | Train loss | Test loss | Train accuracy | Test accuracy |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ];
    for (const point of record.history) {
        lines.push(`| ${[
            point.step,
            formatMetric(point.trainLoss),
            formatMetric(point.testLoss),
            point.trainAccuracy === undefined ? '' : formatMetric(point.trainAccuracy),
            point.testAccuracy === undefined ? '' : formatMetric(point.testAccuracy),
        ].join(' | ')} |`);
    }
    return `${lines.join('\n')}\n`;
}

function downloadMarkdown(record: ExperimentRunRecordV1): void {
    const blob = new Blob([createMarkdownReport(record)], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${record.id}-report.md`;
    a.click();
    URL.revokeObjectURL(url);
}

export const RunHistoryPanel = memo(function RunHistoryPanel({
    onRestore,
    onInitializeArena,
    onStepArena,
}: RunHistoryPanelProps) {
    const records = useExperimentMemoryStore((s) => s.records);
    const saveRecord = useExperimentMemoryStore((s) => s.saveRecord);
    const removeRecord = useExperimentMemoryStore((s) => s.removeRecord);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);

    const handleSave = useCallback(() => {
        const record = captureExperimentRun({
            config: usePlaygroundStore.getState().getConfig(),
            snapshot: useTrainingStore.getState().snapshot,
            history: historyArraysToPoints(readHistory()),
            status: useTrainingStore.getState().status,
            pauseReason: useTrainingStore.getState().pauseReason,
            network: createSerializedNetworkFromFrameBuffer(usePlaygroundStore.getState().network),
        });
        if (record) saveRecord(record);
    }, [saveRecord]);

    const handleRestore = useCallback((record: ExperimentRunRecordV1) => {
        usePlaygroundStore.getState().applyPreset({
            id: `history-${record.id}`,
            title: record.title ?? 'Saved run',
            description: 'Restored config from local run history.',
            config: record.config,
        });
        onRestore();
    }, [onRestore]);

    return (
        <div className="run-history-panel">
            <div className="inspection__empty" role="note" style={{ marginBottom: 8 }}>
                History is the saved-run record surface.
            </div>
            <Tooltip content="Save the current config, final metrics, bounded loss history, and current parameters when available." block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ width: '100%' }}
                    onClick={handleSave}
                    disabled={!snapshot}
                >
                    Save current run
                </button>
            </Tooltip>
            <div className="inspection__empty" role="status" style={{ marginTop: 8 }}>
                {snapshot
                    ? `${status} at step ${snapshot.step.toLocaleString()}`
                    : 'Run or step the model before saving a run.'}
            </div>

            {records.length === 0 ? (
                <div className="inspection__empty" style={{ marginTop: 12 }}>No saved runs</div>
            ) : (
                <>
                    <SideBySideModelArena
                        records={records}
                        onInitializeArena={onInitializeArena}
                        onStepArena={onStepArena}
                    />
                    <div className="inspection__layers" style={{ marginTop: 12 }}>
                        {records.map((record, index) => {
                            const baseline = records[index + 1];
                            return (
                                <article key={record.id} className="inspection__layer" aria-label={getRecordLabel(record)}>
                                    <div className="inspection__layer-name">{record.title ?? 'Saved run'}</div>
                                    <div className="inspection__empty" role="note" style={{ marginTop: 6 }}>
                                        <strong>Saved run reference</strong>
                                        <span>Restore config to make this saved run the current recipe.</span>
                                    </div>
                                    <div className="inspection__stat-row">
                                        <span className="inspection__stat-label">step</span>
                                        <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                            {record.summary.step.toLocaleString()}
                                        </span>
                                    </div>
                                    <div className="inspection__stat-row">
                                        <span className="inspection__stat-label">loss</span>
                                        <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                            {formatMetric(record.summary.trainLoss)} / {formatMetric(record.summary.testLoss)}
                                        </span>
                                    </div>
                                    <RunHistoryThumbnail record={record} />
                                    {baseline && (
                                        <div
                                            role="group"
                                            aria-label={`Comparison for ${getRecordLabel(record)} against ${getRecordLabel(baseline)}`}
                                            style={{ marginTop: 8 }}
                                        >
                                            <div className="inspection__stat-label">Compared with {getRecordLabel(baseline)}</div>
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                                                <span className="inspection__stat-value">
                                                    Train loss {formatSignedMetric(record.summary.trainLoss - baseline.summary.trainLoss)}
                                                </span>
                                                <span className="inspection__stat-value">
                                                    Test loss {formatSignedMetric(record.summary.testLoss - baseline.summary.testLoss)}
                                                </span>
                                                <span className="inspection__stat-value">
                                                    Gap {formatSignedMetric(generalizationGap(record) - generalizationGap(baseline))}
                                                </span>
                                                <span className="inspection__stat-value">
                                                    Steps {formatSignedInteger(record.summary.step - baseline.summary.step)}
                                                </span>
                                            </div>
                                            <div className="inspection__stat-value" style={{ marginTop: 6 }}>
                                                {formatNextAdjustment(record, baseline)}
                                            </div>
                                        </div>
                                    )}
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                                        <button
                                            type="button"
                                            className="btn btn--ghost btn--sm"
                                            onClick={() => handleRestore(record)}
                                            aria-label={`Restore config for ${record.title ?? record.id}`}
                                        >
                                            Restore config
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn--ghost btn--sm"
                                            onClick={() => downloadMarkdown(record)}
                                            aria-label={`Export report for ${record.title ?? record.id}`}
                                        >
                                            Export report
                                        </button>
                                        <button
                                            type="button"
                                            className="btn btn--ghost btn--sm"
                                            onClick={() => removeRecord(record.id)}
                                            aria-label={`Delete ${record.title ?? record.id}`}
                                        >
                                            Delete
                                        </button>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                </>
            )}
            {pauseReason && <span className="sr-only">Last pause reason: {pauseReason}</span>}
        </div>
    );
});
