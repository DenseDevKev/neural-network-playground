import { memo, useCallback } from 'react';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
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

function getRecordLabel(record: ExperimentRunRecordV1): string {
    return record.title ?? record.id;
}

function generalizationGap(record: ExperimentRunRecordV1): number {
    return record.summary.testLoss - record.summary.trainLoss;
}

function formatFeatureList(record: ExperimentRunRecordV1): string {
    const enabled = Object.entries(record.config.features)
        .filter(([, value]) => value)
        .map(([key]) => key);
    return enabled.length ? enabled.join(', ') : 'none';
}

function createLossThumbnailLabel(record: ExperimentRunRecordV1): string {
    const first = record.history[0];
    const last = record.history.at(-1);
    if (!first || !last) return `No loss history thumbnail for ${getRecordLabel(record)}.`;
    return [
        `Loss thumbnail for ${getRecordLabel(record)}: ${record.history.length} points`,
        `train loss ${formatMetric(first.trainLoss)} to ${formatMetric(last.trainLoss)}`,
        `test loss ${formatMetric(first.testLoss)} to ${formatMetric(last.testLoss)}.`,
    ].join(', ');
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

function RunHistoryThumbnail({ record }: { record: ExperimentRunRecordV1 }) {
    if (record.history.length < 2) {
        return (
            <div className="inspection__empty" style={{ marginTop: 8 }}>
                No loss history thumbnail
            </div>
        );
    }
    const trainPath = createSparklinePath(record, 'trainLoss');
    const testPath = createSparklinePath(record, 'testLoss');
    return (
        <div style={{ marginTop: 8 }}>
            <svg
                role="img"
                aria-label={createLossThumbnailLabel(record)}
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

export const RunHistoryPanel = memo(function RunHistoryPanel({ onRestore }: RunHistoryPanelProps) {
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
                <div className="inspection__layers" style={{ marginTop: 12 }}>
                    {records.map((record, index) => {
                        const baseline = records[index + 1];
                        return (
                            <article key={record.id} className="inspection__layer" aria-label={getRecordLabel(record)}>
                                <div className="inspection__layer-name">{record.title ?? 'Saved run'}</div>
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
            )}
            {pauseReason && <span className="sr-only">Last pause reason: {pauseReason}</span>}
        </div>
    );
});
