import { memo, useState } from 'react';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { Tooltip } from '../common/Tooltip.tsx';

interface RunHistoryPanelProps {
    /** @deprecated Legacy V1 records are read-only and never invoke this callback. */
    onRestore?: () => void;
    /** @deprecated Legacy V1 records cannot initialize a live V2 arena. */
    onInitializeArena?: (modelA: ExperimentRunRecordV1, modelB: ExperimentRunRecordV1) => void | Promise<void>;
    /** @deprecated Legacy V1 records cannot step a live V2 arena. */
    onStepArena?: () => void | Promise<void>;
}

function formatMetric(value: number): string {
    return Number.isFinite(value) ? value.toFixed(4) : 'n/a';
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

function SideBySideModelArena({ records }: { records: ExperimentRunRecordV1[] }) {
    const [modelAId, setModelAId] = useState(records[0]?.id ?? '');
    const [modelBId, setModelBId] = useState(records[1]?.id ?? records[0]?.id ?? '');
    if (records.length < 2) return null;

    const modelA = records.find((record) => record.id === modelAId) ?? records[0];
    const modelB = records.find((record) => record.id === modelBId) ?? records[1] ?? records[0];

    return (
        <section className="run-arena" aria-label="Legacy saved-run comparison">
            <div className="run-arena__header">
                <div>
                    <div className="inspection__layer-name">Legacy saved-run comparison</div>
                    <p className="run-arena__summary">
                        Compare preserved V1 metrics and loss thumbnails as historical references.
                    </p>
                </div>
            </div>
            <div className="inspection__empty" role="note">
                Legacy V1 configurations cannot be restored or executed in the live V2 arena.
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
            <div className="run-arena__models">
                <ArenaModelPane label="Model A" record={modelA} />
                <ArenaModelPane label="Model B" record={modelB} />
            </div>
            <div className="run-arena__comparison" role="group" aria-label="Legacy comparison summary">
                <span>
                    Not directly comparable: V1 records do not contain the V2 identities and
                    metric provenance required to name a winner or recommend an adjustment.
                </span>
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
        '## Compatibility',
        '',
        '- Legacy V1 record: read-only and incompatible with the V2 experiment runtime.',
        '- This report is a historical reference and cannot be restored or executed as a V2 experiment.',
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
        `- Legacy parameter snapshot present: ${record.network ? 'yes' : 'no'} (not executable in V2)`,
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

export const RunHistoryPanel = memo(function RunHistoryPanel(_props: RunHistoryPanelProps) {
    const records = useExperimentMemoryStore((s) => s.records);

    return (
        <div className="run-history-panel">
            <div className="inspection__empty" role="note" style={{ marginBottom: 8 }}>
                <span>History is the saved-run record surface.</span>{' '}
                <span>
                    Existing V1 records are preserved as read-only references and are incompatible
                    with the V2 experiment runtime.
                </span>
            </div>
            <Tooltip content="V2 run saving is temporarily unavailable until provenance-aware records land. Existing legacy V1 records remain untouched below." block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ width: '100%' }}
                    disabled
                    aria-describedby="run-history-v2-save-unavailable"
                >
                    Save current run
                </button>
            </Tooltip>
            <div
                id="run-history-v2-save-unavailable"
                className="inspection__empty"
                role="status"
                style={{ marginTop: 8 }}
            >
                V2 run saving is unavailable until provenance-aware records replace the legacy format.
            </div>

            {records.length === 0 ? (
                <div className="inspection__empty" style={{ marginTop: 12 }}>No saved runs</div>
            ) : (
                <>
                    <SideBySideModelArena records={records} />
                    <div className="inspection__layers" style={{ marginTop: 12 }}>
                        {records.map((record, index) => {
                            const baseline = records[index + 1];
                            return (
                                <article key={record.id} className="inspection__layer" aria-label={getRecordLabel(record)}>
                                    <div className="inspection__layer-name">{record.title ?? 'Saved run'}</div>
                                    <div className="inspection__empty" role="note" style={{ marginTop: 6 }}>
                                        <strong>Legacy V1 record</strong>
                                        <span>
                                            Read-only and incompatible with the V2 experiment runtime.
                                            Export is available; restore and live execution are unavailable.
                                        </span>
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
                                            <div className="inspection__stat-value" style={{ marginTop: 6 }}>
                                                Not directly comparable: legacy V1 records lack V2 identity and
                                                metric-provenance evidence.
                                            </div>
                                        </div>
                                    )}
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                                        <button
                                            type="button"
                                            className="btn btn--ghost btn--sm"
                                            onClick={() => downloadMarkdown(record)}
                                            aria-label={`Export report for ${record.title ?? record.id}`}
                                        >
                                            Export report
                                        </button>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
});
