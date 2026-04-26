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

function createMarkdownReport(record: ExperimentRunRecordV1): string {
    const lines = [
        `# ${record.title ?? 'Neural Network Playground Run'}`,
        '',
        `- Dataset: ${record.config.data.dataset}`,
        `- Step: ${record.summary.step}`,
        `- Epoch: ${record.summary.epoch}`,
        `- Train loss: ${formatMetric(record.summary.trainLoss)}`,
        `- Test loss: ${formatMetric(record.summary.testLoss)}`,
        `- Pause reason: ${record.summary.pauseReason ?? 'none'}`,
        `- Hidden layers: [${record.config.network.hiddenLayers.join(', ')}]`,
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
                    {records.map((record) => (
                        <article key={record.id} className="inspection__layer" aria-label={record.title ?? record.id}>
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
                    ))}
                </div>
            )}
            {pauseReason && <span className="sr-only">Last pause reason: {pauseReason}</span>}
        </div>
    );
});
