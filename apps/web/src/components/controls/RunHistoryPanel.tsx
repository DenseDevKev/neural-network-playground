import { memo, useCallback, useState } from 'react';
import type {
    ExperimentRunRecordV2,
    RejectedExperimentRunRecordV2,
} from '@nn-playground/shared';
import { getWorkerApi } from '../../worker/workerBridge.ts';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

function recordLabel(record: ExperimentRunRecordV2): string {
    return record.title ?? record.id;
}

function formatMetric(value: number): string {
    return Number.isFinite(value) ? value.toFixed(4) : 'n/a';
}

function createUuid(): string {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
        return globalThis.crypto.randomUUID();
    }
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-');
}

function downloadText(filename: string, text: string, type = 'application/json'): void {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
}

function RejectedRecord({ record }: { record: RejectedExperimentRunRecordV2 }) {
    const deleteRejectedRecord = useExperimentMemoryStore((state) => state.deleteRejectedRecord);
    return (
        <article className="inspection__layer" aria-label={`Rejected saved record ${record.sourceIndex}`}>
            <div className="inspection__layer-name">Rejected saved record</div>
            <div className="inspection__empty" role="alert" style={{ marginTop: 6 }}>
                {record.issues.map((entry) => `${entry.path}: ${entry.message}`).join(' ')}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => downloadText(
                        `rejected-run-${record.sourceIndex}.json`,
                        record.rawJson,
                    )}
                    aria-label="Download rejected record"
                >
                    Download raw JSON
                </button>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => { void deleteRejectedRecord(record.sourceIndex); }}
                    aria-label="Delete rejected record"
                >
                    Delete rejected record
                </button>
            </div>
        </article>
    );
}

function SavedRunComparison({ records }: { records: readonly ExperimentRunRecordV2[] }) {
    if (records.length < 2) return null;
    const current = records[0];
    const baseline = records[1];
    const currentEvaluation = current.snapshot.evaluation;
    const baselineEvaluation = baseline.snapshot.evaluation;
    const comparable = currentEvaluation.dataset.datasetKey === baselineEvaluation.dataset.datasetKey
        && currentEvaluation.objectiveKey === baselineEvaluation.objectiveKey;

    let summary = 'Not directly comparable';
    if (comparable) {
        const currentLoss = currentEvaluation.test.values.dataLoss;
        const baselineLoss = baselineEvaluation.test.values.dataLoss;
        if (currentLoss === baselineLoss) {
            summary = `Equal test data loss at ${formatMetric(currentLoss)}`;
        } else {
            const winner = currentLoss < baselineLoss ? current : baseline;
            summary = `${recordLabel(winner)} has lower test data loss by ${formatMetric(
                Math.abs(currentLoss - baselineLoss),
            )}`;
        }
    }

    return (
        <section className="inspection__layer" role="group" aria-label="Saved run comparison">
            <div className="inspection__layer-name">Saved run comparison</div>
            <div className="inspection__stat-value" style={{ marginTop: 6 }}>{summary}</div>
            {!comparable && (
                <div className="inspection__empty" style={{ marginTop: 6 }}>
                    Dataset and objective identities must both match before losses can be ranked.
                </div>
            )}
        </section>
    );
}

export const RunHistoryPanel = memo(function RunHistoryPanel() {
    const hydrationStatus = useExperimentMemoryStore((state) => state.hydrationStatus);
    const records = useExperimentMemoryStore((state) => state.records);
    const rejectedRecords = useExperimentMemoryStore((state) => state.rejectedRecords);
    const legacyRaw = useExperimentMemoryStore((state) => state.legacyRaw);
    const legacyNoticeDismissed = useExperimentMemoryStore((state) => state.legacyNoticeDismissed);
    const persistenceError = useExperimentMemoryStore((state) => state.persistenceError);
    const pendingSave = useExperimentMemoryStore((state) => state.pendingSave);
    const saveRecord = useExperimentMemoryStore((state) => state.saveRecord);
    const retryPersistence = useExperimentMemoryStore((state) => state.retryPersistence);
    const dismissPersistenceError = useExperimentMemoryStore((state) => state.dismissPersistenceError);
    const removeRecord = useExperimentMemoryStore((state) => state.removeRecord);
    const dismissLegacyNotice = useExperimentMemoryStore((state) => state.dismissLegacyNotice);
    const deleteLegacyStorage = useExperimentMemoryStore((state) => state.deleteLegacyStorage);
    const prepared = usePlaygroundStore((state) => state.prepared);
    const replaceDocument = usePlaygroundStore((state) => state.replaceDocument);
    const [saving, setSaving] = useState(false);
    const [actionError, setActionError] = useState<string | null>(null);

    const saveCurrentRun = useCallback(async () => {
        if (saving) return;
        setSaving(true);
        setActionError(null);
        try {
            const timestamp = new Date().toISOString();
            const record = await getWorkerApi().captureRunArtifact({
                id: createUuid(),
                createdAt: timestamp,
                updatedAt: timestamp,
            });
            await saveRecord(record);
        } catch (error) {
            setActionError(error instanceof Error ? error.message : 'Saving the current run failed.');
        } finally {
            setSaving(false);
        }
    }, [saveRecord, saving]);

    const applySavedRecipe = useCallback(async (record: ExperimentRunRecordV2) => {
        setActionError(null);
        if (!prepared) {
            setActionError('A compatible version-2 experiment is required before applying a saved recipe.');
            return;
        }
        const result = await replaceDocument({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: record.recipe,
            view: prepared.document.view,
        });
        if (!result.ok) {
            setActionError(result.issues.map((entry) => entry.message).join(' '));
        }
    }, [prepared, replaceDocument]);

    return (
        <div className="run-history-panel">
            <div className="inspection__empty" role="note" style={{ marginBottom: 8 }}>
                Saved runs contain a recipe plus worker-authored evaluation evidence. They do not
                contain trained parameters.
            </div>
            <button
                type="button"
                className="btn btn--ghost btn--sm"
                style={{ width: '100%' }}
                onClick={() => { void saveCurrentRun(); }}
                disabled={saving || hydrationStatus !== 'ready' || prepared === null}
            >
                {saving ? 'Saving…' : 'Save current run'}
            </button>

            {actionError && <div className="inspection__empty" role="alert" style={{ marginTop: 8 }}>{actionError}</div>}
            {persistenceError && (
                <div className="inspection__layer" role="alert" style={{ marginTop: 8 }}>
                    <div className="inspection__empty">{persistenceError.message}</div>
                    <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                        {pendingSave && (
                            <button
                                type="button"
                                className="btn btn--ghost btn--sm"
                                onClick={() => { void retryPersistence(); }}
                            >
                                Retry saving
                            </button>
                        )}
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={dismissPersistenceError}
                        >
                            Dismiss error
                        </button>
                    </div>
                </div>
            )}

            {legacyRaw !== null && !legacyNoticeDismissed && (
                <section
                    className="inspection__layer"
                    role="note"
                    aria-label="Earlier saved runs"
                    style={{ marginTop: 12 }}
                >
                    <div className="inspection__layer-name">Earlier saved runs are incompatible</div>
                    <div className="inspection__empty" style={{ marginTop: 6 }}>
                        Version-1 bytes were not loaded, migrated, or deleted. Download them before
                        deleting if you want to keep a copy.
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={() => downloadText('nn-playground-earlier-runs.json', legacyRaw)}
                        >
                            Download earlier runs
                        </button>
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={dismissLegacyNotice}
                            aria-label="Dismiss earlier runs notice"
                        >
                            Dismiss notice
                        </button>
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={() => { void deleteLegacyStorage(); }}
                        >
                            Delete earlier runs
                        </button>
                    </div>
                </section>
            )}

            {rejectedRecords.length > 0 && (
                <div className="inspection__layers" style={{ marginTop: 12 }}>
                    {rejectedRecords.map((record) => (
                        <RejectedRecord key={`${record.sourceIndex}:${record.rawJson}`} record={record} />
                    ))}
                </div>
            )}

            {records.length === 0 ? (
                <div className="inspection__empty" style={{ marginTop: 12 }}>
                    {hydrationStatus === 'loading' ? 'Loading saved runs…' : 'No saved runs'}
                </div>
            ) : (
                <>
                    <div className="inspection__layers" style={{ marginTop: 12 }}>
                        {records.map((record) => (
                            <article key={record.id} className="inspection__layer" aria-label={recordLabel(record)}>
                                <div className="inspection__layer-name">{recordLabel(record)}</div>
                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">Model revision</span>
                                    <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                        {record.snapshot.model.revision.toLocaleString()}
                                    </span>
                                </div>
                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">Train / test data loss</span>
                                    <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                        {formatMetric(record.snapshot.evaluation.train.values.dataLoss)} /{' '}
                                        {formatMetric(record.snapshot.evaluation.test.values.dataLoss)}
                                    </span>
                                </div>
                                <div className="inspection__empty" style={{ marginTop: 6 }}>
                                    Full evaluation at step {record.snapshot.model.step.toLocaleString()};{' '}
                                    {record.snapshot.evaluation.train.basis.sampleCount.toLocaleString()} train and{' '}
                                    {record.snapshot.evaluation.test.basis.sampleCount.toLocaleString()} test samples.
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                                    <button
                                        type="button"
                                        className="btn btn--ghost btn--sm"
                                        onClick={() => { void applySavedRecipe(record); }}
                                    >
                                        Apply saved recipe
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn--ghost btn--sm"
                                        onClick={() => downloadText(
                                            `${record.id}.json`,
                                            JSON.stringify(record, null, 2),
                                        )}
                                        aria-label={`Download evidence for ${recordLabel(record)}`}
                                    >
                                        Download evidence
                                    </button>
                                    <button
                                        type="button"
                                        className="btn btn--ghost btn--sm"
                                        onClick={() => { void removeRecord(record.id); }}
                                        aria-label={`Delete ${recordLabel(record)}`}
                                    >
                                        Delete
                                    </button>
                                </div>
                            </article>
                        ))}
                    </div>
                    <SavedRunComparison records={records} />
                </>
            )}
        </div>
    );
});
