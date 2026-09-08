import { memo, useCallback, useEffect, useId, useRef, useState } from 'react';
import type {
    ExperimentRunRecordV2,
    RejectedExperimentRunRecordV2,
} from '@nn-playground/shared';
import { EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useSaveCurrentRun, type SaveCurrentRunController } from '../../hooks/useSaveCurrentRun.ts';
import { reconcileComparisonSelection } from './runComparisonSelection.ts';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

function recordLabel(record: ExperimentRunRecordV2): string {
    return record.title ?? record.id;
}

function comparisonRecordLabel(
    record: ExperimentRunRecordV2,
    records: readonly ExperimentRunRecordV2[],
): string {
    const label = recordLabel(record);
    const duplicate = records.some((candidate) => (
        candidate.id !== record.id && recordLabel(candidate) === label
    ));
    return duplicate ? `${label} (${record.id})` : label;
}

function formatMetric(value: number): string {
    return Number.isFinite(value) ? value.toFixed(4) : 'n/a';
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

function SavedRunComparison({
    current,
    baseline,
    currentLabel,
    baselineLabel,
}: {
    current: ExperimentRunRecordV2;
    baseline: ExperimentRunRecordV2;
    currentLabel: string;
    baselineLabel: string;
}) {
    const headingId = useId();
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
            const winnerLabel = currentLoss < baselineLoss ? currentLabel : baselineLabel;
            summary = `${winnerLabel} has lower test data loss by ${formatMetric(
                Math.abs(currentLoss - baselineLoss),
            )}`;
        }
    }

    return (
        <section className="inspection__layer" role="group" aria-labelledby={headingId}>
            <h3 id={headingId} className="inspection__layer-name">
                Saved run comparison: {currentLabel} and {baselineLabel}
            </h3>
            <div className="inspection__stat-value" style={{ marginTop: 6 }}>{summary}</div>
            {!comparable && (
                <div className="inspection__empty" style={{ marginTop: 6 }}>
                    Dataset and objective identities must both match before losses can be ranked.
                </div>
            )}
        </section>
    );
}

interface Props { saveController?: SaveCurrentRunController }

// The standalone adapter is for direct-render/legacy consumers. Production
// receives the single controller owned above the drawer in App.
export const RunHistoryPanel = memo(function RunHistoryPanel({ saveController }: Props) {
    return saveController ? <RunHistoryContent saveController={saveController} /> : <StandaloneRunHistory />;
});
function StandaloneRunHistory() {
    const saveController = useSaveCurrentRun();
    return <RunHistoryContent saveController={saveController} />;
}
function RunHistoryContent({ saveController }: { saveController: SaveCurrentRunController }) {
    const hydrationStatus = useExperimentMemoryStore((state) => state.hydrationStatus);
    const records = useExperimentMemoryStore((state) => state.records);
    const rejectedRecords = useExperimentMemoryStore((state) => state.rejectedRecords);
    const incompatibleEnvelope = useExperimentMemoryStore((state) => state.incompatibleEnvelope);
    const legacyRaw = useExperimentMemoryStore((state) => state.legacyRaw);
    const legacyNoticeDismissed = useExperimentMemoryStore((state) => state.legacyNoticeDismissed);
    const persistenceError = useExperimentMemoryStore((state) => state.persistenceError);
    const pendingSave = useExperimentMemoryStore((state) => state.pendingSave);
    const removeRecord = useExperimentMemoryStore((state) => state.removeRecord);
    const deleteIncompatibleEnvelope = useExperimentMemoryStore((state) => state.deleteIncompatibleEnvelope);
    const dismissLegacyNotice = useExperimentMemoryStore((state) => state.dismissLegacyNotice);
    const deleteLegacyStorage = useExperimentMemoryStore((state) => state.deleteLegacyStorage);
    const prepared = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared
        : null);
    const replaceDocument = usePlaygroundStore((state) => state.replaceDocument);
    const saving = saveController.busy;
    const [actionError, setActionError] = useState<string | null>(null);
    const [runTitle, setRunTitle] = useState('');
    const [comparisonSelection, setComparisonSelection] = useState<readonly string[] | null>(null);
    const panelId = useId();
    const comparisonGuidanceId = `${panelId}-comparison-guidance`;
    const runNameGuidanceId = `${panelId}-run-name-guidance`;
    const savedRecipeEffectsId = `${panelId}-saved-recipe-effects`;
    const mountedRef = useRef(true);
    const trimmedRunTitle = runTitle.trim();
    const runTitleTooLong = Array.from(trimmedRunTitle).length
        > EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS;
    const initialComparisonSelection = hydrationStatus === 'ready'
        ? records.slice(0, 2).map((record) => record.id)
        : [];
    const selectedComparisonIds = reconcileComparisonSelection(
        comparisonSelection ?? initialComparisonSelection,
        records,
        null,
    );
    const selectedComparisonRecords = selectedComparisonIds.flatMap((id) => {
        const record = records.find((candidate) => candidate.id === id);
        return record ? [record] : [];
    });

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    useEffect(() => {
        if (hydrationStatus !== 'ready') return;
        setComparisonSelection((current) => current ?? records.slice(0, 2).map((record) => record.id));
    }, [hydrationStatus, records]);

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

    const deleteSavedRecord = useCallback(async (id: string) => {
        const removed = await removeRecord(id);
        if (!removed || !mountedRef.current) return;
        const currentRecords = useExperimentMemoryStore.getState().records;
        setComparisonSelection((current) => current === null
            ? current
            : reconcileComparisonSelection(current, currentRecords, null));
    }, [removeRecord]);

    return (
        <div className="run-history-panel">
            <div className="inspection__empty" role="note" style={{ marginBottom: 8 }}>
                Saved runs contain a recipe plus worker-authored evaluation evidence. They do not
                contain trained parameters.
            </div>
            <div
                id={savedRecipeEffectsId}
                className="inspection__empty"
                style={{ marginBottom: 8 }}
            >
                {STATE_EFFECTS['saved-recipe-apply']}
            </div>
            <label style={{ display: 'grid', gap: 4, marginBottom: 8 }}>
                <span>Run name</span>
                <input
                    type="text"
                    value={runTitle}
                    onChange={(event) => setRunTitle(event.currentTarget.value)}
                    aria-describedby={runNameGuidanceId}
                    aria-invalid={runTitleTooLong}
                />
            </label>
            <div id={runNameGuidanceId} className="inspection__empty" style={{ marginBottom: 8 }}>
                Optional. Leave blank to use dataset, architecture, and saved step.
            </div>
            {runTitleTooLong && (
                <div className="inspection__empty" role="alert" style={{ marginBottom: 8 }}>
                    Run name must contain at most {EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS} Unicode code points.
                </div>
            )}
            <button
                type="button"
                className="btn btn--ghost btn--sm"
                style={{ width: '100%' }}
                onClick={() => { void saveController.commands.save(trimmedRunTitle); }}
                disabled={saveController.disabledReason !== null
                    || pendingSave !== null
                    || runTitleTooLong
                    || hydrationStatus !== 'ready'
                    || prepared === null}
            >
                {saving ? 'Saving…' : 'Save current run'}
            </button>

            {(actionError || (saveController.error && !persistenceError)) && <div className="inspection__empty" role="alert" style={{ marginTop: 8 }}>{actionError || saveController.error}</div>}
            {(persistenceError || pendingSave) && (
                <div
                    className="inspection__layer"
                    {...(persistenceError ? { role: 'alert' as const } : { role: 'status' as const })}
                    style={{ marginTop: 8 }}
                >
                    {persistenceError && (
                        <div className="inspection__empty">{persistenceError.message}</div>
                    )}
                    <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                        {pendingSave && (
                            <button
                                type="button"
                                className="btn btn--ghost btn--sm"
                                disabled={saving}
                                onClick={() => { void saveController.commands.retry(); }}
                            >
                                Retry saving
                            </button>
                        )}
                        {pendingSave && (
                            <button
                                type="button"
                                className="btn btn--ghost btn--sm"
                                disabled={saving}
                                onClick={() => { void saveController.commands.discard(); }}
                            >
                                Discard pending save
                            </button>
                        )}
                        {persistenceError && (
                            <button
                                type="button"
                                className="btn btn--ghost btn--sm"
                                onClick={saveController.commands.dismiss}
                            >
                                Dismiss error
                            </button>
                        )}
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

            {incompatibleEnvelope !== null && (
                <section
                    className="inspection__layer"
                    role="note"
                    aria-label="Incompatible saved-run file"
                    style={{ marginTop: 12 }}
                >
                    <div className="inspection__layer-name">Saved-run file is incompatible</div>
                    <div className="inspection__empty" style={{ marginTop: 6 }}>
                        These bytes were not changed or loaded as individual runs. Download a copy
                        before deleting the file. New saves remain pending until it is deleted.
                    </div>
                    <div className="inspection__empty" role="alert" style={{ marginTop: 6 }}>
                        {incompatibleEnvelope.issues
                            .map((entry) => `${entry.path}: ${entry.message}`)
                            .join(' ')}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={() => downloadText(
                                'nn-playground-incompatible-saved-runs.json',
                                incompatibleEnvelope.rawJson,
                            )}
                            aria-label="Download incompatible saved-run file"
                        >
                            Download raw file
                        </button>
                        <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={() => { void deleteIncompatibleEnvelope(); }}
                            aria-label="Delete incompatible saved-run file"
                        >
                            Delete incompatible file
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
                    <fieldset
                        aria-describedby={comparisonGuidanceId}
                        style={{ border: 0, margin: '12px 0 0', minWidth: 0, padding: 0 }}
                    >
                        <legend className="inspection__layer-name">Runs to compare</legend>
                        <div id={comparisonGuidanceId} className="inspection__empty" style={{ marginBottom: 8 }}>
                            Choose up to two saved runs. Selecting a third replaces the earliest choice.
                        </div>
                        <div className="inspection__layers">
                            {records.map((record) => {
                                const comparisonLabel = comparisonRecordLabel(record, records);
                                return (
                                    <article
                                        key={record.id}
                                        className="inspection__layer"
                                        aria-label={recordLabel(record)}
                                    >
                                        <div className="inspection__layer-name">{recordLabel(record)}</div>
                                        <label style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                                            <input
                                                type="checkbox"
                                                checked={selectedComparisonIds.includes(record.id)}
                                                onChange={() => setComparisonSelection((current) => (
                                                    reconcileComparisonSelection(
                                                        current ?? initialComparisonSelection,
                                                        records,
                                                        record.id,
                                                    )
                                                ))}
                                            />
                                            <span>Compare {comparisonLabel}</span>
                                        </label>
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
                                            {record.snapshot.evaluation.train.basis.sampleCount.toLocaleString()}{' '}
                                            train and{' '}
                                            {record.snapshot.evaluation.test.basis.sampleCount.toLocaleString()}{' '}
                                            test samples.
                                        </div>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                                            <button
                                                type="button"
                                                className="btn btn--ghost btn--sm"
                                                onClick={() => { void applySavedRecipe(record); }}
                                                aria-describedby={savedRecipeEffectsId}
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
                                                onClick={() => { void deleteSavedRecord(record.id); }}
                                                aria-label={`Delete ${recordLabel(record)}`}
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </article>
                                );
                            })}
                        </div>
                    </fieldset>
                    {selectedComparisonRecords.length === 2 && (
                        <SavedRunComparison
                            current={selectedComparisonRecords[0]}
                            baseline={selectedComparisonRecords[1]}
                            currentLabel={comparisonRecordLabel(selectedComparisonRecords[0], records)}
                            baselineLabel={comparisonRecordLabel(selectedComparisonRecords[1], records)}
                        />
                    )}
                </>
            )}
        </div>
    );
}
