import { memo, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type {
    ExperimentRunRecordV2,
    RejectedExperimentRunRecordV2,
} from '@nn-playground/shared';
import { EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS } from '@nn-playground/shared';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useSaveCurrentRun, type SaveCurrentRunController } from '../../hooks/useSaveCurrentRun.ts';
import { reconcileComparisonSelection } from './runComparisonSelection.ts';
import { SavedRunComparison } from '../atelier/saved/Comparison.tsx';
import { ConfirmAction } from '../atelier/saved/ConfirmAction.tsx';
import { RenameRun } from '../atelier/saved/RenameRun.tsx';
import '../atelier/saved/saved.css';
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

function downloadText(filename: string, text: string, onError: (message: string) => void): void {
    let url: string | null = null;
    try {
        url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        anchor.click();
    } catch (error) { onError(`Download failed: ${String(error)}`); }
    finally { if (url !== null) URL.revokeObjectURL(url); }
}

function RejectedRecord({ record }: { record: RejectedExperimentRunRecordV2 }) {
    const [downloadError, setDownloadError] = useState<string | null>(null);
    const deleteRejectedRecord = useExperimentMemoryStore((state) => state.deleteRejectedRecord);
    return (
        <article className="inspection__layer" aria-label={`Rejected saved record ${record.sourceIndex}`}>
            <div className="inspection__layer-name">Rejected saved record</div>
            {downloadError && <p role="alert">{downloadError}</p>}
            <div className="inspection__empty" role="alert" style={{ marginTop: 6 }}>
                {record.issues.map((entry) => `${entry.path}: ${entry.message}`).join(' ')}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => downloadText(
                        `rejected-run-${record.sourceIndex}.json`,
                        record.rawJson, setDownloadError,
                    )}
                    aria-label="Download rejected record"
                >
                    Download raw JSON
                </button>
                <ConfirmAction label="Delete rejected record" title="Delete rejected record" onConfirm={() => deleteRejectedRecord(record.sourceIndex)}>Delete this rejected record? Download the raw JSON first to keep a copy.</ConfirmAction>
            </div>
        </article>
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
    const compareButtonRef = useRef<HTMLButtonElement>(null);
    const listRef = useRef<HTMLDivElement>(null);
    const listScroll = useRef<Array<{ node: HTMLElement; top: number; left: number }>>([]);
    const [comparing, setComparing] = useState(false);
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
        setComparisonSelection((current) => {
            if (current === null) return records.slice(0, 2).map((record) => record.id);
            const next = reconcileComparisonSelection(current, records, null);
            return next.length === current.length && next.every((id, index) => id === current[index]) ? current : next;
        });
    }, [hydrationStatus, records]);

    useEffect(() => {
        if (comparing && selectedComparisonRecords.length !== 2) setComparing(false);
    }, [comparing, selectedComparisonRecords.length]);

    useLayoutEffect(() => {
        if (comparing) return;
        if (listScroll.current.length) compareButtonRef.current?.focus({ preventScroll: true });
        for (const { node, top, left } of listScroll.current) {
            node.scrollTop = top; node.scrollLeft = left;
        }
    }, [comparing]);

    function openComparison() {
        const nodes = Array.from(listRef.current?.querySelectorAll<HTMLElement>('.saved-table-scroll') ?? []);
        let ancestor: HTMLElement | null = listRef.current;
        while (ancestor) { nodes.push(ancestor); ancestor = ancestor.parentElement; }
        listScroll.current = nodes.map((node) => ({ node, top: node.scrollTop, left: node.scrollLeft }));
        setComparing(true);
    }

    const applySavedRecipe = useCallback(async (record: ExperimentRunRecordV2) => {
        setActionError(null);
        if (!prepared) {
            setActionError('A compatible version-2 experiment is required before applying a saved recipe.');
            return false;
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
        return result.ok;
    }, [prepared, replaceDocument]);

    const deleteSavedRecord = useCallback(async (id: string) => {
        const removed = await removeRecord(id);
        if (!removed || !mountedRef.current) return false;
        const currentRecords = useExperimentMemoryStore.getState().records;
        setComparisonSelection((current) => current === null
            ? current
            : reconcileComparisonSelection(current, currentRecords, null));
        return true;
    }, [removeRecord]);

    return (
        <div className="run-history-panel saved-runs">
            <div ref={listRef} hidden={comparing && selectedComparisonRecords.length === 2}>
            <h2 className="saved-sr-only">Saved runs</h2>
            <p>A notebook of your experiments, stored in this browser. {records.length} / 20 saved runs.</p>
            <p id={savedRecipeEffectsId}>Saved evidence includes a recipe and full evaluation, without trained parameters. Applying a recipe starts a fresh model.</p>
            {(persistenceError || pendingSave) && (
                <div
                    className="inspection__layer"
                    {...(persistenceError ? { role: 'alert' as const } : { role: 'status' as const })}
                    style={{ marginTop: 8 }}
                >
                    {persistenceError && (
                        <div className="inspection__empty">{persistenceError.message}</div>
                    )}
                    {pendingSave && <><h3>This run couldn’t be saved</h3><p>{recordLabel(pendingSave)} · Full evaluation at step {pendingSave.snapshot.model.step.toLocaleString()}</p><p>{pendingSave.snapshot.evaluation.train.basis.sampleCount} train / {pendingSave.snapshot.evaluation.test.basis.sampleCount} test samples · Train loss {formatMetric(pendingSave.snapshot.evaluation.train.values.dataLoss)} · Test loss {formatMetric(pendingSave.snapshot.evaluation.test.values.dataLoss)}</p><p>Retry and download use this exact snapshot, even if training continues. Keep this tab open or download the pending evidence before leaving.</p></>}
                    <div className="saved-actions">
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
                        {pendingSave && <><button type="button" disabled={saving} onClick={() => { void saveController.commands.downloadPending(); }}>Download pending evidence</button><ConfirmAction disabled={saving} label="Discard pending save" title="Discard pending save" onConfirm={saveController.commands.discard}>This removes the retained snapshot from this tab. Download its evidence first to keep a copy.</ConfirmAction></>}
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

            <div className="saved-capture">
            <div>
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
            </div>
            <button
                type="button"
                className="saved-primary"
                onClick={() => { void saveController.commands.save(trimmedRunTitle); }}
                disabled={saveController.disabledReason !== null
                    || pendingSave !== null
                    || runTitleTooLong
                    || hydrationStatus !== 'ready'
                    || prepared === null}
            >
                {saving ? 'Saving…' : 'Save current run'}
            </button>
            </div>

            {(actionError || (saveController.error && saveController.error !== persistenceError?.message)) && <div className="inspection__empty" role="alert" style={{ marginTop: 8 }}>{actionError || saveController.error}</div>}
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
                            onClick={() => downloadText('nn-playground-earlier-runs.json', legacyRaw, setActionError)}
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
                        <ConfirmAction label="Delete earlier runs" title="Delete earlier runs" onConfirm={deleteLegacyStorage}>Delete these stored bytes? Download a copy first; this cannot be undone.</ConfirmAction>
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
                                incompatibleEnvelope.rawJson, setActionError,
                            )}
                            aria-label="Download incompatible saved-run file"
                        >
                            Download raw file
                        </button>
                        <ConfirmAction label="Delete incompatible saved-run file" title="Delete incompatible file" onConfirm={deleteIncompatibleEnvelope}>Delete these stored bytes? Download a copy first; this cannot be undone.</ConfirmAction>
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
                            Choose exactly two saved runs, then compare their stored evidence. Uncheck a run to select another.
                        </div>
                        <div className="saved-table-scroll"><table className="saved-run-table"><thead><tr><th scope="col">Select</th><th scope="col">Run</th><th scope="col">Dataset</th><th scope="col">Step</th><th scope="col">Train loss</th><th scope="col">Test loss</th><th scope="col">Actions</th></tr></thead><tbody>
                            {records.map((record) => <tr key={record.id}>
                                <td data-label="Select"><input type="checkbox" aria-label={`Compare ${comparisonRecordLabel(record, records)}`} checked={selectedComparisonIds.includes(record.id)} disabled={selectedComparisonIds.length === 2 && !selectedComparisonIds.includes(record.id)} onChange={() => setComparisonSelection((current) => reconcileComparisonSelection(current ?? initialComparisonSelection, records, record.id))} /></td>
                                <td data-label="Run"><article aria-label={recordLabel(record)}><strong>{recordLabel(record)}</strong><small>{new Date(record.createdAt).toLocaleString()} · Model revision {record.snapshot.model.revision}</small></article></td>
                                <td data-label="Dataset">{record.recipe.task.dataset}</td>
                                <td data-label="Step">{record.snapshot.model.step.toLocaleString()}</td>
                                <td data-label="Train loss">{formatMetric(record.snapshot.evaluation.train.values.dataLoss)}</td>
                                <td data-label="Test loss">{formatMetric(record.snapshot.evaluation.test.values.dataLoss)}</td>
                                <td data-label="Actions"><details><summary>Actions<span className="saved-sr-only"> for {comparisonRecordLabel(record, records)}</span></summary><div className="saved-actions">
                                    <RenameRun record={record} />
                                    <ConfirmAction label="Apply saved recipe" title="Apply saved recipe" descriptionId={savedRecipeEffectsId} onConfirm={() => applySavedRecipe(record)}>{STATE_EFFECTS['saved-recipe-apply']} This starts a fresh model; saved trained parameters are not available.</ConfirmAction>
                                    <button type="button" aria-label={`Download evidence for ${recordLabel(record)}`} onClick={() => { try { downloadText(`${record.id}.json`, JSON.stringify(record, null, 2), setActionError); } catch (error) { setActionError(`Download failed: ${String(error)}`); } }}>Download evidence</button>
                                    <ConfirmAction label={`Delete ${recordLabel(record)}`} title="Delete saved run" onConfirm={() => deleteSavedRecord(record.id)}>Delete {recordLabel(record)} permanently? Download the evidence first to keep a copy.</ConfirmAction>
                                </div></details></td>
                            </tr>)}
                        </tbody></table></div>
                        <div className="saved-selection"><span>{selectedComparisonIds.length} runs selected</span><button type="button" onClick={() => setComparisonSelection([])}>Clear</button><button ref={compareButtonRef} className="saved-primary" type="button" disabled={selectedComparisonRecords.length !== 2} onClick={openComparison}>Compare selected</button></div>
                    </fieldset>
                </>
            )}
            </div>
            {comparing && selectedComparisonRecords.length === 2 && <SavedRunComparison current={selectedComparisonRecords[0]} baseline={selectedComparisonRecords[1]} currentLabel={comparisonRecordLabel(selectedComparisonRecords[0], records)} baselineLabel={comparisonRecordLabel(selectedComparisonRecords[1], records)} onBack={() => setComparing(false)} />}
        </div>
    );
}
