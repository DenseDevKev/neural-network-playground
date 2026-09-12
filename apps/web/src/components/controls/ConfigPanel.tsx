// ── Config Import/Export Panel ──
// Transports exact, validated version-2 experiment documents.

import {
    memo,
    useCallback,
    useEffect,
    useRef,
    useState,
    type ChangeEvent,
} from 'react';
import {
    MAX_EXPERIMENT_JSON_BYTES,
    decodeExperimentJson,
    encodeExperimentJson,
    type ExperimentSchemaIssue,
    type ValidatedExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { Tooltip } from '../common/Tooltip.tsx';
import './exportUtilities.css';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useTimedState } from '../../hooks/useTimedState.ts';

interface ConfigPanelProps {
    onReset: () => void;
}

const NO_ACTIVE_EXPERIMENT = '$: No compatible version-2 experiment is active';

function formatIssues(issues: readonly ExperimentSchemaIssue[]): string {
    return issues.slice(0, 8).map((issue) => `${issue.path}: ${issue.message}`).join('; ').slice(0, 1800);
}

function errorMessage(error: unknown, fallback: string): string {
    return error instanceof Error && error.message
        ? `${fallback}: ${error.message.slice(0, 1000)}`
        : fallback;
}

export const ConfigPanel = memo(function ConfigPanel(_props: ConfigPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const mounted = useRef(true);
    const copyInFlight = useRef(false);
    const importInFlight = useRef(false);
    const errorGeneration = useRef(0);
    const [stage, setStage] = useState<{ document: ValidatedExperimentDocumentV2; name: string; baseRequestId: number } | null>(null);
    const [setupLink, setSetupLink] = useState<string | null>(null);
    const target = useRef<PreparedExperimentDocumentV2 | null>(null);
    const selection = useRef(0);
    const pending = useTrainingStore((state) => state.pendingConfigSource);
    const configError = useTrainingStore((state) => state.configError);
    const [error, setError] = useState<string | null>(null);
    const [isCopying, setIsCopying] = useState(false);
    const [isImporting, setIsImporting] = useState(false);
    const [status, setStatus] = useTimedState<string | null>(null, 2000);

    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
        };
    }, []);

    const reportError = useCallback((message: string) => {
        if (!mounted.current) return;
        errorGeneration.current += 1;
        setStatus(null);
        setError(message);
    }, [setStatus]);

    const reportSuccess = useCallback((message: string, startErrorGeneration: number) => {
        if (!mounted.current) return;
        if (errorGeneration.current === startErrorGeneration) setError(null);
        setStatus(message);
    }, [setStatus]);

    const finishImport = useCallback(() => {
        importInFlight.current = false;
        if (mounted.current) setIsImporting(false);
    }, []);

    useEffect(() => {
        const check = () => {
            const expected = target.current;
            if (!expected) return;
            const ps = usePlaygroundStore.getState();
            const ts = useTrainingStore.getState();
            if (ps.access.status !== 'ready' || ps.access.prepared !== expected || ps.preparation.status === 'preparing') {
                target.current = null;
                finishImport();
                reportError('The active setup changed during import. Select the file again to review it.');
            } else if (ts.configError || ts.workerError) {
                reportError(`${ts.configError ?? ts.workerError}. Retry synchronization to finish applying this setup.`);
            } else if (ts.pendingConfigSource === null && ts.trainedRecipeSource === 'config-sync'
                && ts.trainedRecipeFingerprint === expected.identities.recipeFingerprint) {
                target.current = null;
                setStage(null);
                finishImport();
                setError(null);
                setStatus('Imported setup applied. Training is paused.');
            }
        };
        const offTraining = useTrainingStore.subscribe(check);
        const offPlayground = usePlaygroundStore.subscribe(check);
        return () => { offTraining(); offPlayground(); };
    }, [finishImport, reportError, setStatus]);

    const handleExport = useCallback(() => {
        const startErrorGeneration = errorGeneration.current;
        const access = usePlaygroundStore.getState().access;
        if (access.status !== 'ready') {
            reportError(NO_ACTIVE_EXPERIMENT);
            return;
        }
        const prepared = access.prepared;

        let url: string | null = null;
        try {
            const json = encodeExperimentJson(prepared.document);
            const blob = new Blob([json], { type: 'application/json' });
            url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = 'nn-playground-experiment-v2.json';
            anchor.click();
            reportSuccess('Exported!', startErrorGeneration);
        } catch (exportError) {
            reportError(errorMessage(exportError, '$: Could not export experiment'));
        } finally {
            if (url !== null) URL.revokeObjectURL(url);
        }
    }, [reportError, reportSuccess]);

    const handleCopyUrl = useCallback(async () => {
        if (copyInFlight.current) return;
        copyInFlight.current = true;
        setIsCopying(true);
        const startErrorGeneration = errorGeneration.current;

        try {
            let syncResult;
            try {
                syncResult = usePlaygroundStore.getState().syncToUrl();
            } catch (syncError) {
                reportError(errorMessage(syncError, '$: Could not synchronize URL'));
                return;
            }

            if (!syncResult.ok) {
                reportError(formatIssues(syncResult.issues) || '$: Could not synchronize URL');
                return;
            }

            const currentUrl = new URL(window.location.href);
            currentUrl.hash = syncResult.value.slice(1);
            const absoluteUrl = currentUrl.href;
            setSetupLink(absoluteUrl);
            const writeText = navigator.clipboard?.writeText;
            if (!writeText) {
                reportError('$: Could not copy setup link: Clipboard API is unavailable');
                return;
            }

            try {
                await writeText.call(navigator.clipboard, absoluteUrl);
                reportSuccess('URL copied!', startErrorGeneration);
            } catch (clipboardError) {
                reportError(errorMessage(clipboardError, '$: Could not copy setup link'));
            }
        } finally {
            copyInFlight.current = false;
            if (mounted.current) setIsCopying(false);
        }
    }, [reportError, reportSuccess]);

    const handleImport = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileSelect = useCallback((event: ChangeEvent<HTMLInputElement>) => {
        const input = event.currentTarget;
        const file = input.files?.[0];
        input.value = ''; // Same-file retry after every outcome.
        if (!file || importInFlight.current) return;
        const token = ++selection.current;
        const baseRequestId = usePlaygroundStore.getState().preparation.requestId;
        setStage(null); setError(null); setStatus(null);
        if (file.size > MAX_EXPERIMENT_JSON_BYTES) {
            reportError(`$: experiment JSON exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`);
            return;
        }
        const reader = new FileReader();
        const current = () => mounted.current && token === selection.current;
        reader.onerror = () => { if (current()) reportError('$: Could not read config file. Select the file again to retry.'); };
        reader.onabort = () => { if (current()) reportError('$: Config file read was canceled. Select the file again to retry.'); };
        reader.onload = (loadEvent) => {
            if (!current()) return;
            if (usePlaygroundStore.getState().preparation.requestId !== baseRequestId) {
                reportError('The active setup changed while reading. Select the file again to review it.'); return;
            }
            const text = loadEvent.target?.result;
            if (typeof text !== 'string') { reportError('$: Could not read config file. Select the file again.'); return; }
            const decoded = decodeExperimentJson(text);
            if (!decoded.ok) { reportError(formatIssues(decoded.issues)); return; }
            setStage({ document: decoded.value, name: file.name, baseRequestId });
        };
        try { reader.readAsText(file); }
        catch (cause) { if (current()) reportError(errorMessage(cause, '$: Could not read config file')); }
    }, [reportError, setStatus]);

    const applyImport = async () => {
        if (!stage || importInFlight.current || useTrainingStore.getState().pendingConfigSource !== null) return;
        const ps = usePlaygroundStore.getState();
        if (ps.preparation.requestId !== stage.baseRequestId || ps.preparation.status === 'preparing') {
            reportError('The active setup changed. Select the file again to review it before applying.'); return;
        }
        importInFlight.current = true; setIsImporting(true); setError(null); setStatus(null);
        const token = selection.current;
        useTrainingStore.getState().beginConfigChange('setup');
        let requestId = ps.preparation.requestId;
        try {
            const promise = ps.replaceDocument(stage.document);
            requestId = usePlaygroundStore.getState().preparation.requestId;
            const result = await promise;
            if (!mounted.current || selection.current !== token || usePlaygroundStore.getState().preparation.requestId !== requestId) return;
            if (!result.ok) {
                const message = formatIssues(result.issues);
                useTrainingStore.getState().failConfigChange(message);
                setStage({ ...stage, baseRequestId: requestId });
                reportError(`${message}. The staged file is preserved; retry Apply or select another file.`);
                return;
            }
            const access = usePlaygroundStore.getState().access;
            if (access.status !== 'ready' || access.prepared !== result.value) return;
            target.current = result.value;
            // The existing config transaction pauses and resets exactly once.
            // Success belongs only to the worker acknowledgement of this publication.
            const ts = useTrainingStore.getState();
            if (ts.configError || ts.workerError) reportError(`${ts.configError ?? ts.workerError}. Retry synchronization.`);
            else if (ts.pendingConfigSource === null && ts.trainedRecipeSource === 'config-sync'
                && ts.trainedRecipeFingerprint === result.value.identities.recipeFingerprint) {
                target.current = null; setStage(null); reportSuccess('Imported setup applied. Training is paused.', errorGeneration.current);
            }
        } catch (cause) {
            if (mounted.current && selection.current === token && usePlaygroundStore.getState().preparation.requestId === requestId) {
                const message = errorMessage(cause, 'Setup could not be prepared');
                useTrainingStore.getState().failConfigChange(message);
                setStage({ ...stage, baseRequestId: requestId }); reportError(`${message}. Retry Apply or select the file again.`);
            }
        } finally { if (!target.current) finishImport(); }
    };

    return (
        <div className="setup-export-panel">
            <h3>Experiment setup</h3><p>Save the recipe and view settings. Evaluated evidence downloads remain in Saved runs.</p>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <Tooltip content="Cause: export saves the exact version-2 experiment recipe and view. Effect: you can replay the same canonical experiment later.">
                    <button className="btn btn--ghost btn--sm" onClick={handleExport}>
                        Export JSON setup
                    </button>
                </Tooltip>
                <Tooltip content="Cause: import strictly validates and prepares a saved version-2 experiment. Effect: the playground changes only after the full experiment is ready.">
                    <button
                        className="btn btn--ghost btn--sm"
                        onClick={handleImport}
                        disabled={isImporting}
                        aria-busy={isImporting}
                    >
                        Import JSON
                    </button>
                </Tooltip>
                <Tooltip content="Cause: copying first synchronizes the exact version-2 experiment into the address. Effect: someone else can open the same canonical setup.">
                    <button
                        className="btn btn--ghost btn--sm"
                        onClick={handleCopyUrl}
                        disabled={isCopying}
                        aria-busy={isCopying}
                    >
                        Copy setup link
                    </button>
                </Tooltip>
            </div>
            {stage && <section className="setup-import-review" aria-label="Imported setup review">
                <h3>Review imported setup</h3><p>{stage.name}</p>
                <dl><div><dt>Dataset</dt><dd>{stage.document.recipe.task.dataset}</dd></div>
                    <div><dt>Samples</dt><dd>{stage.document.recipe.data.sampleCount}</dd></div>
                    <div><dt>Hidden layers</dt><dd>{stage.document.recipe.model.hiddenLayers.join(' → ') || 'None'}</dd></div>
                    <div><dt>Optimizer</dt><dd>{stage.document.recipe.training.optimizer.kind}</dd></div>
                    <div><dt>Learning rate</dt><dd>{stage.document.recipe.training.learningRate}</dd></div></dl>
                <p>Applying replaces the setup and starts a fresh, paused model.</p>
                <div className="export-actions"><button type="button" disabled={isImporting || pending !== null} onClick={() => void applyImport()}>Apply imported setup</button>
                    <button type="button" disabled={isImporting} onClick={() => { selection.current++; setStage(null); setError(null); }}>Cancel</button></div>
            </section>}
            {isImporting && <p role="status">Waiting for the worker to acknowledge this setup…</p>}
            {isImporting && configError && <button type="button" onClick={() => useTrainingStore.getState().retryConfigSync()}>Retry synchronization</button>}
            {setupLink && <label className="setup-link-fallback">Selectable setup link<textarea readOnly value={setupLink} onFocus={(event) => event.currentTarget.select()} /><span>If copying fails, select this link or download the setup JSON above.</span></label>}
            {error && (
                <div className="config-feedback config-feedback--error" role="alert">
                    {error}
                </div>
            )}
            {status && (
                <div className="config-feedback" role="status">
                    {status}
                </div>
            )}
            <input
                ref={fileInputRef}
                aria-label="Import setup JSON file"
                type="file"
                accept=".json,application/json"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
            />
        </div>
    );
});
