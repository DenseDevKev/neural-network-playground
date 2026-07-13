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
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { Tooltip } from '../common/Tooltip.tsx';
import { useTimedState } from '../../hooks/useTimedState.ts';

interface ConfigPanelProps {
    onReset: () => void;
}

const NO_ACTIVE_EXPERIMENT = '$: No compatible version-2 experiment is active';

function formatIssues(issues: readonly ExperimentSchemaIssue[]): string {
    return issues.map((issue) => `${issue.path}: ${issue.message}`).join('; ');
}

function errorMessage(error: unknown, fallback: string): string {
    return error instanceof Error && error.message
        ? `${fallback}: ${error.message}`
        : fallback;
}

export const ConfigPanel = memo(function ConfigPanel({ onReset }: ConfigPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const mounted = useRef(true);
    const copyInFlight = useRef(false);
    const importInFlight = useRef(false);
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
        setStatus(null);
        setError(message);
    }, [setStatus]);

    const reportSuccess = useCallback((message: string) => {
        if (mounted.current) setStatus(message);
    }, [setStatus]);

    const finishImport = useCallback(() => {
        importInFlight.current = false;
        if (mounted.current) setIsImporting(false);
    }, []);

    const rejectFile = useCallback((
        file: File,
        issues: readonly ExperimentSchemaIssue[],
    ) => {
        usePlaygroundStore.getState().markIncompatible(
            { kind: 'file', file },
            issues,
        );
        reportError(formatIssues(issues));
    }, [reportError]);

    const handleExport = useCallback(() => {
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
            reportSuccess('Exported!');
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

            const writeText = navigator.clipboard?.writeText;
            if (!writeText) {
                reportError('$: Could not copy URL: Clipboard API is unavailable');
                return;
            }

            const currentUrl = new URL(window.location.href);
            currentUrl.hash = syncResult.value.slice(1);
            const absoluteUrl = currentUrl.href;
            try {
                await writeText.call(navigator.clipboard, absoluteUrl);
                reportSuccess('URL copied!');
            } catch (clipboardError) {
                reportError(errorMessage(clipboardError, '$: Could not copy URL'));
            }
        } finally {
            copyInFlight.current = false;
            if (mounted.current) setIsCopying(false);
        }
    }, [reportError, reportSuccess]);

    const handleImport = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileSelect = useCallback(
        (event: ChangeEvent<HTMLInputElement>) => {
            const input = event.currentTarget;
            const file = input.files?.[0];
            if (!file) return;

            if (importInFlight.current) {
                input.value = '';
                return;
            }

            if (file.size > MAX_EXPERIMENT_JSON_BYTES) {
                rejectFile(file, [{
                    code: 'resource-limit',
                    path: '$',
                    message: `experiment JSON exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`,
                }]);
                input.value = '';
                return;
            }

            importInFlight.current = true;
            setIsImporting(true);
            const selectionRequestId = usePlaygroundStore.getState()
                .preparation.requestId;

            const reader = new FileReader();
            reader.onerror = () => {
                if (
                    mounted.current
                    && usePlaygroundStore.getState().preparation.requestId
                        === selectionRequestId
                ) {
                    rejectFile(file, [{
                        code: 'invalid-field',
                        path: '$',
                        message: 'Could not read config file',
                    }]);
                }
                finishImport();
            };
            reader.onabort = () => {
                if (
                    mounted.current
                    && usePlaygroundStore.getState().preparation.requestId
                        === selectionRequestId
                ) {
                    rejectFile(file, [{
                        code: 'invalid-field',
                        path: '$',
                        message: 'Config file read was canceled',
                    }]);
                }
                finishImport();
            };
            reader.onload = async (loadEvent) => {
                try {
                    if (
                        !mounted.current
                        || usePlaygroundStore.getState().preparation.requestId
                            !== selectionRequestId
                    ) {
                        return;
                    }

                    const text = loadEvent.target?.result;
                    if (typeof text !== 'string') {
                        rejectFile(file, [{
                            code: 'invalid-field',
                            path: '$',
                            message: 'Could not read config file',
                        }]);
                        return;
                    }

                    const decoded = decodeExperimentJson(text);
                    if (!decoded.ok) {
                        rejectFile(file, decoded.issues);
                        return;
                    }

                    let preparationRequestId = selectionRequestId;
                    let result;
                    try {
                        const pendingResult = usePlaygroundStore.getState()
                            .replaceImportedDocument(decoded.value, file);
                        preparationRequestId = usePlaygroundStore.getState()
                            .preparation.requestId;
                        result = await pendingResult;
                    } catch (preparationError) {
                        if (
                            usePlaygroundStore.getState().preparation.requestId
                            === preparationRequestId
                        ) {
                            rejectFile(file, [{
                                code: 'invalid-field',
                                path: '$',
                                message: errorMessage(
                                    preparationError,
                                    'Experiment could not be prepared',
                                ),
                            }]);
                        }
                        return;
                    }

                    if (
                        usePlaygroundStore.getState().preparation.requestId
                        !== preparationRequestId
                    ) {
                        return;
                    }

                    if (!result.ok) {
                        rejectFile(file, result.issues);
                        return;
                    }

                    // A successful return can belong to an older request. Only the
                    // exact result currently published by the store may reset UI.
                    if (
                        !mounted.current
                        || usePlaygroundStore.getState().access.status !== 'ready'
                        || usePlaygroundStore.getState().access.prepared !== result.value
                    ) {
                        return;
                    }
                    onReset();
                    if (mounted.current) reportSuccess('Imported!');
                } finally {
                    finishImport();
                }
            };

            try {
                reader.readAsText(file);
            } catch (readError) {
                rejectFile(file, [{
                    code: 'invalid-field',
                    path: '$',
                    message: errorMessage(readError, 'Could not read config file'),
                }]);
                finishImport();
            } finally {
                // Permit choosing the same file again, including after any failure.
                input.value = '';
            }
        },
        [finishImport, onReset, rejectFile, reportSuccess],
    );

    return (
        <div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <Tooltip content="Cause: export saves the exact version-2 experiment recipe and view. Effect: you can replay the same canonical experiment later.">
                    <button className="btn btn--ghost btn--sm" onClick={handleExport}>
                        ↓ Export JSON
                    </button>
                </Tooltip>
                <Tooltip content="Cause: import strictly validates and prepares a saved version-2 experiment. Effect: the playground changes only after the full experiment is ready.">
                    <button
                        className="btn btn--ghost btn--sm"
                        onClick={handleImport}
                        disabled={isImporting}
                        aria-busy={isImporting}
                    >
                        ↑ Import JSON
                    </button>
                </Tooltip>
                <Tooltip content="Cause: copying first synchronizes the exact version-2 experiment into the address. Effect: someone else can open the same canonical setup.">
                    <button
                        className="btn btn--ghost btn--sm"
                        onClick={handleCopyUrl}
                        disabled={isCopying}
                        aria-busy={isCopying}
                    >
                        🔗 Copy URL
                    </button>
                </Tooltip>
            </div>
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
                type="file"
                accept=".json,application/json"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
            />
        </div>
    );
});
