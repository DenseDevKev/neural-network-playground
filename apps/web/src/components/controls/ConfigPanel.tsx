// ── Config Import/Export Panel ──
// Allows users to save and restore playground configurations as JSON.

import { useCallback, useRef, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { exportConfigJson, validateImportedConfig } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';
import { useTimedState } from '../../hooks/useTimedState.ts';

interface ConfigPanelProps {
    onReset: () => void;
}

export const ConfigPanel = memo(function ConfigPanel({ onReset }: ConfigPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [feedback, setFeedback] = useTimedState<{ message: string; tone: 'status' | 'error' } | null>(null, 2000);

    const handleExport = useCallback(() => {
        const config = usePlaygroundStore.getState().getConfig();
        const json = exportConfigJson(config);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'nn-playground-config.json';
        a.click();
        URL.revokeObjectURL(url);
        setFeedback({ message: 'Exported!', tone: 'status' });
    }, [setFeedback]);

    const handleCopyUrl = useCallback(async () => {
        try {
            if (!navigator.clipboard?.writeText) {
                throw new Error('Clipboard API unavailable');
            }

            await navigator.clipboard.writeText(window.location.href);
            setFeedback({ message: 'URL copied!', tone: 'status' });
        } catch {
            setFeedback({ message: 'Could not copy URL.', tone: 'error' });
        }
    }, [setFeedback]);

    const handleImport = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;

            if (file.size > 1024 * 1024) {
                setFeedback({ message: 'Config file must be smaller than 1MB', tone: 'error' });
                e.target.value = '';
                return;
            }

            const reader = new FileReader();
            reader.onload = (ev) => {
                const text = ev.target?.result as string;
                let validation;
                try {
                    validation = validateImportedConfig(JSON.parse(text));
                } catch {
                    validation = { config: null, error: 'Invalid JSON file' };
                }

                if (validation.config) {
                    const store = usePlaygroundStore.getState();
                    store.applyPreset({
                        id: 'imported',
                        title: 'Imported Config',
                        description: '',
                        config: {
                            data: validation.config.data,
                            network: validation.config.network,
                            features: validation.config.features,
                            training: validation.config.training,
                            ui: validation.config.ui,
                        },
                    });
                    onReset();
                    setFeedback({ message: 'Imported!', tone: 'status' });
                } else {
                    setFeedback({ message: validation.error ?? 'Invalid config', tone: 'error' });
                }
            };
            reader.readAsText(file);
            // Reset so same file can be re-imported
            e.target.value = '';
        },
        [onReset, setFeedback],
    );

    return (
        <div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <Tooltip content="Download the current playground configuration as JSON">
                    <button type="button" className="btn btn--ghost btn--sm" onClick={handleExport}>
                        ↓ Export JSON
                    </button>
                </Tooltip>
                <Tooltip content="Import a previously saved JSON configuration file">
                    <button type="button" className="btn btn--ghost btn--sm" onClick={handleImport}>
                        ↑ Import JSON
                    </button>
                </Tooltip>
                <Tooltip content="Copy a shareable URL for the current configuration">
                    <button type="button" className="btn btn--ghost btn--sm" onClick={handleCopyUrl}>
                        🔗 Copy URL
                    </button>
                </Tooltip>
            </div>
            {feedback && (
                <div
                    className={`config-feedback ${feedback.tone === 'error' ? 'config-feedback--error' : ''}`}
                    role={feedback.tone === 'error' ? 'alert' : 'status'}
                >
                    {feedback.message}
                </div>
            )}
            <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
            />
        </div>
    );
});
