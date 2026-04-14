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
    const [feedback, setFeedback] = useTimedState<string | null>(null, 2000);

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
        setFeedback('Exported!');
    }, []);

    const handleCopyUrl = useCallback(() => {
        navigator.clipboard.writeText(window.location.href).then(() => {
            setFeedback('URL copied!');
        });
    }, []);

    const handleImport = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;

            if (file.size > 1024 * 1024) {
                setFeedback('Config file must be smaller than 1MB');
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
                    setFeedback('Imported!');
                } else {
                    setFeedback(validation.error ?? 'Invalid config');
                }
            };
            reader.readAsText(file);
            // Reset so same file can be re-imported
            e.target.value = '';
        },
        [onReset],
    );

    return (
        <div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <Tooltip content="Download the current playground configuration as JSON">
                    <button className="btn btn--ghost btn--sm" onClick={handleExport}>
                        ↓ Export JSON
                    </button>
                </Tooltip>
                <Tooltip content="Import a previously saved JSON configuration file">
                    <button className="btn btn--ghost btn--sm" onClick={handleImport}>
                        ↑ Import JSON
                    </button>
                </Tooltip>
                <Tooltip content="Copy a shareable URL for the current configuration">
                    <button className="btn btn--ghost btn--sm" onClick={handleCopyUrl}>
                        🔗 Copy URL
                    </button>
                </Tooltip>
            </div>
            {feedback && (
                <div className="config-feedback">{feedback}</div>
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
