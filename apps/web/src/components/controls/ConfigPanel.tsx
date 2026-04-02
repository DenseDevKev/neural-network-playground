// ── Config Import/Export Panel ──
// Allows users to save and restore playground configurations as JSON.

import { useCallback, useRef, useState, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { exportConfigJson, importConfigJson } from '@nn-playground/shared';

interface ConfigPanelProps {
    onReset: () => void;
}

export const ConfigPanel = memo(function ConfigPanel({ onReset }: ConfigPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [feedback, setFeedback] = useState<string | null>(null);

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
        setTimeout(() => setFeedback(null), 2000);
    }, []);

    const handleCopyUrl = useCallback(() => {
        navigator.clipboard.writeText(window.location.href).then(() => {
            setFeedback('URL copied!');
            setTimeout(() => setFeedback(null), 2000);
        });
    }, []);

    const handleImport = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (ev) => {
                const text = ev.target?.result as string;
                const config = importConfigJson(text);
                if (config) {
                    const store = usePlaygroundStore.getState();
                    store.applyPreset({
                        id: 'imported',
                        title: 'Imported Config',
                        description: '',
                        config: {
                            data: config.data,
                            network: config.network,
                            features: config.features,
                            training: config.training,
                            ui: config.ui,
                        },
                    });
                    onReset();
                    setFeedback('Imported!');
                } else {
                    setFeedback('Invalid config');
                }
                setTimeout(() => setFeedback(null), 2000);
            };
            reader.readAsText(file);
            // Reset so same file can be re-imported
            e.target.value = '';
        },
        [onReset],
    );

    return (
        <div className="panel">
            <div className="panel__title">Config</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <button className="btn btn--ghost btn--sm" onClick={handleExport}>
                    ↓ Export JSON
                </button>
                <button className="btn btn--ghost btn--sm" onClick={handleImport}>
                    ↑ Import JSON
                </button>
                <button className="btn btn--ghost btn--sm" onClick={handleCopyUrl}>
                    🔗 Copy URL
                </button>
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
