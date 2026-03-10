// ── Preset Panel ──
// Dropdown to quickly apply curated experiment presets.

import { useState, useCallback } from 'react';
import { PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

interface PresetPanelProps {
    onReset: () => void;
}

export function PresetPanel({ onReset }: PresetPanelProps) {
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const [selectedId, setSelectedId] = useState('');
    const selectedPreset = PRESETS.find((p) => p.id === selectedId);

    const handleChange = useCallback(
        (e: React.ChangeEvent<HTMLSelectElement>) => {
            const id = e.target.value;
            setSelectedId(id);
            const preset = PRESETS.find((p) => p.id === id);
            if (preset) {
                applyPreset(preset);
                onReset();
            }
        },
        [applyPreset, onReset],
    );

    return (
        <div className="panel preset-panel">
            <div className="panel__title">Presets</div>
            <div className="preset-select-wrapper">
                <select
                    className="select"
                    style={{ width: '100%' }}
                    value={selectedId}
                    onChange={handleChange}
                >
                    <option value="">— Select a preset —</option>
                    {PRESETS.map((p) => (
                        <option key={p.id} value={p.id}>
                            {p.title}
                        </option>
                    ))}
                </select>
            </div>
            {selectedPreset && (
                <>
                    <div className="preset-description">{selectedPreset.description}</div>
                    {selectedPreset.learningGoal && (
                        <div className="preset-goal">💡 {selectedPreset.learningGoal}</div>
                    )}
                </>
            )}
        </div>
    );
}
