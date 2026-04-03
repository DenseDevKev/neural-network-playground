// ── Preset Panel ──
// Card grid to quickly apply curated experiment presets.

import { useState, useCallback, memo } from 'react';
import { PRESETS, type Preset } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { PresetCard } from './PresetCard.tsx';

interface PresetPanelProps {
    onReset: () => void;
}

export const PresetPanel = memo(function PresetPanel({ onReset }: PresetPanelProps) {
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const [selectedId, setSelectedId] = useState('');

    const handleSelect = useCallback(
        (preset: Preset) => {
            setSelectedId(preset.id);
            applyPreset(preset);
            onReset();
        },
        [applyPreset, onReset],
    );

    return (
        <div>
            <div className="preset-grid" role="list" aria-label="Available presets">
                {PRESETS.map((preset) => (
                    <div key={preset.id} role="listitem">
                        <PresetCard
                            preset={preset}
                            isSelected={selectedId === preset.id}
                            onSelect={handleSelect}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
});
