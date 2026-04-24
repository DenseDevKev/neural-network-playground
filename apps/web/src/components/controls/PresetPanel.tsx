// ── Preset Panel ──
// Card grid to quickly apply curated experiment presets.

import { useCallback, memo } from 'react';
import { PRESETS, type Preset } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { PresetCard } from './PresetCard.tsx';

interface PresetPanelProps {
    onReset: () => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null;
}

function valueMatches(expected: unknown, actual: unknown): boolean {
    if (Array.isArray(expected)) {
        return Array.isArray(actual)
            && expected.length === actual.length
            && expected.every((item, index) => valueMatches(item, actual[index]));
    }

    if (isRecord(expected)) {
        if (!isRecord(actual)) return false;
        return Object.entries(expected).every(([key, value]) => valueMatches(value, actual[key]));
    }

    return Object.is(expected, actual);
}

function partialConfigMatches(expected: unknown, actual: unknown): boolean {
    if (!isRecord(expected)) return true;
    if (!isRecord(actual)) return false;
    return Object.entries(expected).every(([key, value]) => valueMatches(value, actual[key]));
}

export const PresetPanel = memo(function PresetPanel({ onReset }: PresetPanelProps) {
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const data = usePlaygroundStore((s) => s.data);
    const network = usePlaygroundStore((s) => s.network);
    const features = usePlaygroundStore((s) => s.features);
    const training = usePlaygroundStore((s) => s.training);
    const ui = usePlaygroundStore((s) => s.ui);

    const handleSelect = useCallback(
        (preset: Preset) => {
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
                            isSelected={
                                partialConfigMatches(preset.config.data, data)
                                && partialConfigMatches(preset.config.network, network)
                                && partialConfigMatches(preset.config.features, features)
                                && partialConfigMatches(preset.config.training, training)
                                && partialConfigMatches(preset.config.ui, ui)
                            }
                            onSelect={handleSelect}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
});
