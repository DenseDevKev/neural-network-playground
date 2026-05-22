// ── Preset Panel ──
// Card grid to quickly apply curated experiment presets.

import { useCallback, memo } from 'react';
import { PRESETS, type Preset } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { LoadingState } from '../common/LoadingState.tsx';
import { PresetCard } from './PresetCard.tsx';

interface PresetPanelProps {
    onReset: () => void;
    onApplied?: () => void;
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

export const PresetPanel = memo(function PresetPanel({ onReset, onApplied }: PresetPanelProps) {
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const data = usePlaygroundStore((s) => s.data);
    const network = usePlaygroundStore((s) => s.network);
    const features = usePlaygroundStore((s) => s.features);
    const training = usePlaygroundStore((s) => s.training);
    const ui = usePlaygroundStore((s) => s.ui);
    const isLoading = useTrainingStore((s) => s.presetConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);

    const handleSelect = useCallback(
        (preset: Preset) => {
            const state = usePlaygroundStore.getState();
            const isSelected = partialConfigMatches(preset.config.data, state.data)
                && partialConfigMatches(preset.config.network, state.network)
                && partialConfigMatches(preset.config.features, state.features)
                && partialConfigMatches(preset.config.training, state.training)
                && partialConfigMatches(preset.config.ui, state.ui);
            if (isSelected) return;

            useTrainingStore.getState().beginConfigChange('preset');
            applyPreset(preset);
            onReset();
            onApplied?.();
        },
        [applyPreset, onApplied, onReset],
    );
    const retryPresetChange = () => useTrainingStore.getState().retryConfigSync();

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Applying preset..." />
            {configError && configErrorSource === 'preset' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryPresetChange}>
                        Retry
                    </button>
                </div>
            )}
            <div className="preset-grid" role="list" aria-label="Available presets" aria-busy={isLoading}>
                {PRESETS.map((preset) => {
                    const isSelected = partialConfigMatches(preset.config.data, data)
                        && partialConfigMatches(preset.config.network, network)
                        && partialConfigMatches(preset.config.features, features)
                        && partialConfigMatches(preset.config.training, training)
                        && partialConfigMatches(preset.config.ui, ui);
                    return (
                    <div key={preset.id} role="listitem">
                        <PresetCard
                            preset={preset}
                            isSelected={isSelected}
                            disabled={isLoading}
                            onSelect={handleSelect}
                        />
                    </div>
                    );
                })}
            </div>
        </div>
    );
});
