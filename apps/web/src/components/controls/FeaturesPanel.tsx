// ── Features Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { ALL_FEATURES } from '@nn-playground/engine';
import type { FeatureFlags } from '@nn-playground/engine';

export const FeaturesPanel = memo(function FeaturesPanel() {
    const features = usePlaygroundStore((s) => s.features);
    const store = usePlaygroundStore;

    return (
        <div className="panel">
            <div className="panel__title">Features</div>
            <div className="chip-group">
                {ALL_FEATURES.map((f) => (
                    <button
                        key={f.id}
                        className={`feature-chip ${features[f.id as keyof FeatureFlags] ? 'active' : ''}`}
                        onClick={() => store.getState().toggleFeature(f.id as keyof FeatureFlags)}
                        aria-pressed={features[f.id as keyof FeatureFlags]}
                        title={`Toggle ${f.label} feature`}
                    >
                        {f.label}
                    </button>
                ))}
            </div>
        </div>
    );
});
