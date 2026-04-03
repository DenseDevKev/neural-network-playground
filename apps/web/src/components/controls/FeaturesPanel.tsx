// ── Features Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { ALL_FEATURES } from '@nn-playground/engine';
import type { FeatureFlags } from '@nn-playground/engine';
import { Tooltip } from '../common/Tooltip.tsx';

export const FeaturesPanel = memo(function FeaturesPanel() {
    const features = usePlaygroundStore((s) => s.features);
    const store = usePlaygroundStore;

    return (
        <div>
            <div className="chip-group">
                {ALL_FEATURES.map((f) => (
                    <Tooltip key={f.id} content={`Toggle the ${f.label} input feature`}>
                        <button
                            className={`feature-chip ${features[f.id as keyof FeatureFlags] ? 'active' : ''}`}
                            onClick={() => store.getState().toggleFeature(f.id as keyof FeatureFlags)}
                            aria-pressed={features[f.id as keyof FeatureFlags]}
                        >
                            {f.label}
                        </button>
                    </Tooltip>
                ))}
            </div>
        </div>
    );
});
