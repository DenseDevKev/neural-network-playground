// ── Features Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { ALL_FEATURES } from '@nn-playground/engine';
import type { FeatureFlags } from '@nn-playground/engine';
import { Tooltip } from '../common/Tooltip.tsx';

const FEATURE_TOOLTIPS: Record<string, string> = {
    x: 'Cause: x gives the model horizontal position. Effect: removing it blinds the network to left-right structure.',
    y: 'Cause: y gives the model vertical position. Effect: removing it blinds the network to up-down structure.',
    xSquared: 'Cause: x squared turns distance from the vertical center into a feature. Effect: circles and rings become easier to separate.',
    ySquared: 'Cause: y squared turns distance from the horizontal center into a feature. Effect: round patterns can be learned with simpler weights.',
    xy: 'Cause: x*y captures diagonal interaction. Effect: quadrant-style patterns such as XOR become easier for shallow models.',
    sinX: 'Cause: sin(x) adds a repeating horizontal signal. Effect: wavy boundaries can fit with less hidden-layer capacity.',
    sinY: 'Cause: sin(y) adds a repeating vertical signal. Effect: periodic vertical structure becomes easier to learn.',
    cosX: 'Cause: cos(x) adds a shifted repeating horizontal signal. Effect: the model can align with wave peaks and valleys.',
    cosY: 'Cause: cos(y) adds a shifted repeating vertical signal. Effect: the model can fit alternating bands with fewer neurons.',
};

export const FeaturesPanel = memo(function FeaturesPanel() {
    const features = usePlaygroundStore((s) => s.features);
    const store = usePlaygroundStore;

    return (
        <div>
            <div className="chip-group">
                {ALL_FEATURES.map((f) => (
                    <Tooltip key={f.id} content={FEATURE_TOOLTIPS[f.id] ?? `Cause: toggling ${f.label} changes the inputs. Effect: the model sees a different representation of the same data.`}>
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
