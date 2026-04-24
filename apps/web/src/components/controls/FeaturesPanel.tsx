// ── Features Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { ALL_FEATURES } from '@nn-playground/engine';
import type { FeatureFlags } from '@nn-playground/engine';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

export const FeaturesPanel = memo(function FeaturesPanel() {
    const features = usePlaygroundStore((s) => s.features);
    const isLoading = useTrainingStore((s) => s.featuresConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const activeFeatureCount = Object.values(features).filter(Boolean).length;
    const store = usePlaygroundStore;

    const beginFeatureChange = () => useTrainingStore.getState().beginConfigChange('features');
    const retryFeatureChange = () => useTrainingStore.getState().retryConfigSync();

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Updating features..." />
            {configError && configErrorSource === 'features' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryFeatureChange}>
                        Retry
                    </button>
                </div>
            )}

            <div className="chip-group">
                {ALL_FEATURES.map((f) => (
                    <Tooltip key={f.id} content={`Toggle the ${f.label} input feature`}>
                        <button
                            type="button"
                            className={`feature-chip ${features[f.id as keyof FeatureFlags] ? 'active' : ''}`}
                            onClick={() => {
                                const featureId = f.id as keyof FeatureFlags;
                                if (features[featureId] && activeFeatureCount === 1) {
                                    return;
                                }

                                beginFeatureChange();
                                store.getState().toggleFeature(featureId);
                            }}
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
