// ── Features Panel ──
import { memo } from 'react';
import { ALL_FEATURES } from '@nn-playground/engine';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { commitRecipeEdit } from '../../store/commitRecipeEdit.ts';
import { toggleFeature } from '../../store/recipeEdits.ts';
import { LoadingState } from '../common/LoadingState.tsx';
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
    const featureIds = usePlaygroundStore(
        (state) => state.access.status === 'ready'
            ? state.access.prepared.document.recipe.inputs.featureIds
            : null,
    );
    const isLoading = useTrainingStore((state) => state.featuresConfigLoading);
    const configError = useTrainingStore((state) => state.configError);
    const configErrorSource = useTrainingStore((state) => state.configErrorSource);

    const retryFeatureChange = () => useTrainingStore.getState().retryConfigSync();

    if (!featureIds) {
        return (
            <div>
                <LoadingState isLoading={isLoading} inline message="Updating features..." />
                <div className="config-feedback config-feedback--error" role="alert">
                    No compatible version-2 experiment is active.
                </div>
            </div>
        );
    }

    const selectedFeatureIds = new Set(featureIds);

    return (
        <div aria-busy={isLoading}>
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
                {ALL_FEATURES.map((feature) => {
                    const isSelected = selectedFeatureIds.has(feature.id);
                    const isFinalFeature = isSelected && featureIds.length === 1;
                    return (
                        <Tooltip
                            key={feature.id}
                            content={FEATURE_TOOLTIPS[feature.id]
                                ?? `Cause: toggling ${feature.label} changes the inputs. Effect: the model sees a different representation of the same data.`}
                        >
                            <button
                                type="button"
                                className={`feature-chip ${isSelected ? 'active' : ''}`}
                                disabled={isFinalFeature}
                                onClick={() => {
                                    if (isFinalFeature) return;
                                    void commitRecipeEdit(
                                        'features',
                                        (current) => toggleFeature(current, feature.id),
                                    );
                                }}
                                aria-pressed={isSelected}
                            >
                                {feature.label}
                            </button>
                        </Tooltip>
                    );
                })}
            </div>
        </div>
    );
});
