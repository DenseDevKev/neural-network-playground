import { memo, useMemo } from 'react';
import type { AppConfig } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getRecipeDrift, summarizeRecipe } from '../../store/recipeIdentity.ts';

function useCurrentRecipeConfig(): AppConfig {
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const ui = usePlaygroundStore((s) => s.ui);

    return useMemo(() => ({ data, features, network, training, ui }), [data, features, network, training, ui]);
}

export const RecipeSummaryCard = memo(function RecipeSummaryCard() {
    const currentConfig = useCurrentRecipeConfig();
    const trainedRecipeConfig = useTrainingStore((s) => s.trainedRecipeConfig);
    const trainedRecipeFingerprint = useTrainingStore((s) => s.trainedRecipeFingerprint);
    const currentRecipeFingerprint = usePlaygroundStore(
        (s) => s.prepared?.identities.recipeFingerprint ?? null,
    );
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const testMetricsStale = useTrainingStore((s) => s.testMetricsStale);
    const summary = useMemo(() => summarizeRecipe(currentConfig), [currentConfig]);
    const drift = useMemo(
        () => getRecipeDrift(
            trainedRecipeConfig,
            currentConfig,
            3,
            trainedRecipeFingerprint === null ? undefined : {
                trainedRecipeFingerprint,
                currentRecipeFingerprint,
            },
        ),
        [trainedRecipeConfig, currentConfig, trainedRecipeFingerprint, currentRecipeFingerprint],
    );
    const showsStaleEvidence = !drift.hasDrift && testMetricsStale;
    const pillLabel = pendingConfigSource
        ? 'Updating'
        : drift.hasDrift
            ? 'Drift'
            : showsStaleEvidence
                ? 'Stale'
                : 'Ready';
    const noteHeadline = showsStaleEvidence
        ? 'Current recipe accepted; evidence metrics are stale.'
        : drift.headline;
    const noteResolution = showsStaleEvidence
        ? 'Run, step, or resume to refresh evidence for this recipe.'
        : drift.resolution;
    const noteClassName = [
        'forge-drift-note',
        drift.hasDrift ? 'forge-drift-note--active' : '',
        showsStaleEvidence ? 'forge-drift-note--stale' : '',
    ].filter(Boolean).join(' ');
    const pillClassName = [
        'forge-context-pill',
        drift.hasDrift ? 'forge-context-pill--drift' : '',
        showsStaleEvidence ? 'forge-context-pill--stale' : '',
    ].filter(Boolean).join(' ');

    return (
        <section className="forge-context-card forge-recipe-card" role="region" aria-label="Recipe summary">
            <div className="forge-context-card__head">
                <span className="forge-context-card__eyebrow">Current recipe</span>
                <span className={pillClassName}>{pillLabel}</span>
            </div>

            <dl className="forge-recipe-card__summary">
                <div>
                    <dt>Data</dt>
                    <dd>{summary.dataset}</dd>
                </div>
                <div>
                    <dt>Network</dt>
                    <dd>{summary.architecture}</dd>
                </div>
                <div>
                    <dt>Training</dt>
                    <dd>{summary.training}</dd>
                </div>
                <div>
                    <dt>Loss</dt>
                    <dd>{summary.lossAndBatch}</dd>
                </div>
                <div>
                    <dt>Features</dt>
                    <dd>{summary.features}</dd>
                </div>
                <div>
                    <dt>Count</dt>
                    <dd>{summary.featureCount}</dd>
                </div>
            </dl>

            <div className={noteClassName}>
                <p>{noteHeadline}</p>
                {drift.groupLabels.length > 0 && (
                    <span className="forge-drift-note__groups">{drift.groupLabels.join(', ')}</span>
                )}
                {drift.visibleItems.length > 0 && (
                    <ul className="forge-drift-list" aria-label="Changed recipe fields">
                        {drift.visibleItems.map((item) => (
                            <li key={`${item.group}-${item.field}`}>
                                <span>{item.label}</span>
                                <strong>{`${item.snapshotValue} -> ${item.currentValue}`}</strong>
                            </li>
                        ))}
                        {drift.remainingCount > 0 && (
                            <li className="forge-drift-list__more">
                                <span>{`+${drift.remainingCount} more`}</span>
                            </li>
                        )}
                    </ul>
                )}
                <small>{noteResolution}</small>
            </div>
        </section>
    );
});
