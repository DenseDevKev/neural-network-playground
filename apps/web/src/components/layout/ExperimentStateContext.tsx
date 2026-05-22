import { memo, useMemo } from 'react';
import type { AppConfig } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore, type ConfigChangeSource } from '../../store/useTrainingStore.ts';
import { getRecipeDrift } from '../../store/recipeIdentity.ts';

function useCurrentRecipeConfig(): AppConfig {
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const ui = usePlaygroundStore((s) => s.ui);

    return useMemo(() => ({ data, features, network, training, ui }), [data, features, network, training, ui]);
}

function sourceText(source: ConfigChangeSource): string {
    return source ? `${source} config` : 'config';
}

const EVIDENCE_EXPLANATIONS: Record<string, string> = {
    Boundary: 'Shows decision regions and sample outcomes.',
    Loss: 'Shows learning curves over time.',
    Confusion: 'Shows class-level prediction errors.',
    Inspection: 'Shows layer activations, gradients, and probes.',
    Code: 'Shows the current model recipe as readable code.',
    History: 'Shows saved run records and comparison paths.',
};

function useExperimentContext() {
    const currentConfig = useCurrentRecipeConfig();
    const trainedRecipeConfig = useTrainingStore((s) => s.trainedRecipeConfig);
    const status = useTrainingStore((s) => s.status);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const testMetricsStale = useTrainingStore((s) => s.testMetricsStale);
    const workerError = useTrainingStore((s) => s.workerError);
    const configError = useTrainingStore((s) => s.configError);
    const drift = useMemo(
        () => getRecipeDrift(trainedRecipeConfig, currentConfig),
        [trainedRecipeConfig, currentConfig],
    );

    return {
        drift,
        status,
        snapshot,
        pendingConfigSource,
        testMetricsStale,
        workerError,
        configError,
    };
}

export const TopologyStateBadge = memo(function TopologyStateBadge() {
    const { drift, status, snapshot, pendingConfigSource } = useExperimentContext();
    const label = status === 'running' && snapshot
        ? 'Live Run'
        : drift.hasDrift
            ? 'Drifted Recipe'
            : snapshot
                ? 'Trained Snapshot'
                : 'Draft Blueprint';

    return (
        <div
            className={`forge-state-badge forge-state-badge--${label.toLowerCase().replace(/\s+/g, '-')}`}
            aria-label="Topology state"
        >
            <span>{label}</span>
            {pendingConfigSource && <small>{`Updating ${sourceText(pendingConfigSource)}`}</small>}
        </div>
    );
});

export const EvidenceContextLine = memo(function EvidenceContextLine({ view }: { view: string }) {
    const { drift, status, snapshot, pendingConfigSource, testMetricsStale, workerError, configError } = useExperimentContext();

    let copy: string;
    let stateLabel = 'Fresh';
    if (workerError || configError) {
        stateLabel = 'Unavailable';
        copy = `${view} evidence is unavailable until the worker reconnects.`;
    } else if (!snapshot) {
        stateLabel = 'Empty';
        copy = `Train to see ${view} evidence.`;
    } else if (pendingConfigSource) {
        stateLabel = 'Updating';
        copy = `${view} evidence reflects trained snapshot step ${snapshot.step.toLocaleString()}; updating ${sourceText(pendingConfigSource)}.`;
    } else if (drift.hasDrift) {
        stateLabel = 'Drift';
        copy = `${view} evidence belongs to trained snapshot step ${snapshot.step.toLocaleString()}; current recipe has drift.`;
    } else if (status === 'running') {
        stateLabel = 'Live';
        copy = `${view} evidence follows live run step ${snapshot.step.toLocaleString()}.`;
    } else if (testMetricsStale) {
        stateLabel = 'Stale';
        copy = `${view} evidence uses cached test metrics from snapshot step ${snapshot.step.toLocaleString()}.`;
    } else {
        copy = `${view} evidence reflects trained snapshot step ${snapshot.step.toLocaleString()}.`;
    }
    const explanation = EVIDENCE_EXPLANATIONS[view] ?? 'Explains the selected model state.';

    return (
        <p className="forge-evidence-context" aria-live="polite">
            <span className="forge-evidence-context__state">{stateLabel}</span>
            <span>{copy}</span>
            <small>{explanation}</small>
        </p>
    );
});
