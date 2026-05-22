import { memo, type ReactNode, useMemo } from 'react';
import type { AppConfig } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore, type ConfigChangeSource } from '../../store/useTrainingStore.ts';
import { useLayoutStore, type EvidenceViewId } from '../../store/useLayoutStore.ts';
import { getRecipeDrift } from '../../store/recipeIdentity.ts';

type EvidenceViewName = 'Boundary' | 'Loss' | 'Confusion' | 'Inspection' | 'Code' | 'History';

interface EvidenceMeta {
    title: EvidenceViewName;
    interactionModel: string;
    explanation: string;
}

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

const EVIDENCE_META: Record<EvidenceViewName, EvidenceMeta> = {
    Boundary: {
        title: 'Boundary',
        interactionModel: 'Plot-based',
        explanation: 'Shows decision regions and sample outcomes.',
    },
    Loss: {
        title: 'Loss',
        interactionModel: 'Time-series',
        explanation: 'Shows learning curves over time.',
    },
    Confusion: {
        title: 'Confusion',
        interactionModel: 'Matrix-based',
        explanation: 'Shows class-level prediction errors.',
    },
    Inspection: {
        title: 'Inspection',
        interactionModel: 'Hierarchical',
        explanation: 'Shows layer activations, gradients, and probes.',
    },
    Code: {
        title: 'Code',
        interactionModel: 'Textual',
        explanation: 'Shows the current model recipe as readable code.',
    },
    History: {
        title: 'History',
        interactionModel: 'Record-based',
        explanation: 'Shows saved run records and comparison paths.',
    },
};

const EVIDENCE_VIEW_LABELS: Record<EvidenceViewId, EvidenceViewName> = {
    boundary: 'Boundary',
    loss: 'Loss',
    confusion: 'Confusion',
    inspection: 'Inspection',
    code: 'Code',
    history: 'History',
};

function getEvidenceMeta(view: string): EvidenceMeta {
    return EVIDENCE_META[view as EvidenceViewName] ?? {
        title: view as EvidenceViewName,
        interactionModel: 'Evidence',
        explanation: 'Explains the selected model state.',
    };
}

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

function formatMetric(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? 'n/a' : value.toFixed(4);
}

function formatSignedMetric(value: number | undefined): string {
    if (value === undefined || !Number.isFinite(value)) return 'n/a';
    return `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
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
    const explanation = getEvidenceMeta(view).explanation;

    return (
        <p className="forge-evidence-context" aria-live="polite">
            <span className="forge-evidence-context__state">{stateLabel}</span>
            <span>{copy}</span>
            <small>{explanation}</small>
        </p>
    );
});

export const EvidenceFrame = memo(function EvidenceFrame({
    view,
    children,
}: {
    view: EvidenceViewName;
    children: ReactNode;
}) {
    const meta = getEvidenceMeta(view);
    return (
        <section className="forge-evidence-frame" aria-label={`${meta.title} evidence view`}>
            <header className="forge-evidence-frame__head">
                <div>
                    <span className="forge-evidence-frame__eyebrow">Evidence</span>
                    <strong>{meta.title}</strong>
                </div>
                <span className="forge-evidence-frame__model">{meta.interactionModel}</span>
            </header>
            <EvidenceContextLine view={meta.title} />
            <div className="forge-evidence-frame__body">
                {children}
            </div>
        </section>
    );
});

export const DiagnosticCockpitStrip = memo(function DiagnosticCockpitStrip() {
    const { drift, status, snapshot, pendingConfigSource, testMetricsStale, workerError, configError } = useExperimentContext();
    const activeEvidenceView = useLayoutStore((s) => s.activeEvidenceView);
    const visibleEvidenceView = activeEvidenceView === 'history' ? 'boundary' : activeEvidenceView;
    const activeEvidence = EVIDENCE_VIEW_LABELS[visibleEvidenceView];

    let stateLabel = 'Draft';
    let copy = `Topology is a draft blueprint; run or step to produce ${activeEvidence} evidence.`;
    if (workerError || configError) {
        stateLabel = 'Unavailable';
        copy = 'Diagnostics are unavailable until the worker reconnects.';
    } else if (pendingConfigSource && snapshot) {
        stateLabel = 'Updating';
        copy = `Topology is syncing ${sourceText(pendingConfigSource)} while ${activeEvidence} evidence still reflects snapshot step ${snapshot.step.toLocaleString()}.`;
    } else if (status === 'running' && snapshot) {
        stateLabel = 'Live';
        copy = `Topology and ${activeEvidence} are reading live run step ${snapshot.step.toLocaleString()}.`;
    } else if (drift.hasDrift && snapshot) {
        stateLabel = 'Mixed';
        copy = `Topology shows the draft recipe while ${activeEvidence} evidence belongs to trained snapshot step ${snapshot.step.toLocaleString()}.`;
    } else if (testMetricsStale && snapshot) {
        stateLabel = 'Stale';
        copy = `${activeEvidence} evidence uses cached metrics from snapshot step ${snapshot.step.toLocaleString()}.`;
    } else if (snapshot) {
        stateLabel = 'Snapshot';
        copy = `Topology and ${activeEvidence} reflect trained snapshot step ${snapshot.step.toLocaleString()}.`;
    }

    const gap = snapshot ? snapshot.testLoss - snapshot.trainLoss : undefined;

    return (
        <div className="forge-cockpit-strip" role="status" aria-label="Diagnostic cockpit state">
            <div className="forge-cockpit-strip__copy">
                <span>{stateLabel}</span>
                <strong>{copy}</strong>
            </div>
            {snapshot && (
                <div className="forge-cockpit-strip__metrics" aria-label="Cockpit metrics">
                    <span>{`train ${formatMetric(snapshot.trainLoss)}`}</span>
                    <span>{`test ${formatMetric(snapshot.testLoss)}`}</span>
                    <span>{`gap ${formatSignedMetric(gap)}`}</span>
                </div>
            )}
        </div>
    );
});
