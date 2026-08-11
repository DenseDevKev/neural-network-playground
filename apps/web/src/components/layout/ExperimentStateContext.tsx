import { memo, type ReactNode, useMemo } from 'react';
import { resolveVisibleEvidenceView } from '../../productShell/visibleShell.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore, type ConfigChangeSource } from '../../store/useTrainingStore.ts';
import { useLayoutStore, type EvidenceViewId } from '../../store/useLayoutStore.ts';
import { getRecipeDrift } from '../../store/recipeIdentity.ts';
import {
    selectScientificEvidence,
    type ScientificEvidence,
} from '../../store/evidenceSelectors.ts';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';

type EvidenceViewName = 'Boundary' | 'Loss' | 'Confusion' | 'Inspection' | 'Code' | 'History';

interface EvidenceMeta {
    title: EvidenceViewName;
    interactionModel: string;
    explanation: string;
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
    const currentRecipe = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared.document.recipe : null
    ));
    const trainedRecipe = useTrainingStore((s) => s.trainedRecipe);
    const trainedRecipeFingerprint = useTrainingStore((s) => s.trainedRecipeFingerprint);
    const currentRecipeFingerprint = usePlaygroundStore(
        (s) => s.access.status === 'ready'
            ? s.access.prepared.identities.recipeFingerprint
            : null,
    );
    const status = useTrainingStore((s) => s.status);
    const latestLiveSignal = useTrainingStore((s) => s.latestLiveSignal);
    const latestEvaluation = useTrainingStore((s) => s.latestEvaluation);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const configError = useTrainingStore((s) => s.configError);
    const evidence = useMemo(
        () => selectScientificEvidence({ latestLiveSignal, latestEvaluation }),
        [latestEvaluation, latestLiveSignal],
    );
    const drift = useMemo(
        () => getRecipeDrift(
            trainedRecipe,
            currentRecipe,
            3,
            trainedRecipeFingerprint === null ? undefined : {
                trainedRecipeFingerprint,
                currentRecipeFingerprint,
            },
        ),
        [trainedRecipe, currentRecipe, trainedRecipeFingerprint, currentRecipeFingerprint],
    );

    return {
        drift,
        status,
        evidence,
        pendingConfigSource,
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
    const { drift, status, evidence, pendingConfigSource } = useExperimentContext();
    const label = status === 'running' && evidence.currentModel
        ? 'Live Run'
        : drift.hasDrift
            ? 'Drifted Recipe'
            : evidence.currentModel
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

function exactEvidenceCopy(view: string, evidence: ScientificEvidence): string {
    const evaluation = evidence.fullEvaluation;
    const currentModel = evidence.currentModel;
    if (!evaluation || !currentModel) {
        return `${view} evidence has no paired full evaluation.`;
    }
    if (view === 'Confusion') {
        return `Confusion uses full evaluation ${evaluation.evaluationId} at step ${evaluation.step.toLocaleString()} across all ${evaluation.testSampleCount.toLocaleString()} test samples; the current model is at step ${currentModel.step.toLocaleString()}.`;
    }
    if (evidence.batchTrend) {
        return `${view} has batch trend through step ${evidence.batchTrend.step.toLocaleString()} and full evaluation ${evaluation.evaluationId} at step ${evaluation.step.toLocaleString()} using all ${evaluation.trainSampleCount.toLocaleString()} train and ${evaluation.testSampleCount.toLocaleString()} test samples.`;
    }
    return `${view} uses full evaluation ${evaluation.evaluationId} at step ${evaluation.step.toLocaleString()} across all ${evaluation.trainSampleCount.toLocaleString()} train and ${evaluation.testSampleCount.toLocaleString()} test samples.`;
}

export const EvidenceContextLine = memo(function EvidenceContextLine({ view }: { view: string }) {
    const guidanceLevel = useAudienceGuidanceLevel();
    const { drift, status, evidence, pendingConfigSource, workerError, configError } = useExperimentContext();
    const currentModel = evidence.currentModel;

    let copy: string;
    let stateLabel = 'Fresh';
    if (workerError || configError) {
        stateLabel = 'Unavailable';
        copy = `${view} evidence is unavailable until the worker reconnects.`;
    } else if (!currentModel) {
        stateLabel = 'Empty';
        copy = `Train to see ${view} evidence.`;
    } else if (pendingConfigSource) {
        stateLabel = 'Updating';
        copy = `${view} evidence reflects trained model step ${currentModel.step.toLocaleString()}; updating ${sourceText(pendingConfigSource)}.`;
    } else if (drift.hasDrift) {
        stateLabel = 'Drift';
        copy = `${view} evidence belongs to trained model step ${currentModel.step.toLocaleString()}; current recipe has drift.`;
    } else if ((evidence.evaluationAgeSteps ?? 0) > 0) {
        stateLabel = 'Evaluation age';
        copy = exactEvidenceCopy(view, evidence);
    } else if (status === 'running') {
        stateLabel = 'Live';
        copy = evidence.fullEvaluation
            ? exactEvidenceCopy(view, evidence)
            : `${view} batch trend follows the live model through step ${currentModel.step.toLocaleString()}; no paired full evaluation has been published yet.`;
    } else {
        copy = evidence.fullEvaluation
            ? exactEvidenceCopy(view, evidence)
            : `${view} evidence reflects trained model step ${currentModel.step.toLocaleString()}; no paired full evaluation has been published yet.`;
    }
    const explanation = getEvidenceMeta(view).explanation;

    return (
        <p className="forge-evidence-context">
            <span className="forge-evidence-context__state">{stateLabel}</span>
            <span>{copy}</span>
            <small>{explanation}</small>
            {view === 'Inspection' ? (
                <ConceptHelp
                    conceptId="activation"
                    guidanceLevel={guidanceLevel}
                    className="concept-help--end"
                />
            ) : null}
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
    const { drift, status, evidence, pendingConfigSource, workerError, configError } = useExperimentContext();
    const activeEvidenceView = useLayoutStore((s) => s.activeEvidenceView);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const advancedToolsOpen = useLayoutStore((s) => s.advancedToolsOpen);
    const visibleEvidenceView = resolveVisibleEvidenceView(
        audienceMode,
        advancedToolsOpen,
        activeEvidenceView,
    );
    const activeEvidence = EVIDENCE_VIEW_LABELS[visibleEvidenceView];

    let stateLabel = 'Draft';
    let copy = `Topology is a draft blueprint; run or step to produce ${activeEvidence} evidence.`;
    if (workerError || configError) {
        stateLabel = 'Unavailable';
        copy = 'Diagnostics are unavailable until the worker reconnects.';
    } else if (pendingConfigSource && evidence.currentModel) {
        stateLabel = 'Updating';
        copy = `Topology is syncing ${sourceText(pendingConfigSource)} while ${activeEvidence} evidence still reflects model step ${evidence.currentModel.step.toLocaleString()}.`;
    } else if (drift.hasDrift && evidence.currentModel) {
        stateLabel = 'Mixed';
        copy = `Topology shows the draft recipe while ${activeEvidence} evidence belongs to trained model step ${evidence.currentModel.step.toLocaleString()}.`;
    } else if ((evidence.evaluationAgeSteps ?? 0) > 0) {
        stateLabel = 'Evaluation age';
        copy = exactEvidenceCopy(activeEvidence, evidence);
    } else if (status === 'running' && evidence.currentModel) {
        stateLabel = 'Live';
        copy = evidence.fullEvaluation
            ? exactEvidenceCopy(activeEvidence, evidence)
            : `Topology and ${activeEvidence} follow the batch trend through live model step ${evidence.currentModel.step.toLocaleString()}.`;
    } else if (evidence.currentModel) {
        stateLabel = 'Snapshot';
        copy = evidence.fullEvaluation
            ? exactEvidenceCopy(activeEvidence, evidence)
            : `Topology and ${activeEvidence} reflect trained model step ${evidence.currentModel.step.toLocaleString()}.`;
    }

    return (
        <div className="forge-cockpit-strip" role="group" aria-label="Diagnostic cockpit state">
            <div className="forge-cockpit-strip__copy">
                <span>{stateLabel}</span>
                <strong>{copy}</strong>
            </div>
            {evidence.currentModel && (
                <div className="forge-cockpit-strip__metrics" aria-label="Cockpit metrics">
                    {evidence.batchTrend && (
                        <span>{`Batch trend (EMA) ${formatMetric(evidence.batchTrend.dataLoss)}`}</span>
                    )}
                    {evidence.fullEvaluation && (
                        <>
                            <span>{`Train data loss (full split) ${formatMetric(evidence.fullEvaluation.trainDataLoss)}`}</span>
                            <span>{`Test data loss (full split) ${formatMetric(evidence.fullEvaluation.testDataLoss)}`}</span>
                            <span>{`Training objective ${formatMetric(evidence.fullEvaluation.trainingObjective)}`}</span>
                            <span>{`gap ${formatSignedMetric(evidence.generalizationGap ?? undefined)}`}</span>
                        </>
                    )}
                </div>
            )}
        </div>
    );
});
