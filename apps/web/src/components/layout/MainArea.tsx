// ── MainArea ── canvas + right-panel content
// Named exports (CanvasContent, BoundaryContent, etc.) are the primary
// integration points consumed by App.tsx via RegionShell.
// The legacy MainArea default export is preserved for tests and fallback contexts.

import { lazy, memo, Suspense, useState } from 'react';
import { TrainingControls } from '../controls/TrainingControls.tsx';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { DecisionBoundary, getDecisionOverlayCopy } from '../visualization/DecisionBoundary.tsx';
import type { DecisionOverlayMode } from '../visualization/DecisionBoundary.tsx';
import { LossChart } from '../visualization/LossChart.tsx';
import { TrainingExplanationPanel } from '../visualization/TrainingExplanationPanel.tsx';
import { ConfusionMatrix } from '../visualization/ConfusionMatrix.tsx';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { Panel } from '../common/Panel.tsx';
import { ErrorBoundary } from '../common/ErrorBoundary.tsx';
import { LoadingState } from '../common/LoadingState.tsx';
import { DiagnosticCockpitStrip, EvidenceFrame, TopologyStateBadge } from './ExperimentStateContext.tsx';

interface MainAreaProps { training: TrainingHook }

const InspectionPanel = lazy(() =>
    import('../controls/InspectionPanel.tsx').then((m) => ({ default: m.InspectionPanel })),
);
const CodeExportPanel = lazy(() =>
    import('../controls/CodeExportPanel.tsx').then((m) => ({ default: m.CodeExportPanel })),
);
const RunHistoryPanel = lazy(() =>
    import('../controls/RunHistoryPanel.tsx').then((m) => ({ default: m.RunHistoryPanel })),
);

const DECISION_OVERLAY_MODES: readonly DecisionOverlayMode[] = [
    'none',
    'uncertainty',
    'misclassification',
    'split',
];

function getDecisionOverlayButtonLabel(mode: DecisionOverlayMode): string {
    switch (mode) {
        case 'uncertainty':
            return 'Uncertain';
        case 'misclassification':
            return 'Errors';
        case 'split':
            return 'Split';
        case 'none':
        default:
            return 'Output';
    }
}

function Fallback({ msg }: { msg: string }) {
    return <LoadingState isLoading inline message={msg} />;
}

export const TopologyStage = memo(function TopologyStage() {
    return (
        <div className="forge-topology-stage">
            <div className="network-graph-wrapper">
                <TopologyStateBadge />
                <NetworkGraph />
            </div>
            <DiagnosticCockpitStrip />
        </div>
    );
});

// ── Canvas content (network topology) ────────────────────────────────────
export const CanvasContent = memo(function CanvasContent() {
    return <TopologyStage />;
});

// ── Right-panel tab contents ──────────────────────────────────────────────
export const BoundaryContent = memo(function BoundaryContent() {
    const showTestData = usePlaygroundStore((s) => s.ui.showTestData);
    const discretize   = usePlaygroundStore((s) => s.ui.discretizeOutput);
    const trainPoints  = useTrainingStore((s) => s.trainPoints);
    const testPoints   = useTrainingStore((s) => s.testPoints);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');
    const overlayCopy = getDecisionOverlayCopy(overlayMode, showTestData, discretize);
    return (
        <ErrorBoundary title="Decision boundary unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <EvidenceFrame view="Boundary">
                <DecisionBoundary
                    trainPoints={trainPoints}
                    testPoints={testPoints}
                    showTestData={showTestData}
                    discretize={discretize}
                    overlayMode={overlayMode}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
                    <label className="checkbox-row">
                        <input type="checkbox" checked={showTestData}
                            onChange={(e) => usePlaygroundStore.getState().setShowTestData(e.target.checked)} />
                        Show test data
                    </label>
                    <label className="checkbox-row">
                        <input type="checkbox" checked={discretize}
                            onChange={(e) => usePlaygroundStore.getState().setDiscretize(e.target.checked)} />
                        Discretize output
                    </label>
                    <div className="decision-overlay-controls" aria-label="Decision overlay controls">
                        {DECISION_OVERLAY_MODES.map((mode) => (
                            <button
                                key={mode}
                                type="button"
                                aria-pressed={overlayMode === mode}
                                onClick={() => setOverlayMode(mode)}
                            >
                                {getDecisionOverlayButtonLabel(mode)}
                            </button>
                        ))}
                    </div>
                    <p className="decision-overlay-note" aria-live="polite">
                        {overlayCopy.description}
                    </p>
                </div>
            </EvidenceFrame>
        </ErrorBoundary>
    );
});

export const LossContent = memo(function LossContent() {
    return (
        <ErrorBoundary title="Loss chart unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <EvidenceFrame view="Loss">
                <LossChart />
                <TrainingExplanationPanel />
            </EvidenceFrame>
        </ErrorBoundary>
    );
});

export const ConfusionContent = memo(function ConfusionContent() {
    return (
        <ErrorBoundary title="Confusion matrix unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <EvidenceFrame view="Confusion">
                <ConfusionMatrix />
            </EvidenceFrame>
        </ErrorBoundary>
    );
});

export const InspectContent = memo(function InspectContent() {
    return (
        <EvidenceFrame view="Inspection">
            <Suspense fallback={<Fallback msg="Loading inspection…" />}>
                <InspectionPanel />
            </Suspense>
        </EvidenceFrame>
    );
});

export const CodeContent = memo(function CodeContent() {
    return (
        <EvidenceFrame view="Code">
            <Suspense fallback={<Fallback msg="Loading code export…" />}>
                <CodeExportPanel />
            </Suspense>
        </EvidenceFrame>
    );
});

interface HistoryContentProps {
    onRestore: () => void;
    onInitializeArena?: (modelA: ExperimentRunRecordV1, modelB: ExperimentRunRecordV1) => void | Promise<void>;
    onStepArena?: () => void | Promise<void>;
}

export const HistoryContent = memo(function HistoryContent({
    onRestore,
    onInitializeArena,
    onStepArena,
}: HistoryContentProps) {
    return (
        <EvidenceFrame view="History">
            <Suspense fallback={<Fallback msg="Loading run history…" />}>
                <RunHistoryPanel
                    onRestore={onRestore}
                    onInitializeArena={onInitializeArena}
                    onStepArena={onStepArena}
                />
            </Suspense>
        </EvidenceFrame>
    );
});

// ── Legacy MainArea (for direct-render tests and fallback contexts) ────────
export const MainArea = memo(function MainArea({ training }: MainAreaProps) {
    const showTestData = usePlaygroundStore((s) => s.ui.showTestData);
    const discretize   = usePlaygroundStore((s) => s.ui.discretizeOutput);
    const trainPoints  = useTrainingStore((s) => s.trainPoints);
    const testPoints   = useTrainingStore((s) => s.testPoints);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');
    const overlayCopy = getDecisionOverlayCopy(overlayMode, showTestData, discretize);

    return (
        <>
            <main id="main-content" className="center-area" role="main" tabIndex={-1}>
                <TrainingControls training={training} />
                <TopologyStage />
            </main>
            <aside className="right-panel" aria-label="Output">
                <ErrorBoundary title="Decision boundary unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <EvidenceFrame view="Boundary">
                        <DecisionBoundary
                            trainPoints={trainPoints}
                            testPoints={testPoints}
                            showTestData={showTestData}
                            discretize={discretize}
                            overlayMode={overlayMode}
                        />
                    </EvidenceFrame>
                </ErrorBoundary>
                <div className="decision-overlay-controls" aria-label="Decision overlay controls">
                    {DECISION_OVERLAY_MODES.map((mode) => (
                        <button
                            key={mode}
                            type="button"
                            aria-pressed={overlayMode === mode}
                            onClick={() => setOverlayMode(mode)}
                        >
                            {getDecisionOverlayButtonLabel(mode)}
                        </button>
                    ))}
                </div>
                <p className="decision-overlay-note" aria-live="polite">
                    {overlayCopy.description}
                </p>
                <ErrorBoundary title="Loss chart unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <EvidenceFrame view="Loss">
                        <LossChart />
                        <TrainingExplanationPanel />
                    </EvidenceFrame>
                </ErrorBoundary>
                <ErrorBoundary title="Confusion matrix unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <EvidenceFrame view="Confusion">
                        <ConfusionMatrix />
                    </EvidenceFrame>
                </ErrorBoundary>
                <Panel title="Inspection" phase="run">
                    <EvidenceFrame view="Inspection">
                        <Suspense fallback={<Fallback msg="Loading inspection…" />}>
                            <InspectionPanel />
                        </Suspense>
                    </EvidenceFrame>
                </Panel>
                <Panel title="Code Export" phase="both">
                    <EvidenceFrame view="Code">
                        <Suspense fallback={<Fallback msg="Loading code export…" />}>
                            <CodeExportPanel />
                        </Suspense>
                    </EvidenceFrame>
                </Panel>
                <Panel title="Run History" phase="both">
                    <EvidenceFrame view="History">
                        <Suspense fallback={<Fallback msg="Loading run history…" />}>
                            <RunHistoryPanel
                                onRestore={training.reset}
                                onInitializeArena={(modelA, modelB) => training.initializeArena(
                                    {
                                        label: modelA.title ?? modelA.id,
                                        network: modelA.config.network,
                                        training: modelA.config.training,
                                        data: modelA.config.data,
                                        features: modelA.config.features,
                                    },
                                    {
                                        label: modelB.title ?? modelB.id,
                                        network: modelB.config.network,
                                        training: modelB.config.training,
                                        data: modelB.config.data,
                                        features: modelB.config.features,
                                    },
                                )}
                                onStepArena={() => training.stepArena(1)}
                            />
                        </Suspense>
                    </EvidenceFrame>
                </Panel>
            </aside>
        </>
    );
});
