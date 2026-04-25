// ── MainArea ── canvas + right-panel content
// Named exports (CanvasContent, BoundaryContent, etc.) are the primary
// integration points consumed by App.tsx via RegionShell.
// The legacy MainArea default export is preserved for tests and fallback contexts.

import { lazy, memo, Suspense, useState } from 'react';
import { TrainingControls } from '../controls/TrainingControls.tsx';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { DecisionBoundary } from '../visualization/DecisionBoundary.tsx';
import type { DecisionOverlayMode } from '../visualization/DecisionBoundary.tsx';
import { LossChart } from '../visualization/LossChart.tsx';
import { ConfusionMatrix } from '../visualization/ConfusionMatrix.tsx';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { Panel } from '../common/Panel.tsx';
import { ErrorBoundary } from '../common/ErrorBoundary.tsx';
import { LoadingState } from '../common/LoadingState.tsx';

interface MainAreaProps { training: TrainingHook }

const InspectionPanel = lazy(() =>
    import('../controls/InspectionPanel.tsx').then((m) => ({ default: m.InspectionPanel })),
);
const CodeExportPanel = lazy(() =>
    import('../controls/CodeExportPanel.tsx').then((m) => ({ default: m.CodeExportPanel })),
);

function Fallback({ msg }: { msg: string }) {
    return <LoadingState isLoading inline message={msg} />;
}

// ── Canvas content (network topology) ────────────────────────────────────
export const CanvasContent = memo(function CanvasContent() {
    return (
        <div
            className="network-graph-wrapper"
            style={{ flex: 1, borderRadius: 'var(--radius-md)', overflow: 'hidden' }}
        >
            <NetworkGraph />
        </div>
    );
});

// ── Right-panel tab contents ──────────────────────────────────────────────
export const BoundaryContent = memo(function BoundaryContent() {
    const showTestData = usePlaygroundStore((s) => s.ui.showTestData);
    const discretize   = usePlaygroundStore((s) => s.ui.discretizeOutput);
    const trainPoints  = useTrainingStore((s) => s.trainPoints);
    const testPoints   = useTrainingStore((s) => s.testPoints);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');
    return (
        <ErrorBoundary title="Decision boundary unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <>
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
                        {(['none', 'uncertainty', 'misclassification'] as const).map((mode) => (
                            <button
                                key={mode}
                                type="button"
                                aria-pressed={overlayMode === mode}
                                onClick={() => setOverlayMode(mode)}
                            >
                                {mode === 'none' ? 'Output' : mode === 'uncertainty' ? 'Uncertain' : 'Errors'}
                            </button>
                        ))}
                    </div>
                </div>
            </>
        </ErrorBoundary>
    );
});

export const LossContent = memo(function LossContent() {
    return (
        <ErrorBoundary title="Loss chart unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <LossChart />
        </ErrorBoundary>
    );
});

export const ConfusionContent = memo(function ConfusionContent() {
    return (
        <ErrorBoundary title="Confusion matrix unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <ConfusionMatrix />
        </ErrorBoundary>
    );
});

export const InspectContent = memo(function InspectContent() {
    return (
        <Suspense fallback={<Fallback msg="Loading inspection…" />}>
            <InspectionPanel />
        </Suspense>
    );
});

export const CodeContent = memo(function CodeContent() {
    return (
        <Suspense fallback={<Fallback msg="Loading code export…" />}>
            <CodeExportPanel />
        </Suspense>
    );
});

// ── Legacy MainArea (for direct-render tests and fallback contexts) ────────
export const MainArea = memo(function MainArea({ training }: MainAreaProps) {
    const showTestData = usePlaygroundStore((s) => s.ui.showTestData);
    const discretize   = usePlaygroundStore((s) => s.ui.discretizeOutput);
    const trainPoints  = useTrainingStore((s) => s.trainPoints);
    const testPoints   = useTrainingStore((s) => s.testPoints);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');

    return (
        <>
            <main id="main-content" className="center-area" role="main" tabIndex={-1}>
                <TrainingControls training={training} />
                <div className="network-graph-wrapper">
                    <NetworkGraph />
                </div>
            </main>
            <aside className="right-panel" aria-label="Output">
                <ErrorBoundary title="Decision boundary unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <DecisionBoundary
                        trainPoints={trainPoints}
                        testPoints={testPoints}
                        showTestData={showTestData}
                        discretize={discretize}
                        overlayMode={overlayMode}
                    />
                </ErrorBoundary>
                <div className="decision-overlay-controls" aria-label="Decision overlay controls">
                    {(['none', 'uncertainty', 'misclassification'] as const).map((mode) => (
                        <button
                            key={mode}
                            type="button"
                            aria-pressed={overlayMode === mode}
                            onClick={() => setOverlayMode(mode)}
                        >
                            {mode === 'none' ? 'Output' : mode === 'uncertainty' ? 'Uncertain' : 'Errors'}
                        </button>
                    ))}
                </div>
                <ErrorBoundary title="Loss chart unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <LossChart />
                </ErrorBoundary>
                <ErrorBoundary title="Confusion matrix unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
                    <ConfusionMatrix />
                </ErrorBoundary>
                <Panel title="Inspection" phase="run">
                    <Suspense fallback={<Fallback msg="Loading inspection…" />}>
                        <InspectionPanel />
                    </Suspense>
                </Panel>
                <Panel title="Code Export" phase="both">
                    <Suspense fallback={<Fallback msg="Loading code export…" />}>
                        <CodeExportPanel />
                    </Suspense>
                </Panel>
            </aside>
        </>
    );
});
