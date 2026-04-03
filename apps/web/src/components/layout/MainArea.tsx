// ── MainArea Component ──
import { memo } from 'react';
import { TrainingControls } from '../controls/TrainingControls.tsx';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { DecisionBoundary } from '../visualization/DecisionBoundary.tsx';
import { LossChart } from '../visualization/LossChart.tsx';
import { ConfusionMatrix } from '../visualization/ConfusionMatrix.tsx';
import { CodeExportPanel } from '../controls/CodeExportPanel.tsx';
import { InspectionPanel } from '../controls/InspectionPanel.tsx';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { HEX_BLUE, HEX_ORANGE } from '@nn-playground/shared';
import { CollapsiblePanel } from '../common/CollapsiblePanel.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import { ErrorBoundary } from '../common/ErrorBoundary.tsx';

interface MainAreaProps {
    training: TrainingHook;
}

export const MainArea = memo(function MainArea({ training }: MainAreaProps) {
    const showTestData = usePlaygroundStore((s) => s.ui.showTestData);
    const discretize = usePlaygroundStore((s) => s.ui.discretizeOutput);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);

    return (
        <>
            <main id="main-content" className="center-area" role="main" tabIndex={-1}>
                <TrainingControls training={training} />
                <div className="network-graph-wrapper">
                    <NetworkGraph />
                </div>
            </main>
            <aside className="right-panel" aria-label="Output">
                <ErrorBoundary
                    title="Decision boundary unavailable"
                    description="The decision boundary visualization hit a rendering error."
                    actionLabel="Retry visualization"
                    className="panel panel--error"
                >
                    <DecisionBoundary
                        trainPoints={trainPoints}
                        testPoints={testPoints}
                        showTestData={showTestData}
                        discretize={discretize}
                    />
                </ErrorBoundary>
                <div className="legend">
                    <div className="legend__item">
                        <div className="legend__swatch" style={{ background: HEX_BLUE }} />
                        <span>Negative / Class 0</span>
                    </div>
                    <div className="legend__item">
                        <div className="legend__swatch" style={{ background: HEX_ORANGE }} />
                        <span>Positive / Class 1</span>
                    </div>
                </div>
                <ErrorBoundary
                    title="Loss chart unavailable"
                    description="The training history chart could not be rendered."
                    actionLabel="Retry chart"
                    className="panel panel--error"
                >
                    <LossChart />
                </ErrorBoundary>
                <ErrorBoundary
                    title="Confusion matrix unavailable"
                    description="The evaluation metrics panel hit an error."
                    actionLabel="Retry metrics"
                    className="panel panel--error"
                >
                    <ConfusionMatrix />
                </ErrorBoundary>
                <ErrorBoundary
                    title="Analysis tools unavailable"
                    description="One of the right-panel tools failed to render."
                    actionLabel="Reload tools"
                    className="panel panel--error"
                >
                    <>
                        <CollapsiblePanel title="Options">
                            <Tooltip content="Overlay the held-out test samples on the decision boundary">
                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={showTestData}
                                        onChange={(e) => usePlaygroundStore.getState().setShowTestData(e.target.checked)}
                                    />
                                    Show test data
                                </label>
                            </Tooltip>
                            <Tooltip content="Snap the decision boundary to class regions instead of smooth probabilities">
                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={discretize}
                                        onChange={(e) => usePlaygroundStore.getState().setDiscretize(e.target.checked)}
                                    />
                                    Discretize output
                                </label>
                            </Tooltip>
                        </CollapsiblePanel>
                        <CollapsiblePanel title="Inspection" defaultExpanded={false} className="inspection-panel">
                            <InspectionPanel />
                        </CollapsiblePanel>
                        <CollapsiblePanel title="Code Export" defaultExpanded={false} className="code-export-panel">
                            <CodeExportPanel />
                        </CollapsiblePanel>
                    </>
                </ErrorBoundary>
            </aside>
        </>
    );
});
