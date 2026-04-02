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
            <main className="center-area" role="main">
                <TrainingControls training={training} />
                <div className="network-graph-wrapper">
                    <NetworkGraph />
                </div>
            </main>
            <aside className="right-panel" aria-label="Output">
                <DecisionBoundary
                    trainPoints={trainPoints}
                    testPoints={testPoints}
                    showTestData={showTestData}
                    discretize={discretize}
                />
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
                <LossChart />
                <ConfusionMatrix />
                <div className="panel">
                    <div className="panel__title">Options</div>
                    <label className="checkbox-row">
                        <input
                            type="checkbox"
                            checked={showTestData}
                            onChange={(e) => usePlaygroundStore.getState().setShowTestData(e.target.checked)}
                        />
                        Show test data
                    </label>
                    <label className="checkbox-row">
                        <input
                            type="checkbox"
                            checked={discretize}
                            onChange={(e) => usePlaygroundStore.getState().setDiscretize(e.target.checked)}
                        />
                        Discretize output
                    </label>
                </div>
                <InspectionPanel />
                <CodeExportPanel />
            </aside>
        </>
    );
});

