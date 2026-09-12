import { lazy, memo, Suspense } from 'react';
import type { SaveCurrentRunController } from '../../hooks/useSaveCurrentRun.ts';
import type { NetworkSelectionController } from '../visualization/useNetworkSelectionController.ts';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { LossChart } from '../visualization/LossChart.tsx';
import { ConfusionMatrix } from '../visualization/ConfusionMatrix.tsx';
import { ErrorBoundary } from '../common/ErrorBoundary.tsx';
import { LoadingState } from '../common/LoadingState.tsx';
import { DiagnosticCockpitStrip, EvidenceFrame, TopologyStateBadge } from './ExperimentStateContext.tsx';

const TrainingExplanationPanel = lazy(() => import('../EducationContent.ts').then((module) => ({ default: module.TrainingExplanationPanel })));
const InspectionPanel = lazy(() =>
    import('../controls/InspectionPanel.tsx').then((module) => ({ default: module.InspectionPanel })),
);
const CodeExportPanel = lazy(() =>
    import('../controls/CodeExportPanel.tsx').then((module) => ({ default: module.CodeExportPanel })),
);
const RunHistoryPanel = lazy(() =>
    import('../WorkspaceUtilities.ts').then((module) => ({ default: module.RunHistoryPanel })),
);
const ConfigPanel = lazy(() =>
    import('../WorkspaceUtilities.ts').then((module) => ({ default: module.ConfigPanel })),
);

function Fallback({ message }: { message: string }) {
    return <LoadingState isLoading inline message={message} />;
}

export const TopologyStage = memo(function TopologyStage({
    selectionController,
}: {
    selectionController?: NetworkSelectionController;
}) {
    return (
        <div className="forge-topology-stage">
            <div className="network-graph-wrapper">
                <TopologyStateBadge />
                <NetworkGraph selectionController={selectionController} />
            </div>
            <DiagnosticCockpitStrip />
        </div>
    );
});

export const TopologyContent = memo(function TopologyContent({
    selectionController,
}: {
    selectionController?: NetworkSelectionController;
}) {
    return <TopologyStage selectionController={selectionController} />;
});

export const LossContent = memo(function LossContent() {
    return (
        <ErrorBoundary title="Loss chart unavailable" description="Rendering error." actionLabel="Retry" className="panel panel--error">
            <EvidenceFrame view="Loss">
                <LossChart />
                <Suspense fallback={<Fallback message="Loading explanations…" />}>
                    <TrainingExplanationPanel />
                </Suspense>
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
            <Suspense fallback={<Fallback message="Loading inspection…" />}>
                <InspectionPanel />
            </Suspense>
        </EvidenceFrame>
    );
});

export const CodeContent = memo(function CodeContent() {
    return (
        <EvidenceFrame view="Code">
            <Suspense fallback={<Fallback message="Loading code export…" />}>
                <CodeExportPanel />
            </Suspense>
        </EvidenceFrame>
    );
});

export const HistoryContent = memo(function HistoryContent({
    saveController,
}: {
    saveController?: SaveCurrentRunController;
}) {
    return (
        <EvidenceFrame view="History">
            <Suspense fallback={<Fallback message="Loading run history…" />}>
                <RunHistoryPanel saveController={saveController} />
            </Suspense>
        </EvidenceFrame>
    );
});

export const ConfigurationContent = memo(function ConfigurationContent({
    onReset,
}: {
    onReset: () => void;
}) {
    return (
        <Suspense fallback={<Fallback message="Loading configuration…" />}>
            <ConfigPanel onReset={onReset} />
        </Suspense>
    );
});
