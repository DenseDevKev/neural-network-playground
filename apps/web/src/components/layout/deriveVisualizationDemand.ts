import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { EvidenceViewId, WorkspaceView } from '../../store/useLayoutStore.ts';

export function deriveVisualizationDemand(args: {
    view: WorkspaceView;
    activeEvidenceView: EvidenceViewId;
    historyDrawerOpen?: boolean;
    graphRenderer: 'canvas' | 'svg';
}): VisualizationDemand {
    const activeEvidenceVisible = args.view === 'run';
    const boundaryVisible = activeEvidenceVisible && args.activeEvidenceView === 'boundary';
    const confusionVisible = activeEvidenceVisible && args.activeEvidenceView === 'confusion';
    const inspectionVisible = activeEvidenceVisible && args.activeEvidenceView === 'inspection';
    const graphConsumesNeuronGrids = args.graphRenderer === 'canvas' || args.graphRenderer === 'svg';

    return {
        ...DEFAULT_DEMAND,
        needDecisionBoundary: boundaryVisible,
        needNeuronGrids: graphConsumesNeuronGrids,
        needLayerStats: inspectionVisible,
        needActivationHistograms: inspectionVisible,
        needConfusionMatrix: confusionVisible,
    };
}
