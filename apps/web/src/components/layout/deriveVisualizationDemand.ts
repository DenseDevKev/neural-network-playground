import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { AudienceMode } from '../../productShell/audienceProfiles.ts';
import { resolveVisibleEvidenceView } from '../../productShell/visibleShell.ts';
import type { EvidenceViewId, WorkspaceView } from '../../productShell/shellTypes.ts';

export function deriveVisualizationDemand(args: {
    view: WorkspaceView;
    activeEvidenceView: EvidenceViewId;
    audienceMode: AudienceMode;
    advancedToolsOpen: boolean;
    graphRenderer: 'canvas' | 'svg';
}): VisualizationDemand {
    const activeEvidenceVisible = args.view === 'run';
    const visibleEvidenceView = resolveVisibleEvidenceView(
        args.audienceMode,
        args.advancedToolsOpen,
        args.activeEvidenceView,
    );
    const boundaryVisible = activeEvidenceVisible && visibleEvidenceView === 'boundary';
    const confusionVisible = activeEvidenceVisible && visibleEvidenceView === 'confusion';
    const inspectionVisible = activeEvidenceVisible && visibleEvidenceView === 'inspection';
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
