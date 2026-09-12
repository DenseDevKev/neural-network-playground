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
    boundaryRailMounted: boolean;
}): VisualizationDemand {
    const visibleEvidenceView = resolveVisibleEvidenceView(
        args.audienceMode,
        args.advancedToolsOpen,
        args.activeEvidenceView,
    );
    const confusionVisible = visibleEvidenceView === 'confusion';
    const inspectionVisible = visibleEvidenceView === 'inspection';
    const graphConsumesNeuronGrids = args.graphRenderer === 'canvas' || args.graphRenderer === 'svg';

    return {
        ...DEFAULT_DEMAND,
        needDecisionBoundary: args.boundaryRailMounted,
        needNeuronGrids: graphConsumesNeuronGrids,
        needLayerStats: inspectionVisible,
        needActivationHistograms: inspectionVisible,
        needConfusionMatrix: confusionVisible,
    };
}
