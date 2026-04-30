import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { LayoutVariant, PhaseMode } from '../../store/useLayoutStore.ts';

export function deriveVisualizationDemand(args: {
    layout: LayoutVariant;
    phase: PhaseMode;
    activeTabRight: string;
    graphRenderer: 'canvas' | 'svg';
}): VisualizationDemand {
    const rightTab = args.activeTabRight;
    const allRightPanelsVisible = args.layout === 'focus' || args.layout === 'grid';
    const selectedRightPanelVisible = args.layout === 'dock';
    const splitRunVisible = args.layout === 'split' && args.phase === 'run';

    const boundaryVisible =
        allRightPanelsVisible ||
        splitRunVisible ||
        (selectedRightPanelVisible && rightTab === 'boundary');
    const confusionVisible =
        allRightPanelsVisible ||
        splitRunVisible ||
        (selectedRightPanelVisible && rightTab === 'confusion');
    const inspectionVisible =
        allRightPanelsVisible ||
        splitRunVisible ||
        (selectedRightPanelVisible && rightTab === 'inspection');
    const graphConsumesNeuronGrids = args.graphRenderer === 'canvas' || args.graphRenderer === 'svg';

    return {
        ...DEFAULT_DEMAND,
        needDecisionBoundary: boundaryVisible,
        needNeuronGrids: graphConsumesNeuronGrids,
        needLayerStats: inspectionVisible,
        needConfusionMatrix: confusionVisible,
    };
}
