import { describe, expect, it } from 'vitest';
import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { EvidenceViewId, WorkspaceView } from '../../store/useLayoutStore.ts';
import { deriveVisualizationDemand } from './deriveVisualizationDemand.ts';

function demand(overrides: Partial<VisualizationDemand>): VisualizationDemand {
    return {
        ...DEFAULT_DEMAND,
        ...overrides,
    };
}

describe('deriveVisualizationDemand', () => {
    it.each<{
        name: string;
        view: WorkspaceView;
        activeEvidenceView: EvidenceViewId;
        historyDrawerOpen?: boolean;
        expected: VisualizationDemand;
    }>([
        {
            name: 'Build view asks only for topology graph data',
            view: 'build',
            activeEvidenceView: 'boundary',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'Run boundary evidence requests decision boundary',
            view: 'run',
            activeEvidenceView: 'boundary',
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'Run loss evidence keeps graph demand without boundary expansion',
            view: 'run',
            activeEvidenceView: 'loss',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'Run confusion evidence requests confusion matrix',
            view: 'run',
            activeEvidenceView: 'confusion',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: true,
            }),
        },
        {
            name: 'Run inspection evidence requests layer diagnostics',
            view: 'run',
            activeEvidenceView: 'inspection',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: true,
                needActivationHistograms: true,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'Run code evidence does not expand visualization demand',
            view: 'run',
            activeEvidenceView: 'code',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'History drawer does not expand worker demand by itself',
            view: 'run',
            activeEvidenceView: 'loss',
            historyDrawerOpen: true,
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needActivationHistograms: false,
                needConfusionMatrix: false,
            }),
        },
    ])('$name', ({ view, activeEvidenceView, historyDrawerOpen, expected }) => {
        expect(deriveVisualizationDemand({
            view,
            activeEvidenceView,
            historyDrawerOpen,
            graphRenderer: 'canvas',
        })).toEqual(expected);
    });

    it('requests neuron grids for both graph renderers because topology can use activation tiles', () => {
        expect(deriveVisualizationDemand({
            view: 'build',
            activeEvidenceView: 'boundary',
            graphRenderer: 'svg',
        }).needNeuronGrids).toBe(true);
    });
});
