import { describe, expect, it } from 'vitest';
import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { AudienceMode } from '../../productShell/audienceProfiles.ts';
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
        audienceMode: AudienceMode;
        advancedToolsOpen: boolean;
        expected: VisualizationDemand;
    }>([
        {
            name: 'Build view asks only for topology graph data',
            view: 'build',
            activeEvidenceView: 'boundary',
            audienceMode: 'explore',
            advancedToolsOpen: false,
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
            audienceMode: 'explore',
            advancedToolsOpen: false,
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
            audienceMode: 'explore',
            advancedToolsOpen: false,
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
            audienceMode: 'explore',
            advancedToolsOpen: false,
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
            audienceMode: 'explore',
            advancedToolsOpen: true,
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
            audienceMode: 'explore',
            advancedToolsOpen: true,
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'Legacy History resolves to the visible Boundary evidence',
            view: 'run',
            activeEvidenceView: 'history',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: false,
                needActivationHistograms: false,
                needConfusionMatrix: false,
            }),
        },
    ])('$name', ({ view, activeEvidenceView, audienceMode, advancedToolsOpen, expected }) => {
        expect(deriveVisualizationDemand({
            view,
            activeEvidenceView,
            audienceMode,
            advancedToolsOpen,
            graphRenderer: 'canvas',
        })).toEqual(expected);
    });

    it.each([
        ['inspection', 'needLayerStats'],
        ['confusion', 'needConfusionMatrix'],
        ['code', null],
    ] as const)(
        'falls back from hidden Beginner %s evidence without requesting its diagnostic artifact',
        (activeEvidenceView, hiddenDemandKey) => {
            const result = deriveVisualizationDemand({
                view: 'run',
                activeEvidenceView,
                audienceMode: 'beginner',
                advancedToolsOpen: false,
                graphRenderer: 'canvas',
            });

            expect(result.needDecisionBoundary).toBe(true);
            if (hiddenDemandKey) expect(result[hiddenDemandKey]).toBe(false);
            expect(result.needActivationHistograms).toBe(false);
        },
    );

    it.each(['boundary', 'loss'] as const)(
        'opening Advanced Tools while %s is active does not request diagnostic artifacts',
        (activeEvidenceView) => {
            const closed = deriveVisualizationDemand({
                view: 'run',
                activeEvidenceView,
                audienceMode: 'explore',
                advancedToolsOpen: false,
                graphRenderer: 'canvas',
            });
            const open = deriveVisualizationDemand({
                view: 'run',
                activeEvidenceView,
                audienceMode: 'explore',
                advancedToolsOpen: true,
                graphRenderer: 'canvas',
            });

            expect(open).toEqual(closed);
            expect(open).toMatchObject({
                needLayerStats: false,
                needActivationHistograms: false,
                needConfusionMatrix: false,
            });
        },
    );

    it('requests neuron grids for both graph renderers because topology can use activation tiles', () => {
        expect(deriveVisualizationDemand({
            view: 'build',
            activeEvidenceView: 'boundary',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            graphRenderer: 'svg',
        }).needNeuronGrids).toBe(true);
    });
});
