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
            name: 'Build asks for topology and the mounted live boundary',
            view: 'build',
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
            name: 'Run loss evidence keeps graph and pinned-boundary demand',
            view: 'run',
            activeEvidenceView: 'loss',
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
            name: 'Run confusion evidence requests confusion matrix',
            view: 'run',
            activeEvidenceView: 'confusion',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            expected: demand({
                needDecisionBoundary: true,
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
                needDecisionBoundary: true,
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
                needDecisionBoundary: true,
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
            boundaryRailMounted: true,
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
            boundaryRailMounted: true,
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
            boundaryRailMounted: true,
            });
            const open = deriveVisualizationDemand({
                view: 'run',
                activeEvidenceView,
                audienceMode: 'explore',
                advancedToolsOpen: true,
                graphRenderer: 'canvas',
            boundaryRailMounted: true,
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
            boundaryRailMounted: true,
        }).needNeuronGrids).toBe(true);
    });
});

it.each(['build', 'run'] as const)('only explicit rail mount controls boundary demand in %s', (view) => {
    for (const activeEvidenceView of ['boundary', 'loss', 'confusion', 'inspection', 'code'] as const) {
        for (const boundaryRailMounted of [false, true]) {
            expect(deriveVisualizationDemand({ view, activeEvidenceView, audienceMode: 'lab', advancedToolsOpen: true, graphRenderer: 'canvas', boundaryRailMounted }).needDecisionBoundary).toBe(boundaryRailMounted);
        }
    }
});

it.each(['build', 'run'] as const)('requests diagnostics for the visible evidence panel in %s', (view) => {
    const input = { view, audienceMode: 'lab' as const, advancedToolsOpen: true, graphRenderer: 'canvas' as const, boundaryRailMounted: true };
    expect(deriveVisualizationDemand({ ...input, activeEvidenceView: 'inspection' })).toMatchObject({ needLayerStats: true, needActivationHistograms: true });
    expect(deriveVisualizationDemand({ ...input, activeEvidenceView: 'confusion' })).toMatchObject({ needConfusionMatrix: true });
});
