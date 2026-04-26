import { describe, expect, it } from 'vitest';
import { DEFAULT_DEMAND, type VisualizationDemand } from '@nn-playground/shared';
import type { LayoutVariant, PhaseMode } from '../../store/useLayoutStore.ts';
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
        layout: LayoutVariant;
        phase: PhaseMode;
        activeTabRight: string;
        expected: VisualizationDemand;
    }>([
        {
            name: 'dock boundary tab',
            layout: 'dock',
            phase: 'build',
            activeTabRight: 'boundary',
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'dock loss tab',
            layout: 'dock',
            phase: 'build',
            activeTabRight: 'loss',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'dock confusion tab',
            layout: 'dock',
            phase: 'build',
            activeTabRight: 'confusion',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: true,
            }),
        },
        {
            name: 'dock inspection tab',
            layout: 'dock',
            phase: 'build',
            activeTabRight: 'inspection',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: true,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'focus layout',
            layout: 'focus',
            phase: 'build',
            activeTabRight: 'loss',
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: true,
                needConfusionMatrix: true,
            }),
        },
        {
            name: 'grid layout',
            layout: 'grid',
            phase: 'build',
            activeTabRight: 'loss',
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: true,
                needConfusionMatrix: true,
            }),
        },
        {
            name: 'split build phase',
            layout: 'split',
            phase: 'build',
            activeTabRight: 'boundary',
            expected: demand({
                needDecisionBoundary: false,
                needNeuronGrids: true,
                needLayerStats: false,
                needConfusionMatrix: false,
            }),
        },
        {
            name: 'split run phase',
            layout: 'split',
            phase: 'run',
            activeTabRight: 'loss',
            expected: demand({
                needDecisionBoundary: true,
                needNeuronGrids: true,
                needLayerStats: true,
                needConfusionMatrix: true,
            }),
        },
    ])('$name', ({ layout, phase, activeTabRight, expected }) => {
        expect(deriveVisualizationDemand({
            layout,
            phase,
            activeTabRight,
            graphRenderer: 'canvas',
        })).toEqual(expected);
    });
});
