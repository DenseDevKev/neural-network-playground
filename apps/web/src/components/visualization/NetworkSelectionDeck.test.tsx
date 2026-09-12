import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { NetworkSelectionDeck } from './NetworkSelectionDeck.tsx';
import type { NetworkSelectionDisplayModel } from './networkSelectionModel.ts';

const model: NetworkSelectionDisplayModel = { kind: 'selected', node: { layerIdx: 1, nodeIdx: 1 }, nodeLabel: 'Hidden 1 · neuron 2', bias: 0,
    grid: null, gridSize: 0, activation: null, parameterStep: 20, activationStep: null,
    incoming: [{ edgeKey: '1:1:0', direction: 'incoming', peer: { layerIdx: 0, nodeIdx: 0 }, peerLabel: 'Input 1', weight: -2, magnitude: 2, sign: 'negative' }],
    outgoing: [], highlightedEdgeKeys: new Set(['1:1:0']) };

describe('NetworkSelectionDeck', () => {
    it('explains the empty state without presenting pretend measurements', () => {
        render(<NetworkSelectionDeck model={{ kind: 'empty' }} onClear={vi.fn()} />);
        expect(screen.getByText(/Select a neuron/)).toBeVisible();
        expect(screen.queryByRole('button', { name: 'Clear selection' })).toBeNull();
    });
    it('preserves zero bias, unavailable activation, signed influences and artifact basis', () => {
        render(<NetworkSelectionDeck model={model} onClear={vi.fn()} />);
        const region = screen.getByRole('region', { name: 'Selected neuron details' });
        expect(within(region).getByText('0.000')).toBeVisible();
        expect(within(region).getByText(/Activation grid not available/)).toBeVisible();
        expect(within(region).getByText(/negative.*-2.000/)).toBeVisible();
        expect(within(region).getByText(/Weights at step 20/)).toBeVisible();
        expect(within(region).getByText(/not a full-split evaluation/)).toBeVisible();
    });
    it('forwards clear exactly once and renders actual summary statistics', () => {
        const clear = vi.fn();
        render(<NetworkSelectionDeck model={{ ...model, activation: { minimum: -1, maximum: 3, mean: 1, standardDeviation: 2 }, activationStep: 18 }} onClear={clear} />);
        expect(screen.getByText(/activation grid at step 18/)).toBeVisible();
        fireEvent.click(screen.getByRole('button', { name: 'Clear selection' }));
        expect(clear).toHaveBeenCalledOnce();
    });
});
