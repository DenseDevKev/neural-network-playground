import { describe, expect, it } from 'vitest';
import { getFrameBuffer, type FrameBuffer } from '../../worker/frameBuffer.ts';
import { deriveNetworkSelectionModel } from './networkSelectionModel.ts';

function snapshot(): FrameBuffer {
    return { ...getFrameBuffer(), weightLayout: { layerSizes: [2, 3, 2] },
        weights: new Float32Array([0.1, 0.2, 1, -4, 0.5, 0, 2, 3, 1, -2, -3, 0]),
        biases: new Float32Array([0.25, -0.5, 0, 0.75, 1]),
        neuronGrids: new Float32Array(Array.from({ length: 20 }, (_, i) => i)),
        neuronGridLayout: { count: 5, gridSize: 2 } };
}

describe('networkSelectionModel', () => {
    it('reads packed destination-major parameters and borrows the exact neuron grid', () => {
        const input = snapshot();
        const result = deriveNetworkSelectionModel(input, { layerIdx: 1, nodeIdx: 1 });
        expect(result.kind).toBe('selected');
        if (result.kind !== 'selected') throw new Error('missing selection');
        expect(result.bias).toBe(-0.5);
        expect(result.incoming.map((x) => [x.edgeKey, x.weight, x.sign])).toEqual([
            ['1:1:1', -4, 'negative'], ['1:1:0', 1, 'positive'],
        ]);
        expect(result.outgoing.map((x) => [x.edgeKey, x.weight])).toEqual([['2:0:1', 3], ['2:1:1', -3]]);
        expect(result.grid?.buffer).toBe(input.neuronGrids!.buffer);
        expect([...result.grid!]).toEqual([4, 5, 6, 7]);
        expect(result.activation).toMatchObject({ minimum: 4, maximum: 7, mean: 5.5 });
        expect(result.activation?.standardDeviation).toBeCloseTo(Math.sqrt(1.25));
        expect([...result.highlightedEdgeKeys]).toEqual(['1:1:1', '1:1:0', '2:0:1', '2:1:1']);
    });
    it('treats input nodes as biasless and outputs as having no outgoing weights', () => {
        const input = deriveNetworkSelectionModel(snapshot(), { layerIdx: 0, nodeIdx: 0 });
        expect(input).toMatchObject({ kind: 'selected', nodeLabel: 'Input 1', bias: null, grid: null, incoming: [] });
        const output = deriveNetworkSelectionModel(snapshot(), { layerIdx: 2, nodeIdx: 1 });
        expect(output).toMatchObject({ kind: 'selected', nodeLabel: 'Output neuron 2', bias: 1, outgoing: [] });
    });
    it.each([null, { layerIdx: -1, nodeIdx: 0 }, { layerIdx: 1, nodeIdx: 3 }, { layerIdx: 3, nodeIdx: 0 }, { layerIdx: 1, nodeIdx: 0.5 }])('rejects absent or invalid node %j', (node) => {
        expect(deriveNetworkSelectionModel(snapshot(), node)).toEqual({ kind: 'empty' });
    });
    it('ranks no more than three peers per direction with deterministic numeric ties', () => {
        const input = { ...snapshot(), weightLayout: { layerSizes: [4, 1, 4] }, weights: new Float32Array([2, -2, 0, 4, 1, -1, 1, 0]) };
        const result = deriveNetworkSelectionModel(input, { layerIdx: 1, nodeIdx: 0 });
        if (result.kind !== 'selected') throw new Error('missing selection');
        expect(result.incoming.map((x) => x.peer.nodeIdx)).toEqual([3, 0, 1]);
        expect(result.outgoing.map((x) => x.peer.nodeIdx)).toEqual([0, 1, 2]);
        expect(result.highlightedEdgeKeys.size).toBe(6);
    });
    it('keeps zero-valued parameters and zero-spread activation meaningful', () => {
        const input = snapshot(); input.neuronGrids!.fill(0); input.weights!.fill(0);
        const result = deriveNetworkSelectionModel(input, { layerIdx: 1, nodeIdx: 2 });
        expect(result).toMatchObject({ kind: 'selected', bias: 0, activation: { minimum: 0, maximum: 0, mean: 0, standardDeviation: 0 } });
        if (result.kind === 'selected') expect(result.incoming.every((x) => x.sign === 'zero')).toBe(true);
    });
    it('never presents malformed buffers or non-finite samples as real evidence', () => {
        const input = snapshot(); input.weights = new Float32Array(1); input.biases![1] = NaN; input.neuronGrids![4] = NaN;
        expect(deriveNetworkSelectionModel(input, { layerIdx: 1, nodeIdx: 1 })).toMatchObject({ kind: 'selected', bias: null, grid: null, activation: null, incoming: [], outgoing: [] });
    });
    it('returns an empty display for an unavailable or invalid architecture', () => {
        expect(deriveNetworkSelectionModel({ ...snapshot(), weightLayout: null }, { layerIdx: 0, nodeIdx: 0 })).toEqual({ kind: 'empty' });
        expect(deriveNetworkSelectionModel({ ...snapshot(), weightLayout: { layerSizes: [2, -1] } }, { layerIdx: 0, nodeIdx: 0 })).toEqual({ kind: 'empty' });
    });
});
