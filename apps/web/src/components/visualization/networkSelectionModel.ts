import type { FrameBuffer } from '../../worker/frameBuffer.ts';
import { layerBiasOffset, layerWeightOffset } from '../../worker/frameBufferLayout.ts';
import { edgeRefKey } from './networkGraphPainter.ts';

export interface NetworkNodeRef { readonly layerIdx: number; readonly nodeIdx: number }
export interface NetworkInfluence {
    readonly edgeKey: string;
    readonly direction: 'incoming' | 'outgoing';
    readonly peer: NetworkNodeRef;
    readonly peerLabel: string;
    readonly weight: number;
    readonly magnitude: number;
    readonly sign: 'positive' | 'negative' | 'zero';
}
export type NetworkSelectionDisplayModel = { readonly kind: 'empty' } | {
    readonly kind: 'selected';
    readonly node: NetworkNodeRef;
    readonly nodeLabel: string;
    readonly bias: number | null;
    readonly grid: Float32Array | null;
    readonly gridSize: number;
    readonly activation: { readonly minimum: number; readonly maximum: number; readonly mean: number; readonly standardDeviation: number } | null;
    readonly incoming: readonly NetworkInfluence[];
    readonly outgoing: readonly NetworkInfluence[];
    readonly highlightedEdgeKeys: ReadonlySet<string>;
    readonly parameterStep: number | null;
    readonly activationStep: number | null;
};

export function networkNodeLabel(node: NetworkNodeRef, layerCount: number): string {
    if (node.layerIdx === 0) return `Input ${node.nodeIdx + 1}`;
    return node.layerIdx === layerCount - 1 ? `Output neuron ${node.nodeIdx + 1}` : `Hidden ${node.layerIdx} · neuron ${node.nodeIdx + 1}`;
}
export function isNetworkNode(node: NetworkNodeRef | null, layers: readonly number[]): node is NetworkNodeRef {
    return node !== null && Number.isInteger(node.layerIdx) && Number.isInteger(node.nodeIdx)
        && node.layerIdx >= 0 && node.layerIdx < layers.length && node.nodeIdx >= 0 && node.nodeIdx < layers[node.layerIdx];
}

/** Read accepted packed views. Never clones the large activation grid. */
export function deriveNetworkSelectionModel(snapshot: Readonly<FrameBuffer>, node: NetworkNodeRef | null): NetworkSelectionDisplayModel {
    const layers = snapshot.weightLayout?.layerSizes;
    if (!layers || layers.length < 2 || !layers.every((n) => Number.isInteger(n) && n > 0) || !isNetworkNode(node, layers)) return { kind: 'empty' };
    const incoming: NetworkInfluence[] = [];
    const outgoing: NetworkInfluence[] = [];
    const expectedWeights = layers.slice(1).reduce((sum, size, i) => sum + size * layers[i], 0);
    const weights = snapshot.weights?.length === expectedWeights ? snapshot.weights : null;
    const add = (direction: NetworkInfluence['direction'], peer: NetworkNodeRef, weight: number, edgeKey: string) => {
        if (!Number.isFinite(weight)) return;
        (direction === 'incoming' ? incoming : outgoing).push({ direction, peer, peerLabel: networkNodeLabel(peer, layers.length), weight, magnitude: Math.abs(weight), sign: weight === 0 ? 'zero' : weight > 0 ? 'positive' : 'negative', edgeKey });
    };
    if (weights && node.layerIdx > 0) {
        const offset = layerWeightOffset(layers, node.layerIdx - 1) + node.nodeIdx * layers[node.layerIdx - 1];
        for (let i = 0; i < layers[node.layerIdx - 1]; i++) add('incoming', { layerIdx: node.layerIdx - 1, nodeIdx: i }, weights[offset + i], edgeRefKey(node.layerIdx, node.nodeIdx, i));
    }
    if (weights && node.layerIdx < layers.length - 1) {
        const offset = layerWeightOffset(layers, node.layerIdx);
        for (let i = 0; i < layers[node.layerIdx + 1]; i++) add('outgoing', { layerIdx: node.layerIdx + 1, nodeIdx: i }, weights[offset + i * layers[node.layerIdx] + node.nodeIdx], edgeRefKey(node.layerIdx + 1, i, node.nodeIdx));
    }
    const rank = (values: NetworkInfluence[]) => Object.freeze(values.sort((a, b) => b.magnitude - a.magnitude || a.peer.layerIdx - b.peer.layerIdx || a.peer.nodeIdx - b.peer.nodeIdx).slice(0, 3));
    const inputs = rank(incoming);
    const outputs = rank(outgoing);
    const neuronCount = layers.slice(1).reduce((sum, n) => sum + n, 0);
    const offset = node.layerIdx > 0 ? layerBiasOffset(layers, node.layerIdx - 1) + node.nodeIdx : -1;
    const rawBias = snapshot.biases?.length === neuronCount && offset >= 0 ? snapshot.biases[offset] : undefined;
    let grid: Float32Array | null = null;
    let gridSize = 0;
    let activation: Extract<NetworkSelectionDisplayModel, { kind: 'selected' }>['activation'] = null;
    const layout = snapshot.neuronGridLayout;
    if (offset >= 0 && layout && Number.isInteger(layout.gridSize) && layout.gridSize > 0 && layout.count === neuronCount && snapshot.neuronGrids?.length === neuronCount * layout.gridSize ** 2) {
        const cells = layout.gridSize ** 2;
        const candidate = snapshot.neuronGrids.subarray(offset * cells, (offset + 1) * cells);
        if (candidate.every(Number.isFinite)) {
            grid = candidate; gridSize = layout.gridSize;
            let minimum = Infinity, maximum = -Infinity, mean = 0, m2 = 0;
            // Welford's population variance avoids cancellation for narrow distributions.
            for (let i = 0; i < grid.length; i++) {
                const value = grid[i]; minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
                const delta = value - mean; mean += delta / (i + 1); m2 += delta * (value - mean);
            }
            activation = { minimum, maximum, mean, standardDeviation: Math.sqrt(Math.max(0, m2 / grid.length)) };
        }
    }
    return { kind: 'selected', node, nodeLabel: networkNodeLabel(node, layers.length), bias: rawBias !== undefined && Number.isFinite(rawBias) ? rawBias : null,
        grid, gridSize, activation, incoming: inputs, outgoing: outputs,
        highlightedEdgeKeys: new Set([...inputs, ...outputs].map((x) => x.edgeKey)),
        parameterStep: snapshot.parameterProvenance?.model.step ?? null,
        activationStep: grid ? snapshot.neuronGridsProvenance?.model.step ?? null : null };
}
