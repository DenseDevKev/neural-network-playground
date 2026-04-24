// ── Frame Buffer ──
// Module-level mutable state for heavy per-frame typed arrays.
// Lives OUTSIDE of React/Zustand to avoid unnecessary re-renders and
// structured-clone overhead. Components subscribe to a version counter
// in useTrainingStore as a render trigger, then read from here imperatively.

import type { LayerStats, ConfusionMatrixData } from '@nn-playground/engine';

export interface FrameVersions {
    frameVersion: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
}

export interface FrameBuffer {
    // Decision boundary grid (gridSize × gridSize predictions)
    outputGrid: Float32Array | null;
    gridSize: number;

    // Per-neuron activation grids (flattened, all neurons concatenated)
    neuronGrids: Float32Array | null;
    neuronGridLayout: { count: number; gridSize: number } | null;

    // Flattened weights and biases
    weights: Float32Array | null;
    biases: Float32Array | null;
    weightLayout: { layerSizes: number[] } | null;

    // Layer statistics (small enough to keep here)
    layerStats: LayerStats[] | null;

    // Confusion matrix
    confusionMatrix: ConfusionMatrixData | null;

    // Version counters — `version` is the legacy broad frame version.
    version: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
}

let _buffer: FrameBuffer = {
    outputGrid: null,
    gridSize: 0,
    neuronGrids: null,
    neuronGridLayout: null,
    weights: null,
    biases: null,
    weightLayout: null,
    layerStats: null,
    confusionMatrix: null,
    version: 0,
    outputGridVersion: 0,
    neuronGridsVersion: 0,
    paramsVersion: 0,
    layerStatsVersion: 0,
    confusionMatrixVersion: 0,
};

/** Get a readonly view of the current frame buffer. */
export function getFrameBuffer(): Readonly<FrameBuffer> {
    return _buffer;
}

/** Get the current frame buffer version. */
export function getFrameVersion(): number {
    return _buffer.version;
}

/** Get the current frame buffer versions. */
export function getFrameVersions(): FrameVersions {
    return {
        frameVersion: _buffer.version,
        outputGridVersion: _buffer.outputGridVersion,
        neuronGridsVersion: _buffer.neuronGridsVersion,
        paramsVersion: _buffer.paramsVersion,
        layerStatsVersion: _buffer.layerStatsVersion,
        confusionMatrixVersion: _buffer.confusionMatrixVersion,
    };
}

type FrameBufferPatch = Partial<Omit<
    FrameBuffer,
    | 'version'
    | 'outputGridVersion'
    | 'neuronGridsVersion'
    | 'paramsVersion'
    | 'layerStatsVersion'
    | 'confusionMatrixVersion'
>>;

function hasOwn(patch: FrameBufferPatch, key: keyof FrameBufferPatch): boolean {
    return Object.prototype.hasOwnProperty.call(patch, key);
}

/** Update the frame buffer with new data and increment affected version counters. */
export function updateFrameBuffer(patch: FrameBufferPatch): number {
    const outputGridChanged = hasOwn(patch, 'outputGrid');
    const neuronGridsChanged =
        hasOwn(patch, 'neuronGrids') || hasOwn(patch, 'neuronGridLayout');
    const paramsChanged =
        hasOwn(patch, 'weights') || hasOwn(patch, 'biases') || hasOwn(patch, 'weightLayout');
    const layerStatsChanged = hasOwn(patch, 'layerStats');
    const confusionMatrixChanged = hasOwn(patch, 'confusionMatrix');
    const anyDomainChanged =
        outputGridChanged ||
        neuronGridsChanged ||
        paramsChanged ||
        layerStatsChanged ||
        confusionMatrixChanged;

    _buffer = {
        ..._buffer,
        ...patch,
        version: _buffer.version + (anyDomainChanged ? 1 : 0),
        outputGridVersion: _buffer.outputGridVersion + (outputGridChanged ? 1 : 0),
        neuronGridsVersion: _buffer.neuronGridsVersion + (neuronGridsChanged ? 1 : 0),
        paramsVersion: _buffer.paramsVersion + (paramsChanged ? 1 : 0),
        layerStatsVersion: _buffer.layerStatsVersion + (layerStatsChanged ? 1 : 0),
        confusionMatrixVersion: _buffer.confusionMatrixVersion + (confusionMatrixChanged ? 1 : 0),
    };
    return _buffer.version;
}

/** Reset the frame buffer to its initial empty state. */
export function resetFrameBuffer(): void {
    _buffer = {
        outputGrid: null,
        gridSize: 0,
        neuronGrids: null,
        neuronGridLayout: null,
        weights: null,
        biases: null,
        weightLayout: null,
        layerStats: null,
        confusionMatrix: null,
        version: _buffer.version + 1,
        outputGridVersion: _buffer.outputGridVersion + 1,
        neuronGridsVersion: _buffer.neuronGridsVersion + 1,
        paramsVersion: _buffer.paramsVersion + 1,
        layerStatsVersion: _buffer.layerStatsVersion + 1,
        confusionMatrixVersion: _buffer.confusionMatrixVersion + 1,
    };
}

export function flattenWeights(
    weights: number[][][],
): { buffer: Float32Array; layerSizes: number[] } {
    if (weights.length === 0) {
        return {
            buffer: new Float32Array(0),
            layerSizes: [],
        };
    }

    let totalWeights = 0;
    const layerSizes = [weights[0][0]?.length ?? 0];

    for (const layer of weights) {
        layerSizes.push(layer.length);
        for (const neuron of layer) {
            totalWeights += neuron.length;
        }
    }

    const buffer = new Float32Array(totalWeights);
    let offset = 0;
    for (const layer of weights) {
        for (const neuron of layer) {
            buffer.set(neuron, offset);
            offset += neuron.length;
        }
    }

    return { buffer, layerSizes };
}

export function flattenBiases(biases: number[][]): Float32Array {
    let totalBiases = 0;
    for (const layer of biases) {
        totalBiases += layer.length;
    }

    const buffer = new Float32Array(totalBiases);
    let offset = 0;
    for (const layer of biases) {
        buffer.set(layer, offset);
        offset += layer.length;
    }

    return buffer;
}

export function flattenNeuronGrids(
    neuronGrids: number[][] | Float32Array[],
    gridSize: number,
): { buffer: Float32Array; layout: { count: number; gridSize: number } } {
    const count = neuronGrids.length;
    const buffer = new Float32Array(count * gridSize * gridSize);

    for (let i = 0; i < count; i++) {
        buffer.set(neuronGrids[i], i * gridSize * gridSize);
    }

    return {
        buffer,
        layout: { count, gridSize },
    };
}

/**
 * Helper: Reconstruct nested weight arrays from a flat Float32Array + layout.
 * Returns weights[layer][neuron][prevNeuron].
 */
export function unflattenWeights(
    flat: Float32Array,
    layerSizes: number[],
): number[][][] {
    const weights: number[][][] = [];
    let offset = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        const fanIn = layerSizes[l];
        const fanOut = layerSizes[l + 1];
        const layer: number[][] = [];
        for (let n = 0; n < fanOut; n++) {
            const neuronWeights: number[] = [];
            for (let w = 0; w < fanIn; w++) {
                neuronWeights.push(flat[offset++]);
            }
            layer.push(neuronWeights);
        }
        weights.push(layer);
    }
    return weights;
}

/**
 * Helper: Reconstruct nested bias arrays from a flat Float32Array + layout.
 * Returns biases[layer][neuron].
 */
export function unflattenBiases(
    flat: Float32Array,
    layerSizes: number[],
): number[][] {
    const biases: number[][] = [];
    let offset = 0;
    for (let l = 1; l < layerSizes.length; l++) {
        const layerBiases: number[] = [];
        for (let n = 0; n < layerSizes[l]; n++) {
            layerBiases.push(flat[offset++]);
        }
        biases.push(layerBiases);
    }
    return biases;
}

/**
 * Helper: Extract a single neuron's grid from the flattened neuronGrids buffer.
 * Neuron grids are stored as: neuronGrids[neuronIndex * gridLength .. (neuronIndex+1) * gridLength].
 */
export function extractNeuronGrid(
    neuronGrids: Float32Array,
    neuronIndex: number,
    gridLength: number,
): Float32Array {
    const start = neuronIndex * gridLength;
    return neuronGrids.subarray(start, start + gridLength);
}

// ── Flat-buffer offset helpers ─────────────────────────────────────────────
// `layerSizes` is [inputSize, h1, ..., outputSize]. Layer index `l` refers
// to the weight matrix mapping layer l → layer l+1 (so l ranges over
// 0..layerSizes.length-2). These helpers let UI components read the packed
// weight / bias Float32Arrays emitted by the worker without unflattening
// into nested number[][][] arrays on every frame.

/** Start offset (in weights buffer) of the weight row for a given layer. */
export function layerWeightOffset(layerSizes: number[], layerIdx: number): number {
    let offset = 0;
    for (let l = 0; l < layerIdx; l++) {
        offset += layerSizes[l] * layerSizes[l + 1];
    }
    return offset;
}

/** Read a single weight from the flat buffer.
 *  `layerIdx` is the *source* layer (weights between layer `layerIdx` and
 *  `layerIdx + 1`); `neuronIdx` indexes the destination neuron; `prevIdx`
 *  indexes the source neuron. Matches the Network.getWeightsFlat layout:
 *  row-major [fanOut × fanIn], one block per layer. */
export function readWeight(
    flat: Float32Array,
    layerSizes: number[],
    layerIdx: number,
    neuronIdx: number,
    prevIdx: number,
): number {
    const fanIn = layerSizes[layerIdx];
    return flat[layerWeightOffset(layerSizes, layerIdx) + neuronIdx * fanIn + prevIdx];
}

/** Start offset (in biases buffer) of the bias row for a given layer. */
export function layerBiasOffset(layerSizes: number[], layerIdx: number): number {
    let offset = 0;
    for (let l = 1; l <= layerIdx; l++) {
        offset += layerSizes[l];
    }
    return offset;
}

/** Read a single bias from the flat buffer.
 *  `layerIdx` is the source layer (weights between `layerIdx` and
 *  `layerIdx + 1`); the bias lives on the *destination* layer `layerIdx+1`. */
export function readBias(
    flat: Float32Array,
    layerSizes: number[],
    layerIdx: number,
    neuronIdx: number,
): number {
    return flat[layerBiasOffset(layerSizes, layerIdx) + neuronIdx];
}
