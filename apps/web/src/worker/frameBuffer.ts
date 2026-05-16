// ── Frame Buffer ──
// Module-level mutable state for heavy per-frame typed arrays.
// Lives OUTSIDE of React/Zustand to avoid unnecessary re-renders and
// structured-clone overhead. Components subscribe to a version counter
// in useTrainingStore as a render trigger, then read from here imperatively.

import type {
    LayerStats,
    ConfusionMatrixData,
    MulticlassConfusionMatrixData,
} from '@nn-playground/engine';
import type {
    ActivationHistogramLayout,
    ArenaModelSummary,
    MulticlassBoundaryLayout,
} from '@nn-playground/shared';

export interface FrameVersions {
    frameVersion: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
    activationHistogramsVersion: number;
    multiclassBoundaryVersion: number;
    arenaSummariesVersion: number;
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

    // Bounded activation histograms (flattened bins + compact layer layout)
    activationHistogramBins: Float32Array | null;
    activationHistogramLayout: ActivationHistogramLayout | null;

    // Bounded multiclass decision-boundary payload (class index + confidence)
    multiclassClassGrid: Uint8Array | null;
    multiclassConfidenceGrid: Float32Array | null;
    multiclassBoundaryLayout: MulticlassBoundaryLayout | null;

    // Confusion matrix
    confusionMatrix: ConfusionMatrixData | null;
    multiclassConfusionMatrix: MulticlassConfusionMatrixData | null;

    // Scalar-only live arena summaries. Heavy per-model arrays stay out of React state.
    arenaSummaries: ArenaModelSummary[] | null;

    // Version counters — `version` is the legacy broad frame version.
    version: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
    activationHistogramsVersion: number;
    multiclassBoundaryVersion: number;
    multiclassConfusionMatrixVersion: number;
    arenaSummariesVersion: number;
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
    activationHistogramBins: null,
    activationHistogramLayout: null,
    multiclassClassGrid: null,
    multiclassConfidenceGrid: null,
    multiclassBoundaryLayout: null,
    confusionMatrix: null,
    multiclassConfusionMatrix: null,
    arenaSummaries: null,
    version: 0,
    outputGridVersion: 0,
    neuronGridsVersion: 0,
    paramsVersion: 0,
    layerStatsVersion: 0,
    confusionMatrixVersion: 0,
    activationHistogramsVersion: 0,
    multiclassBoundaryVersion: 0,
    multiclassConfusionMatrixVersion: 0,
    arenaSummariesVersion: 0,
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
        activationHistogramsVersion: _buffer.activationHistogramsVersion,
        multiclassBoundaryVersion: _buffer.multiclassBoundaryVersion,
        arenaSummariesVersion: _buffer.arenaSummariesVersion,
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
    | 'activationHistogramsVersion'
    | 'multiclassBoundaryVersion'
    | 'multiclassConfusionMatrixVersion'
    | 'arenaSummariesVersion'
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
    const multiclassConfusionMatrixChanged =
        hasOwn(patch, 'multiclassConfusionMatrix') &&
        patch.multiclassConfusionMatrix !== _buffer.multiclassConfusionMatrix;
    const activationHistogramsChanged =
        hasOwn(patch, 'activationHistogramBins') ||
        hasOwn(patch, 'activationHistogramLayout');
    const multiclassBoundaryChanged =
        (
            hasOwn(patch, 'multiclassClassGrid') &&
            patch.multiclassClassGrid !== _buffer.multiclassClassGrid
        ) ||
        (
            hasOwn(patch, 'multiclassConfidenceGrid') &&
            patch.multiclassConfidenceGrid !== _buffer.multiclassConfidenceGrid
        ) ||
        (
            hasOwn(patch, 'multiclassBoundaryLayout') &&
            patch.multiclassBoundaryLayout !== _buffer.multiclassBoundaryLayout
        );
    const arenaSummariesChanged = hasOwn(patch, 'arenaSummaries');
    const anyDomainChanged =
        outputGridChanged ||
        neuronGridsChanged ||
        paramsChanged ||
        layerStatsChanged ||
        confusionMatrixChanged ||
        multiclassConfusionMatrixChanged ||
        activationHistogramsChanged ||
        multiclassBoundaryChanged ||
        arenaSummariesChanged;

    _buffer = {
        ..._buffer,
        ...patch,
        version: _buffer.version + (anyDomainChanged ? 1 : 0),
        outputGridVersion: _buffer.outputGridVersion + (outputGridChanged ? 1 : 0),
        neuronGridsVersion: _buffer.neuronGridsVersion + (neuronGridsChanged ? 1 : 0),
        paramsVersion: _buffer.paramsVersion + (paramsChanged ? 1 : 0),
        layerStatsVersion: _buffer.layerStatsVersion + (layerStatsChanged ? 1 : 0),
        confusionMatrixVersion: _buffer.confusionMatrixVersion + (confusionMatrixChanged ? 1 : 0),
        multiclassConfusionMatrixVersion:
            _buffer.multiclassConfusionMatrixVersion + (multiclassConfusionMatrixChanged ? 1 : 0),
        activationHistogramsVersion:
            _buffer.activationHistogramsVersion + (activationHistogramsChanged ? 1 : 0),
        multiclassBoundaryVersion:
            _buffer.multiclassBoundaryVersion + (multiclassBoundaryChanged ? 1 : 0),
        arenaSummariesVersion: _buffer.arenaSummariesVersion + (arenaSummariesChanged ? 1 : 0),
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
        activationHistogramBins: null,
        activationHistogramLayout: null,
        multiclassClassGrid: null,
        multiclassConfidenceGrid: null,
        multiclassBoundaryLayout: null,
        confusionMatrix: null,
        multiclassConfusionMatrix: null,
        arenaSummaries: null,
        version: _buffer.version + 1,
        outputGridVersion: _buffer.outputGridVersion + 1,
        neuronGridsVersion: _buffer.neuronGridsVersion + 1,
        paramsVersion: _buffer.paramsVersion + 1,
        layerStatsVersion: _buffer.layerStatsVersion + 1,
        confusionMatrixVersion: _buffer.confusionMatrixVersion + 1,
        multiclassConfusionMatrixVersion: _buffer.multiclassConfusionMatrixVersion + 1,
        activationHistogramsVersion: _buffer.activationHistogramsVersion + 1,
        multiclassBoundaryVersion: _buffer.multiclassBoundaryVersion + 1,
        arenaSummariesVersion: _buffer.arenaSummariesVersion + 1,
    };
}
