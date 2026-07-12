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
    ArtifactProvenance,
    ActivationHistogramLayout,
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
}

export interface FrameBuffer {
    // Decision boundary grid (gridSize × gridSize predictions)
    outputGrid: Float32Array | null;
    gridSize: number;
    decisionBoundaryProvenance: ArtifactProvenance | null;

    // Per-neuron activation grids (flattened, all neurons concatenated)
    neuronGrids: Float32Array | null;
    neuronGridLayout: { count: number; gridSize: number } | null;
    neuronGridsProvenance: ArtifactProvenance | null;

    // Flattened weights and biases
    weights: Float32Array | null;
    biases: Float32Array | null;
    weightLayout: { layerSizes: number[] } | null;

    // Layer statistics (small enough to keep here)
    layerStats: LayerStats[] | null;
    layerStatsProvenance: ArtifactProvenance | null;
    layerStatsGradientRevision: number | null;

    // Bounded activation histograms (flattened bins + compact layer layout)
    activationHistogramBins: Float32Array | null;
    activationHistogramLayout: ActivationHistogramLayout | null;
    activationHistogramProvenance: ArtifactProvenance | null;

    // Bounded multiclass decision-boundary payload (class index + confidence)
    multiclassClassGrid: Uint8Array | null;
    multiclassConfidenceGrid: Float32Array | null;
    multiclassBoundaryLayout: MulticlassBoundaryLayout | null;

    // Confusion matrix
    confusionMatrix: ConfusionMatrixData | null;
    multiclassConfusionMatrix: MulticlassConfusionMatrixData | null;
    confusionMatrixProvenance: ArtifactProvenance | null;
    confusionMatrixEvaluationId: number | null;

    // Version counters — `version` is the broad frame version.
    version: number;
    outputGridVersion: number;
    neuronGridsVersion: number;
    paramsVersion: number;
    layerStatsVersion: number;
    confusionMatrixVersion: number;
    activationHistogramsVersion: number;
    multiclassBoundaryVersion: number;
    multiclassConfusionMatrixVersion: number;
}

let _buffer: FrameBuffer = {
    outputGrid: null,
    gridSize: 0,
    decisionBoundaryProvenance: null,
    neuronGrids: null,
    neuronGridLayout: null,
    neuronGridsProvenance: null,
    weights: null,
    biases: null,
    weightLayout: null,
    layerStats: null,
    layerStatsProvenance: null,
    layerStatsGradientRevision: null,
    activationHistogramBins: null,
    activationHistogramLayout: null,
    activationHistogramProvenance: null,
    multiclassClassGrid: null,
    multiclassConfidenceGrid: null,
    multiclassBoundaryLayout: null,
    confusionMatrix: null,
    multiclassConfusionMatrix: null,
    confusionMatrixProvenance: null,
    confusionMatrixEvaluationId: null,
    version: 0,
    outputGridVersion: 0,
    neuronGridsVersion: 0,
    paramsVersion: 0,
    layerStatsVersion: 0,
    confusionMatrixVersion: 0,
    activationHistogramsVersion: 0,
    multiclassBoundaryVersion: 0,
    multiclassConfusionMatrixVersion: 0,
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
    };
}

export type FrameBufferPatch = Partial<Omit<
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
>>;

function hasOwn(patch: FrameBufferPatch, key: keyof FrameBufferPatch): boolean {
    return Object.prototype.hasOwnProperty.call(patch, key);
}

export interface FrameBufferUpdateOptions {
    readonly requireArtifactProvenance?: boolean;
}

function assertArtifactPair(
    label: string,
    payloadMutated: boolean,
    payloadPresent: boolean,
    provenanceMutated: boolean,
    provenance: ArtifactProvenance | null | undefined,
): void {
    if (!payloadMutated && !provenanceMutated) return;
    if (!payloadMutated
        || !provenanceMutated
        || (payloadPresent && provenance == null)
        || (!payloadPresent && provenance !== null)) {
        throw new Error(`${label} payload and provenance must update atomically`);
    }
}

function assertStrictArtifactPairs(patch: FrameBufferPatch): void {
    const paramsMutated = hasOwn(patch, 'weights')
        || hasOwn(patch, 'biases')
        || hasOwn(patch, 'weightLayout');
    if (paramsMutated && (
        !hasOwn(patch, 'weights')
        || !hasOwn(patch, 'biases')
        || !hasOwn(patch, 'weightLayout')
    )) throw new Error('parameter strict transaction must replace the complete domain');
    const boundaryMutated = hasOwn(patch, 'outputGrid')
        || hasOwn(patch, 'gridSize')
        || hasOwn(patch, 'multiclassClassGrid')
        || hasOwn(patch, 'multiclassConfidenceGrid')
        || hasOwn(patch, 'multiclassBoundaryLayout');
    if (boundaryMutated && ![
        'outputGrid',
        'gridSize',
        'multiclassClassGrid',
        'multiclassConfidenceGrid',
        'multiclassBoundaryLayout',
    ].every((key) => hasOwn(patch, key as keyof FrameBufferPatch))) {
        throw new Error('decision boundary strict transaction must replace the complete domain');
    }
    const scalarBoundary = patch.outputGrid;
    const multiclassClassGrid = patch.multiclassClassGrid;
    const multiclassConfidenceGrid = patch.multiclassConfidenceGrid;
    const multiclassLayout = patch.multiclassBoundaryLayout;
    const hasScalarBoundary = scalarBoundary != null;
    const hasAnyMulticlassBoundary = multiclassClassGrid != null
        || multiclassConfidenceGrid != null
        || multiclassLayout != null;
    const hasCompleteMulticlassBoundary = multiclassClassGrid != null
        && multiclassConfidenceGrid != null
        && multiclassLayout != null;
    if (boundaryMutated && (
        (hasScalarBoundary && hasAnyMulticlassBoundary)
        || (hasAnyMulticlassBoundary && !hasCompleteMulticlassBoundary)
        || (hasScalarBoundary && (
            !Number.isSafeInteger(patch.gridSize)
            || (patch.gridSize as number) < 1
            || scalarBoundary.length !== (patch.gridSize as number) ** 2
        ))
        || (hasCompleteMulticlassBoundary && (
            patch.gridSize !== multiclassLayout.gridSize
            || multiclassClassGrid.length !== multiclassLayout.gridSize ** 2
            || multiclassConfidenceGrid.length !== multiclassLayout.gridSize ** 2
        ))
        || (!hasScalarBoundary && !hasAnyMulticlassBoundary && patch.gridSize !== 0)
    )) {
        throw new Error('decision boundary strict transaction has an incomplete or inconsistent payload');
    }
    const boundaryPresent = hasScalarBoundary || hasCompleteMulticlassBoundary;
    assertArtifactPair(
        'decision boundary',
        boundaryMutated,
        boundaryPresent,
        hasOwn(patch, 'decisionBoundaryProvenance'),
        patch.decisionBoundaryProvenance,
    );

    const neuronMutated = hasOwn(patch, 'neuronGrids') || hasOwn(patch, 'neuronGridLayout');
    if (neuronMutated
        && (!hasOwn(patch, 'neuronGrids') || !hasOwn(patch, 'neuronGridLayout'))) {
        throw new Error('neuron grids strict transaction must replace the complete domain');
    }
    if (neuronMutated && (
        (patch.neuronGrids === null) !== (patch.neuronGridLayout === null)
        || (patch.neuronGrids != null && patch.neuronGridLayout != null && (
            patch.neuronGrids.length
                !== patch.neuronGridLayout.count * patch.neuronGridLayout.gridSize ** 2
            || patch.neuronGrids.some((value) => !Number.isFinite(value))
        ))
    )) throw new Error('neuron grids strict transaction has an inconsistent payload');
    assertArtifactPair(
        'neuron grids',
        neuronMutated,
        (hasOwn(patch, 'neuronGrids') ? patch.neuronGrids : _buffer.neuronGrids) != null,
        hasOwn(patch, 'neuronGridsProvenance'),
        patch.neuronGridsProvenance,
    );

    const layerMutated = hasOwn(patch, 'layerStats')
        || hasOwn(patch, 'layerStatsGradientRevision');
    if (layerMutated
        && (!hasOwn(patch, 'layerStats') || !hasOwn(patch, 'layerStatsGradientRevision'))) {
        throw new Error('layer statistics strict transaction must replace the complete domain');
    }
    const resultingLayerStats = hasOwn(patch, 'layerStats')
        ? patch.layerStats
        : _buffer.layerStats;
    const resultingGradientRevision = hasOwn(patch, 'layerStatsGradientRevision')
        ? patch.layerStatsGradientRevision
        : _buffer.layerStatsGradientRevision;
    if (layerMutated && (
        (resultingLayerStats === null) !== (resultingGradientRevision === null)
        || (resultingLayerStats !== null && (
            !Number.isSafeInteger(resultingGradientRevision)
            || (resultingGradientRevision as number) < 0
        ))
    )) throw new Error('layer statistics strict transaction has an inconsistent payload');
    const layerPresent = resultingLayerStats != null
        && Number.isSafeInteger(resultingGradientRevision)
        && (resultingGradientRevision as number) >= 0;
    assertArtifactPair(
        'layer statistics',
        layerMutated,
        layerPresent,
        hasOwn(patch, 'layerStatsProvenance'),
        patch.layerStatsProvenance,
    );

    const histogramMutated = hasOwn(patch, 'activationHistogramBins')
        || hasOwn(patch, 'activationHistogramLayout');
    if (histogramMutated && (
        !hasOwn(patch, 'activationHistogramBins')
        || !hasOwn(patch, 'activationHistogramLayout')
    )) throw new Error('activation histogram strict transaction must replace the complete domain');
    if (histogramMutated && (
        (patch.activationHistogramBins === null) !== (patch.activationHistogramLayout === null)
        || (patch.activationHistogramBins != null
            && patch.activationHistogramLayout != null
            && (
                patch.activationHistogramBins.length
                    !== patch.activationHistogramLayout.layers.reduce(
                        (sum, layer) => sum + layer.binCount,
                        0,
                    )
                || patch.activationHistogramBins.some(
                    (value) => !Number.isSafeInteger(value) || value < 0,
                )
            ))
    )) throw new Error('activation histogram strict transaction has an inconsistent payload');
    if (histogramMutated
        && patch.activationHistogramBins != null
        && patch.activationHistogramLayout != null) {
        const provenance = patch.activationHistogramProvenance;
        const layerSizes = (patch.weightLayout ?? _buffer.weightLayout)?.layerSizes;
        const sampleCount = provenance?.basis.kind === 'bounded-sample'
            ? provenance.basis.sampleCount
            : null;
        let offset = 0;
        const malformed = patch.activationHistogramLayout.binCount !== 12
            || layerSizes === undefined
            || sampleCount === null
            || patch.activationHistogramLayout.layers.length !== layerSizes.length - 1
            || patch.activationHistogramLayout.layers.some((layer, index) => {
                let sum = 0;
                for (let bin = 0; bin < layer.binCount; bin++) {
                    sum += patch.activationHistogramBins![offset + bin];
                }
                offset += layer.binCount;
                return layer.layerIndex !== index
                    || layer.binCount !== patch.activationHistogramLayout!.binCount
                    || !Number.isFinite(layer.binStart)
                    || !Number.isFinite(layer.binWidth)
                    || layer.binWidth <= 0
                    || !Number.isFinite(layer.minActivation)
                    || !Number.isFinite(layer.maxActivation)
                    || layer.minActivation > layer.maxActivation
                    || layer.totalCount !== sampleCount * layerSizes[index + 1]
                    || sum !== layer.totalCount;
            });
        if (malformed) {
            throw new Error('activation histogram strict transaction has malformed semantics');
        }
    }
    assertArtifactPair(
        'activation histogram',
        histogramMutated,
        (
            hasOwn(patch, 'activationHistogramBins')
                ? patch.activationHistogramBins
                : _buffer.activationHistogramBins
        ) != null && (
            hasOwn(patch, 'activationHistogramLayout')
                ? patch.activationHistogramLayout
                : _buffer.activationHistogramLayout
        ) != null,
        hasOwn(patch, 'activationHistogramProvenance'),
        patch.activationHistogramProvenance,
    );

    const confusionMutated = hasOwn(patch, 'confusionMatrix')
        || hasOwn(patch, 'multiclassConfusionMatrix')
        || hasOwn(patch, 'confusionMatrixProvenance')
        || hasOwn(patch, 'confusionMatrixEvaluationId');
    if (confusionMutated
        && (!hasOwn(patch, 'confusionMatrix')
            || !hasOwn(patch, 'multiclassConfusionMatrix')
            || !hasOwn(patch, 'confusionMatrixProvenance')
            || !hasOwn(patch, 'confusionMatrixEvaluationId'))) {
        throw new Error('confusion matrix strict transaction must replace the complete domain');
    }
    if (confusionMutated
        && patch.confusionMatrix != null
        && patch.multiclassConfusionMatrix != null) {
        throw new Error('confusion matrix strict transaction cannot contain both matrix kinds');
    }
    if (confusionMutated) {
        const matrixPresent = patch.confusionMatrix != null
            || patch.multiclassConfusionMatrix != null;
        if ((matrixPresent && (
            patch.confusionMatrixProvenance == null
            || !Number.isSafeInteger(patch.confusionMatrixEvaluationId)
            || (patch.confusionMatrixEvaluationId as number) < 1
        )) || (!matrixPresent && (
            patch.confusionMatrixProvenance !== null
            || patch.confusionMatrixEvaluationId !== null
        ))) {
            throw new Error('confusion matrix provenance and evaluation ID must update atomically');
        }
    }
}

/** Validate a prospective frame transaction without publishing it. */
export function validateFrameBufferPatch(
    patch: FrameBufferPatch,
    options: FrameBufferUpdateOptions = {},
): void {
    if (options.requireArtifactProvenance === true) assertStrictArtifactPairs(patch);
}

/** Update the frame buffer with new data and increment affected version counters. */
export function updateFrameBuffer(
    patch: FrameBufferPatch,
    options: FrameBufferUpdateOptions = {},
): number {
    validateFrameBufferPatch(patch, options);
    const outputGridChanged = hasOwn(patch, 'outputGrid');
    const neuronGridsChanged =
        hasOwn(patch, 'neuronGrids') || hasOwn(patch, 'neuronGridLayout');
    const paramsChanged =
        hasOwn(patch, 'weights') || hasOwn(patch, 'biases') || hasOwn(patch, 'weightLayout');
    const layerStatsChanged = hasOwn(patch, 'layerStats')
        || hasOwn(patch, 'layerStatsGradientRevision')
        || hasOwn(patch, 'layerStatsProvenance');
    const confusionMatrixChanged = (
        hasOwn(patch, 'confusionMatrix')
        && patch.confusionMatrix !== _buffer.confusionMatrix
    ) || (
        hasOwn(patch, 'multiclassConfusionMatrix')
        && patch.multiclassConfusionMatrix !== _buffer.multiclassConfusionMatrix
    ) || (
        hasOwn(patch, 'confusionMatrixProvenance')
        && patch.confusionMatrixProvenance !== _buffer.confusionMatrixProvenance
    ) || (
        hasOwn(patch, 'confusionMatrixEvaluationId')
        && patch.confusionMatrixEvaluationId !== _buffer.confusionMatrixEvaluationId
    );
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
    const anyDomainChanged =
        outputGridChanged ||
        neuronGridsChanged ||
        paramsChanged ||
        layerStatsChanged ||
        confusionMatrixChanged ||
        multiclassConfusionMatrixChanged ||
        activationHistogramsChanged ||
        multiclassBoundaryChanged;

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
    };
    return _buffer.version;
}

/** Reset the frame buffer to its initial empty state. */
export function resetFrameBuffer(): void {
    _buffer = {
        outputGrid: null,
        gridSize: 0,
        decisionBoundaryProvenance: null,
        neuronGrids: null,
        neuronGridLayout: null,
        neuronGridsProvenance: null,
        weights: null,
        biases: null,
        weightLayout: null,
        layerStats: null,
        layerStatsProvenance: null,
        layerStatsGradientRevision: null,
        activationHistogramBins: null,
        activationHistogramLayout: null,
        activationHistogramProvenance: null,
        multiclassClassGrid: null,
        multiclassConfidenceGrid: null,
        multiclassBoundaryLayout: null,
        confusionMatrix: null,
        multiclassConfusionMatrix: null,
        confusionMatrixProvenance: null,
        confusionMatrixEvaluationId: null,
        version: _buffer.version + 1,
        outputGridVersion: _buffer.outputGridVersion + 1,
        neuronGridsVersion: _buffer.neuronGridsVersion + 1,
        paramsVersion: _buffer.paramsVersion + 1,
        layerStatsVersion: _buffer.layerStatsVersion + 1,
        confusionMatrixVersion: _buffer.confusionMatrixVersion + 1,
        multiclassConfusionMatrixVersion: _buffer.multiclassConfusionMatrixVersion + 1,
        activationHistogramsVersion: _buffer.activationHistogramsVersion + 1,
        multiclassBoundaryVersion: _buffer.multiclassBoundaryVersion + 1,
    };
}
