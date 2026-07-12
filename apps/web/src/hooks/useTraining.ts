// ── useTraining hook ──
// Manages the training loop, worker communication, and data synchronization.
// Phase 3: Uses useTrainingStore for runtime state, usePlaygroundStore for config.

import { useEffect, useRef, useCallback } from 'react';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import {
    useTrainingStore,
    type TrainedRecipeSource,
    type TrainingStore,
} from '../store/useTrainingStore.ts';
import {
    getWorkerApi,
    setupStreamChannel,
    postStreamCommand,
    startRenderLoop,
    stopRenderLoop,
    onSnapshot,
    newRunTo,
    discardPendingSnapshot,
    terminateWorker,
} from '../worker/workerBridge.ts';
import {
    getFrameBuffer,
    getFrameVersions,
    updateFrameBuffer,
    validateFrameBufferPatch,
    type FrameBufferPatch,
    type FrameVersions,
} from '../worker/frameBuffer.ts';
import {
    flattenBiases,
    flattenNeuronGrids,
    flattenWeights,
} from '../worker/frameBufferLayout.ts';
import type {
    DataPoint,
} from '@nn-playground/engine';
import { getDatasetContract } from '@nn-playground/engine';
import type {
    ArtifactProvenance,
    CheckpointTimeline,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerEvidenceMessageV2,
    WorkerArtifactProvenanceV2,
    WorkerExperimentRequestV2,
    WorkerProtocolErrorMessageV2,
    WorkerToMainMessage,
} from '@nn-playground/shared';
import {
    GRID_SIZE,
    WORKER_PROTOCOL_VERSION,
    parseArtifactProvenance,
    parseWorkerEvidenceMessageV2,
    parseCheckpointTimelineV2,
} from '@nn-playground/shared';
import type { WorkerExperimentResultV2 } from '../worker/training.worker.ts';

export interface TrainingHook {
    play: () => void;
    pause: () => void;
    step: () => Promise<void>;
    reset: () => Promise<void>;
    restoreCheckpoint: (id: number) => Promise<void>;
}

function getErrorMessage(error: unknown, fallback: string): string {
    if (error instanceof Error) return error.message;
    return typeof error === 'string' && error.length > 0 ? error : fallback;
}

function requirePreparedExperiment(): PreparedExperimentDocumentV2 {
    const access = usePlaygroundStore.getState().access;
    if (access.status === 'ready') return access.prepared;
    throw new Error(
        'Training is unavailable because the shared experiment URL is incompatible with version 2.',
    );
}

function currentPreparedExperiment(): PreparedExperimentDocumentV2 | null {
    const access = usePlaygroundStore.getState().access;
    return access.status === 'ready' ? access.prepared : null;
}

export function createWorkerExperimentRequestV2(
    prepared: PreparedExperimentDocumentV2,
    requestId: number,
): WorkerExperimentRequestV2 {
    if (!Number.isSafeInteger(requestId) || requestId < 1) {
        throw new RangeError('worker requestId must be a positive safe integer');
    }
    return {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId,
        document: prepared.document,
        claimedIdentities: prepared.identities,
    };
}

/**
 * Direct manual-step RPCs can resolve before cadence evidence already queued
 * on the ordered MessagePort. Never jump the evaluation series over that gap.
 */
export function evidenceForDirectStep(
    evidence: WorkerEvidenceMessageV2,
    latestEvaluation: PairedEvaluation | null,
): WorkerEvidenceMessageV2 | null {
    const evaluation = evidence.latestEvaluation;
    if (!evaluation) return evidence;
    const latestId = latestEvaluation?.evaluationId ?? 0;
    if (evaluation.evaluationId === latestId
        || evaluation.evaluationId === latestId + 1) {
        return evidence;
    }
    if (!evidence.liveSignal && !evidence.artifacts) return null;
    return {
        type: 'evidence',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        ...(evidence.liveSignal ? { liveSignal: evidence.liveSignal } : {}),
        ...(evidence.artifacts ? { artifacts: evidence.artifacts } : {}),
    };
}

function getTotalNeuronCount(layerSizes: number[]): number {
    let total = 0;
    for (let i = 1; i < layerSizes.length; i++) {
        total += layerSizes[i];
    }
    return total;
}

function deepFreezeDirectArtifact<T>(value: T): Readonly<T> {
    if (typeof value !== 'object' || value === null || Object.isFrozen(value)) return value;
    for (const key of Reflect.ownKeys(value)) {
        deepFreezeDirectArtifact((value as Record<PropertyKey, unknown>)[key]);
    }
    return Object.freeze(value);
}

function directArtifact(
    result: WorkerExperimentResultV2,
    key: keyof WorkerArtifactProvenanceV2,
    expectedBasis: ArtifactProvenance['basis'],
): ArtifactProvenance {
    const raw = result.artifacts?.[key];
    if (raw === undefined) throw new Error(`direct V2 ${key} payload requires provenance`);
    const parsed = parseArtifactProvenance(raw);
    const evaluation = result.evidence.latestEvaluation;
    const sameModel = evaluation !== undefined
        && parsed.model.generationId === evaluation.model.generationId
        && parsed.model.revision === evaluation.model.revision
        && parsed.model.step === evaluation.model.step
        && parsed.model.epoch === evaluation.model.epoch;
    const sameBasis = (() => {
        if (parsed.basis.kind !== expectedBasis.kind) return false;
        switch (expectedBasis.kind) {
            case 'prediction-grid':
                return parsed.basis.kind === 'prediction-grid'
                    && parsed.basis.pointCount === expectedBasis.pointCount
                    && parsed.basis.domain.every(
                        (value, index) => value === expectedBasis.domain[index],
                    );
            case 'bounded-sample':
                return parsed.basis.kind === 'bounded-sample'
                    && parsed.basis.split === expectedBasis.split
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.populationCount === expectedBasis.populationCount;
            case 'full-split':
                return parsed.basis.kind === 'full-split'
                    && parsed.basis.split === expectedBasis.split
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.populationCount === expectedBasis.populationCount;
            case 'parameter-grid':
                return parsed.basis.kind === 'parameter-grid'
                    && parsed.basis.sampleCount === expectedBasis.sampleCount
                    && parsed.basis.parameterPositions === expectedBasis.parameterPositions;
        }
    })();
    if (parsed.model.generationId !== result.runId
        || evaluation === undefined
        || !sameModel
        || parsed.dataset.generatorVersion !== evaluation.dataset.generatorVersion
        || parsed.dataset.datasetKey !== evaluation.dataset.datasetKey
        || parsed.dataset.trainCount !== evaluation.dataset.trainCount
        || parsed.dataset.testCount !== evaluation.dataset.testCount
        || parsed.objectiveKey !== evaluation.objectiveKey
        || !sameBasis) {
        throw new Error(`direct V2 ${key} provenance identity or basis mismatch`);
    }
    return deepFreezeDirectArtifact(parsed) as ArtifactProvenance;
}

function validateEvidenceAgainstPrepared(
    value: unknown,
    prepared: PreparedExperimentDocumentV2,
): WorkerEvidenceMessageV2 {
    const evidence = parseWorkerEvidenceMessageV2(value);
    const sampleCount = prepared.compiled.data.sampleCount;
    const trainCount = sampleCount === 1
        ? 1
        : Math.min(sampleCount - 1, Math.max(
            1,
            Math.floor(sampleCount * prepared.compiled.data.trainFraction),
        ));
    const testCount = sampleCount - trainCount;
    const generatorVersion = getDatasetContract(prepared.compiled.data.dataset).generatorVersion;
    const identities = [
        evidence.liveSignal,
        evidence.latestEvaluation,
        ...Object.values(evidence.artifacts ?? {}),
    ].filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
    if (identities.some((entry) => (
        entry.dataset.generatorVersion !== generatorVersion
        || entry.dataset.datasetKey !== prepared.identities.datasetKey
        || entry.dataset.trainCount !== trainCount
        || entry.dataset.testCount !== testCount
        || entry.objectiveKey !== prepared.identities.objectiveKey
    ))) {
        throw new Error('worker evidence does not match the active prepared experiment');
    }
    const evaluation = evidence.latestEvaluation;
    if (evaluation !== undefined) {
        const validSide = (
            side: PairedEvaluation['train'] | PairedEvaluation['test'],
            count: number,
        ): boolean => {
            const { accuracy, confusionMatrix } = side.values;
            if (prepared.compiled.task.kind === 'regression') {
                return accuracy === undefined && confusionMatrix === undefined;
            }
            if (accuracy === undefined || accuracy < 0 || accuracy > 1
                || confusionMatrix === undefined) return false;
            if (prepared.compiled.task.kind === 'binary-classification') {
                return !('classCount' in confusionMatrix)
                    && confusionMatrix.tp + confusionMatrix.tn
                        + confusionMatrix.fp + confusionMatrix.fn === count
                    && accuracy === (confusionMatrix.tp + confusionMatrix.tn) / count;
            }
            return 'classCount' in confusionMatrix
                && confusionMatrix.counts.reduce((sum, value) => sum + value, 0) === count
                && accuracy === (
                    confusionMatrix.counts[0]
                    + confusionMatrix.counts[4]
                    + confusionMatrix.counts[8]
                ) / count;
        };
        if (!validSide(evaluation.train, trainCount)
            || !validSide(evaluation.test, testCount)) {
            throw new Error('worker evidence task metrics do not match the active prepared task');
        }
    }
    return evidence;
}

function preflightStrictV2Result(
    result: WorkerExperimentResultV2,
    expectedPrepared: PreparedExperimentDocumentV2,
    expectedTrigger: 'initial' | 'manual-step' | 'checkpoint' | 'restore',
): WorkerEvidenceMessageV2 {
    const { snapshot } = result;
    const evidence = validateEvidenceAgainstPrepared(result.evidence, expectedPrepared);
    const evaluation = evidence.latestEvaluation;
    if (result.identities?.canonicalRecipeKey
            !== expectedPrepared.identities.canonicalRecipeKey
        || result.identities?.recipeFingerprint
            !== expectedPrepared.identities.recipeFingerprint
        || result.identities?.datasetKey !== expectedPrepared.identities.datasetKey
        || result.identities?.objectiveKey !== expectedPrepared.identities.objectiveKey) {
        throw new Error('direct V2 result prepared identities do not match the active recipe');
    }
    const expectedTrainCount = Math.min(
        expectedPrepared.compiled.data.sampleCount - 1,
        Math.max(
            1,
            Math.floor(
                expectedPrepared.compiled.data.sampleCount
                * expectedPrepared.compiled.data.trainFraction,
            ),
        ),
    );
    const expectedTestCount = expectedPrepared.compiled.data.sampleCount - expectedTrainCount;
    if (evaluation === undefined
        || result.runId !== evaluation.model.generationId
        || snapshot.step !== evaluation.model.step
        || snapshot.epoch !== evaluation.model.epoch
        || (evidence.liveSignal !== undefined && (
            evidence.liveSignal.model.generationId !== result.runId
            || evidence.liveSignal.model.revision !== evaluation.model.revision
            || evidence.liveSignal.model.step !== evaluation.model.step
            || evidence.liveSignal.model.epoch !== evaluation.model.epoch
        ))
        || evaluation.dataset.datasetKey !== expectedPrepared.identities.datasetKey
        || evaluation.dataset.generatorVersion !== getDatasetContract(
            expectedPrepared.compiled.data.dataset,
        ).generatorVersion
        || evaluation.objectiveKey !== expectedPrepared.identities.objectiveKey
        || evaluation.dataset.trainCount !== expectedTrainCount
        || evaluation.dataset.testCount !== expectedTestCount) {
        throw new Error('direct V2 result requires an exact current evidence pair');
    }
    const checkpointTimeline = parseCheckpointTimelineV2(result.checkpointTimeline);
    if (expectedTrigger === 'initial' && (
        evaluation.evaluationId !== 1
        || evaluation.trigger !== 'initial'
        || evaluation.model.revision !== 0
        || evaluation.model.step !== 0
        || evaluation.model.epoch !== 0
        || evidence.liveSignal !== undefined
    )) {
        throw new Error('fresh direct V2 result requires the initial revision-zero evaluation');
    }
    if (expectedTrigger !== 'initial' && evaluation.trigger !== expectedTrigger) {
        throw new Error(`direct V2 result requires a ${expectedTrigger} evaluation`);
    }
    if (expectedTrigger === 'restore' && evidence.liveSignal !== undefined) {
        throw new Error('restore direct V2 result cannot retain a live training signal');
    }
    if (expectedTrigger === 'initial' && (
        checkpointTimeline.checkpoints.length !== 1
        || checkpointTimeline.checkpoints[0]?.step !== 0
        || checkpointTimeline.liveCheckpointId !== checkpointTimeline.checkpoints[0]?.id
        || checkpointTimeline.restoredCheckpointId !== null
    )) {
        throw new Error('fresh direct V2 result requires one initial checkpoint summary');
    }
    const validateTaskSide = (
        side: PairedEvaluation['train'] | PairedEvaluation['test'],
        expectedCount: number,
    ): boolean => {
        const { accuracy, confusionMatrix } = side.values;
        if (expectedPrepared.compiled.task.kind === 'regression') {
            return accuracy === undefined && confusionMatrix === undefined;
        }
        if (accuracy === undefined || !Number.isFinite(accuracy) || accuracy < 0 || accuracy > 1
            || confusionMatrix === undefined) return false;
        if (expectedPrepared.compiled.task.kind === 'binary-classification') {
            return !('classCount' in confusionMatrix)
                && confusionMatrix.tp + confusionMatrix.tn
                    + confusionMatrix.fp + confusionMatrix.fn === expectedCount
                && accuracy === (confusionMatrix.tp + confusionMatrix.tn) / expectedCount;
        }
        return 'classCount' in confusionMatrix
            && confusionMatrix.classCount === 3
            && confusionMatrix.classLabels.every((label, index) => label === index)
            && confusionMatrix.counts.reduce((sum, count) => sum + count, 0) === expectedCount
            && accuracy === (
                confusionMatrix.counts[0]
                + confusionMatrix.counts[4]
                + confusionMatrix.counts[8]
            ) / expectedCount;
    };
    if (!validateTaskSide(evaluation.train, expectedTrainCount)
        || !validateTaskSide(evaluation.test, expectedTestCount)) {
        throw new Error('direct V2 result task metrics do not match the prepared task');
    }
    if (!Number.isSafeInteger(snapshot.step)
        || snapshot.step < 0
        || !Number.isSafeInteger(snapshot.epoch)
        || snapshot.epoch < 0
        || snapshot.gridSize !== GRID_SIZE
    ) {
        throw new Error('direct V2 result contains invalid snapshot scalars');
    }

    const layerSizes = [
        expectedPrepared.compiled.network.inputSize,
        ...expectedPrepared.compiled.network.hiddenLayers,
        expectedPrepared.compiled.network.outputSize,
    ];
    if (!Array.isArray(snapshot.weights)
        || snapshot.weights.length !== layerSizes.length - 1
        || !Array.isArray(snapshot.biases)
        || snapshot.biases.length !== layerSizes.length - 1) {
        throw new Error('direct V2 result parameter topology mismatch');
    }
    for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex++) {
        const weights = snapshot.weights[layerIndex];
        const biases = snapshot.biases[layerIndex];
        if (!Array.isArray(weights)
            || weights.length !== layerSizes[layerIndex + 1]
            || !Array.isArray(biases)
            || biases.length !== layerSizes[layerIndex + 1]) {
            throw new Error('direct V2 result parameter topology mismatch');
        }
        for (const row of weights) {
            if (!Array.isArray(row)
                || row.length !== layerSizes[layerIndex]
                || row.some((value) => (
                    !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
                ))) {
                throw new Error('direct V2 result contains malformed or non-finite weights');
            }
        }
        if (biases.some((value) => (
            !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
        ))) {
            throw new Error('direct V2 result contains non-finite biases');
        }
    }

    const gridPointCount = snapshot.gridSize * snapshot.gridSize;
    const outputGrid = snapshot.outputGrid;
    if (!(outputGrid instanceof Float32Array) && !Array.isArray(outputGrid)) {
        throw new Error('direct V2 result output grid has an invalid representation');
    }
    if ((outputGrid.length !== 0 && outputGrid.length !== gridPointCount)
        || Array.from(outputGrid).some((value) => (
            !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
        ))) {
        throw new Error('direct V2 result output grid has an invalid size or value');
    }
    if (snapshot.multiclassBoundary !== undefined) {
        const boundary = snapshot.multiclassBoundary;
        if (outputGrid.length > 0
            || boundary.gridSize !== snapshot.gridSize
            || !(boundary.classGrid instanceof Uint8Array)
            || !(boundary.confidenceGrid instanceof Float32Array)
            || boundary.classGrid.length !== gridPointCount
            || boundary.confidenceGrid.length !== gridPointCount
            || boundary.classGrid.some((value) => value > 2)
            || boundary.confidenceGrid.some((value) => (
                !Number.isFinite(value) || value < 0 || value > 1
            ))) {
            throw new Error('direct V2 result multiclass boundary is malformed');
        }
    }
    if (snapshot.neuronGrids !== undefined) {
        const expectedNeuronValues = layerSizes.slice(1).reduce((sum, size) => sum + size, 0)
            * gridPointCount;
        const validFlat = snapshot.neuronGrids instanceof Float32Array
            && snapshot.neuronGrids.length === expectedNeuronValues
            && !snapshot.neuronGrids.some((value) => !Number.isFinite(value));
        const validNested = Array.isArray(snapshot.neuronGrids)
            && snapshot.neuronGrids.length === expectedNeuronValues / gridPointCount
            && snapshot.neuronGrids.every((grid) => (
                (Array.isArray(grid) || grid instanceof Float32Array)
                && grid.length === gridPointCount
                && !Array.from(grid).some((value) => (
                    !Number.isFinite(value) || !Number.isFinite(Math.fround(value))
                ))
            ));
        if (!validFlat && !validNested) {
            throw new Error('direct V2 result neuron grids have an invalid size or value');
        }
    }
    if (snapshot.layerStats === undefined && result.layerStatsGradientRevision !== undefined) {
        throw new Error('direct V2 result has a surplus layer statistics revision');
    }
    if (snapshot.layerStats !== undefined && (
        snapshot.layerStats.length !== layerSizes.length - 1
        || snapshot.layerStats.some((stats) => (
            Object.keys(stats).sort().join(',')
                !== 'activationStd,meanAbsGradient,meanAbsWeight,meanActivation'
            || Object.values(stats).some((value) => !Number.isFinite(value))
            || stats.activationStd < 0
            || stats.meanAbsWeight < 0
            || stats.meanAbsGradient < 0
        ))
    )) {
        throw new Error('direct V2 result layer statistics are malformed');
    }
    if (snapshot.activationHistograms !== undefined) {
        const histogram = snapshot.activationHistograms;
        const histogramSampleCount = Math.min(128, expectedTrainCount);
        const validLayers = histogram.layers.length === layerSizes.length - 1
            && histogram.layers.every((layer, index) => (
                layer.layerIndex === index
                && layer.binCount === 12
                && Number.isFinite(layer.binStart)
                && Number.isFinite(layer.binWidth)
                && layer.binWidth > 0
                && Number.isFinite(layer.minActivation)
                && Number.isFinite(layer.maxActivation)
                && layer.minActivation <= layer.maxActivation
                && layer.totalCount === histogramSampleCount * layerSizes[index + 1]
                && Number.isSafeInteger(layer.nearZeroCount)
                && layer.nearZeroCount >= 0
                && layer.nearZeroCount <= layer.totalCount
                && Number.isSafeInteger(layer.saturatedCount)
                && layer.saturatedCount >= 0
                && layer.saturatedCount <= layer.totalCount
                && Array.from(
                    histogram.bins.subarray(index * 12, (index + 1) * 12),
                ).reduce((sum, count) => sum + count, 0) === layer.totalCount
            ));
        if (!(histogram.bins instanceof Float32Array)
            || histogram.bins.length !== histogram.layers.length * 12
            || histogram.bins.some((value) => (
                !Number.isSafeInteger(value) || value < 0
            ))
            || !validLayers) {
            throw new Error('direct V2 result activation histogram is malformed');
        }
    }
    return evidence;
}

function buildStrictV2FramePatch(
    result: WorkerExperimentResultV2,
    replaceAbsentArtifacts: boolean,
    expectedPrepared: PreparedExperimentDocumentV2,
    expectedTrigger: 'initial' | 'manual-step' | 'checkpoint' | 'restore' = (
        replaceAbsentArtifacts ? 'initial' : 'manual-step'
    ),
): FrameBufferPatch {
    const { snapshot } = result;
    preflightStrictV2Result(result, expectedPrepared, expectedTrigger);
    const currentFrame = getFrameBuffer();
    const { buffer: weights, layerSizes } = flattenWeights(snapshot.weights);
    const biases = flattenBiases(snapshot.biases);
    const patch: Parameters<typeof updateFrameBuffer>[0] = {
        weights,
        biases,
        weightLayout: { layerSizes },
    };

    const hasScalarBoundary = snapshot.outputGrid.length > 0;
    const hasMulticlassBoundary = snapshot.multiclassBoundary !== undefined;
    if (hasScalarBoundary && hasMulticlassBoundary) {
        throw new Error('direct V2 snapshot cannot contain scalar and multiclass boundaries');
    }
    const hasCurrentBoundary = currentFrame.outputGrid !== null
        || currentFrame.multiclassClassGrid !== null
        || currentFrame.multiclassConfidenceGrid !== null
        || currentFrame.multiclassBoundaryLayout !== null
        || currentFrame.decisionBoundaryProvenance !== null;
    const canReuseCurrentBoundary = (
        currentFrame.outputGrid !== null
        || currentFrame.multiclassClassGrid !== null
        || currentFrame.multiclassConfidenceGrid !== null
    ) && currentFrame.decisionBoundaryProvenance?.model.generationId === result.runId;
    const clearCurrentBoundary = !hasScalarBoundary
        && !hasMulticlassBoundary
        && (replaceAbsentArtifacts || (hasCurrentBoundary && !canReuseCurrentBoundary));
    if (hasScalarBoundary || hasMulticlassBoundary || clearCurrentBoundary) {
        patch.outputGrid = hasScalarBoundary
            ? snapshot.outputGrid instanceof Float32Array
                ? snapshot.outputGrid
                : new Float32Array(snapshot.outputGrid)
            : null;
        patch.gridSize = hasScalarBoundary || hasMulticlassBoundary ? snapshot.gridSize : 0;
        patch.multiclassClassGrid = snapshot.multiclassBoundary?.classGrid ?? null;
        patch.multiclassConfidenceGrid = snapshot.multiclassBoundary?.confidenceGrid ?? null;
        patch.multiclassBoundaryLayout = snapshot.multiclassBoundary === undefined
            ? null
            : {
                gridSize: snapshot.multiclassBoundary.gridSize,
                classCount: 3,
                classLabels: [0, 1, 2],
            };
        patch.decisionBoundaryProvenance = hasScalarBoundary || hasMulticlassBoundary
            ? directArtifact(result, 'decisionBoundary', {
                kind: 'prediction-grid',
                pointCount: snapshot.gridSize * snapshot.gridSize,
                domain: [-1, 1, -1, 1],
            })
            : null;
    }

    let neuronGrids: Float32Array | null = null;
    let neuronGridLayout: { count: number; gridSize: number } | null = null;
    if (snapshot.neuronGrids !== undefined && snapshot.neuronGrids.length > 0) {
        if (snapshot.neuronGrids instanceof Float32Array) {
            neuronGrids = snapshot.neuronGrids;
            neuronGridLayout = { count: getTotalNeuronCount(layerSizes), gridSize: snapshot.gridSize };
        } else {
            const flattened = flattenNeuronGrids(snapshot.neuronGrids, snapshot.gridSize);
            neuronGrids = flattened.buffer;
            neuronGridLayout = flattened.layout;
        }
    }
    const hasCurrentNeuronGrids = currentFrame.neuronGrids !== null
        || currentFrame.neuronGridLayout !== null
        || currentFrame.neuronGridsProvenance !== null;
    const canReuseCurrentNeuronGrids = currentFrame.neuronGrids !== null
        && currentFrame.neuronGridLayout !== null
        && currentFrame.neuronGridsProvenance?.model.generationId === result.runId;
    const clearCurrentNeuronGrids = neuronGrids === null
        && (replaceAbsentArtifacts || (hasCurrentNeuronGrids && !canReuseCurrentNeuronGrids));
    if (neuronGrids !== null || clearCurrentNeuronGrids) {
        patch.neuronGrids = neuronGrids;
        patch.neuronGridLayout = neuronGridLayout;
        patch.neuronGridsProvenance = neuronGrids === null
            ? null
            : directArtifact(result, 'neuronGrids', {
                kind: 'prediction-grid',
                pointCount: snapshot.gridSize * snapshot.gridSize,
                domain: [-1, 1, -1, 1],
            });
    }

    if (snapshot.layerStats !== undefined) {
        patch.layerStats = snapshot.layerStats;
        const populationCount = result.evidence.latestEvaluation!.dataset.trainCount;
        const provenance = directArtifact(result, 'activationStatistics', {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, populationCount),
            populationCount,
        });
        const gradientRevision = result.layerStatsGradientRevision;
        if (!Number.isSafeInteger(gradientRevision)
            || (gradientRevision as number) < 0
            || (gradientRevision as number) > provenance.model.revision) {
            throw new Error('direct V2 layer statistics require a valid gradient revision');
        }
        patch.layerStatsProvenance = provenance;
        patch.layerStatsGradientRevision = gradientRevision!;
    } else {
        const hasCurrentLayerStats = currentFrame.layerStats !== null
            || currentFrame.layerStatsProvenance !== null
            || currentFrame.layerStatsGradientRevision !== null;
        const canReuseCurrentLayerStats = currentFrame.layerStats !== null
            && currentFrame.layerStatsProvenance?.model.generationId === result.runId
            && Number.isSafeInteger(currentFrame.layerStatsGradientRevision)
            && (currentFrame.layerStatsGradientRevision as number) >= 0;
        if (replaceAbsentArtifacts || (hasCurrentLayerStats && !canReuseCurrentLayerStats)) {
            patch.layerStats = null;
            patch.layerStatsProvenance = null;
            patch.layerStatsGradientRevision = null;
        }
    }

    if (snapshot.activationHistograms !== undefined) {
        patch.activationHistogramBins = snapshot.activationHistograms.bins;
        patch.activationHistogramLayout = {
            binCount: snapshot.activationHistograms.layers[0]?.binCount ?? 0,
            layers: snapshot.activationHistograms.layers,
        };
        patch.activationHistogramProvenance = directArtifact(result, 'activationHistogram', {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(
                128,
                result.evidence.latestEvaluation!.dataset.trainCount,
            ),
            populationCount: result.evidence.latestEvaluation!.dataset.trainCount,
        });
    } else {
        const hasCurrentHistogram = currentFrame.activationHistogramBins !== null
            || currentFrame.activationHistogramLayout !== null
            || currentFrame.activationHistogramProvenance !== null;
        const canReuseCurrentHistogram = currentFrame.activationHistogramBins !== null
            && currentFrame.activationHistogramLayout !== null
            && currentFrame.activationHistogramProvenance?.model.generationId === result.runId;
        if (replaceAbsentArtifacts || (hasCurrentHistogram && !canReuseCurrentHistogram)) {
            patch.activationHistogramBins = null;
            patch.activationHistogramLayout = null;
            patch.activationHistogramProvenance = null;
        }
    }

    const binaryConfusion = snapshot.testMetrics.confusionMatrix;
    const multiclassConfusion = snapshot.testMetrics.multiclassConfusionMatrix;
    if (binaryConfusion !== undefined && multiclassConfusion !== undefined) {
        throw new Error('direct V2 snapshot cannot contain two confusion matrix kinds');
    }
    if (binaryConfusion !== undefined || multiclassConfusion !== undefined) {
        const pairedConfusion = result.evidence.latestEvaluation?.test.values.confusionMatrix;
        const binaryMatches = binaryConfusion !== undefined
            && pairedConfusion !== undefined
            && !('classCount' in pairedConfusion)
            && binaryConfusion.tp === pairedConfusion.tp
            && binaryConfusion.tn === pairedConfusion.tn
            && binaryConfusion.fp === pairedConfusion.fp
            && binaryConfusion.fn === pairedConfusion.fn;
        const multiclassMatches = multiclassConfusion !== undefined
            && pairedConfusion !== undefined
            && 'classCount' in pairedConfusion
            && multiclassConfusion.classCount === pairedConfusion.classCount
            && multiclassConfusion.classLabels.every(
                (label, index) => label === pairedConfusion.classLabels[index],
            )
            && multiclassConfusion.counts.length === pairedConfusion.counts.length
            && multiclassConfusion.counts.every(
                (count, index) => count === pairedConfusion.counts[index],
            );
        if (!binaryMatches && !multiclassMatches) {
            throw new Error('direct V2 confusion matrix does not match latest paired evaluation');
        }
        patch.confusionMatrix = binaryMatches ? pairedConfusion : null;
        patch.multiclassConfusionMatrix = multiclassMatches ? pairedConfusion : null;
        patch.confusionMatrixProvenance = directArtifact(result, 'confusionMatrix', {
            kind: 'full-split',
            split: 'test',
            sampleCount: result.evidence.latestEvaluation!.dataset.testCount,
            populationCount: result.evidence.latestEvaluation!.dataset.testCount,
        });
        patch.confusionMatrixEvaluationId = result.evidence.latestEvaluation!.evaluationId;
    } else {
        const hasCurrentConfusion = currentFrame.confusionMatrix !== null
            || currentFrame.multiclassConfusionMatrix !== null
            || currentFrame.confusionMatrixProvenance !== null
            || currentFrame.confusionMatrixEvaluationId !== null;
        const canReuseCurrentConfusion = (
            currentFrame.confusionMatrix !== null
            || currentFrame.multiclassConfusionMatrix !== null
        ) && currentFrame.confusionMatrixProvenance?.model.generationId === result.runId;
        if (replaceAbsentArtifacts || (hasCurrentConfusion && !canReuseCurrentConfusion)) {
            patch.confusionMatrix = null;
            patch.multiclassConfusionMatrix = null;
            patch.confusionMatrixProvenance = null;
            patch.confusionMatrixEvaluationId = null;
        }
    }

    const expectedKeys = new Set<keyof WorkerArtifactProvenanceV2>();
    if (hasScalarBoundary || hasMulticlassBoundary) expectedKeys.add('decisionBoundary');
    if (neuronGrids !== null) expectedKeys.add('neuronGrids');
    if (snapshot.layerStats !== undefined) expectedKeys.add('activationStatistics');
    if (snapshot.activationHistograms !== undefined) expectedKeys.add('activationHistogram');
    if (binaryConfusion !== undefined || multiclassConfusion !== undefined) {
        expectedKeys.add('confusionMatrix');
    }
    for (const key of Object.keys(result.artifacts ?? {}) as Array<keyof WorkerArtifactProvenanceV2>) {
        if (!expectedKeys.has(key)) throw new Error(`direct V2 ${key} provenance has no payload`);
    }

    validateFrameBufferPatch(patch, { requireArtifactProvenance: true });
    return patch;
}

function applyFreshV2SnapshotToStore(
    ts: TrainingStore,
    _result: WorkerExperimentResultV2,
    frameVersions: FrameVersions,
): void {
    ts.setFrameVersions(frameVersions);
}

const EMPTY_CHECKPOINT_TIMELINE: CheckpointTimeline = {
    checkpoints: [],
    maxCheckpoints: 8,
    evictedCount: 0,
    liveCheckpointId: null,
    restoredCheckpointId: null,
};

export function useTraining(): TrainingHook {
    // All refs first (stable hook order)
    const mountedRef = useRef(true);
    const initializedRef = useRef(false);
    const activePreparedRef = useRef<PreparedExperimentDocumentV2 | null>(null);
    const prevPreparedRef = useRef<PreparedExperimentDocumentV2 | null>(null);
    const prevConfigSyncNonceRef = useRef(0);
    const requestIdRef = useRef(0);
    const activeRequestIdRef = useRef<number | null>(null);
    const pendingPreparationRequestIdRef = useRef<number | null>(null);
    const rejectedPreparationRequestIdsRef = useRef(new Set<number>());
    const lifecycleEpochRef = useRef(0);
    const streamSetupPromiseRef = useRef<Promise<void> | null>(null);
    const initializationPromiseRef = useRef<Promise<boolean> | null>(null);
    const mutationPausePromiseRef = useRef<Promise<void> | null>(null);
    const mutationPauseResolveRef = useRef<(() => void) | null>(null);
    const mutationPauseRejectRef = useRef<((error: Error) => void) | null>(null);
    const manualActionPendingRef = useRef(false);
    const stepsPerFrameRef = useRef(5);
    const isPlayingRef = useRef(false);
    const configSyncSeqRef = useRef(0);
    const activeConfigSyncSeqRef = useRef(0);
    const configSyncPendingRef = useRef(false);
    const restoreBarrierRef = useRef<Promise<void> | null>(null);

    // Config selectors (from playground store — stable, rarely changes)
    const prepared = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared : null
    ));
    const demand = usePlaygroundStore((s) => s.demand);
    const webgpuGrid = usePlaygroundStore((s) => s.featuresUI.webgpuGrid);

    // Runtime selectors (from training store — volatile)
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const configSyncNonce = useTrainingStore((s) => s.configSyncNonce);

    const reportWorkerError = useCallback((error: unknown, fallback: string) => {
        if (!mountedRef.current) return;
        const message = getErrorMessage(error, fallback);
        mutationPauseRejectRef.current?.(new Error(message));
        mutationPausePromiseRef.current = null;
        mutationPauseResolveRef.current = null;
        mutationPauseRejectRef.current = null;
        const ts = useTrainingStore.getState();
        isPlayingRef.current = false;
        stopRenderLoop();
        ts.setWorkerError(message);
        ts.setPauseReason('error');
        ts.setStatus('paused');
        initializedRef.current = false;
    }, []);

    const beginConfigSync = useCallback(() => {
        const seq = configSyncSeqRef.current + 1;
        configSyncSeqRef.current = seq;
        activeConfigSyncSeqRef.current = seq;
        configSyncPendingRef.current = true;
        return seq;
    }, []);

    const isCurrentConfigSync = useCallback((seq: number) => (
        configSyncPendingRef.current && activeConfigSyncSeqRef.current === seq
    ), []);

    const finishConfigSyncIfCurrent = useCallback((seq: number) => {
        if (activeConfigSyncSeqRef.current === seq) {
            configSyncPendingRef.current = false;
        }
    }, []);

    const nextRequest = useCallback((nextPrepared: PreparedExperimentDocumentV2) => {
        const requestId = requestIdRef.current + 1;
        requestIdRef.current = requestId;
        activeRequestIdRef.current = requestId;
        pendingPreparationRequestIdRef.current = requestId;
        return createWorkerExperimentRequestV2(nextPrepared, requestId);
    }, []);

    const ensureStreamChannel = useCallback((): Promise<void> => {
        if (!streamSetupPromiseRef.current) {
            const promise = setupStreamChannel().catch((error) => {
                if (streamSetupPromiseRef.current === promise) {
                    streamSetupPromiseRef.current = null;
                }
                throw error;
            });
            streamSetupPromiseRef.current = promise;
        }
        return streamSetupPromiseRef.current;
    }, []);

    const pauseForMutation = useCallback((): Promise<void> => {
        if (mutationPausePromiseRef.current) return mutationPausePromiseRef.current;
        if (!isPlayingRef.current) return Promise.resolve();
        isPlayingRef.current = false;
        stopRenderLoop();
        let rejectPause!: (error: Error) => void;
        const promise = new Promise<void>((resolve, reject) => {
            mutationPauseResolveRef.current = resolve;
            mutationPauseRejectRef.current = reject;
            rejectPause = reject;
        });
        mutationPausePromiseRef.current = promise;
        try {
            postStreamCommand({ type: 'stopTraining', protocolVersion: WORKER_PROTOCOL_VERSION });
        } catch (error) {
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            rejectPause(error instanceof Error ? error : new Error(String(error)));
        }
        return promise;
    }, []);

    const applyFreshV2Run = useCallback((
        result: Awaited<ReturnType<ReturnType<typeof getWorkerApi>['initializeExperimentV2']>>,
        owner: PreparedExperimentDocumentV2,
    ) => {
        // Preflight both strict boundaries before changing the active bridge
        // generation or scientific store. A forged direct artifact must leave
        // the entire previously accepted run observable and intact.
        const evidence = parseWorkerEvidenceMessageV2(result.evidence);
        const framePatch = buildStrictV2FramePatch(
            result,
            true,
            owner,
        );
        const ts = useTrainingStore.getState();
        const evidenceReplacement = ts.prepareEvidenceReplacement(evidence);
        newRunTo(result.runId);
        ts.commitEvidenceReplacement(evidenceReplacement);
        updateFrameBuffer(framePatch, { requireArtifactProvenance: true });
        const frameVersions = getFrameVersions();
        applyFreshV2SnapshotToStore(ts, result, frameVersions);
        activePreparedRef.current = owner;
        ts.setCheckpointTimeline(parseCheckpointTimelineV2(result.checkpointTimeline));
        ts.clearWorkerError();
        ts.clearPauseReason();
    }, []);

    const publishCommittedV2Run = useCallback((
        result: Awaited<ReturnType<ReturnType<typeof getWorkerApi>['initializeExperimentV2']>>,
        owner: PreparedExperimentDocumentV2,
        source: TrainedRecipeSource,
    ): boolean => {
        const ts = useTrainingStore.getState();
        if (result.runId <= (ts.evidenceGenerationId ?? 0)) {
            throw new Error('fresh V2 result must advance the active generation monotonically');
        }
        applyFreshV2Run(result, owner);
        ts.setTrainPoints([]);
        ts.setTestPoints([]);
        ts.markTrainedRecipe(
            owner.document.recipe,
            source,
            owner.identities.recipeFingerprint,
        );
        ts.setStatus('idle');
        initializedRef.current = false;
        return true;
    }, [applyFreshV2Run]);

    const initializePrepared = useCallback(async (
        requestedPrepared: PreparedExperimentDocumentV2 = requirePreparedExperiment(),
    ): Promise<boolean> => {
        const lifecycleEpoch = lifecycleEpochRef.current;
        prevPreparedRef.current = requestedPrepared;
        const request = nextRequest(requestedPrepared);
        const api = getWorkerApi();
        let result;
        try {
            await ensureStreamChannel();
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || activeRequestIdRef.current !== request.requestId) {
                return false;
            }
            result = await api.initializeExperimentV2(request);
        } catch (error) {
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || activeRequestIdRef.current !== request.requestId) {
                return false;
            }
            pendingPreparationRequestIdRef.current = null;
            rejectedPreparationRequestIdsRef.current.delete(request.requestId);
            throw error;
        }
        if (!mountedRef.current || lifecycleEpochRef.current !== lifecycleEpoch) return false;
        if (rejectedPreparationRequestIdsRef.current.delete(request.requestId)) return false;
        publishCommittedV2Run(result, requestedPrepared, 'initialize');
        if (activeRequestIdRef.current !== request.requestId) return false;
        pendingPreparationRequestIdRef.current = null;
        if (currentPreparedExperiment() !== requestedPrepared) return false;
        const ts = useTrainingStore.getState();

        // Hydration does not own generation identity. A failure below reports
        // a runtime error, but never rolls the committed run back to stale UI.
        const trainPts = await api.getTrainPointsV2();
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        const testPts = await api.getTestPointsV2();
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        const latestState = usePlaygroundStore.getState();
        if (latestState.access.status !== 'ready'
            || latestState.access.prepared !== requestedPrepared) return false;

        // Send initial demand
        await api.updateDemand(latestState.demand);
        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;

        // AS-4: tell the worker whether the user has opted in to the
        // WebGPU grid path. Capability detection still gates this; the
        // worker silently falls back to CPU when the device isn't
        // available or the network shape exceeds the shader caps.
        try {
            await api.setWebGpuEnabled(usePlaygroundStore.getState().featuresUI.webgpuGrid);
        } catch {
            // Older worker bundles won't expose setWebGpuEnabled — ignore.
        }

        if (!mountedRef.current || activeRequestIdRef.current !== request.requestId) return false;
        if (currentPreparedExperiment() !== requestedPrepared) return false;
        ts.setTrainPoints(trainPts);
        ts.setTestPoints(testPts);
        initializedRef.current = true;
        return true;
    }, [ensureStreamChannel, nextRequest, publishCommittedV2Run]);

    const initializeWorker = useCallback((
        requestedPrepared: PreparedExperimentDocumentV2 = requirePreparedExperiment(),
    ): Promise<boolean> => {
        const inFlight = initializationPromiseRef.current;
        if (inFlight && prevPreparedRef.current === requestedPrepared) return inFlight;
        const promise = initializePrepared(requestedPrepared);
        initializationPromiseRef.current = promise;
        const clear = () => {
            if (initializationPromiseRef.current === promise) {
                initializationPromiseRef.current = null;
            }
        };
        void promise.then(clear, clear);
        return promise;
    }, [initializePrepared]);

    // Keep ref in sync so streaming commands use current speed.
    useEffect(() => {
        stepsPerFrameRef.current = stepsPerFrame;
        // If currently playing, update the worker's speed
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateSpeed', protocolVersion: WORKER_PROTOCOL_VERSION, stepsPerFrame });
        }
    }, [stepsPerFrame]);

    // ── Snapshot handler: applies streamed snapshots to training store ──
    useEffect(() => {
        const unsubscribe = onSnapshot((msg: WorkerToMainMessage) => {
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();

            if (msg.type === 'evidence') {
                try {
                    const activePrepared = activePreparedRef.current;
                    if (activePrepared === null) {
                        throw new Error('scientific evidence arrived before an active prepared run');
                    }
                    ts.applyEvidence(validateEvidenceAgainstPrepared(msg, activePrepared));
                } catch (error) {
                    reportWorkerError(error, 'Received invalid scientific evidence from the worker.');
                }
            } else if (msg.type === 'worker-error') {
                const error = msg as WorkerProtocolErrorMessageV2;
                if (error.generationId === null) {
                    if (error.requestId === null
                        || error.requestId !== pendingPreparationRequestIdRef.current) return;
                    pendingPreparationRequestIdRef.current = null;
                    rejectedPreparationRequestIdsRef.current.add(error.requestId);
                    activeRequestIdRef.current = null;
                    if (configSyncPendingRef.current) {
                        configSyncPendingRef.current = false;
                        ts.failConfigChange(error.message);
                        return;
                    }
                    reportWorkerError(error.message, 'Failed to prepare the experiment.');
                    return;
                }
                if (error.generationId !== ts.evidenceGenerationId) return;
                reportWorkerError(error.message, 'The training worker failed.');
            } else if (msg.type === 'snapshot') {
                try {
                    const checkpointTimeline = parseCheckpointTimelineV2(msg.checkpointTimeline);
                    const frameVersions = getFrameVersions();
                    ts.applyStreamedFrame({
                        frameVersions,
                        checkpointTimeline,
                    });
                } catch (error) {
                    reportWorkerError(error, 'Received invalid checkpoint metadata from the worker.');
                }
            } else if (msg.type === 'status') {
                if (msg.status === 'paused') {
                    const resolveMutationPause = mutationPauseResolveRef.current;
                    if (resolveMutationPause) {
                        mutationPausePromiseRef.current = null;
                        mutationPauseResolveRef.current = null;
                        mutationPauseRejectRef.current = null;
                        resolveMutationPause();
                    }
                    if (msg.pauseReason) {
                        isPlayingRef.current = false;
                        stopRenderLoop();
                        ts.setPauseReason(msg.pauseReason);
                    }
                    ts.setStatus('paused');
                } else if (msg.status === 'idle') {
                    isPlayingRef.current = false;
                    stopRenderLoop();
                    ts.clearPauseReason();
                    ts.setStatus(msg.status);
                } else if (msg.status === 'running') {
                    ts.clearPauseReason();
                    ts.setStatus('running');
                }
            } else if (msg.type === 'error') {
                reportWorkerError(msg.message, 'The training worker failed.');
            }
        });

        return unsubscribe;
    }, [reportWorkerError]);

    // Initialize worker on mount
    useEffect(() => {
        lifecycleEpochRef.current++;
        mountedRef.current = true;
        if (!prepared) {
            reportWorkerError(
                new Error('Training is unavailable because the shared experiment URL is incompatible with version 2.'),
                'Failed to initialize training worker.',
            );
            return;
        }
        initializeWorker(prepared).catch((error) => {
            reportWorkerError(error, 'Failed to initialize training worker.');
        });
        // Mount initialization is intentionally one-shot. Later prepared
        // documents are handled by the ordered config transaction below.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Every accepted prepared-document replacement starts a fresh generation.
    useEffect(() => {
        const isRetry = configSyncNonce !== prevConfigSyncNonceRef.current;
        if (!prepared) return;
        if (!isRetry && prevPreparedRef.current === prepared) return;
        const previousPrepared = prevPreparedRef.current;
        prevPreparedRef.current = prepared;
        prevConfigSyncNonceRef.current = configSyncNonce;
        const seq = beginConfigSync();
        const request = nextRequest(prepared);
        const lifecycleEpoch = lifecycleEpochRef.current;

        const sync = async () => {
            const restoreBarrier = restoreBarrierRef.current;
            if (restoreBarrier) {
                await restoreBarrier;
                if (!isCurrentConfigSync(seq)) return;
            }
            // Serialize MessagePort pause with the Comlink mutation. The
            // paused status is the acknowledgement that the old loop and its
            // forced evaluation have completed.
            try {
                await pauseForMutation();
            } catch (error) {
                if (mountedRef.current && isCurrentConfigSync(seq)) {
                    useTrainingStore.getState().failConfigChange(
                        getErrorMessage(error, 'Failed to pause the current experiment.'),
                    );
                    finishConfigSyncIfCurrent(seq);
                }
                return;
            }
            if (!mountedRef.current
                || lifecycleEpochRef.current !== lifecycleEpoch
                || !isCurrentConfigSync(seq)) return;

            const api = getWorkerApi();
            const ts = useTrainingStore.getState();
            let result: Awaited<ReturnType<typeof api.initializeExperimentV2>>;
            try {
                await ensureStreamChannel();
                if (!mountedRef.current
                    || lifecycleEpochRef.current !== lifecycleEpoch
                    || !isCurrentConfigSync(seq)
                    || activeRequestIdRef.current !== request.requestId) return;
                result = await api.initializeExperimentV2(request);
                if (!mountedRef.current || lifecycleEpochRef.current !== lifecycleEpoch) return;
                if (rejectedPreparationRequestIdsRef.current.delete(request.requestId)) return;
                publishCommittedV2Run(result, prepared, 'config-sync');
                if (!isCurrentConfigSync(seq)
                    || activeRequestIdRef.current !== request.requestId) return;
                pendingPreparationRequestIdRef.current = null;
                const committed = usePlaygroundStore.getState();
                if (committed.access.status !== 'ready'
                    || committed.access.prepared !== prepared) return;
            } catch (error) {
                if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
                if (pendingPreparationRequestIdRef.current === request.requestId) {
                    pendingPreparationRequestIdRef.current = null;
                }
                rejectedPreparationRequestIdsRef.current.delete(request.requestId);
                prevPreparedRef.current = previousPrepared;
                ts.failConfigChange(error instanceof Error ? error.message : 'Failed to update configuration');
                finishConfigSyncIfCurrent(seq);
                return;
            }

            // The worker/app identity commit above cannot be rolled back.
            // Auxiliary failures become runtime errors, never config rollback.
            let auxiliaryError: unknown = null;
            let trainPts: DataPoint[] = [];
            let testPts: DataPoint[] = [];
            try {
                usePlaygroundStore.getState().syncToUrl();
            } catch (error) {
                auxiliaryError = error;
            }
            try {
                trainPts = await api.getTrainPointsV2();
            } catch (error) {
                auxiliaryError = error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            try {
                testPts = await api.getTestPointsV2();
            } catch (error) {
                auxiliaryError ??= error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            const latest = usePlaygroundStore.getState();
            if (latest.access.status !== 'ready'
                || latest.access.prepared !== prepared) return;
            try {
                await api.updateDemand(latest.demand);
            } catch (error) {
                auxiliaryError ??= error;
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            try {
                await api.setWebGpuEnabled(latest.featuresUI.webgpuGrid);
            } catch {
                // Capability and older-bundle fallbacks remain non-fatal.
            }
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;

            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);
            initializedRef.current = true;
            ts.finishConfigChange();
            finishConfigSyncIfCurrent(seq);
            if (auxiliaryError !== null) {
                reportWorkerError(auxiliaryError, 'Failed to hydrate the new experiment runtime.');
            }
        };
        void sync();
    }, [
        prepared,
        configSyncNonce,
        beginConfigSync,
        finishConfigSyncIfCurrent,
        ensureStreamChannel,
        isCurrentConfigSync,
        nextRequest,
        pauseForMutation,
        publishCommittedV2Run,
        reportWorkerError,
    ]);

    // Sync demand changes to worker
    useEffect(() => {
        if (!initializedRef.current) return;
        if (isPlayingRef.current) {
            postStreamCommand({ type: 'updateDemand', protocolVersion: WORKER_PROTOCOL_VERSION, demand });
            return;
        }
        void getWorkerApi().updateDemand(demand).catch((error: unknown) => {
            reportWorkerError(error, 'Failed to update visualization demand.');
        });
    }, [demand, reportWorkerError]);

    // AS-4: live-toggle the WebGPU grid path when the user flips the
    // featuresUI flag. Disabling immediately disposes the GPU predictor
    // (frees device memory); enabling lets the next snapshot lazily
    // re-allocate.
    useEffect(() => {
        if (!initializedRef.current) return;
        const api = getWorkerApi();
        api.setWebGpuEnabled(webgpuGrid).catch(() => {
            // Ignore — capability detection inside the worker handles
            // any per-device fallback. A toggle that doesn't reach the
            // worker just means the next snapshot still uses whatever
            // path the worker last knew about.
        });
    }, [webgpuGrid]);

    const play = useCallback(() => {
        if (configSyncPendingRef.current
            || mutationPausePromiseRef.current !== null
            || manualActionPendingRef.current
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }

        const startTraining = () => {
            isPlayingRef.current = true;
            const ts = useTrainingStore.getState();
            ts.clearPauseReason();
            ts.setStatus('running');
            startRenderLoop();
            postStreamCommand({
                type: 'startTraining',
                protocolVersion: WORKER_PROTOCOL_VERSION,
                stepsPerFrame: stepsPerFrameRef.current,
            });
        };

        if (!initializedRef.current) {
            initializeWorker().catch((error) => {
                reportWorkerError(error, 'Failed to initialize training worker.');
            }).then(() => {
                if (initializedRef.current) {
                    startTraining();
                }
            });
            return;
        }
        startTraining();
    }, [initializeWorker, reportWorkerError]);

    const pause = useCallback(() => {
        if (!isPlayingRef.current && useTrainingStore.getState().status !== 'running') {
            return;
        }
        const ts = useTrainingStore.getState();
        ts.setPauseReason('manual');
        ts.setStatus('paused');
        void pauseForMutation().catch((error) => {
            reportWorkerError(error, 'Failed to pause training.');
        });
    }, [pauseForMutation, reportWorkerError]);

    const step = useCallback(async () => {
        if (configSyncPendingRef.current || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        if (manualActionPendingRef.current) return;
        manualActionPendingRef.current = true;
        const wasPlaying = isPlayingRef.current;
        try {
            await pauseForMutation();
            if (configSyncPendingRef.current
                || useTrainingStore.getState().pendingConfigSource !== null) return;
            if (wasPlaying) {
                useTrainingStore.getState().setPauseReason('manual');
                useTrainingStore.getState().setStatus('paused');
            }
            if (!initializedRef.current) {
                await initializeWorker();
                if (!initializedRef.current) return;
            }
            const api = getWorkerApi();
            const result = await api.stepExperimentV2(1);
            if (!mountedRef.current) return;
            const ts = useTrainingStore.getState();
            if (configSyncPendingRef.current
                || ts.pendingConfigSource !== null
                || result.runId !== ts.evidenceGenerationId) return;
            const resultRevision = result.evidence.liveSignal?.model.revision
                ?? result.evidence.latestEvaluation?.model.revision;
            const currentRevision = ts.latestLiveSignal?.model.revision
                ?? ts.latestEvaluation?.model.revision;
            if (resultRevision !== undefined
                && currentRevision !== undefined
                && resultRevision < currentRevision) return;
            const expectedPrepared = activePreparedRef.current;
            if (expectedPrepared === null) {
                throw new Error('direct V2 step requires an active prepared experiment');
            }
            const framePatch = buildStrictV2FramePatch(
                result,
                false,
                expectedPrepared,
            );
            const directEvidence = evidenceForDirectStep(
                result.evidence,
                ts.latestEvaluation,
            );
            const preparedEvidence = directEvidence === null
                ? null
                : ts.prepareEvidenceAppend(
                    validateEvidenceAgainstPrepared(directEvidence, expectedPrepared),
                );
            updateFrameBuffer(framePatch, { requireArtifactProvenance: true });
            const frameVersions = getFrameVersions();
            ts.setFrameVersions(frameVersions);
            if (preparedEvidence !== null) ts.commitEvidenceAppend(preparedEvidence);
            ts.setCheckpointTimeline(parseCheckpointTimelineV2(result.checkpointTimeline));
        } catch (error) {
            reportWorkerError(error, 'Failed to run a training step.');
        } finally {
            manualActionPendingRef.current = false;
        }
    }, [initializeWorker, pauseForMutation, reportWorkerError]);

    const reset = useCallback(async () => {
        if (configSyncPendingRef.current
            || manualActionPendingRef.current
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        const seq = beginConfigSync();
        useTrainingStore.getState().clearPauseReason();
        try {
            await pauseForMutation();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            if (!initializedRef.current) {
                await initializeWorker();
                if (!isCurrentConfigSync(seq)) return;
                useTrainingStore.getState().setStatus('idle');
                finishConfigSyncIfCurrent(seq);
                return;
            }
            const api = getWorkerApi();
            const resetPrepared = requirePreparedExperiment();
            const result = await api.resetExperimentV2();
            if (!mountedRef.current) return;
            publishCommittedV2Run(result, resetPrepared, 'reset');
            if (!isCurrentConfigSync(seq)) return;
            if (currentPreparedExperiment() !== resetPrepared) return;
            const ts = useTrainingStore.getState();

            // Auxiliary reset hydration cannot roll generation identity back.
            const trainPts = await api.getTrainPointsV2();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            const testPts = await api.getTestPointsV2();
            if (!mountedRef.current || !isCurrentConfigSync(seq)) return;
            if (currentPreparedExperiment() !== resetPrepared) return;
            ts.setTrainPoints(trainPts);
            ts.setTestPoints(testPts);
            initializedRef.current = true;
            finishConfigSyncIfCurrent(seq);
        } catch (error) {
            if (!isCurrentConfigSync(seq)) return;
            reportWorkerError(error, 'Failed to reset training.');
            finishConfigSyncIfCurrent(seq);
        }
    }, [beginConfigSync, finishConfigSyncIfCurrent, initializeWorker, isCurrentConfigSync, pauseForMutation, publishCommittedV2Run, reportWorkerError]);

    const restoreCheckpoint = useCallback(async (id: number) => {
        if (!Number.isSafeInteger(id) || id < 1) return;
        if (configSyncPendingRef.current
            || manualActionPendingRef.current
            || restoreBarrierRef.current !== null
            || useTrainingStore.getState().pendingConfigSource !== null) {
            return;
        }
        if (!useTrainingStore.getState().checkpointTimeline.checkpoints.some(
            (checkpoint) => checkpoint.id === id,
        )) return;
        manualActionPendingRef.current = true;
        let releaseRestore!: () => void;
        const restoreBarrier = new Promise<void>((resolve) => {
            releaseRestore = resolve;
        });
        restoreBarrierRef.current = restoreBarrier;
        try {
            await pauseForMutation();
            if (!mountedRef.current || configSyncPendingRef.current) return;
            if (!initializedRef.current) {
                await initializeWorker();
                if (!initializedRef.current) return;
            }
            const restoredPrepared = requirePreparedExperiment();
            const before = useTrainingStore.getState();
            const currentGeneration = before.evidenceGenerationId;
            const currentRevision = before.latestLiveSignal?.model.revision
                ?? before.latestEvaluation?.model.revision;
            if (currentGeneration === null || currentRevision === undefined) {
                throw new Error('checkpoint restore requires current scientific evidence');
            }
            const requestId = requestIdRef.current + 1;
            requestIdRef.current = requestId;
            const result = await getWorkerApi().restoreCheckpointV2({
                type: 'restore-checkpoint',
                protocolVersion: WORKER_PROTOCOL_VERSION,
                requestId,
                checkpointId: id,
            });
            const currentPrepared = currentPreparedExperiment();
            if (!mountedRef.current
                || configSyncPendingRef.current
                || currentPrepared?.identities.recipeFingerprint
                    !== restoredPrepared.identities.recipeFingerprint) {
                return;
            }
            const ts = useTrainingStore.getState();
            const evaluation = result.evidence.latestEvaluation;
            if (result.runId !== currentGeneration
                || evaluation?.model.generationId !== currentGeneration
                || evaluation.model.revision !== currentRevision + 1) {
                throw new Error('checkpoint restore must advance the current revision exactly once');
            }
            const timeline = parseCheckpointTimelineV2(result.checkpointTimeline);
            if (timeline.restoredCheckpointId !== id
                || !timeline.checkpoints.some((checkpoint) => checkpoint.id === id)) {
                throw new Error('checkpoint restore timeline does not identify the restored checkpoint');
            }
            const evidence = validateEvidenceAgainstPrepared(result.evidence, restoredPrepared);
            const evidenceReplacement = ts.prepareEvidenceReplacement(evidence);
            const framePatch = buildStrictV2FramePatch(
                result,
                true,
                restoredPrepared,
                'restore',
            );

            discardPendingSnapshot(result.runId, evaluation.model.revision);
            ts.commitEvidenceReplacement(evidenceReplacement);
            updateFrameBuffer(framePatch, { requireArtifactProvenance: true });
            const frameVersions = getFrameVersions();
            ts.setFrameVersions(frameVersions);
            ts.setCheckpointTimeline(timeline);
            ts.markTrainedRecipe(
                restoredPrepared.document.recipe,
                'restore',
                restoredPrepared.identities.recipeFingerprint,
            );
            ts.setPauseReason('manual');
            ts.setStatus('paused');
            ts.clearWorkerError();
        } catch (error) {
            reportWorkerError(error, 'Failed to restore checkpoint.');
        } finally {
            if (restoreBarrierRef.current === restoreBarrier) {
                restoreBarrierRef.current = null;
            }
            manualActionPendingRef.current = false;
            releaseRestore();
        }
    }, [initializeWorker, pauseForMutation, reportWorkerError]);

    // Cleanup on unmount
    useEffect(() => {
        const lifecycleEpoch = lifecycleEpochRef;
        const rejectedPreparationRequestIds = rejectedPreparationRequestIdsRef;
        return () => {
            mountedRef.current = false;
            lifecycleEpoch.current++;
            activeRequestIdRef.current = null;
            pendingPreparationRequestIdRef.current = null;
            rejectedPreparationRequestIds.current.clear();
            streamSetupPromiseRef.current = null;
            initializationPromiseRef.current = null;
            mutationPauseRejectRef.current?.(new Error('Training hook unmounted'));
            mutationPausePromiseRef.current = null;
            mutationPauseResolveRef.current = null;
            mutationPauseRejectRef.current = null;
            activeConfigSyncSeqRef.current = configSyncSeqRef.current + 1;
            configSyncSeqRef.current = activeConfigSyncSeqRef.current;
            configSyncPendingRef.current = false;
            manualActionPendingRef.current = false;
            isPlayingRef.current = false;
            terminateWorker();
            const ts = useTrainingStore.getState();
            ts.resetEvidence();
            const frameVersions = getFrameVersions();
            useTrainingStore.setState({
                status: 'idle',
                frameVersion: frameVersions.frameVersion,
                outputGridVersion: frameVersions.outputGridVersion,
                neuronGridsVersion: frameVersions.neuronGridsVersion,
                paramsVersion: frameVersions.paramsVersion,
                layerStatsVersion: frameVersions.layerStatsVersion,
                confusionMatrixVersion: frameVersions.confusionMatrixVersion,
                activationHistogramsVersion: frameVersions.activationHistogramsVersion,
                multiclassBoundaryVersion: frameVersions.multiclassBoundaryVersion,
                trainPoints: [],
                testPoints: [],
                dataConfigLoading: false,
                networkConfigLoading: false,
                featuresConfigLoading: false,
                trainingConfigLoading: false,
                presetConfigLoading: false,
                pendingConfigSource: null,
                configError: null,
                configErrorSource: null,
                workerError: null,
                pauseReason: null,
                checkpointTimeline: EMPTY_CHECKPOINT_TIMELINE,
                trainedRecipe: null,
                trainedRecipeFingerprint: null,
                trainedRecipeRecordedAt: null,
                trainedRecipeSource: null,
            });
        };
    }, []);

    return { play, pause, step, reset, restoreCheckpoint };
}
