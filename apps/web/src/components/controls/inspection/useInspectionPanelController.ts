import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ModelRevision } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { selectScientificEvidence } from '../../../store/evidenceSelectors.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../../worker/frameBuffer.ts';
import { getWorkerApi } from '../../../worker/workerBridge.ts';
import type {
    BackpropExplanationResponseV2,
    ObjectiveLandscapeResponseV2,
    PredictionTraceResponseV2,
} from '../../../worker/training.worker.ts';
import {
    createInspectionPanelDisplayModel,
    normalizeInspectionSampleIndex,
    resolveInspectionEffectiveSampleIndex,
    type InspectionPanelCommands,
    type InspectionPanelDisplayModel,
    type InspectionTraceSource,
} from './inspectionPanelModel.ts';

export interface InspectionPanelController {
    readonly model: InspectionPanelDisplayModel;
    readonly commands: InspectionPanelCommands;
}

function sameModelRevision(
    left: ModelRevision | null,
    right: ModelRevision | null,
): boolean {
    return left !== null
        && right !== null
        && left.generationId === right.generationId
        && left.revision === right.revision;
}

function activeModelRevision(): ModelRevision | null {
    const state = useTrainingStore.getState();
    return selectScientificEvidence({
        latestLiveSignal: state.latestLiveSignal,
        latestEvaluation: state.latestEvaluation,
    }).currentModel;
}

export function useInspectionPanelController(): InspectionPanelController {
    const layerStatsVersion = useTrainingStore((state) => state.layerStatsVersion);
    const activationHistogramsVersion = useTrainingStore(
        (state) => state.activationHistogramsVersion,
    );
    const trainPointCount = useTrainingStore((state) => state.trainPoints.length);
    const testPointCount = useTrainingStore((state) => state.testPoints.length);
    const hiddenLayerCount = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared.compiled.network.hiddenLayers.length
        : 0);
    const latestLiveSignal = useTrainingStore((state) => state.latestLiveSignal);
    const latestEvaluation = useTrainingStore((state) => state.latestEvaluation);
    const evidence = useMemo(() => selectScientificEvidence({
        latestLiveSignal,
        latestEvaluation,
    }), [latestEvaluation, latestLiveSignal]);
    const currentModel = evidence.currentModel;
    const currentModelKey = currentModel === null
        ? 'none'
        : `${currentModel.generationId}:${currentModel.revision}`;

    const [selectedHistogramLayer, setSelectedHistogramLayer] = useState(0);
    const [traceSource, setTraceSource] = useState<InspectionTraceSource>('train');
    const [sampleIndex, setSampleIndex] = useState(0);
    const traceSourceRef = useRef<InspectionTraceSource>('train');
    const sampleIndexRef = useRef(0);
    const [traceResult, setTraceResult] = useState<PredictionTraceResponseV2 | null>(null);
    const [traceError, setTraceError] = useState<string | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const [backpropResult, setBackpropResult] = useState<BackpropExplanationResponseV2 | null>(null);
    const [backpropError, setBackpropError] = useState<string | null>(null);
    const [backpropLoading, setBackpropLoading] = useState(false);
    const [landscapeResult, setLandscapeResult] = useState<ObjectiveLandscapeResponseV2 | null>(null);
    const [landscapeError, setLandscapeError] = useState<string | null>(null);
    const [landscapeLoading, setLandscapeLoading] = useState(false);
    const traceRequestRef = useRef(0);
    const backpropRequestRef = useRef(0);
    const landscapeRequestRef = useRef(0);
    const traceResultRef = useRef<PredictionTraceResponseV2 | null>(null);
    const previousModelKeyRef = useRef(currentModelKey);

    useEffect(() => {
        if (previousModelKeyRef.current === currentModelKey) return;
        previousModelKeyRef.current = currentModelKey;
        traceRequestRef.current++;
        backpropRequestRef.current++;
        landscapeRequestRef.current++;
        const hadTrace = traceResultRef.current !== null;
        traceResultRef.current = null;
        setTraceResult(null);
        setTraceError(hadTrace ? 'Trace cleared because the active model changed.' : null);
        setTraceLoading(false);
        setBackpropResult(null);
        setBackpropError(null);
        setBackpropLoading(false);
        setLandscapeResult(null);
        setLandscapeError(null);
        setLandscapeLoading(false);
    }, [currentModelKey]);

    const layerStatsState = useMemo(() => {
        void layerStatsVersion;
        const frame = getFrameBuffer();
        return {
            values: frame.layerStats,
            provenance: frame.layerStatsProvenance,
            gradientRevision: frame.layerStatsGradientRevision,
        };
    }, [layerStatsVersion]);
    const activationBasis = layerStatsState.provenance?.basis.kind === 'bounded-sample'
        ? layerStatsState.provenance.basis
        : null;

    const activationHistograms = useMemo(() => {
        void activationHistogramsVersion;
        const frame = getFrameBuffer();
        if (frame.activationHistogramBins && frame.activationHistogramLayout) {
            return {
                bins: frame.activationHistogramBins,
                layout: frame.activationHistogramLayout,
            };
        }
        return null;
    }, [activationHistogramsVersion]);

    const requestTrace = useCallback(async () => {
        if (traceLoading || currentModel === null) return;

        const requestSource = traceSourceRef.current;
        const requestSampleIndex = sampleIndexRef.current;
        const trainingState = useTrainingStore.getState();
        const pointsSnapshot = requestSource === 'test'
            ? trainingState.testPoints
            : trainingState.trainPoints;
        const requestIndex = resolveInspectionEffectiveSampleIndex(
            requestSampleIndex,
            pointsSnapshot.length,
        );
        if (pointsSnapshot[requestIndex] === undefined) return;

        const requestModel = currentModel;
        const requestId = ++traceRequestRef.current;
        setTraceLoading(true);
        setTraceError(null);
        try {
            const api = await getWorkerApi();
            const response = await api.getPredictionTraceV2({
                source: requestSource,
                index: requestIndex,
            });
            if (
                traceRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)
            ) {
                traceResultRef.current = response;
                setTraceResult(response);
            }
        } catch (error) {
            if (
                traceRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
            ) {
                traceResultRef.current = null;
                setTraceResult(null);
                setTraceError(error instanceof Error ? error.message : String(error));
            }
        } finally {
            if (traceRequestRef.current === requestId) setTraceLoading(false);
        }
    }, [currentModel, traceLoading]);

    const requestBackprop = useCallback(async () => {
        if (backpropLoading || currentModel === null) return;
        const requestModel = currentModel;
        const requestId = ++backpropRequestRef.current;
        setBackpropLoading(true);
        setBackpropError(null);
        setBackpropResult(null);
        try {
            const api = await getWorkerApi();
            const response = await api.getBackpropExplanationV2();
            if (
                backpropRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)
            ) {
                setBackpropResult(response);
            }
        } catch (error) {
            if (
                backpropRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
            ) {
                setBackpropResult(null);
                setBackpropError(error instanceof Error ? error.message : String(error));
            }
        } finally {
            if (backpropRequestRef.current === requestId) setBackpropLoading(false);
        }
    }, [backpropLoading, currentModel]);

    const requestLandscape = useCallback(async () => {
        if (landscapeLoading || currentModel === null) return;
        const requestModel = currentModel;
        const requestId = ++landscapeRequestRef.current;
        setLandscapeLoading(true);
        setLandscapeError(null);
        setLandscapeResult(null);
        try {
            const api = await getWorkerApi();
            const response = await api.getObjectiveLandscapeV2();
            if (
                landscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)
            ) {
                setLandscapeResult(response);
            }
        } catch (error) {
            if (
                landscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
            ) {
                setLandscapeResult(null);
                setLandscapeError(error instanceof Error ? error.message : String(error));
            }
        } finally {
            if (landscapeRequestRef.current === requestId) setLandscapeLoading(false);
        }
    }, [currentModel, landscapeLoading]);

    const model = useMemo(() => createInspectionPanelDisplayModel({
        hiddenLayerCount,
        layerStats: layerStatsState.values,
        activationBasis: activationBasis
            ? {
                sampleCount: activationBasis.sampleCount,
                populationCount: activationBasis.populationCount,
                modelRevision: layerStatsState.provenance?.model.revision ?? 0,
                gradientRevision: layerStatsState.gradientRevision,
            }
            : null,
        histogram: activationHistograms
            ? {
                bins: activationHistograms.bins,
                layers: activationHistograms.layout.layers,
            }
            : null,
        selectedHistogramLayer,
        traceSource,
        sampleIndex,
        trainPointCount,
        testPointCount,
        hasCurrentModel: currentModel !== null,
        traceLoading,
        traceError,
        traceResult: traceResult
            ? {
                source: traceResult.sample.source,
                sampleIndex: traceResult.sample.index,
                modelStep: traceResult.model.step,
                modelRevision: traceResult.model.revision,
                output: traceResult.trace.output,
                sampleDataLoss: traceResult.trace.sampleDataLoss,
                regularizationPenalty: traceResult.trace.regularizationPenalty,
                layers: traceResult.trace.layers,
            }
            : null,
        backpropLoading,
        backpropError,
        backpropResult: backpropResult
            ? {
                modelStep: backpropResult.model.step,
                modelEpoch: backpropResult.model.epoch,
                summary: backpropResult.explanation.summary,
                batchSize: backpropResult.explanation.batchSize,
                learningRate: backpropResult.explanation.learningRate,
                dataLoss: backpropResult.explanation.objective.dataLoss,
                regularizationPenalty: backpropResult.explanation.objective.regularizationPenalty,
                totalObjective: backpropResult.explanation.objective.totalObjective,
                totalGradientNorm: backpropResult.explanation.gradients.totalGradientNorm,
                clippedGradientNorm: backpropResult.explanation.gradients.clippedGradientNorm,
                clipScale: backpropResult.explanation.gradients.clipScale,
                layers: backpropResult.explanation.layers,
            }
            : null,
        landscapeLoading,
        landscapeError,
        landscapeResult: landscapeResult
            ? {
                modelStep: landscapeResult.model.step,
                modelEpoch: landscapeResult.model.epoch,
                summary: landscapeResult.probe.summary,
                gridSize: landscapeResult.probe.gridSize,
                sampleCount: landscapeResult.probe.sampleCount,
                parameterPositionCount: landscapeResult.probe.parameterPositionCount,
                objectives: landscapeResult.probe.objectives,
                centerObjective: landscapeResult.probe.centerObjective,
                minObjective: landscapeResult.probe.minObjective,
                maxObjective: landscapeResult.probe.maxObjective,
                axisALabel: landscapeResult.probe.axisA.parameter.label,
                axisBLabel: landscapeResult.probe.axisB.parameter.label,
                bestOffsetA: landscapeResult.probe.best.offsetA,
                bestOffsetB: landscapeResult.probe.best.offsetB,
            }
            : null,
    }), [
        activationBasis,
        activationHistograms,
        backpropError,
        backpropLoading,
        backpropResult,
        currentModel,
        hiddenLayerCount,
        landscapeError,
        landscapeLoading,
        landscapeResult,
        layerStatsState,
        sampleIndex,
        selectedHistogramLayer,
        testPointCount,
        traceError,
        traceLoading,
        traceResult,
        traceSource,
        trainPointCount,
    ]);

    const selectTraceSource = useCallback((source: InspectionTraceSource) => {
        traceRequestRef.current++;
        traceResultRef.current = null;
        traceSourceRef.current = source;
        setTraceSource(source);
        setTraceResult(null);
        setTraceError(null);
        setTraceLoading(false);
    }, []);

    const selectSampleIndex = useCallback((index: number) => {
        const normalizedIndex = normalizeInspectionSampleIndex(index);
        traceRequestRef.current++;
        traceResultRef.current = null;
        sampleIndexRef.current = normalizedIndex;
        setTraceResult(null);
        setTraceError(null);
        setTraceLoading(false);
        setSampleIndex(normalizedIndex);
    }, []);

    const commands = useMemo<InspectionPanelCommands>(() => ({
        selectHistogramLayer: setSelectedHistogramLayer,
        selectTraceSource,
        selectSampleIndex,
        requestTrace,
        requestBackprop,
        requestLandscape,
    }), [
        requestBackprop,
        requestLandscape,
        requestTrace,
        selectSampleIndex,
        selectTraceSource,
    ]);

    return useMemo(() => ({ model, commands }), [commands, model]);
}
