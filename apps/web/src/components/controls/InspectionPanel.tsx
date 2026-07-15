// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { ModelRevision } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { getWorkerApi } from '../../worker/workerBridge.ts';
import type {
    BackpropExplanationResponseV2,
    ObjectiveLandscapeResponseV2,
    PredictionTraceResponseV2,
} from '../../worker/training.worker.ts';
import { InspectionPanelView } from './inspection/InspectionPanelView.tsx';
import {
    createInspectionPanelDisplayModel,
    normalizeInspectionSampleIndex,
    type InspectionPanelCommands,
    type InspectionTraceSource,
} from './inspection/inspectionPanelModel.ts';

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

export const InspectionPanel = memo(function InspectionPanel() {
    const frameVersion = useTrainingStore((state) => state.frameVersion);
    const activationHistogramsVersion = useTrainingStore(
        (state) => state.activationHistogramsVersion,
    );
    const trainPoints = useTrainingStore((state) => state.trainPoints);
    const testPoints = useTrainingStore((state) => state.testPoints);
    const hiddenLayers = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared.compiled.network.hiddenLayers
        : []);
    const [selectedHistogramLayer, setSelectedHistogramLayer] = useState(0);
    const [traceSource, setTraceSource] = useState<InspectionTraceSource>('train');
    const [sampleIndex, setSampleIndex] = useState(0);
    const [traceResult, setTraceResult] = useState<PredictionTraceResponseV2 | null>(null);
    const [traceError, setTraceError] = useState<string | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const [backpropResult, setBackpropResult] = useState<BackpropExplanationResponseV2 | null>(null);
    const [backpropError, setBackpropError] = useState<string | null>(null);
    const [backpropLoading, setBackpropLoading] = useState(false);
    const [lossLandscapeResult, setLossLandscapeResult] = useState<ObjectiveLandscapeResponseV2 | null>(null);
    const [lossLandscapeError, setLossLandscapeError] = useState<string | null>(null);
    const [lossLandscapeLoading, setLossLandscapeLoading] = useState(false);
    const traceRequestRef = useRef(0);
    const backpropRequestRef = useRef(0);
    const lossLandscapeRequestRef = useRef(0);
    const traceResultRef = useRef<PredictionTraceResponseV2 | null>(null);
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
    const previousModelKeyRef = useRef(currentModelKey);

    useEffect(() => {
        if (previousModelKeyRef.current === currentModelKey) return;
        previousModelKeyRef.current = currentModelKey;
        traceRequestRef.current++;
        backpropRequestRef.current++;
        lossLandscapeRequestRef.current++;
        const hadTrace = traceResultRef.current !== null;
        traceResultRef.current = null;
        setTraceResult(null);
        setTraceError(hadTrace ? 'Trace cleared because the active model changed.' : null);
        setTraceLoading(false);
        setBackpropResult(null);
        setBackpropError(null);
        setBackpropLoading(false);
        setLossLandscapeResult(null);
        setLossLandscapeError(null);
        setLossLandscapeLoading(false);
    }, [currentModelKey]);

    useEffect(() => {
        const enableInspectionDemand = (enabled: boolean) => {
            const { demand, setDemand } = usePlaygroundStore.getState();
            if (
                demand.needLayerStats === enabled
                && demand.needActivationHistograms === enabled
            ) {
                return;
            }
            setDemand({
                ...demand,
                needLayerStats: enabled,
                needActivationHistograms: enabled,
            });
        };

        enableInspectionDemand(true);
        return () => enableInspectionDemand(false);
    }, []);

    const layerStatsState = useMemo(() => {
        void frameVersion;
        const frame = getFrameBuffer();
        return {
            values: frame.layerStats,
            provenance: frame.layerStatsProvenance,
            gradientRevision: frame.layerStatsGradientRevision,
        };
    }, [frameVersion]);
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

    const selectedPoints = traceSource === 'test' ? testPoints : trainPoints;
    const selectedSample = selectedPoints[
        Math.min(sampleIndex, Math.max(0, selectedPoints.length - 1))
    ];
    const canTrace = selectedSample !== undefined && currentModel !== null;

    const handleTrace = async () => {
        if (!canTrace || traceLoading) return;
        const requestModel = currentModel;
        const requestId = ++traceRequestRef.current;
        setTraceLoading(true);
        setTraceError(null);
        try {
            const api = await getWorkerApi();
            const response = await api.getPredictionTraceV2({
                source: traceSource,
                index: Math.min(sampleIndex, selectedPoints.length - 1),
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
    };

    const handleBackpropPreview = async () => {
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
    };

    const handleLossLandscapeProbe = async () => {
        if (lossLandscapeLoading || currentModel === null) return;
        const requestModel = currentModel;
        const requestId = ++lossLandscapeRequestRef.current;
        setLossLandscapeLoading(true);
        setLossLandscapeError(null);
        setLossLandscapeResult(null);
        try {
            const api = await getWorkerApi();
            const response = await api.getObjectiveLandscapeV2();
            if (
                lossLandscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)
            ) {
                setLossLandscapeResult(response);
            }
        } catch (error) {
            if (
                lossLandscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
            ) {
                setLossLandscapeResult(null);
                setLossLandscapeError(error instanceof Error ? error.message : String(error));
            }
        } finally {
            if (lossLandscapeRequestRef.current === requestId) setLossLandscapeLoading(false);
        }
    };

    const model = createInspectionPanelDisplayModel({
        hiddenLayerCount: hiddenLayers.length,
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
        trainPointCount: trainPoints.length,
        testPointCount: testPoints.length,
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
        landscapeLoading: lossLandscapeLoading,
        landscapeError: lossLandscapeError,
        landscapeResult: lossLandscapeResult
            ? {
                modelStep: lossLandscapeResult.model.step,
                modelEpoch: lossLandscapeResult.model.epoch,
                summary: lossLandscapeResult.probe.summary,
                gridSize: lossLandscapeResult.probe.gridSize,
                sampleCount: lossLandscapeResult.probe.sampleCount,
                parameterPositionCount: lossLandscapeResult.probe.parameterPositionCount,
                objectives: lossLandscapeResult.probe.objectives,
                centerObjective: lossLandscapeResult.probe.centerObjective,
                minObjective: lossLandscapeResult.probe.minObjective,
                maxObjective: lossLandscapeResult.probe.maxObjective,
                axisALabel: lossLandscapeResult.probe.axisA.parameter.label,
                axisBLabel: lossLandscapeResult.probe.axisB.parameter.label,
                bestOffsetA: lossLandscapeResult.probe.best.offsetA,
                bestOffsetB: lossLandscapeResult.probe.best.offsetB,
            }
            : null,
    });

    const commands: InspectionPanelCommands = {
        selectHistogramLayer: setSelectedHistogramLayer,
        selectTraceSource(source) {
            traceRequestRef.current++;
            traceResultRef.current = null;
            setTraceSource(source);
            setTraceResult(null);
            setTraceError(null);
            setTraceLoading(false);
        },
        selectSampleIndex(index) {
            traceRequestRef.current++;
            traceResultRef.current = null;
            setTraceResult(null);
            setTraceError(null);
            setTraceLoading(false);
            setSampleIndex(normalizeInspectionSampleIndex(index));
        },
        requestTrace: handleTrace,
        requestBackprop: handleBackpropPreview,
        requestLandscape: handleLossLandscapeProbe,
    };

    return <InspectionPanelView model={model} commands={commands} />;
});
