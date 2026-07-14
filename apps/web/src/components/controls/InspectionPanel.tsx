// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { getWorkerApi } from '../../worker/workerBridge.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import type { ModelRevision } from '@nn-playground/shared';
import type {
    BackpropExplanationResponseV2,
    ObjectiveLandscapeResponseV2,
    PredictionTraceResponseV2,
    PredictionTraceSampleSource,
} from '../../worker/training.worker.ts';

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

function formatPercent(count: number, total: number): string {
    return total > 0 ? `${((count / total) * 100).toFixed(1)}%` : '0.0%';
}

function formatRange(value: number): string {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
        return value.toExponential(1);
    }
    return value.toFixed(3);
}

function formatBackpropMetric(value: number): string {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
        return value.toExponential(1);
    }
    return value.toFixed(3);
}

function formatSignedOffset(value: number): string {
    const formatted = formatBackpropMetric(value);
    return value > 0 ? `+${formatted}` : formatted;
}

function describeActivationShape(
    nearZeroCount: number,
    saturatedCount: number,
    totalCount: number,
): string {
    if (totalCount === 0) return 'No activation samples yet';
    const nearZeroRatio = nearZeroCount / totalCount;
    const saturatedRatio = saturatedCount / totalCount;
    if (saturatedRatio >= 0.45) return 'Many activations are near the activation limits';
    if (nearZeroRatio >= 0.45) return 'Many activations are near zero or inactive';
    return 'Activations are spread across the sampled range';
}

export const InspectionPanel = memo(function InspectionPanel() {
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const activationHistogramsVersion = useTrainingStore((s) => s.activationHistogramsVersion);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const hiddenLayers = usePlaygroundStore((s) => s.access.status === 'ready'
        ? s.access.prepared.compiled.network.hiddenLayers
        : []);
    const [selectedHistogramLayer, setSelectedHistogramLayer] = useState(0);
    const [traceSource, setTraceSource] = useState<PredictionTraceSampleSource>('train');
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
                demand.needLayerStats === enabled &&
                demand.needActivationHistograms === enabled
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
    const layerStats = layerStatsState.values;
    const activationBasis = layerStatsState.provenance?.basis.kind === 'bounded-sample'
        ? layerStatsState.provenance.basis
        : null;

    const activationHistograms = useMemo(
        () => {
            void activationHistogramsVersion;
            const frame = getFrameBuffer();
            if (frame.activationHistogramBins && frame.activationHistogramLayout) {
                return {
                    bins: frame.activationHistogramBins,
                    layout: frame.activationHistogramLayout,
                    provenance: frame.activationHistogramProvenance,
                };
            }
            return null;
        },
        [activationHistogramsVersion],
    );

    const layerNames = useMemo(() => {
        const names: string[] = [];
        for (let i = 0; i < hiddenLayers.length; i++) {
            names.push(`Hidden ${i + 1}`);
        }
        names.push('Output');
        return names;
    }, [hiddenLayers]);

    const layerStatMaxima = useMemo(() => {
        if (!layerStats || layerStats.length === 0) {
            return { maxGrad: 0.001, maxWeight: 0.001 };
        }

        let maxGrad = 0.001;
        let maxWeight = 0.001;
        for (const stats of layerStats) {
            maxGrad = Math.max(maxGrad, stats.meanAbsGradient);
            maxWeight = Math.max(maxWeight, stats.meanAbsWeight);
        }
        return { maxGrad, maxWeight };
    }, [layerStats]);

    const selectedHistogram = activationHistograms?.layout.layers[
        Math.min(selectedHistogramLayer, Math.max(0, activationHistograms.layout.layers.length - 1))
    ];
    const selectedHistogramIndex = selectedHistogram?.layerIndex ?? 0;
    const selectedHistogramBins = activationHistograms && selectedHistogram
        ? activationHistograms.bins.slice(
            selectedHistogramIndex * selectedHistogram.binCount,
            selectedHistogramIndex * selectedHistogram.binCount + selectedHistogram.binCount,
        )
        : null;
    const maxHistogramCount = selectedHistogramBins
        ? Math.max(1, ...Array.from(selectedHistogramBins))
        : 1;
    const histogramSummary = selectedHistogram
        ? `${layerNames[selectedHistogramIndex] ?? `Layer ${selectedHistogramIndex + 1}`} activations: ${formatPercent(selectedHistogram.nearZeroCount, selectedHistogram.totalCount)} near zero, ${formatPercent(selectedHistogram.saturatedCount, selectedHistogram.totalCount)} near activation limits. ${describeActivationShape(selectedHistogram.nearZeroCount, selectedHistogram.saturatedCount, selectedHistogram.totalCount)}.`
        : 'Activation histograms are not available yet.';

    const selectedPoints = traceSource === 'test' ? testPoints : trainPoints;
    const selectedSample = selectedPoints[Math.min(sampleIndex, Math.max(0, selectedPoints.length - 1))];
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
            if (traceRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)) {
                traceResultRef.current = response;
                setTraceResult(response);
            }
        } catch (err) {
            if (traceRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())) {
                traceResultRef.current = null;
                setTraceResult(null);
                setTraceError(err instanceof Error ? err.message : String(err));
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
            if (backpropRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)) {
                setBackpropResult(response);
            }
        } catch (err) {
            if (backpropRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())) {
                setBackpropResult(null);
                setBackpropError(err instanceof Error ? err.message : String(err));
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
            if (lossLandscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())
                && sameModelRevision(requestModel, response.model)) {
                setLossLandscapeResult(response);
            }
        } catch (err) {
            if (lossLandscapeRequestRef.current === requestId
                && sameModelRevision(requestModel, activeModelRevision())) {
                setLossLandscapeResult(null);
                setLossLandscapeError(err instanceof Error ? err.message : String(err));
            }
        } finally {
            if (lossLandscapeRequestRef.current === requestId) setLossLandscapeLoading(false);
        }
    };

    const lossLandscapeSummary = lossLandscapeResult
        ? `Training-objective parameter grid: center ${formatBackpropMetric(lossLandscapeResult.probe.centerObjective)}, min ${formatBackpropMetric(lossLandscapeResult.probe.minObjective)}, max ${formatBackpropMetric(lossLandscapeResult.probe.maxObjective)}. Best direction ${lossLandscapeResult.probe.axisA.parameter.label} ${formatSignedOffset(lossLandscapeResult.probe.best.offsetA)}, ${lossLandscapeResult.probe.axisB.parameter.label} ${formatSignedOffset(lossLandscapeResult.probe.best.offsetB)}.`
        : 'Training-objective landscape probe has not run yet.';
    const lossLandscapeSpread = lossLandscapeResult
        ? Math.max(0.000001, lossLandscapeResult.probe.maxObjective - lossLandscapeResult.probe.minObjective)
        : 1;

    return (
        <div className="inspection-panel">
            {activationBasis && (
                <p className="inspection__basis">
                    <span>{`Activation statistics across ${activationBasis.sampleCount.toLocaleString()} of ${activationBasis.populationCount.toLocaleString()} training examples`}</span>
                    {layerStatsState.gradientRevision !== null
                        && layerStatsState.gradientRevision !== layerStatsState.provenance?.model.revision
                        ? `; gradient summary comes from model revision ${layerStatsState.gradientRevision.toLocaleString()}.`
                        : '.'}
                </p>
            )}
            {!layerStats || layerStats.length === 0 ? (
                <div className="inspection__empty">Train the model to see stats</div>
            ) : (
                <div className="inspection__layers">
                    {layerStats.map((stats, idx) => {
                        const gradPct = (stats.meanAbsGradient / layerStatMaxima.maxGrad) * 100;
                        const weightPct = (stats.meanAbsWeight / layerStatMaxima.maxWeight) * 100;

                        return (
                            <div key={idx} className="inspection__layer">
                                <div className="inspection__layer-name">{layerNames[idx]}</div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">|∇w|</span>
                                    <div className="inspection__bar-track">
                                        <div
                                            className="inspection__bar-fill inspection__bar-fill--grad"
                                            style={{ width: `${Math.max(2, gradPct)}%` }}
                                        />
                                    </div>
                                    <span className="inspection__stat-value">
                                        {stats.meanAbsGradient < 0.0001
                                            ? stats.meanAbsGradient.toExponential(1)
                                            : stats.meanAbsGradient.toFixed(4)}
                                    </span>
                                </div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">|w|</span>
                                    <div className="inspection__bar-track">
                                        <div
                                            className="inspection__bar-fill inspection__bar-fill--weight"
                                            style={{ width: `${Math.max(2, weightPct)}%` }}
                                        />
                                    </div>
                                    <span className="inspection__stat-value">
                                        {stats.meanAbsWeight.toFixed(4)}
                                    </span>
                                </div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">μ(a)</span>
                                    <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                        {stats.meanActivation.toFixed(4)}
                                    </span>
                                    <span className="inspection__stat-label" style={{ marginLeft: 8 }}>σ</span>
                                    <span className="inspection__stat-value">
                                        {stats.activationStd.toFixed(4)}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
            <section className="inspection__layer" aria-label="Activation histogram explorer">
                <div className="inspection__layer-name">Activation Histogram</div>
                {!activationHistograms || activationHistograms.layout.layers.length === 0 || !selectedHistogram ? (
                    <div className="inspection__empty">Open inspection while training to sample layer activations.</div>
                ) : (
                    <>
                        <div className="control-row">
                            <label htmlFor="activation-histogram-layer">Histogram layer</label>
                            <select
                                id="activation-histogram-layer"
                                className="select"
                                value={selectedHistogramIndex}
                                onChange={(event) => {
                                    const next = Number(event.currentTarget.value);
                                    setSelectedHistogramLayer(Number.isFinite(next) ? next : 0);
                                }}
                            >
                                {activationHistograms.layout.layers.map((layer) => (
                                    <option key={layer.layerIndex} value={layer.layerIndex}>
                                        {layerNames[layer.layerIndex] ?? `Layer ${layer.layerIndex + 1}`}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div
                            className="inspection__histogram"
                            role="img"
                            aria-label={histogramSummary}
                        >
                            {selectedHistogramBins
                                ? Array.from(selectedHistogramBins).map((count, binIndex) => (
                                    <div
                                        key={binIndex}
                                        className="inspection__histogram-bin"
                                        style={{ height: `${Math.max(6, (count / maxHistogramCount) * 100)}%` }}
                                        aria-hidden="true"
                                    />
                                ))
                                : null}
                        </div>
                        <div className="inspection__histogram-summary">
                            <strong>{layerNames[selectedHistogramIndex] ?? `Layer ${selectedHistogramIndex + 1}`} activations</strong>
                            <span>{formatPercent(selectedHistogram.nearZeroCount, selectedHistogram.totalCount)} near zero</span>
                            <span>{formatPercent(selectedHistogram.saturatedCount, selectedHistogram.totalCount)} near activation limits</span>
                            <span>
                                range {formatRange(selectedHistogram.minActivation)} to {formatRange(selectedHistogram.maxActivation)}
                            </span>
                        </div>
                        <p className="sr-only">{histogramSummary}</p>
                    </>
                )}
            </section>
            <div className="inspection__layer" aria-label="Prediction trace">
                <div className="inspection__layer-name">Prediction Trace</div>
                <div className="control-row">
                    <label htmlFor="trace-source">Sample</label>
                    <select
                        id="trace-source"
                        className="select"
                        value={traceSource}
                        onChange={(event) => {
                            traceRequestRef.current++;
                            traceResultRef.current = null;
                            setTraceSource(event.currentTarget.value as PredictionTraceSampleSource);
                            setTraceResult(null);
                            setTraceError(null);
                            setTraceLoading(false);
                        }}
                    >
                        <option value="train">Training</option>
                        <option value="test">Test</option>
                    </select>
                </div>
                <div className="control-row">
                    <label htmlFor="trace-index">Index</label>
                    <input
                        id="trace-index"
                        className="input"
                        type="number"
                        min={0}
                        max={Math.max(0, selectedPoints.length - 1)}
                        value={sampleIndex}
                        onChange={(event) => {
                            const next = Number(event.currentTarget.value);
                            traceRequestRef.current++;
                            traceResultRef.current = null;
                            setTraceResult(null);
                            setTraceError(null);
                            setTraceLoading(false);
                            setSampleIndex(Number.isFinite(next) ? Math.max(0, Math.trunc(next)) : 0);
                        }}
                    />
                </div>

                {!canTrace ? (
                    <div className="inspection__empty">
                        No {traceSource === 'test' ? 'test' : 'training'} samples are available yet.
                    </div>
                ) : null}
                <button
                    type="button"
                    className="btn"
                    onClick={handleTrace}
                    disabled={!canTrace || traceLoading}
                >
                    {traceLoading ? 'Tracing…' : 'Trace prediction'}
                </button>
                <div aria-live="polite">
                    {traceError ? (
                        <div className="inspection__empty">
                            {traceError.startsWith('Trace cleared') ? traceError : `Trace failed: ${traceError}`}
                        </div>
                    ) : null}
                    {traceResult ? (
                        <div className="inspection__layers">
                            <p className="inspection__basis">
                                {`Trace from ${traceResult.sample.source === 'train' ? 'training' : 'test'} sample ${traceResult.sample.index.toLocaleString()} · model step ${traceResult.model.step.toLocaleString()} · revision ${traceResult.model.revision.toLocaleString()}`}
                            </p>
                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">Output</span>
                                <span className="inspection__stat-value">
                                    {traceResult.trace.output.map((value) => value.toFixed(4)).join(', ')}
                                </span>
                                <span className="inspection__stat-label" style={{ marginLeft: 8 }}>sample data loss</span>
                                <span className="inspection__stat-value">
                                    {traceResult.trace.sampleDataLoss.toFixed(4)}
                                </span>
                                <span className="inspection__stat-label" style={{ marginLeft: 8 }}>model penalty</span>
                                <span className="inspection__stat-value">
                                    {traceResult.trace.regularizationPenalty.toFixed(4)}
                                </span>
                            </div>
                            {traceResult.trace.layers.map((layer) => (
                                <div key={layer.layerIndex} className="inspection__stat-row" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: 2, marginBottom: 6 }}>
                                    <span className="inspection__stat-label">Layer {layer.layerIndex + 1}</span>
                                    <div style={{
                                        width: '100%',
                                        background: 'rgba(0,0,0,0.2)',
                                        padding: '4px 6px',
                                        borderRadius: 4,
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: 9,
                                        color: 'var(--text-secondary)',
                                        overflowX: 'auto',
                                        whiteSpace: 'nowrap',
                                        border: '1px solid rgba(255,255,255,0.05)',
                                    }}>
                                        {layer.activations.map((value) => value.toFixed(3)).join(', ')}
                                    </div>
                                </div>
                            ))}

                        </div>
                    ) : null}
                </div>
            </div>
            <section className="inspection__layer" aria-label="Slow-motion backprop preview">
                <div className="inspection__layer-name">Slow-Motion Backprop</div>
                <button
                    type="button"
                    className="btn"
                    onClick={handleBackpropPreview}
                    disabled={backpropLoading}
                >
                    {backpropLoading ? 'Previewing backprop' : 'Preview backprop'}
                </button>
                <div
                    role="status"
                    aria-live="polite"
                    aria-label="Backprop preview status"
                    className="inspection__empty"
                >
                    {backpropLoading
                        ? 'Preparing backprop preview...'
                        : backpropError
                            ? `Backprop preview failed: ${backpropError}`
                            : backpropResult?.explanation.summary ?? ''}
                </div>
                {backpropResult ? (
                    <div className="inspection__layers">
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                Preview from step {backpropResult.model.step} / epoch {backpropResult.model.epoch}
                            </span>
                        </div>
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">batch {backpropResult.explanation.batchSize}</span>
                            <span className="inspection__stat-value">
                                data loss {formatBackpropMetric(backpropResult.explanation.objective.dataLoss)}
                            </span>
                            <span className="inspection__stat-value">
                                penalty {formatBackpropMetric(backpropResult.explanation.objective.regularizationPenalty)}
                            </span>
                            <span className="inspection__stat-value">
                                training objective {formatBackpropMetric(backpropResult.explanation.objective.totalObjective)}
                            </span>
                            <span className="inspection__stat-value">
                                lr {formatBackpropMetric(backpropResult.explanation.learningRate)}
                            </span>
                        </div>
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                complete objective gradient {formatBackpropMetric(backpropResult.explanation.gradients.totalGradientNorm)}
                            </span>
                            <span className="inspection__stat-value">
                                {backpropResult.explanation.gradients.clipScale < 1
                                    ? `clipped ${formatBackpropMetric(backpropResult.explanation.gradients.clipScale)}x to ${formatBackpropMetric(backpropResult.explanation.gradients.clippedGradientNorm)}`
                                    : 'not clipped'}
                            </span>
                        </div>
                        <ul className="inspection__backprop-list" aria-label="Backprop layer summaries">
                            {backpropResult.explanation.layers.map((layer) => (
                                <li key={layer.layerIndex} className="inspection__backprop-item">
                                    <div className="inspection__backprop-heading">
                                        <span className="inspection__stat-label">
                                            {layerNames[layer.layerIndex] ?? `Layer ${layer.layerIndex + 1}`}
                                        </span>
                                        <span className="inspection__stat-value">
                                            {layer.status}
                                        </span>
                                    </div>
                                    <div className="inspection__backprop-metrics">
                                        <span className="inspection__stat-value">
                                            mean update {formatBackpropMetric(layer.meanAbsUpdate)}
                                        </span>
                                        <span className="inspection__stat-value">
                                            mean gradient {formatBackpropMetric(layer.meanAbsGradient)}
                                        </span>
                                        <span className="inspection__stat-value">
                                            error signal {formatBackpropMetric(layer.meanAbsErrorSignal)}
                                        </span>
                                    </div>
                                    <div className="inspection__backprop-note">
                                        {layer.note}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                ) : null}
            </section>
            <section className="inspection__layer" aria-label="Loss landscape probe">
                <div className="inspection__layer-name">Loss Landscape Probe</div>
                <button
                    type="button"
                    className="btn"
                    onClick={handleLossLandscapeProbe}
                    disabled={lossLandscapeLoading}
                >
                    {lossLandscapeLoading ? 'Probing loss surface' : 'Probe loss surface'}
                </button>
                <div
                    role="status"
                    aria-live="polite"
                    aria-label="Loss landscape probe status"
                    className="inspection__empty"
                >
                    {lossLandscapeLoading
                        ? 'Probing a bounded local loss surface...'
                        : lossLandscapeError
                            ? `Loss landscape probe failed: ${lossLandscapeError}`
                            : lossLandscapeResult?.probe.summary ?? ''}
                </div>
                {lossLandscapeResult ? (
                    <div className="inspection__layers">
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                Probe from step {lossLandscapeResult.model.step} / epoch {lossLandscapeResult.model.epoch}
                            </span>
                        </div>
                        <div
                            className="inspection__loss-heatmap"
                            role="img"
                            aria-label={lossLandscapeSummary}
                            style={{
                                gridTemplateColumns: `repeat(${lossLandscapeResult.probe.gridSize}, minmax(0, 1fr))`,
                            }}
                        >
                            {lossLandscapeResult.probe.objectives.map((objective, index) => {
                                const intensity = 1 - ((objective - lossLandscapeResult.probe.minObjective) / lossLandscapeSpread);
                                return (
                                    <span
                                        key={`${index}-${objective}`}
                                        className="inspection__loss-cell"
                                        style={{ opacity: 0.28 + Math.max(0, Math.min(1, intensity)) * 0.72 }}
                                        aria-hidden="true"
                                    />
                                );
                            })}
                        </div>
                        <div className="inspection__histogram-summary">
                            <strong>Training objective on a parameter grid</strong>
                            <span>center {formatBackpropMetric(lossLandscapeResult.probe.centerObjective)}</span>
                            <span>min {formatBackpropMetric(lossLandscapeResult.probe.minObjective)}</span>
                            <span>max {formatBackpropMetric(lossLandscapeResult.probe.maxObjective)}</span>
                            <span>
                                sampled {lossLandscapeResult.probe.sampleCount} training examples across {lossLandscapeResult.probe.parameterPositionCount} parameter positions on a {lossLandscapeResult.probe.gridSize} by {lossLandscapeResult.probe.gridSize} grid
                            </span>
                            <span>
                                best direction {lossLandscapeResult.probe.axisA.parameter.label} {formatSignedOffset(lossLandscapeResult.probe.best.offsetA)}, {lossLandscapeResult.probe.axisB.parameter.label} {formatSignedOffset(lossLandscapeResult.probe.best.offsetB)}
                            </span>
                        </div>
                    </div>
                ) : null}
            </section>
        </div>
    );
});
