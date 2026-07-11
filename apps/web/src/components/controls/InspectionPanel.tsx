// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { getWorkerApi } from '../../worker/workerBridge.ts';
import type {
    BackpropExplanationResponse,
    LossLandscapeProbeResponse,
    PredictionTraceResponse,
    PredictionTraceSampleSource,
} from '../../worker/training.worker.ts';
import type { ActivationHistogramLayout } from '@nn-playground/shared';

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
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const activationHistogramsVersion = useTrainingStore((s) => s.activationHistogramsVersion);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const [selectedHistogramLayer, setSelectedHistogramLayer] = useState(0);
    const [traceSource, setTraceSource] = useState<PredictionTraceSampleSource>('train');
    const [sampleIndex, setSampleIndex] = useState(0);
    const [customX, setCustomX] = useState(0);
    const [customY, setCustomY] = useState(0);
    const [traceResult, setTraceResult] = useState<PredictionTraceResponse | null>(null);
    const [traceError, setTraceError] = useState<string | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);
    const [backpropResult, setBackpropResult] = useState<BackpropExplanationResponse | null>(null);
    const [backpropError, setBackpropError] = useState<string | null>(null);
    const [backpropLoading, setBackpropLoading] = useState(false);
    const backpropRequestInFlightRef = useRef(false);
    const [lossLandscapeResult, setLossLandscapeResult] = useState<LossLandscapeProbeResponse | null>(null);
    const [lossLandscapeError, setLossLandscapeError] = useState<string | null>(null);
    const [lossLandscapeLoading, setLossLandscapeLoading] = useState(false);
    const lossLandscapeRequestInFlightRef = useRef(false);

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

    const layerStats = useMemo(
        () => {
            void frameVersion;
            return getFrameBuffer().layerStats ?? snapshot?.layerStats;
        },
        [frameVersion, snapshot?.layerStats],
    );

    const activationHistograms = useMemo(
        () => {
            void activationHistogramsVersion;
            const frame = getFrameBuffer();
            if (frame.activationHistogramBins && frame.activationHistogramLayout) {
                return {
                    bins: frame.activationHistogramBins,
                    layout: frame.activationHistogramLayout,
                };
            }
            if (snapshot?.activationHistograms) {
                return {
                    bins: snapshot.activationHistograms.bins,
                    layout: {
                        binCount: snapshot.activationHistograms.layers[0]?.binCount ?? 0,
                        layers: snapshot.activationHistograms.layers,
                    } satisfies ActivationHistogramLayout,
                };
            }
            return null;
        },
        [activationHistogramsVersion, snapshot?.activationHistograms],
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
    const selectedSample = traceSource === 'custom'
        ? null
        : selectedPoints[Math.min(sampleIndex, Math.max(0, selectedPoints.length - 1))];
    const canTrace = traceSource === 'custom' || selectedSample !== undefined;

    const handleTrace = async () => {
        if (!canTrace || traceLoading) return;
        setTraceLoading(true);
        setTraceError(null);
        try {
            const response = await getWorkerApi().getPredictionTrace(
                traceSource === 'custom'
                    ? { source: 'custom', x: customX, y: customY }
                    : { source: traceSource, index: Math.min(sampleIndex, selectedPoints.length - 1) },
            );
            setTraceResult(response);
        } catch (err) {
            setTraceResult(null);
            setTraceError(err instanceof Error ? err.message : String(err));
        } finally {
            setTraceLoading(false);
        }
    };

    const handleBackpropPreview = async () => {
        if (backpropRequestInFlightRef.current) return;
        backpropRequestInFlightRef.current = true;
        setBackpropLoading(true);
        setBackpropError(null);
        setBackpropResult(null);
        try {
            const response = await getWorkerApi().getBackpropExplanation();
            setBackpropResult(response);
        } catch (err) {
            setBackpropResult(null);
            setBackpropError(err instanceof Error ? err.message : String(err));
        } finally {
            backpropRequestInFlightRef.current = false;
            setBackpropLoading(false);
        }
    };

    const handleLossLandscapeProbe = async () => {
        if (lossLandscapeRequestInFlightRef.current) return;
        lossLandscapeRequestInFlightRef.current = true;
        setLossLandscapeLoading(true);
        setLossLandscapeError(null);
        setLossLandscapeResult(null);
        try {
            const response = await getWorkerApi().getLossLandscapeProbe();
            setLossLandscapeResult(response);
        } catch (err) {
            setLossLandscapeResult(null);
            setLossLandscapeError(err instanceof Error ? err.message : String(err));
        } finally {
            lossLandscapeRequestInFlightRef.current = false;
            setLossLandscapeLoading(false);
        }
    };

    const lossLandscapeSummary = lossLandscapeResult
        ? `Local 2D loss slice: center ${formatBackpropMetric(lossLandscapeResult.probe.centerLoss)}, min ${formatBackpropMetric(lossLandscapeResult.probe.minLoss)}, max ${formatBackpropMetric(lossLandscapeResult.probe.maxLoss)}. Best direction ${lossLandscapeResult.probe.axisA.parameter.label} ${formatSignedOffset(lossLandscapeResult.probe.best.offsetA)}, ${lossLandscapeResult.probe.axisB.parameter.label} ${formatSignedOffset(lossLandscapeResult.probe.best.offsetB)}.`
        : 'Loss landscape probe has not run yet.';
    const lossLandscapeSpread = lossLandscapeResult
        ? Math.max(0.000001, lossLandscapeResult.probe.maxLoss - lossLandscapeResult.probe.minLoss)
        : 1;

    return (
        <div className="inspection-panel">
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
                            setTraceSource(event.currentTarget.value as PredictionTraceSampleSource);
                            setTraceResult(null);
                            setTraceError(null);
                        }}
                    >
                        <option value="train">Training</option>
                        <option value="test">Test</option>
                        <option value="custom">Custom</option>
                    </select>
                </div>
                {traceSource === 'custom' ? (
                    <>
                        <div className="control-row">
                            <label htmlFor="trace-x">x</label>
                            <input
                                id="trace-x"
                                className="input"
                                type="number"
                                step="0.1"
                                value={customX}
                                onChange={(event) => setCustomX(Number(event.currentTarget.value))}
                            />
                        </div>
                        <div className="control-row">
                            <label htmlFor="trace-y">y</label>
                            <input
                                id="trace-y"
                                className="input"
                                type="number"
                                step="0.1"
                                value={customY}
                                onChange={(event) => setCustomY(Number(event.currentTarget.value))}
                            />
                        </div>
                    </>
                ) : (
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
                                setSampleIndex(Number.isFinite(next) ? Math.max(0, Math.trunc(next)) : 0);
                            }}
                        />
                    </div>
                )}

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
                        <div className="inspection__empty">Trace failed: {traceError}</div>
                    ) : null}
                    {traceResult ? (
                        <div className="inspection__layers">
                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">Output</span>
                                <span className="inspection__stat-value">
                                    {traceResult.trace.output.map((value) => value.toFixed(4)).join(', ')}
                                </span>
                                {traceResult.trace.lossContribution !== undefined ? (
                                    <>
                                        <span className="inspection__stat-label" style={{ marginLeft: 8 }}>loss</span>
                                        <span className="inspection__stat-value">
                                            {traceResult.trace.lossContribution.toFixed(4)}
                                        </span>
                                    </>
                                ) : null}
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
                                Preview from step {backpropResult.step} / epoch {backpropResult.epoch}
                            </span>
                        </div>
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">batch {backpropResult.explanation.batchSize}</span>
                            <span className="inspection__stat-value">
                                loss {formatBackpropMetric(backpropResult.explanation.loss)}
                            </span>
                            <span className="inspection__stat-value">
                                lr {formatBackpropMetric(backpropResult.explanation.learningRate)}
                            </span>
                        </div>
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                gradient norm {formatBackpropMetric(backpropResult.explanation.globalGradientNorm)}
                            </span>
                            <span className="inspection__stat-value">
                                {backpropResult.explanation.clipped
                                    ? `clipped ${formatBackpropMetric(backpropResult.explanation.globalClipScale)}x`
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
                                Probe from step {lossLandscapeResult.step} / epoch {lossLandscapeResult.epoch}
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
                            {lossLandscapeResult.probe.losses.map((loss, index) => {
                                const intensity = 1 - ((loss - lossLandscapeResult.probe.minLoss) / lossLandscapeSpread);
                                return (
                                    <span
                                        key={`${index}-${loss}`}
                                        className="inspection__loss-cell"
                                        style={{ opacity: 0.28 + Math.max(0, Math.min(1, intensity)) * 0.72 }}
                                        aria-hidden="true"
                                    />
                                );
                            })}
                        </div>
                        <div className="inspection__histogram-summary">
                            <strong>Local loss slice</strong>
                            <span>center {formatBackpropMetric(lossLandscapeResult.probe.centerLoss)}</span>
                            <span>min {formatBackpropMetric(lossLandscapeResult.probe.minLoss)}</span>
                            <span>max {formatBackpropMetric(lossLandscapeResult.probe.maxLoss)}</span>
                            <span>
                                sampled {lossLandscapeResult.probe.sampleCount} points on a {lossLandscapeResult.probe.gridSize} by {lossLandscapeResult.probe.gridSize} grid
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
