// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { useEffect, useMemo, useState, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { getWorkerApi } from '../../worker/workerBridge.ts';
import type {
    PredictionTraceResponse,
    PredictionTraceSampleSource,
} from '../../worker/training.worker.ts';

export const InspectionPanel = memo(function InspectionPanel() {
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const [traceSource, setTraceSource] = useState<PredictionTraceSampleSource>('train');
    const [sampleIndex, setSampleIndex] = useState(0);
    const [customX, setCustomX] = useState(0);
    const [customY, setCustomY] = useState(0);
    const [traceResult, setTraceResult] = useState<PredictionTraceResponse | null>(null);
    const [traceError, setTraceError] = useState<string | null>(null);
    const [traceLoading, setTraceLoading] = useState(false);

    useEffect(() => {
        const enableLayerStats = (needLayerStats: boolean) => {
            const { demand, setDemand } = usePlaygroundStore.getState();
            if (demand.needLayerStats === needLayerStats) return;
            setDemand({ ...demand, needLayerStats });
        };

        enableLayerStats(true);
        return () => enableLayerStats(false);
    }, []);

    const layerStats = useMemo(
        () => {
            void frameVersion;
            return getFrameBuffer().layerStats ?? snapshot?.layerStats;
        },
        [frameVersion, snapshot?.layerStats],
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
            <div className="inspection__layer" aria-label="Prediction trace">
                <div className="inspection__layer-name">Prediction Trace</div>
                <div className="control-row">
                    <label htmlFor="trace-source">Sample</label>
                    <select
                        id="trace-source"
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
                                <div key={layer.layerIndex} className="inspection__stat-row">
                                    <span className="inspection__stat-label">Layer {layer.layerIndex + 1}</span>
                                    <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                        {layer.activations.map((value) => value.toFixed(3)).join(', ')}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
});
