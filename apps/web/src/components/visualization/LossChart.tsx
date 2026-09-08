import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { EvaluationPoint, TrainingTrendPoint } from '@nn-playground/shared';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { metricHistoryBuffer } from '../../store/metricHistoryBuffer.ts';
import { EmptyState } from '../common/EmptyState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';

type ChartTab = 'loss' | 'accuracy';
export function deriveLossChartViewport(width: number) {
    const roundedWidth = Number.isFinite(width) ? Math.max(1, Math.round(width)) : 1;
    return { width: roundedWidth, height: Math.max(96, Math.min(140, Math.round(roundedWidth * 0.35))) };
}
const PADDING = { top: 18, right: 16, bottom: 24, left: 42 } as const;

interface ScalarPoint {
    readonly step: number;
    readonly value: number;
}

function finitePoints(points: readonly ScalarPoint[]): readonly ScalarPoint[] {
    return points.filter((point) => Number.isFinite(point.value));
}

function drawSeries(
    ctx: CanvasRenderingContext2D,
    points: readonly ScalarPoint[],
    color: string,
    stepMin: number,
    stepMax: number,
    valueMax: number,
    width: number,
    height: number,
): void {
    const values = finitePoints(points);
    if (values.length === 0) return;
    const plotWidth = Math.max(1, width - PADDING.left - PADDING.right);
    const plotHeight = height - PADDING.top - PADDING.bottom;
    const stepRange = Math.max(1, stepMax - stepMin);
    const scaleX = (step: number) => PADDING.left + ((step - stepMin) / stepRange) * plotWidth;
    const scaleY = (value: number) => PADDING.top + plotHeight - (value / valueMax) * plotHeight;

    ctx.beginPath();
    values.forEach((point, index) => {
        const x = scaleX(point.step);
        const y = scaleY(point.value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
}

function drawChart(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    tab: ChartTab,
    trends: readonly TrainingTrendPoint[],
    evaluations: readonly EvaluationPoint[],
): void {
    const trend = tab === 'loss'
        ? trends.map((point) => ({ step: point.model.step, value: point.dataLoss }))
        : [];
    const train = evaluations.map((point) => ({
        step: point.model.step,
        value: tab === 'loss'
            ? point.train.values.dataLoss
            : point.train.values.accuracy ?? Number.NaN,
    }));
    const test = evaluations.map((point) => ({
        step: point.model.step,
        value: tab === 'loss'
            ? point.test.values.dataLoss
            : point.test.values.accuracy ?? Number.NaN,
    }));
    const objective = tab === 'loss'
        ? evaluations.map((point) => ({
            step: point.model.step,
            value: point.objective.trainTotalObjective,
        }))
        : [];
    const all = [...trend, ...train, ...test, ...objective];
    if (all.length === 0) return;
    const steps = all.map((point) => point.step);
    const values = all.map((point) => point.value).filter(Number.isFinite);
    const stepMin = Math.min(...steps);
    const stepMax = Math.max(...steps);
    const valueMax = tab === 'accuracy'
        ? 1
        : Math.max(0.000001, ...values) * 1.05;

    ctx.fillStyle = 'rgba(12, 16, 28, 0.9)';
    ctx.fillRect(0, 0, width, height);
    drawSeries(ctx, trend, '#f6c85f', stepMin, stepMax, valueMax, width, height);
    drawSeries(ctx, train, '#00e5c3', stepMin, stepMax, valueMax, width, height);
    drawSeries(ctx, test, '#7c5cfc', stepMin, stepMax, valueMax, width, height);
    drawSeries(ctx, objective, '#ff8f70', stepMin, stepMax, valueMax, width, height);

    ctx.fillStyle = 'rgba(255,255,255,0.58)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(stepMin.toLocaleString(), PADDING.left, height - 7);
    ctx.textAlign = 'right';
    ctx.fillText(stepMax.toLocaleString(), width - PADDING.right, height - 7);
}

function formatSigned(value: number): string {
    return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`;
}

function isPlateau(evaluations: readonly EvaluationPoint[]): boolean {
    if (evaluations.length < 8) return false;
    const recent = evaluations.slice(-8).map((point) => point.test.values.dataLoss);
    return Math.max(...recent) - Math.min(...recent) <= 1e-6;
}

export const LossChart = memo(function LossChart() {
    const guidanceLevel = useAudienceGuidanceLevel();
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const trainingTrendVersion = useTrainingStore((state) => state.trainingTrendVersion);
    const evaluationHistoryVersion = useTrainingStore((state) => state.evaluationHistoryVersion);
    const taskKind = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared.document.recipe.task.kind
        : null);
    const [tab, setTab] = useState<ChartTab>('loss');
    const [viewport, setViewport] = useState(() => deriveLossChartViewport(320));

    const history = useMemo(() => {
        void trainingTrendVersion;
        void evaluationHistoryVersion;
        return metricHistoryBuffer.read();
    }, [evaluationHistoryVersion, trainingTrendVersion]);

    useEffect(() => {
        if (taskKind === 'regression') setTab('loss');
    }, [taskKind]);

    useEffect(() => {
        const container = containerRef.current;
        const Observer = typeof window === 'undefined' ? undefined : window.ResizeObserver;
        if (!container || !Observer) return;
        let frame: number | null = null;
        let pendingWidth = container.clientWidth;
        let stopped = false;
        const schedule = (width: number) => {
            if (stopped || !Number.isFinite(width) || width <= 0) return;
            pendingWidth = width;
            if (frame !== null) return;
            frame = window.requestAnimationFrame(() => {
                frame = null;
                if (stopped) return;
                const next = deriveLossChartViewport(pendingWidth);
                setViewport((current) => current.width === next.width && current.height === next.height
                    ? current : next);
            });
        };
        const observer = new Observer((entries) => {
            for (const entry of entries) schedule(entry.contentRect.width);
        });
        observer.observe(container);
        schedule(container.clientWidth);
        return () => {
            stopped = true;
            observer.disconnect();
            if (frame !== null) window.cancelAnimationFrame(frame);
        };
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(viewport.width * dpr));
        canvas.height = Math.max(1, Math.round(viewport.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawChart(
            ctx,
            viewport.width,
            viewport.height,
            tab,
            history.trendHistory,
            history.evaluationHistory,
        );
    }, [viewport, history, tab]);

    if (history.trendHistory.length === 0 && history.evaluationHistory.length === 0) {
        return (
            <div className="loss-chart" ref={containerRef}>
                <EmptyState
                    icon="📉"
                    title="No scientific metric history"
                    description="Start or step training to plot the batch trend and paired full evaluations."
                />
            </div>
        );
    }

    const latestTrend = history.trendHistory.at(-1) ?? null;
    const latestEvaluation = history.evaluationHistory.at(-1) ?? null;
    const bestTest = history.evaluationHistory.length === 0
        ? null
        : Math.min(...history.evaluationHistory.map((point) => point.test.values.dataLoss));
    const gap = latestEvaluation === null
        ? null
        : latestEvaluation.test.values.dataLoss - latestEvaluation.train.values.dataLoss;

    return (
        <div className="loss-chart" ref={containerRef}>
            <div className="chart-tabs">
                <Tooltip content="View data-loss and training-objective evidence by actual model step">
                    <button
                        className={`chart-tab ${tab === 'loss' ? 'active' : ''}`}
                        onClick={() => setTab('loss')}
                        aria-pressed={tab === 'loss'}
                    >
                        Loss
                    </button>
                </Tooltip>
                {taskKind !== 'regression' && (
                    <Tooltip content="View full-split classification accuracy by evaluation step">
                        <button
                            className={`chart-tab ${tab === 'accuracy' ? 'active' : ''}`}
                            onClick={() => setTab('accuracy')}
                            aria-pressed={tab === 'accuracy'}
                        >
                            Accuracy
                        </button>
                    </Tooltip>
                )}
            </div>
            <div className="loss-chart__legend" aria-label="Metric series">
                {tab === 'loss' ? (
                    <>
                        <span>
                            <span className="loss-chart__swatch loss-chart__swatch--trend" aria-hidden="true" />
                            Batch trend (EMA)
                        </span>
                        <span>
                            <span className="loss-chart__swatch loss-chart__swatch--train" aria-hidden="true" />
                            Train data loss (full split)
                        </span>
                        <span>
                            <span className="loss-chart__swatch loss-chart__swatch--test" aria-hidden="true" />
                            Test data loss (full split)
                        </span>
                        <span className="loss-chart__legend-concept">
                            <span className="loss-chart__swatch loss-chart__swatch--objective" aria-hidden="true" />
                            <span>Training objective</span>
                            <ConceptHelp
                                conceptId="training-objective"
                                guidanceLevel={guidanceLevel}
                                className="concept-help--end"
                            />
                        </span>
                    </>
                ) : (
                    <>
                        <span>
                            <span className="loss-chart__swatch loss-chart__swatch--train" aria-hidden="true" />
                            Train accuracy (full split)
                        </span>
                        <span>
                            <span className="loss-chart__swatch loss-chart__swatch--test" aria-hidden="true" />
                            Test accuracy (full split)
                        </span>
                    </>
                )}
            </div>
            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: viewport.height, display: 'block' }}
                aria-label={tab === 'loss'
                    ? 'Scientific loss evidence by actual model step'
                    : 'Full-split accuracy by evaluation model step'}
            />
            <p className="loss-chart__basis">
                {latestTrend
                    ? `Batch trend through step ${latestTrend.model.step.toLocaleString()} using latest batch size ${latestTrend.basis.latestBatchSize}.`
                    : 'No batch trend has been published.'}
                {' '}
                {latestEvaluation
                    ? `Full evaluation ${latestEvaluation.evaluationId} at step ${latestEvaluation.model.step.toLocaleString()} using all ${latestEvaluation.train.basis.sampleCount} train and ${latestEvaluation.test.basis.sampleCount} test examples.`
                    : 'No paired full evaluation has been published.'}
            </p>
            {tab === 'loss' && latestEvaluation && (
                <div aria-label="Loss diagnostics" className="loss-chart__diagnostics">
                    <span>{`Best test ${bestTest!.toFixed(4)}`}</span>
                    <span>{`Gap ${gap === null ? 'n/a' : formatSigned(gap)}`}</span>
                    <span>{`Penalty ${latestEvaluation.objective.regularizationPenalty.toFixed(4)}`}</span>
                    {isPlateau(history.evaluationHistory) && <span>Plateau</span>}
                </div>
            )}
        </div>
    );
});
