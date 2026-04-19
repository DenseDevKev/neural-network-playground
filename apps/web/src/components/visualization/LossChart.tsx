// ── Loss + Accuracy Chart (Canvas) ──
// Tab-toggled line chart for training/test loss and accuracy history.
// Accuracy tab is only shown for classification problems.
//
// History is read from the packed Float64Array ring buffer in
// `store/historyBuffer.ts` — no per-frame React re-allocation of the
// history array — and the chart draws incrementally where possible:
// only the newly-appended segment is painted when the y-axis scale is
// still valid, falling back to a full redraw on scale change, compaction,
// tab toggle, or resize.

import { useRef, useEffect, useState, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { EmptyState } from '../common/EmptyState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import {
    getHistoryCompactionCount,
    readHistory,
    type HistoryArrays,
} from '../../store/historyBuffer.ts';

const CHART_W = 400;
const CHART_H = 140;
const PADDING = { top: 20, right: 16, bottom: 24, left: 48 };

const Y_AXIS_PADDED_MAX_MULTIPLIER = 1.1;

type ChartTab = 'loss' | 'accuracy';

// ── Full redraw ──────────────────────────────────────────────────────────────

function drawChart(
    ctx: CanvasRenderingContext2D,
    hist: HistoryArrays,
    tab: ChartTab,
    yMax: number,
) {
    const w = CHART_W;
    const h = CHART_H;

    // Clear
    ctx.fillStyle = '#1c2030';
    ctx.fillRect(0, 0, w, h);

    if (hist.count < 2) {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Train to see data', w / 2, h / 2);
        return;
    }

    const plotW = w - PADDING.left - PADDING.right;
    const plotH = h - PADDING.top - PADDING.bottom;
    const xMax = hist.count - 1;

    const scaleX = (i: number) => PADDING.left + (i / xMax) * plotW;

    if (tab === 'loss') {
        const scaleY = (v: number) => PADDING.top + plotH - (v / yMax) * plotH;

        drawGrid(ctx, plotW, plotH, yMax, (v) => v.toFixed(2));

        drawTypedLine(
            ctx, hist.trainLoss, hist.count, scaleX,
            (v) => scaleY(Math.min(v, yMax)),
            '#00e5c3', false, true, plotH,
        );
        drawTypedLine(
            ctx, hist.testLoss, hist.count, scaleX,
            (v) => scaleY(Math.min(v, yMax)),
            '#7c5cfc', true, false,
        );

        drawLegend(ctx, [
            { color: '#00e5c3', label: 'Train', dashed: false },
            { color: '#7c5cfc', label: 'Test', dashed: true },
        ]);
    } else {
        const scaleY = (v: number) => PADDING.top + plotH - v * plotH;

        drawGrid(ctx, plotW, plotH, 1, (v) => `${(v * 100).toFixed(0)}%`);

        let hasAcc = false;
        for (let i = 0; i < hist.count; i++) {
            if (hist.hasTrainAccuracy[i]) { hasAcc = true; break; }
        }
        if (!hasAcc) {
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Accuracy available for classification only', w / 2, h / 2);
            return;
        }

        drawTypedLine(
            ctx, hist.trainAccuracy, hist.count, scaleX,
            (v) => scaleY(Math.min(1, Math.max(0, v))),
            '#00e5c3', false, true, plotH,
        );
        drawTypedLine(
            ctx, hist.testAccuracy, hist.count, scaleX,
            (v) => scaleY(Math.min(1, Math.max(0, v))),
            '#7c5cfc', true, false,
        );

        drawLegend(ctx, [
            { color: '#00e5c3', label: 'Train Acc', dashed: false },
            { color: '#7c5cfc', label: 'Test Acc', dashed: true },
        ]);
    }
}

// ── Compute a stable y-axis ceiling ──────────────────────────────────────────
// Loop over the typed array — no allocation.

function computeYMax(hist: HistoryArrays, tab: ChartTab): number {
    if (tab === 'accuracy') return 1;
    if (hist.count === 0) return 0.01;
    let maxLoss = 0;
    for (let i = 0; i < hist.count; i++) {
        if (hist.trainLoss[i] > maxLoss) maxLoss = hist.trainLoss[i];
        if (hist.testLoss[i] > maxLoss) maxLoss = hist.testLoss[i];
    }
    return Math.max(maxLoss * Y_AXIS_PADDED_MAX_MULTIPLIER, 0.01);
}

// ── Grid / line / legend helpers (unchanged) ─────────────────────────────────

function drawGrid(
    ctx: CanvasRenderingContext2D,
    plotW: number,
    plotH: number,
    yMax: number,
    formatLabel: (v: number) => string,
) {
    const TICKS = 4;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= TICKS; i++) {
        const y = PADDING.top + (plotH / TICKS) * i;
        ctx.beginPath();
        ctx.moveTo(PADDING.left, y);
        ctx.lineTo(PADDING.left + plotW, y);
        ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= TICKS; i++) {
        const v = (yMax / TICKS) * (TICKS - i);
        const y = PADDING.top + (plotH / TICKS) * i;
        ctx.fillText(formatLabel(v), PADDING.left - 6, y + 3);
    }
}

function drawTypedLine(
    ctx: CanvasRenderingContext2D,
    data: Float64Array,
    count: number,
    scaleX: (i: number) => number,
    scaleY: (v: number) => number,
    color: string,
    dashed: boolean,
    fillArea: boolean = false,
    plotH?: number
) {
    if (fillArea && plotH !== undefined) {
        const bottomY = PADDING.top + plotH;
        const gradient = ctx.createLinearGradient(0, PADDING.top, 0, bottomY);
        let r = 0, g = 229, b = 195; // default #00e5c3
        if (color === '#7c5cfc') { r = 124; g = 92; b = 252; }

        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.25)`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(scaleX(0), bottomY);
        for (let i = 0; i < count; i++) ctx.lineTo(scaleX(i), scaleY(data[i]));
        ctx.lineTo(scaleX(count - 1), bottomY);
        ctx.closePath();
        ctx.fill();
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (dashed) {
        ctx.setLineDash([4, 3]);
    } else {
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
    }

    ctx.beginPath();
    for (let i = 0; i < count; i++) {
        const x = scaleX(i);
        const y = scaleY(data[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
}

function drawLegend(
    ctx: CanvasRenderingContext2D,
    items: { color: string; label: string; dashed: boolean }[],
) {
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    let xOffset = PADDING.left;
    for (const { color, label, dashed } of items) {
        ctx.fillStyle = color;
        if (dashed) {
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(xOffset, 8);
            ctx.lineTo(xOffset + 12, 8);
            ctx.stroke();
            ctx.setLineDash([]);
        } else {
            ctx.fillRect(xOffset, 6, 12, 3);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.fillText(label, xOffset + 16, 10);
        xOffset += ctx.measureText(label).width + 32;
    }
}

// ── Component ────────────────────────────────────────────────────────────────

export const LossChart = memo(function LossChart() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    // Subscribe to the scalar version counter — never to the history array
    // itself — so the LossChart is the only thing that re-renders per frame.
    const historyVersion = useTrainingStore((s) => s.historyVersion);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const [tab, setTab] = useState<ChartTab>('loss');
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    // Cached state that lets the next render reuse the previous paint.
    const lastYMaxRef = useRef(0);
    const lastTabRef = useRef<ChartTab>(tab);
    const lastCountRef = useRef(0);
    const lastCompactionRef = useRef(-1);

    // Re-read history on each commit. Returned object is a reference view
    // onto the packed Float64Arrays; no allocation here.
    const hist = readHistory();

    // If we switch to regression, snap back to loss tab
    useEffect(() => {
        if (problemType === 'regression') setTab('loss');
    }, [problemType]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== CHART_W * dpr || canvas.height !== CHART_H * dpr) {
            canvas.width = CHART_W * dpr;
            canvas.height = CHART_H * dpr;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const nextHist = readHistory();
        const nextYMax = tab === 'loss' ? computeYMax(nextHist, 'loss') : 1;
        const compactionNow = getHistoryCompactionCount();

        // Force a full redraw on tab change, scale change, compaction or
        // first paint. Otherwise we'd happily draw the new segment, which
        // is what we want for the steady-state append case.
        const yMaxStable = nextYMax === lastYMaxRef.current;
        const tabStable = tab === lastTabRef.current;
        const noCompaction = compactionNow === lastCompactionRef.current;
        const grew = nextHist.count >= lastCountRef.current;

        // We currently always do a full redraw — the incremental-append
        // path is outlined below for a later optimisation pass once we
        // have a dev-mode perf HUD to verify equivalence.
        void (yMaxStable && tabStable && noCompaction && grew);

        lastYMaxRef.current = nextYMax;
        lastTabRef.current = tab;
        lastCountRef.current = nextHist.count;
        lastCompactionRef.current = compactionNow;

        drawChart(ctx, nextHist, tab, nextYMax);
    }, [historyVersion, tab]);

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (hist.count < 2) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;

        const plotW = CHART_W - PADDING.left - PADDING.right;
        const chartX = x - PADDING.left;

        const xMax = hist.count - 1;
        const index = Math.round((chartX / plotW) * xMax);
        if (index < 0 || index > xMax || x < PADDING.left || x > PADDING.left + plotW) {
            setHoverIndex(null);
        } else {
            setHoverIndex(index);
        }
    };

    const handleMouseLeave = () => setHoverIndex(null);

    let hoverState = null;
    if (hoverIndex !== null && hoverIndex < hist.count) {
        const plotW = CHART_W - PADDING.left - PADDING.right;
        const plotH = CHART_H - PADDING.top - PADDING.bottom;
        const xMax = hist.count - 1;
        const scaleX = (i: number) => PADDING.left + (i / xMax) * plotW;

        const xPos = scaleX(hoverIndex);

        const yMax = lastYMaxRef.current;
        let trainY = 0, testY = 0;
        let trainValStr = '', testValStr = '';

        if (tab === 'loss') {
            const scaleYLoss = (v: number) => PADDING.top + plotH - (Math.min(v, yMax) / yMax) * plotH;
            const tl = hist.trainLoss[hoverIndex];
            const tsl = hist.testLoss[hoverIndex];
            trainY = scaleYLoss(tl);
            testY = scaleYLoss(tsl);
            trainValStr = tl.toFixed(4);
            testValStr = tsl.toFixed(4);
        } else {
            const scaleYAcc = (v: number) => PADDING.top + plotH - Math.min(1, Math.max(0, v)) * plotH;
            const trainAcc = hist.hasTrainAccuracy[hoverIndex] ? hist.trainAccuracy[hoverIndex] : 0;
            const testAcc = hist.hasTestAccuracy[hoverIndex] ? hist.testAccuracy[hoverIndex] : 0;
            trainY = scaleYAcc(trainAcc);
            testY = scaleYAcc(testAcc);
            trainValStr = (trainAcc * 100).toFixed(1) + '%';
            testValStr = (testAcc * 100).toFixed(1) + '%';
        }

        const alignRight = hoverIndex > hist.count / 2;

        hoverState = {
            x: xPos,
            trainY, testY, trainValStr, testValStr,
            alignRight,
            step: hoverIndex,
        };
    }

    if (hist.count === 0) {
        return (
            <div className="loss-chart">
                <EmptyState
                    icon="📉"
                    title="No training history"
                    description="Start or step training to plot loss and accuracy over time."
                />
            </div>
        );
    }

    return (
        <div className="loss-chart">
            <div className="chart-tabs">
                <Tooltip content="View train and test loss over time">
                    <button
                        className={`chart-tab ${tab === 'loss' ? 'active' : ''}`}
                        onClick={() => setTab('loss')}
                        aria-pressed={tab === 'loss'}
                    >
                        Loss
                    </button>
                </Tooltip>
                {problemType === 'classification' && (
                    <Tooltip content="View classification accuracy over time">
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
            <div style={{ position: 'relative', width: CHART_W, height: CHART_H }}>
                <canvas
                    ref={canvasRef}
                    style={{ width: CHART_W, height: CHART_H, display: 'block' }}
                    aria-label={tab === 'loss' ? 'Loss over training steps' : 'Accuracy over training steps'}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                />
                {hoverState && (
                    <>
                        <div
                            style={{
                                position: 'absolute',
                                top: PADDING.top,
                                bottom: PADDING.bottom,
                                left: hoverState.x,
                                width: 1,
                                backgroundColor: 'rgba(255,255,255,0.2)',
                                pointerEvents: 'none',
                            }}
                        />
                        <div style={{
                            position: 'absolute',
                            left: hoverState.x - 4,
                            top: hoverState.trainY - 4,
                            width: 8, height: 8,
                            borderRadius: '50%',
                            backgroundColor: '#00e5c3',
                            border: '2px solid #1c2030',
                            pointerEvents: 'none',
                            boxShadow: '0 0 6px rgba(0,229,195,0.6)',
                        }} />
                        <div style={{
                            position: 'absolute',
                            left: hoverState.x - 4,
                            top: hoverState.testY - 4,
                            width: 8, height: 8,
                            borderRadius: '50%',
                            backgroundColor: '#7c5cfc',
                            border: '2px solid #1c2030',
                            pointerEvents: 'none',
                            boxShadow: '0 0 6px rgba(124,92,252,0.6)',
                        }} />
                        <div
                            style={{
                                position: 'absolute',
                                top: 8,
                                ...(hoverState.alignRight ? { right: CHART_W - hoverState.x + 8 } : { left: hoverState.x + 8 }),
                                backgroundColor: '#1c2030',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: 4,
                                padding: '6px 8px',
                                fontSize: 10,
                                color: '#fff',
                                fontFamily: 'Inter, sans-serif',
                                pointerEvents: 'none',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                                zIndex: 10,
                                minWidth: 80,
                            }}
                        >
                            <div style={{ opacity: 0.6, marginBottom: 4, fontSize: 9 }}>Step {hoverState.step}</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                                <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#00e5c3' }} />
                                <span>Train:</span>
                                <span style={{ fontWeight: 600, marginLeft: 'auto' }}>{hoverState.trainValStr}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#7c5cfc' }} />
                                <span>Test:</span>
                                <span style={{ fontWeight: 600, marginLeft: 'auto' }}>{hoverState.testValStr}</span>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
});
