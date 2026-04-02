// ── Loss + Accuracy Chart (Canvas) ──
// Tab-toggled line chart for training/test loss and accuracy history.
// Accuracy tab is only shown for classification problems.
//
// Incremental drawing: each frame only appends new line segments to the canvas.
// Full redraw is triggered when: tab changes, y-axis scale expands, or history resets.

import { useRef, useEffect, useState, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

const CHART_W = 400;
const CHART_H = 140;
const PADDING = { top: 20, right: 16, bottom: 24, left: 48 };

type ChartTab = 'loss' | 'accuracy';
type HistoryPoint = { trainLoss: number; testLoss: number; trainAccuracy?: number; testAccuracy?: number };

// ── Full redraw ──────────────────────────────────────────────────────────────

function drawChart(
    ctx: CanvasRenderingContext2D,
    history: HistoryPoint[],
    tab: ChartTab,
) {
    const w = CHART_W;
    const h = CHART_H;

    // Clear
    ctx.fillStyle = '#1c2030';
    ctx.fillRect(0, 0, w, h);

    if (history.length < 2) {
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Train to see data', w / 2, h / 2);
        return;
    }

    const plotW = w - PADDING.left - PADDING.right;
    const plotH = h - PADDING.top - PADDING.bottom;
    const xMax = history.length - 1;

    const scaleX = (i: number) => PADDING.left + (i / xMax) * plotW;

    if (tab === 'loss') {
        // ── Loss chart ──
        const yMax = computeYMax(history, 'loss');
        const scaleY = (v: number) => PADDING.top + plotH - (v / yMax) * plotH;

        drawGrid(ctx, plotW, plotH, yMax, (v) => v.toFixed(2));

        // Train loss
        drawLine(ctx, history, scaleX, (p) => scaleY(Math.min(p.trainLoss, yMax)), '#00e5c3', false);
        // Test loss
        drawLine(ctx, history, scaleX, (p) => scaleY(Math.min(p.testLoss, yMax)), '#7c5cfc', true);

        drawLegend(ctx, [
            { color: '#00e5c3', label: 'Train', dashed: false },
            { color: '#7c5cfc', label: 'Test', dashed: true },
        ]);
    } else {
        // ── Accuracy chart ──
        // y-axis is fixed 0–100%
        const scaleY = (v: number) => PADDING.top + plotH - v * plotH;

        drawGrid(ctx, plotW, plotH, 1, (v) => `${(v * 100).toFixed(0)}%`);

        const hasAcc = history.some((p) => p.trainAccuracy !== undefined);
        if (!hasAcc) {
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Accuracy available for classification only', w / 2, h / 2);
            return;
        }

        // Train accuracy
        drawLine(
            ctx, history, scaleX,
            (p) => scaleY(Math.min(1, Math.max(0, p.trainAccuracy ?? 0))),
            '#00e5c3', false,
        );
        // Test accuracy
        drawLine(
            ctx, history, scaleX,
            (p) => scaleY(Math.min(1, Math.max(0, p.testAccuracy ?? 0))),
            '#7c5cfc', true,
        );

        drawLegend(ctx, [
            { color: '#00e5c3', label: 'Train Acc', dashed: false },
            { color: '#7c5cfc', label: 'Test Acc', dashed: true },
        ]);
    }
}

// ── Incremental draw: append only new line segments ──────────────────────────
// Uses the same yMax as the last full redraw (caller guarantees it hasn't changed).
// Slightly re-draws the last segment from (fromIndex-1) to anchor the new segments,
// which handles the sub-pixel x-axis shift from appending one more point.

function drawChartIncremental(
    ctx: CanvasRenderingContext2D,
    history: HistoryPoint[],
    fromIndex: number,
    tab: ChartTab,
    yMax: number,
) {
    if (fromIndex < 1 || fromIndex >= history.length) return;

    const plotW = CHART_W - PADDING.left - PADDING.right;
    const plotH = CHART_H - PADDING.top - PADDING.bottom;
    const xMax = history.length - 1;
    const scaleX = (i: number) => PADDING.left + (i / xMax) * plotW;

    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (tab === 'loss') {
        const scaleY = (v: number) => PADDING.top + plotH - (Math.min(v, yMax) / yMax) * plotH;

        // Extend train loss line from the last drawn point
        ctx.strokeStyle = '#00e5c3';
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainLoss));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainLoss));
        }
        ctx.stroke();

        // Extend test loss line (dashed)
        ctx.strokeStyle = '#7c5cfc';
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].testLoss));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].testLoss));
        }
        ctx.stroke();
        ctx.setLineDash([]);
    } else {
        // Accuracy tab — y-axis is fixed 0–1, no yMax needed from caller
        const scaleY = (v: number) => PADDING.top + plotH - Math.min(1, Math.max(0, v)) * plotH;
        const hasAcc = history.some((p) => p.trainAccuracy !== undefined);
        if (!hasAcc) return;

        // Extend train accuracy line
        ctx.strokeStyle = '#00e5c3';
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainAccuracy ?? 0));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainAccuracy ?? 0));
        }
        ctx.stroke();

        // Extend test accuracy line (dashed)
        ctx.strokeStyle = '#7c5cfc';
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].testAccuracy ?? 0));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].testAccuracy ?? 0));
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// ── Compute a stable y-axis ceiling ──────────────────────────────────────────
// Loop instead of spread+map to avoid GC pressure at 2000 history points.

function computeYMax(history: HistoryPoint[], tab: ChartTab): number {
    if (tab === 'accuracy') return 1;
    if (history.length === 0) return 0.01;
    let maxLoss = 0;
    for (const p of history) {
        if (p.trainLoss > maxLoss) maxLoss = p.trainLoss;
        if (p.testLoss > maxLoss) maxLoss = p.testLoss;
    }
    return Math.max(maxLoss * 1.1, 0.01);
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

function drawLine<T>(
    ctx: CanvasRenderingContext2D,
    data: T[],
    scaleX: (i: number) => number,
    scaleY: (d: T) => number,
    color: string,
    dashed: boolean,
) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    data.forEach((d, i) => {
        const x = scaleX(i);
        const y = scaleY(d);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
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
    const history = useTrainingStore((s) => s.history);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const [tab, setTab] = useState<ChartTab>('loss');

    // Incremental draw state — track what was already on the canvas
    const lastDrawnIndexRef = useRef(0);
    const lastYMaxRef = useRef(0);
    const lastTabRef = useRef<ChartTab>('loss');

    // If we switch to regression, snap back to loss tab
    useEffect(() => {
        if (problemType === 'regression') setTab('loss');
    }, [problemType]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const len = history.length;
        const tabChanged = tab !== lastTabRef.current;
        const historyShrank = len < lastDrawnIndexRef.current;
        const notEnoughData = len <= 1;

        // Determine if we need a full redraw
        let needFullRedraw = tabChanged || historyShrank || notEnoughData;
        let currentYMax = lastYMaxRef.current;

        if (!needFullRedraw && tab === 'loss') {
            // Fast path: only check NEW points to see if y-axis must expand.
            // This avoids O(N) scan every frame when the scale is stable.
            for (let i = lastDrawnIndexRef.current; i < len; i++) {
                const p = history[i];
                const maxThis = Math.max(p.trainLoss, p.testLoss);
                if (maxThis * 1.1 > currentYMax * 1.05) {
                    // Y-axis needs to grow — recompute full yMax and force full redraw
                    currentYMax = computeYMax(history, 'loss');
                    needFullRedraw = true;
                    break;
                }
            }
        }

        if (needFullRedraw) {
            // Recompute yMax from scratch for the full draw
            if (tab === 'loss') currentYMax = computeYMax(history, 'loss');
            drawChart(ctx, history, tab);
            lastDrawnIndexRef.current = len;
            lastYMaxRef.current = currentYMax;
            lastTabRef.current = tab;
        } else if (len > lastDrawnIndexRef.current) {
            // Incremental: append only the new line segments
            drawChartIncremental(ctx, history, lastDrawnIndexRef.current, tab, lastYMaxRef.current);
            lastDrawnIndexRef.current = len;
        }
        // else: len === lastDrawnIndex → no new data, skip canvas update entirely
    }, [history, tab]);

    return (
        <div className="loss-chart">
            <div className="chart-tabs">
                <button
                    className={`chart-tab ${tab === 'loss' ? 'active' : ''}`}
                    onClick={() => setTab('loss')}
                    aria-pressed={tab === 'loss'}
                >
                    Loss
                </button>
                {problemType === 'classification' && (
                    <button
                        className={`chart-tab ${tab === 'accuracy' ? 'active' : ''}`}
                        onClick={() => setTab('accuracy')}
                        aria-pressed={tab === 'accuracy'}
                    >
                        Accuracy
                    </button>
                )}
            </div>
            <canvas
                ref={canvasRef}
                width={CHART_W}
                height={CHART_H}
                aria-label={tab === 'loss' ? 'Loss over training steps' : 'Accuracy over training steps'}
            />
        </div>
    );
});
