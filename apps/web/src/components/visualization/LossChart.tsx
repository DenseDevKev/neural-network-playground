// ── Loss + Accuracy Chart (Canvas) ──
// Tab-toggled line chart for training/test loss and accuracy history.
// Accuracy tab is only shown for classification problems.
//
// Incremental drawing: each frame only appends new line segments to the canvas.
// Full redraw is triggered when: tab changes, y-axis scale expands, or history resets.

import { useRef, useEffect, useState, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { EmptyState } from '../common/EmptyState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

const CHART_W = 400;
const CHART_H = 140;
const PADDING = { top: 20, right: 16, bottom: 24, left: 48 };

const Y_AXIS_PADDED_MAX_MULTIPLIER = 1.1;
const Y_AXIS_REDRAW_THRESHOLD_MULTIPLIER = 1.05;

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
        drawLine(ctx, history, scaleX, (p) => scaleY(Math.min(p.trainLoss, yMax)), '#00e5c3', false, true, plotH);
        // Test loss
        drawLine(ctx, history, scaleX, (p) => scaleY(Math.min(p.testLoss, yMax)), '#7c5cfc', true, false);

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
            '#00e5c3', false, true, plotH
        );
        // Test accuracy
        drawLine(
            ctx, history, scaleX,
            (p) => scaleY(Math.min(1, Math.max(0, p.testAccuracy ?? 0))),
            '#7c5cfc', true, false
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
        const bottomY = PADDING.top + plotH + 1;

        // Area fill for train loss
        const gradient = ctx.createLinearGradient(0, PADDING.top, 0, bottomY);
        gradient.addColorStop(0, 'rgba(0, 229, 195, 0.25)'); // #00e5c3
        gradient.addColorStop(1, 'rgba(0, 229, 195, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        // overlap prior segment by 0.5px to suppress vertical seam
        ctx.moveTo(scaleX(fromIndex - 1) - 0.5, bottomY);
        ctx.lineTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainLoss));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainLoss));
        }
        ctx.lineTo(scaleX(history.length - 1), bottomY);
        ctx.closePath();
        ctx.fill();

        // Extend train loss line from the last drawn point
        ctx.strokeStyle = '#00e5c3';
        ctx.shadowColor = '#00e5c3';
        ctx.shadowBlur = 6;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainLoss));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainLoss));
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

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
        const bottomY = PADDING.top + plotH + 1;
        const hasAcc = history.some((p) => p.trainAccuracy !== undefined);
        if (!hasAcc) return;

        // Area fill for train accuracy
        const gradient = ctx.createLinearGradient(0, PADDING.top, 0, bottomY);
        gradient.addColorStop(0, 'rgba(0, 229, 195, 0.25)'); // #00e5c3
        gradient.addColorStop(1, 'rgba(0, 229, 195, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        // overlap prior segment by 0.5px
        ctx.moveTo(scaleX(fromIndex - 1) - 0.5, bottomY);
        ctx.lineTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainAccuracy ?? 0));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainAccuracy ?? 0));
        }
        ctx.lineTo(scaleX(history.length - 1), bottomY);
        ctx.closePath();
        ctx.fill();

        // Extend train accuracy line
        ctx.strokeStyle = '#00e5c3';
        ctx.shadowColor = '#00e5c3';
        ctx.shadowBlur = 6;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(scaleX(fromIndex - 1), scaleY(history[fromIndex - 1].trainAccuracy ?? 0));
        for (let i = fromIndex; i < history.length; i++) {
            ctx.lineTo(scaleX(i), scaleY(history[i].trainAccuracy ?? 0));
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

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

function drawLine<T>(
    ctx: CanvasRenderingContext2D,
    data: T[],
    scaleX: (i: number) => number,
    scaleY: (d: T) => number,
    color: string,
    dashed: boolean,
    fillArea: boolean = false,
    plotH?: number
) {
    if (fillArea && plotH !== undefined) {
        const bottomY = PADDING.top + plotH;
        const gradient = ctx.createLinearGradient(0, PADDING.top, 0, bottomY);
        // Manually map the two main colors to rgba for gradient
        let r = 0, g = 229, b = 195; // default #00e5c3
        if (color === '#7c5cfc') { r = 124; g = 92; b = 252; }

        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.25)`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(scaleX(0), bottomY);
        data.forEach((d, i) => {
            ctx.lineTo(scaleX(i), scaleY(d));
        });
        ctx.lineTo(scaleX(data.length - 1), bottomY);
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
    data.forEach((d, i) => {
        const x = scaleX(i);
        const y = scaleY(d);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
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
    const history = useTrainingStore((s) => s.history);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const [tab, setTab] = useState<ChartTab>('loss');
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

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
        const justGotEnoughData = len > 1 && lastDrawnIndexRef.current <= 1;

        // Determine if we need a full redraw
        let needFullRedraw = tabChanged || historyShrank || notEnoughData || justGotEnoughData;
        let currentYMax = lastYMaxRef.current;

        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== CHART_W * dpr || canvas.height !== CHART_H * dpr) {
            canvas.width = CHART_W * dpr;
            canvas.height = CHART_H * dpr;
            ctx.scale(dpr, dpr);
            needFullRedraw = true; // Canvas was cleared when width/height changed
        }

        if (!needFullRedraw && tab === 'loss') {
            // Fast path: only check NEW points to see if y-axis must expand.
            // This avoids O(N) scan every frame when the scale is stable.
            for (let i = lastDrawnIndexRef.current; i < len; i++) {
                const p = history[i];
                const maxThis = Math.max(p.trainLoss, p.testLoss);
                if (maxThis * Y_AXIS_PADDED_MAX_MULTIPLIER > currentYMax * Y_AXIS_REDRAW_THRESHOLD_MULTIPLIER) {
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

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (history.length < 2) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        
        const plotW = CHART_W - PADDING.left - PADDING.right;
        const chartX = x - PADDING.left;
        
        const xMax = history.length - 1;
        let index = Math.round((chartX / plotW) * xMax);
        if (index < 0 || index > xMax || x < PADDING.left || x > PADDING.left + plotW) {
            setHoverIndex(null);
        } else {
            setHoverIndex(index);
        }
    };

    const handleMouseLeave = () => setHoverIndex(null);

    let hoverState = null;
    if (hoverIndex !== null && hoverIndex < history.length) {
        const plotW = CHART_W - PADDING.left - PADDING.right;
        const plotH = CHART_H - PADDING.top - PADDING.bottom;
        const xMax = history.length - 1;
        const scaleX = (i: number) => PADDING.left + (i / xMax) * plotW;
        
        const xPos = scaleX(hoverIndex);
        const point = history[hoverIndex];
        
        const yMax = lastYMaxRef.current;
        let trainY = 0, testY = 0;
        let trainValStr = '', testValStr = '';
        
        if (tab === 'loss') {
            const scaleYLoss = (v: number) => PADDING.top + plotH - (Math.min(v, yMax) / yMax) * plotH;
            trainY = scaleYLoss(point.trainLoss);
            testY = scaleYLoss(point.testLoss);
            trainValStr = point.trainLoss.toFixed(4);
            testValStr = point.testLoss.toFixed(4);
        } else {
            const scaleYAcc = (v: number) => PADDING.top + plotH - Math.min(1, Math.max(0, v)) * plotH;
            trainY = scaleYAcc(point.trainAccuracy ?? 0);
            testY = scaleYAcc(point.testAccuracy ?? 0);
            trainValStr = ((point.trainAccuracy ?? 0) * 100).toFixed(1) + '%';
            testValStr = ((point.testAccuracy ?? 0) * 100).toFixed(1) + '%';
        }

        const alignRight = hoverIndex > history.length / 2;
        
        hoverState = {
            x: xPos,
            trainY, testY, trainValStr, testValStr,
            alignRight,
            step: hoverIndex,
        };
    }

    if (history.length === 0) {
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
