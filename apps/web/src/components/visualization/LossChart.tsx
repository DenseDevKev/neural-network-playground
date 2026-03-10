// ── Loss + Accuracy Chart (Canvas) ──
// Tab-toggled line chart for training/test loss and accuracy history.
// Accuracy tab is only shown for classification problems.

import { useRef, useEffect, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

const CHART_W = 400;
const CHART_H = 140;
const PADDING = { top: 20, right: 16, bottom: 24, left: 48 };

type ChartTab = 'loss' | 'accuracy';

function drawChart(
    ctx: CanvasRenderingContext2D,
    history: { trainLoss: number; testLoss: number; trainAccuracy?: number; testAccuracy?: number }[],
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
        const maxLoss = Math.max(...history.map((p) => Math.max(p.trainLoss, p.testLoss)));
        const yMax = Math.max(maxLoss * 1.1, 0.01);
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

export function LossChart() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const history = usePlaygroundStore((s) => s.history);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const [tab, setTab] = useState<ChartTab>('loss');

    // If we switch to regression, snap back to loss tab
    useEffect(() => {
        if (problemType === 'regression') setTab('loss');
    }, [problemType]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        drawChart(ctx, history, tab);
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
}
