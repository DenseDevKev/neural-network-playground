// ── Decision Boundary Heatmap (Canvas) ──
// Renders the neural network's prediction grid as a smooth heatmap
// with training/test data points overlaid.

import { useRef, useEffect, useCallback, useState, memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { writeGridToImageData, HEX_BLUE, HEX_ORANGE } from '@nn-playground/shared';
import type { DataPoint } from '@nn-playground/engine';
import { EmptyState } from '../common/EmptyState.tsx';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';

// ── Constants ──
const BG_COLOR = '#151822';
const TRAIN_RADIUS = 3.5;
const TEST_RADIUS = 3;
const POINT_STROKE_DARK = 'rgba(0,0,0,0.5)';
const POINT_STROKE_LIGHT = '#fff';
const HEATMAP_ALPHA = 255;

interface Props {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
}

// ── Drawing helpers (pure functions, no hooks) ──

function drawHeatmap(
    ctx: CanvasRenderingContext2D,
    grid: ArrayLike<number>,
    canvasW: number,
    canvasH: number,
    discretize: boolean,
    tempCanvas: HTMLCanvasElement,
    imageData: ImageData,
): void {
    const tempCtx = tempCanvas.getContext('2d')!;

    writeGridToImageData(grid, imageData, HEATMAP_ALPHA, discretize);
    tempCtx.putImageData(imageData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, canvasW, canvasH);
}

function drawPoints(
    ctx: CanvasRenderingContext2D,
    points: DataPoint[],
    canvasW: number,
    canvasH: number,
    isTest: boolean,
): void {
    if (points.length === 0) return;

    const radius = isTest ? TEST_RADIUS : TRAIN_RADIUS;
    const orangeBatch: DataPoint[] = [];
    const blueBatch: DataPoint[] = [];

    for (const p of points) {
        if (p.label >= 0.5) {
            orangeBatch.push(p);
        } else {
            blueBatch.push(p);
        }
    }

    const renderBatch = (batch: DataPoint[], color: string) => {
        if (batch.length === 0) return;

        ctx.fillStyle = color;
        ctx.beginPath();

        for (const p of batch) {
            const px = ((p.x + 1) / 2) * canvasW;
            const py = (1 - (p.y + 1) / 2) * canvasH;
            ctx.moveTo(px + radius, py);
            ctx.arc(px, py, radius, 0, Math.PI * 2);
        }

        // Add subtle shadow for premium look
        ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 2;

        ctx.fill();

        // Reset shadow for stroke
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        if (isTest) {
            ctx.strokeStyle = POINT_STROKE_LIGHT;
            ctx.lineWidth = 1;
        } else {
            ctx.strokeStyle = POINT_STROKE_DARK;
            ctx.lineWidth = 0.5;
        }
        ctx.stroke();
    };

    renderBatch(orangeBatch, HEX_ORANGE);
    renderBatch(blueBatch, HEX_BLUE);
}

// ── Component ──

export const DecisionBoundary = memo(function DecisionBoundary({
    trainPoints,
    testPoints,
    showTestData,
    discretize,
}: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Off-screen resources (reused between frames)
    const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const imageDataRef = useRef<ImageData | null>(null);
    const lastGridSizeRef = useRef(0);

    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);

    // Track container size for responsive canvas
    const [canvasSize, setCanvasSize] = useState(320);

    // Observe container size changes
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const width = Math.round(entry.contentRect.width);
                if (width > 0) {
                    setCanvasSize(width);
                }
            }
        });

        observer.observe(container);
        // Set initial size
        const initialWidth = container.clientWidth;
        if (initialWidth > 0) {
            setCanvasSize(initialWidth);
        }

        return () => observer.disconnect();
    }, []);

    // Main paint callback – extracted so useEffect stays clean
    const paint = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const logicalW = canvasSize;
        const logicalH = canvasSize;
        const physicalW = Math.round(logicalW * dpr);
        const physicalH = Math.round(logicalH * dpr);

        // Only resize the backing buffer when dimensions actually change
        if (canvas.width !== physicalW || canvas.height !== physicalH) {
            canvas.width = physicalW;
            canvas.height = physicalH;
        }

        // Reset transform and scale for DPR
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Clear to background
        ctx.fillStyle = BG_COLOR;
        ctx.fillRect(0, 0, logicalW, logicalH);

        // Draw heatmap when grid data exists
        const frameBuffer = getFrameBuffer();
        const grid = frameBuffer.outputGrid ?? snapshot?.outputGrid;
        const gridSize = frameBuffer.outputGrid ? frameBuffer.gridSize : snapshot?.gridSize ?? 0;

        if (grid && gridSize > 0) {
            // Allocate / reuse offscreen canvas + ImageData
            if (!tempCanvasRef.current || lastGridSizeRef.current !== gridSize) {
                const tc = document.createElement('canvas');
                tc.width = gridSize;
                tc.height = gridSize;
                tempCanvasRef.current = tc;
                imageDataRef.current = tc.getContext('2d')!.createImageData(gridSize, gridSize);
                lastGridSizeRef.current = gridSize;
            }

            drawHeatmap(
                ctx,
                grid,
                logicalW,
                logicalH,
                discretize,
                tempCanvasRef.current!,
                imageDataRef.current!,
            );
        }

        // Draw data points on top
        if (trainPoints.length > 0) {
            drawPoints(ctx, trainPoints, logicalW, logicalH, false);
        }
        if (showTestData && testPoints.length > 0) {
            drawPoints(ctx, testPoints, logicalW, logicalH, true);
        }
    }, [snapshot, frameVersion, trainPoints, testPoints, showTestData, discretize, canvasSize]);

    // Paint immediately after React commits the latest frame. Snapshot delivery
    // is already rAF-gated in workerBridge, so an extra rAF here can keep
    // canceling the pending paint while training is running.
    useEffect(() => {
        paint();
    }, [paint]);

    // ── Early-return AFTER all hooks ──
    if (trainPoints.length === 0) {
        return (
            <div className="decision-boundary" ref={containerRef}>
                <EmptyState
                    icon="🎯"
                    title="No training data"
                    description="Generate data or reset the playground to populate the decision boundary."
                />
            </div>
        );
    }

    return (
        <div className="decision-boundary" ref={containerRef}>
            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%' }}
                aria-label="Decision boundary visualization showing the neural network's classification regions"
            />
            <div className="decision-boundary__legend">
                <div className="decision-boundary__legend-item">
                    <div className="decision-boundary__swatch" style={{ background: HEX_BLUE }} />
                    <span>Negative</span>
                </div>
                <div className="decision-boundary__legend-item">
                    <div className="decision-boundary__swatch" style={{ background: HEX_ORANGE }} />
                    <span>Positive</span>
                </div>
            </div>
        </div>
    );
});
