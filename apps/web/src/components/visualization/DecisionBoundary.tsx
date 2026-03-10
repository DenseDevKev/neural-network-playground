// ── Decision Boundary Heatmap (Canvas) ──
// Renders the prediction grid as a heatmap with data points overlaid.

import { useRef, useEffect } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import type { DataPoint } from '@nn-playground/engine';

const CANVAS_SIZE = 320;

interface Props {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
}

function valueToColor(v: number, discretize: boolean): [number, number, number] {
    // v is typically 0–1 for classification (sigmoid output)
    // Map to blue (0) → dark (0.5) → orange (1)
    let t = v;
    if (discretize) {
        t = t > 0.5 ? 1 : 0;
    }
    // Clamp
    t = Math.max(0, Math.min(1, t));

    if (t < 0.5) {
        // Blue to dark
        const p = t / 0.5;
        return [
            Math.round(59 * (1 - p) + 28 * p),
            Math.round(130 * (1 - p) + 32 * p),
            Math.round(246 * (1 - p) + 48 * p),
        ];
    } else {
        // Dark to orange
        const p = (t - 0.5) / 0.5;
        return [
            Math.round(28 * (1 - p) + 249 * p),
            Math.round(32 * (1 - p) + 115 * p),
            Math.round(48 * (1 - p) + 22 * p),
        ];
    }
}

export function DecisionBoundary({ trainPoints, testPoints, showTestData, discretize }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const snapshot = usePlaygroundStore((s) => s.snapshot);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const grid = snapshot?.outputGrid;
        const gridSize = snapshot?.gridSize ?? 0;

        // Clear
        ctx.fillStyle = '#151822';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        // Draw heatmap (only when grid data is available)
        if (grid && gridSize > 0) {
            const imageData = ctx.createImageData(gridSize, gridSize);

            for (let i = 0; i < grid.length; i++) {
                const [r, g, b] = valueToColor(grid[i], discretize);
                const idx = i * 4;
                imageData.data[idx] = r;
                imageData.data[idx + 1] = g;
                imageData.data[idx + 2] = b;
                imageData.data[idx + 3] = 200;
            }

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = gridSize;
            tempCanvas.height = gridSize;
            const tempCtx = tempCanvas.getContext('2d')!;
            tempCtx.putImageData(imageData, 0, 0);

            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(tempCanvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
        }

        // Always draw data points (even before training)
        const drawPoints = (points: DataPoint[], isTest: boolean) => {
            for (const p of points) {
                const px = ((p.x + 1) / 2) * CANVAS_SIZE;
                const py = ((1 - (p.y + 1) / 2)) * CANVAS_SIZE;

                ctx.beginPath();
                ctx.arc(px, py, isTest ? 3 : 3.5, 0, 2 * Math.PI);

                if (p.label >= 0.5) {
                    ctx.fillStyle = '#f97316';
                } else {
                    ctx.fillStyle = '#3b82f6';
                }

                if (isTest) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.fill();
                } else {
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        };

        drawPoints(trainPoints, false);
        if (showTestData) {
            drawPoints(testPoints, true);
        }
    }, [snapshot, trainPoints, testPoints, showTestData, discretize]);

    return (
        <div className="decision-boundary">
            <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                aria-label="Decision boundary visualization"
            />
        </div>
    );
}
