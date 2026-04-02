// ── Decision Boundary Heatmap (Canvas) ──
// Renders the prediction grid as a heatmap with data points overlaid.

import { useRef, useEffect, memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { writeGridToImageData, HEX_BLUE, HEX_ORANGE } from '@nn-playground/shared';
import type { DataPoint } from '@nn-playground/engine';

const CANVAS_SIZE = 320;

interface Props {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
}

export const DecisionBoundary = memo(function DecisionBoundary({ trainPoints, testPoints, showTestData, discretize }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const imageDataRef = useRef<ImageData | null>(null);
    const lastGridSizeRef = useRef(0);
    const snapshot = useTrainingStore((s) => s.snapshot);

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
            // Reuse temp canvas and ImageData if dimensions match
            if (!tempCanvasRef.current || lastGridSizeRef.current !== gridSize) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = gridSize;
                tempCanvas.height = gridSize;
                tempCanvasRef.current = tempCanvas;
                const tempCtx = tempCanvas.getContext('2d')!;
                imageDataRef.current = tempCtx.createImageData(gridSize, gridSize);
                lastGridSizeRef.current = gridSize;
            }

            const tempCanvas = tempCanvasRef.current!;
            const tempCtx = tempCanvas.getContext('2d')!;
            const imageData = imageDataRef.current!;

            writeGridToImageData(grid, imageData, 200, discretize);
            tempCtx.putImageData(imageData, 0, 0);

            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(tempCanvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
        }

        // Always draw data points (even before training)
        const drawPoints = (points: DataPoint[], isTest: boolean): void => {
            const orangePoints: DataPoint[] = [];
            const bluePoints: DataPoint[] = [];

            for (const p of points) {
                if (p.label >= 0.5) {
                    orangePoints.push(p);
                } else {
                    bluePoints.push(p);
                }
            }

            const drawBatch = (batch: DataPoint[], color: string) => {
                if (batch.length === 0) return;

                ctx.fillStyle = color;
                ctx.beginPath();

                const radius = isTest ? 3 : 3.5;

                for (const p of batch) {
                    const px = ((p.x + 1) / 2) * CANVAS_SIZE;
                    const py = ((1 - (p.y + 1) / 2)) * CANVAS_SIZE;
                    ctx.moveTo(px + radius, py);
                    ctx.arc(px, py, radius, 0, 2 * Math.PI);
                }

                ctx.fill();

                if (isTest) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                } else {
                    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            };

            drawBatch(orangePoints, HEX_ORANGE);
            drawBatch(bluePoints, HEX_BLUE);
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
});
