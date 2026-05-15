// ── Decision Boundary Heatmap (Canvas) ──
// Renders the neural network's prediction grid as a smooth heatmap
// with training/test data points overlaid.

import { useRef, useEffect, useCallback, useState, memo, useId } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { writeGridToImageData, HEX_BLUE, HEX_ORANGE } from '@nn-playground/shared';
import type { MulticlassBoundaryLayout } from '@nn-playground/shared';
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
const UNCERTAINTY_THRESHOLD = 0.12;
const MULTICLASS_LOW_CONFIDENCE_THRESHOLD = 0.6;
const MULTICLASS_PALETTE = [
    { label: 'Class 0', color: '#4f8cff', rgb: [79, 140, 255] },
    { label: 'Class 1', color: '#ff9f43', rgb: [255, 159, 67] },
    { label: 'Class 2', color: '#52d273', rgb: [82, 210, 115] },
] as const;

interface Props {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
    overlayMode?: DecisionOverlayMode;
}

export type DecisionOverlayMode = 'none' | 'uncertainty' | 'misclassification' | 'split';

export interface DecisionOverlayCopy {
    label: string;
    description: string;
}

interface MulticlassBoundaryFrame {
    classGrid: Uint8Array;
    confidenceGrid: Float32Array;
    layout: MulticlassBoundaryLayout;
}

interface MulticlassBoundarySummary {
    dominantLabel: string;
    dominantShare: number;
    averageConfidence: number;
    lowConfidenceShare: number;
    description: string;
}

export function getDecisionOverlayCopy(
    mode: DecisionOverlayMode,
    showTestData: boolean,
    discretize: boolean,
): DecisionOverlayCopy {
    switch (mode) {
        case 'uncertainty':
            return {
                label: 'Uncertainty',
                description:
                    'Uncertainty mode brightens regions near 50% probability, where the model is least sure which class to predict.',
            };
        case 'misclassification':
            return {
                label: 'Misclassified',
                description: showTestData
                    ? 'Errors mode marks training and visible test points whose predicted class does not match the label.'
                    : 'Errors mode marks training points whose predicted class does not match the label.',
            };
        case 'split':
            return {
                label: 'Train/test split',
                description:
                    'Split mode keeps held-out test points visible beside training points so you can compare fit against generalization.',
            };
        case 'none':
        default:
            return {
                label: 'Output',
                description: discretize
                    ? 'Output mode shows hard class regions: blue for negative predictions and orange for positive predictions.'
                    : 'Output mode shows predicted probability as a smooth field from negative blue to positive orange.',
            };
    }
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

function writeMulticlassBoundaryImageData(
    classGrid: Uint8Array,
    confidenceGrid: Float32Array,
    imageData: ImageData,
): void {
    for (let i = 0; i < classGrid.length; i++) {
        const palette = MULTICLASS_PALETTE[classGrid[i]] ?? MULTICLASS_PALETTE[0];
        const confidence = Math.max(0, Math.min(1, confidenceGrid[i]));
        const mix = 0.35 + confidence * 0.65;
        const idx = i * 4;
        imageData.data[idx] = Math.round(21 + (palette.rgb[0] - 21) * mix);
        imageData.data[idx + 1] = Math.round(24 + (palette.rgb[1] - 24) * mix);
        imageData.data[idx + 2] = Math.round(34 + (palette.rgb[2] - 34) * mix);
        imageData.data[idx + 3] = HEATMAP_ALPHA;
    }
}

function drawMulticlassHeatmap(
    ctx: CanvasRenderingContext2D,
    classGrid: Uint8Array,
    confidenceGrid: Float32Array,
    canvasW: number,
    canvasH: number,
    tempCanvas: HTMLCanvasElement,
    imageData: ImageData,
): void {
    const tempCtx = tempCanvas.getContext('2d')!;
    writeMulticlassBoundaryImageData(classGrid, confidenceGrid, imageData);
    tempCtx.putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, canvasW, canvasH);
}

export function classifyPointFromGrid(
    point: DataPoint,
    grid: ArrayLike<number>,
    gridSize: number,
): 0 | 1 {
    const gx = Math.max(0, Math.min(gridSize - 1, Math.round(((point.x + 1) / 2) * (gridSize - 1))));
    const gy = Math.max(0, Math.min(gridSize - 1, Math.round((1 - (point.y + 1) / 2) * (gridSize - 1))));
    return grid[gy * gridSize + gx] >= 0.5 ? 1 : 0;
}

function writeUncertaintyOverlay(
    grid: ArrayLike<number>,
    imageData: ImageData,
): void {
    for (let i = 0; i < grid.length; i++) {
        const distance = Math.abs(grid[i] - 0.5);
        const strength = Math.max(0, 1 - distance / UNCERTAINTY_THRESHOLD);
        const idx = i * 4;
        imageData.data[idx] = 255;
        imageData.data[idx + 1] = 255;
        imageData.data[idx + 2] = 255;
        imageData.data[idx + 3] = Math.round(strength * 105);
    }
}

function drawUncertaintyOverlay(
    ctx: CanvasRenderingContext2D,
    grid: ArrayLike<number>,
    canvasW: number,
    canvasH: number,
    tempCanvas: HTMLCanvasElement,
    imageData: ImageData,
): void {
    const tempCtx = tempCanvas.getContext('2d')!;
    writeUncertaintyOverlay(grid, imageData);
    tempCtx.putImageData(imageData, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.drawImage(tempCanvas, 0, 0, canvasW, canvasH);
    ctx.restore();
}

function drawMisclassificationOverlay(
    ctx: CanvasRenderingContext2D,
    points: DataPoint[],
    grid: ArrayLike<number>,
    gridSize: number,
    canvasW: number,
    canvasH: number,
    isTest: boolean,
): void {
    const radius = (isTest ? TEST_RADIUS : TRAIN_RADIUS) + 4;
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = isTest ? 1.5 : 2;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.45)';
    ctx.shadowBlur = 8;

    for (const p of points) {
        if (p.label !== 0 && p.label !== 1) continue;
        if (classifyPointFromGrid(p, grid, gridSize) === p.label) continue;
        const px = ((p.x + 1) / 2) * canvasW;
        const py = (1 - (p.y + 1) / 2) * canvasH;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.moveTo(px - radius * 0.65, py - radius * 0.65);
        ctx.lineTo(px + radius * 0.65, py + radius * 0.65);
        ctx.moveTo(px + radius * 0.65, py - radius * 0.65);
        ctx.lineTo(px - radius * 0.65, py + radius * 0.65);
        ctx.stroke();
    }

    ctx.restore();
}

function drawPoints(
    ctx: CanvasRenderingContext2D,
    points: DataPoint[],
    canvasW: number,
    canvasH: number,
    isTest: boolean,
    multiclass: boolean,
): void {
    if (points.length === 0) return;

    const radius = isTest ? TEST_RADIUS : TRAIN_RADIUS;
    const batches = new Map<string, DataPoint[]>();

    for (const p of points) {
        const classIndex = Number.isInteger(p.label) ? p.label : -1;
        const color = multiclass && classIndex >= 0 && classIndex < MULTICLASS_PALETTE.length
            ? MULTICLASS_PALETTE[classIndex].color
            : p.label >= 0.5 ? HEX_ORANGE : HEX_BLUE;
        const batch = batches.get(color);
        if (batch) {
            batch.push(p);
        } else {
            batches.set(color, [p]);
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

    for (const [color, batch] of batches) {
        renderBatch(batch, color);
    }
}

function hasNonBinaryLabels(points: DataPoint[]): boolean {
    return points.some((point) => point.label !== 0 && point.label !== 1);
}

function getMulticlassBoundaryFrame(): MulticlassBoundaryFrame | null {
    const frame = getFrameBuffer();
    if (
        !frame.multiclassClassGrid ||
        !frame.multiclassConfidenceGrid ||
        !frame.multiclassBoundaryLayout
    ) {
        return null;
    }
    return {
        classGrid: frame.multiclassClassGrid,
        confidenceGrid: frame.multiclassConfidenceGrid,
        layout: frame.multiclassBoundaryLayout,
    };
}

function formatPercent(value: number): string {
    return `${Math.round(value * 100)}%`;
}

function summarizeMulticlassBoundary(frame: MulticlassBoundaryFrame): MulticlassBoundarySummary | null {
    const { classGrid, confidenceGrid, layout } = frame;
    const expectedLength = layout.gridSize * layout.gridSize;
    if (classGrid.length !== expectedLength || confidenceGrid.length !== expectedLength || expectedLength === 0) {
        return null;
    }

    const counts = new Array(layout.classCount).fill(0) as number[];
    let confidenceTotal = 0;
    let lowConfidenceCount = 0;

    for (let i = 0; i < expectedLength; i++) {
        const classIndex = classGrid[i];
        if (classIndex >= layout.classCount) return null;
        counts[classIndex]++;
        confidenceTotal += confidenceGrid[i];
        if (confidenceGrid[i] < MULTICLASS_LOW_CONFIDENCE_THRESHOLD) {
            lowConfidenceCount++;
        }
    }

    let dominantIndex = 0;
    for (let i = 1; i < counts.length; i++) {
        if (counts[i] > counts[dominantIndex]) {
            dominantIndex = i;
        }
    }

    const dominantLabel = `Class ${layout.classLabels[dominantIndex]}`;
    const dominantShare = counts[dominantIndex] / expectedLength;
    const averageConfidence = confidenceTotal / expectedLength;
    const lowConfidenceShare = lowConfidenceCount / expectedLength;
    const description = `Multiclass decision boundary. ${dominantLabel} covers ${formatPercent(dominantShare)} of sampled cells with ${formatPercent(averageConfidence)} average winning confidence. ${formatPercent(lowConfidenceShare)} of cells are below ${formatPercent(MULTICLASS_LOW_CONFIDENCE_THRESHOLD)} confidence.`;

    return {
        dominantLabel,
        dominantShare,
        averageConfidence,
        lowConfidenceShare,
        description,
    };
}

// ── Component ──

export const DecisionBoundary = memo(function DecisionBoundary({
    trainPoints,
    testPoints,
    showTestData,
    discretize,
    overlayMode = 'none',
}: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const descriptionId = useId();
    const overlayCopy = getDecisionOverlayCopy(overlayMode, showTestData, discretize);

    // Off-screen resources (reused between frames)
    const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const imageDataRef = useRef<ImageData | null>(null);
    const overlayImageDataRef = useRef<ImageData | null>(null);
    const lastGridSizeRef = useRef(0);

    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const multiclassBoundaryVersion = useTrainingStore((s) => s.multiclassBoundaryVersion);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const outputSize = usePlaygroundStore((s) => s.network.outputSize);
    const outputActivation = usePlaygroundStore((s) => s.network.outputActivation);
    const supportsMulticlassBoundary =
        problemType === 'classification' &&
        outputSize === 3 &&
        outputActivation === 'softmax';
    const multiclassBoundary = supportsMulticlassBoundary ? getMulticlassBoundaryFrame() : null;
    const multiclassSummary = multiclassBoundary ? summarizeMulticlassBoundary(multiclassBoundary) : null;

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
        void frameVersion;
        void multiclassBoundaryVersion;
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
        const currentMulticlassBoundary = supportsMulticlassBoundary ? getMulticlassBoundaryFrame() : null;
        const grid = frameBuffer.outputGrid ?? snapshot?.outputGrid;
        const gridSize = frameBuffer.outputGrid ? frameBuffer.gridSize : snapshot?.gridSize ?? 0;

        if (currentMulticlassBoundary) {
            const { classGrid, confidenceGrid, layout } = currentMulticlassBoundary;
            if (!tempCanvasRef.current || lastGridSizeRef.current !== layout.gridSize) {
                const tc = document.createElement('canvas');
                tc.width = layout.gridSize;
                tc.height = layout.gridSize;
                tempCanvasRef.current = tc;
                imageDataRef.current = tc.getContext('2d')!.createImageData(layout.gridSize, layout.gridSize);
                overlayImageDataRef.current = tc.getContext('2d')!.createImageData(layout.gridSize, layout.gridSize);
                lastGridSizeRef.current = layout.gridSize;
            }

            drawMulticlassHeatmap(
                ctx,
                classGrid,
                confidenceGrid,
                logicalW,
                logicalH,
                tempCanvasRef.current!,
                imageDataRef.current!,
            );
        }

        if (!currentMulticlassBoundary && grid && gridSize > 0) {
            // Allocate / reuse offscreen canvas + ImageData
            if (!tempCanvasRef.current || lastGridSizeRef.current !== gridSize) {
                const tc = document.createElement('canvas');
                tc.width = gridSize;
                tc.height = gridSize;
                tempCanvasRef.current = tc;
                imageDataRef.current = tc.getContext('2d')!.createImageData(gridSize, gridSize);
                overlayImageDataRef.current = tc.getContext('2d')!.createImageData(gridSize, gridSize);
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

            if (overlayMode === 'uncertainty' && overlayImageDataRef.current) {
                drawUncertaintyOverlay(
                    ctx,
                    grid,
                    logicalW,
                    logicalH,
                    tempCanvasRef.current!,
                    overlayImageDataRef.current,
                );
            }

            if (overlayMode === 'misclassification') {
                drawMisclassificationOverlay(ctx, trainPoints, grid, gridSize, logicalW, logicalH, false);
                if (showTestData && testPoints.length > 0) {
                    drawMisclassificationOverlay(ctx, testPoints, grid, gridSize, logicalW, logicalH, true);
                }
            }
        }

        // Draw data points on top
        if (trainPoints.length > 0) {
            drawPoints(ctx, trainPoints, logicalW, logicalH, false, Boolean(currentMulticlassBoundary));
        }
        if ((showTestData || overlayMode === 'split') && testPoints.length > 0) {
            drawPoints(ctx, testPoints, logicalW, logicalH, true, Boolean(currentMulticlassBoundary));
        }
    }, [
        snapshot,
        frameVersion,
        multiclassBoundaryVersion,
        trainPoints,
        testPoints,
        showTestData,
        discretize,
        canvasSize,
        overlayMode,
        supportsMulticlassBoundary,
    ]);

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

    const isNonBinaryClassification =
        problemType === 'classification' &&
        (outputSize !== 1 ||
            outputActivation === 'softmax' ||
            hasNonBinaryLabels(trainPoints) ||
            hasNonBinaryLabels(testPoints));

    if (isNonBinaryClassification && !multiclassSummary) {
        return (
            <div className="decision-boundary" ref={containerRef}>
                <EmptyState
                    icon="🎯"
                    title="Binary decision boundary unavailable"
                    description="This visualization supports two-class outputs unless bounded multiclass boundary data is available from the worker."
                />
            </div>
        );
    }

    if (multiclassSummary && multiclassBoundary) {
        return (
            <div className="decision-boundary" ref={containerRef}>
                <canvas
                    ref={canvasRef}
                    style={{ width: '100%', height: '100%' }}
                    role="img"
                    aria-label="Multiclass decision boundary visualization showing predicted class regions and confidence"
                    aria-describedby={descriptionId}
                />
                <p id={descriptionId} className="sr-only">
                    {multiclassSummary.description}
                </p>
                <div className="decision-boundary__overlay-badge" data-overlay-mode="multiclass">
                    3 classes
                </div>
                <div className="decision-boundary__summary" aria-hidden="true">
                    <span>Dominant class: {multiclassSummary.dominantLabel}</span>
                    <span>Average confidence: {formatPercent(multiclassSummary.averageConfidence)}</span>
                </div>
                <div className="decision-boundary__legend">
                    {MULTICLASS_PALETTE.map((entry, index) => (
                        <div className="decision-boundary__legend-item" key={entry.label}>
                            <div className="decision-boundary__swatch" style={{ background: entry.color }} />
                            <span>Class {multiclassBoundary.layout.classLabels[index]}</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="decision-boundary" ref={containerRef}>
            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%' }}
                role="img"
                aria-label="Decision boundary visualization showing the neural network's classification regions"
                aria-describedby={descriptionId}
            />
            <p id={descriptionId} className="sr-only">
                {overlayCopy.description}
            </p>
            <div className="decision-boundary__overlay-badge" data-overlay-mode={overlayMode}>
                {overlayCopy.label}
            </div>
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
