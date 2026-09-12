import { memo, useEffect, useRef } from 'react';
import { readPlotPalette, useThemeStore } from '../../store/theme.ts';
import { CLASS_COLORS, fieldColor, hexRgb } from '../visualization/plotColors.ts';
import type { DatasetPreviewModel } from './datasetPreviewModel.ts';


export function paintDatasetPreview(ctx: CanvasRenderingContext2D, model: DatasetPreviewModel, width: number, height: number, background = '#17191B'): void {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
    const { x: [xMin, xMax], y: [yMin, yMax] } = model.inputDomain;
    const [valueMin, valueMax] = model.valueDomain;
    for (const point of model.points) {
        if (![point.x, point.y, point.label].every(Number.isFinite)) continue;
        if (model.taskKind === 'regression') {
            const rgb = fieldColor((point.label - valueMin) / (valueMax - valueMin || 1),hexRgb(background));
            ctx.fillStyle = `rgb(${rgb.join(',')})`;
        } else {
            ctx.fillStyle = model.taskKind === 'multiclass-classification'
                ? CLASS_COLORS[point.label] ?? '#a0a4b8'
                : point.label === 1 ? CLASS_COLORS[1] : CLASS_COLORS[0];
        }
        ctx.beginPath();
        ctx.arc(2 + (point.x - xMin) / (xMax - xMin) * Math.max(0, width - 4),
            2 + (yMax - point.y) / (yMax - yMin) * Math.max(0, height - 4), 1.6, 0, Math.PI * 2);
        ctx.fill();
    }
}

/** Owns paint resources only. The dataset button owns its accessible name. */
export const DatasetPreviewCanvas = memo(function DatasetPreviewCanvas({ model }: { readonly model: DatasetPreviewModel }) {
    const theme = useThemeStore((state) => state.resolved);
    const ref = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
        const canvas = ref.current;
        if (!canvas) return;
        let frame: number | null = null;
        let width = 0;
        let height = 0;
        const paint = () => {
            frame = null;
            if (width <= 0 || height <= 0) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            const dpr = Math.max(1, window.devicePixelRatio || 1);
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            paintDatasetPreview(ctx, model, width, height,readPlotPalette().background);
        };
        const measure = (nextWidth: number, nextHeight: number) => {
            if (Math.abs(nextWidth - width) < 1 && Math.abs(nextHeight - height) < 1) return;
            width = Math.max(0, nextWidth);
            height = Math.max(0, nextHeight);
            if (frame === null) frame = requestAnimationFrame(paint);
        };
        const rect = canvas.getBoundingClientRect();
        measure(rect.width, rect.height);
        const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(([entry]) => {
            if (entry) measure(entry.contentRect.width, entry.contentRect.height);
        });
        observer?.observe(canvas);
        return () => { observer?.disconnect(); if (frame !== null) cancelAnimationFrame(frame); };
    }, [model,theme]);
    return <canvas ref={ref} className="precision-dataset-preview" aria-hidden="true" style={{ width: '100%', height: '100%' }} />;
});
