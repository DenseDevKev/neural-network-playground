import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { DatasetPreviewCanvas, paintDatasetPreview } from './DatasetPreviewCanvas.tsx';
import { deriveDatasetPreviewModel } from './datasetPreviewModel.ts';

const context = { setTransform: vi.fn(), clearRect: vi.fn(), fillRect: vi.fn(), beginPath: vi.fn(), arc: vi.fn(), fill: vi.fn(), fillStyle: '' };
let resize: ResizeObserverCallback;
let nextFrame: FrameRequestCallback | null;
const disconnect = vi.fn();
const model = () => deriveDatasetPreviewModel({ datasetId: 'three-class-clusters', seed: 42, noise: 0 });

describe('DatasetPreviewCanvas', () => {
    beforeEach(() => {
        vi.clearAllMocks(); nextFrame = null;
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
        vi.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 80, height: 60 } as DOMRect);
        vi.stubGlobal('devicePixelRatio', 2);
        vi.stubGlobal('ResizeObserver', class { constructor(cb: ResizeObserverCallback) { resize = cb; } observe() {} disconnect = disconnect; });
        vi.stubGlobal('requestAnimationFrame', vi.fn((cb: FrameRequestCallback) => { nextFrame = cb; return 1; }));
        vi.stubGlobal('cancelAnimationFrame', vi.fn());
    });
    afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
    it('keeps the button as accessibility owner and scales only its backing store', () => {
        const { container, unmount } = render(<DatasetPreviewCanvas model={model()} />);
        const canvas = container.querySelector('canvas')!;
        expect(canvas).toHaveAttribute('aria-hidden', 'true');
        expect(canvas).toHaveStyle({ width: '100%', height: '100%' });
        act(() => nextFrame?.(0));
        expect([canvas.width, canvas.height]).toEqual([160, 120]);
        expect(context.arc).toHaveBeenCalledTimes(72);
        unmount(); expect(disconnect).toHaveBeenCalledOnce();
    });
    it('batches resize delivery, ignores subpixel changes, and cancels on unmount', () => {
        const { unmount } = render(<DatasetPreviewCanvas model={model()} />);
        act(() => nextFrame?.(0));
        vi.mocked(requestAnimationFrame).mockClear();
        const emit = (width: number) => resize([{ contentRect: { width, height: 60 } } as ResizeObserverEntry], {} as ResizeObserver);
        act(() => { emit(80.4); });
        expect(requestAnimationFrame).not.toHaveBeenCalled();
        act(() => { emit(120); emit(121); });
        expect(requestAnimationFrame).toHaveBeenCalledOnce();
        unmount(); expect(cancelAnimationFrame).toHaveBeenCalled();
    });
    it('paints the neutral background for empty evidence, never fake points', () => {
        paintDatasetPreview(context as unknown as CanvasRenderingContext2D, { ...model(), points: [] }, 80, 60);
        expect(context.fillRect).toHaveBeenCalledOnce();
        expect(context.arc).not.toHaveBeenCalled();
    });
    it('uses class colors for classification and a continuous scale for regression', () => {
        const fills: string[] = [];
        context.fill.mockImplementation(() => fills.push(context.fillStyle));
        paintDatasetPreview(context as unknown as CanvasRenderingContext2D, model(), 80, 60);
        expect(new Set(fills).size).toBe(3);
        fills.length = 0;
        paintDatasetPreview(context as unknown as CanvasRenderingContext2D, deriveDatasetPreviewModel({ datasetId: 'reg-plane', seed: 42, noise: 0 }), 80, 60);
        expect(new Set(fills).size).toBeGreaterThan(3);
    });
});
