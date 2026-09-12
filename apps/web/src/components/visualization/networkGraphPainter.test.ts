import { describe, expect, it, vi } from 'vitest';
import { deriveNodeGeometry, hitTestNode, paintEdges, type FlatNetworkView } from './networkGraphPainter.ts';

describe('Precision Lab graph geometry', () => {
    it('keeps usable square tiles and lets the viewport pan dense layers', () => {
        expect(deriveNodeGeometry(360, 6)).toEqual({ width: 56, height: 56, cornerRadius: 3, hitPadding: 2 });
        expect(6 * deriveNodeGeometry(360, 6).height + 5 * 8 + 56).toBeGreaterThan(360);
    });
    it.each([[100, 100], [10000, 1], [0, 0], [NaN, Infinity], [-10, -1]])('clamps geometry for %s / %s', (height, count) => {
        const geometry = deriveNodeGeometry(height, count);
        expect(geometry.width).toBeGreaterThanOrEqual(56);
        expect(geometry.height).toBeLessThanOrEqual(72);
        expect(geometry.cornerRadius).toBe(3);
        expect(geometry.cornerRadius).toBeLessThanOrEqual(3);
    });
    it('includes tile corners and padded hit area without changing visual size', () => {
        const geometry = deriveNodeGeometry(360, 6);
        const nodes = [[{ x: 100, y: 100 }]];
        expect(hitTestNode(124, 124, nodes, geometry)).toEqual({ layerIdx: 0, nodeIdx: 0 });
        expect(hitTestNode(131, 100, nodes, geometry)).toBeNull();
        expect(geometry.width).toBe(56);
    });
});

describe('selected paths', () => {
    function context() {
        const strokes: Array<{ alpha: number; width: number; dash: number[] }> = [];
        let dash: number[] = [];
        const ctx = { globalAlpha: 1, lineWidth: 1, strokeStyle: '', save: vi.fn(), restore: vi.fn(),
            beginPath: vi.fn(), moveTo: vi.fn(), bezierCurveTo: vi.fn(),
            setLineDash: vi.fn((value: number[]) => { dash = value; }),
            stroke: vi.fn(() => { strokes.push({ alpha: ctx.globalAlpha, width: ctx.lineWidth, dash: [...dash] }); }),
        };
        return { ctx: ctx as unknown as CanvasRenderingContext2D, strokes };
    }
    const positions = [[{ x: 10, y: 10 }, { x: 10, y: 60 }], [{ x: 120, y: 30 }]];
    const flat: FlatNetworkView = { weights: new Float32Array([2, -0.5]), biases: new Float32Array([0]), layerSizes: [2, 1] };
    it('keeps ordinary edges visible and paints selected signed paths last at actual magnitude', () => {
        const { ctx, strokes } = context();
        paintEdges(ctx, positions, flat, null, 'all', { highlightedEdgeKeys: new Set(['1:0:0', '1:0:1']) });
        expect(strokes.length).toBeGreaterThan(2);
        expect(strokes.slice(0, -2).every((s) => s.alpha > 0 && s.alpha < 1)).toBe(true);
        expect(strokes.slice(-2).map((s) => s.dash)).toEqual([[], [6, 4]]);
        expect(strokes.at(-2)!.width).toBeGreaterThan(strokes.at(-1)!.width);
        expect(ctx.globalAlpha).toBe(1);
        expect(ctx.setLineDash).toHaveBeenLastCalledWith([]);
    });
    it('keeps signed filters effective on selected paths', () => {
        const { ctx, strokes } = context();
        paintEdges(ctx, positions, flat, null, 'positive', { highlightedEdgeKeys: new Set(['1:0:0', '1:0:1']) });
        expect(strokes.some((s) => s.dash.length > 0)).toBe(false);
    });
});
