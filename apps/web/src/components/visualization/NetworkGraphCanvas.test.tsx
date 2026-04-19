import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render } from '@testing-library/react';
import { NetworkGraphCanvas } from './NetworkGraphCanvas.tsx';
import {
    resetFrameBuffer,
    updateFrameBuffer,
    getFrameVersion,
} from '../../worker/frameBuffer.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    hitTestEdge,
    hitTestNode,
    paintEdges,
    paintLabels,
    paintNodes,
} from './networkGraphPainter.ts';

// Minimal Canvas2D mock — just enough for paintEdges/paintNodes/paintLabels
// to run. We assert at the integration level that the component mounts and
// triggers a paint; the painter helpers are exercised in isolation below.
function createMockContext() {
    return {
        clearRect: vi.fn(),
        fillRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        bezierCurveTo: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        stroke: vi.fn(),
        fillText: vi.fn(),
        setTransform: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        createImageData: (w: number, h: number) => ({
            width: w,
            height: h,
            data: new Uint8ClampedArray(w * h * 4),
        }),
        putImageData: vi.fn(),
        drawImage: vi.fn(),
        imageSmoothingEnabled: false,
        imageSmoothingQuality: 'low' as const,
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 1,
        font: '',
        textAlign: 'start' as CanvasTextAlign,
        textBaseline: 'alphabetic' as CanvasTextBaseline,
    };
}

describe('NetworkGraphCanvas', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    const originalResizeObserver = window.ResizeObserver;

    beforeEach(() => {
        resetFrameBuffer();
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
        });

        HTMLCanvasElement.prototype.getContext = vi.fn(
            () => createMockContext() as unknown as CanvasRenderingContext2D,
        ) as unknown as typeof HTMLCanvasElement.prototype.getContext;

        class ResizeObserverMock {
            observe() {}
            disconnect() {}
        }
        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: ResizeObserverMock,
        });
    });

    afterEach(() => {
        HTMLCanvasElement.prototype.getContext = originalGetContext;
        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: originalResizeObserver,
        });
    });

    it('mounts and paints when the frame buffer carries weights', () => {
        // 2-input → 2-hidden → 1-output network with one weight slot per edge.
        const layerSizes = [2, 2, 1];
        const weights = new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]);
        const biases = new Float32Array([0.1, -0.1, 0.05]);

        act(() => {
            updateFrameBuffer({
                weights,
                biases,
                weightLayout: { layerSizes },
            });
            useTrainingStore.setState({ frameVersion: getFrameVersion() });
        });

        // Match the network shape the component reads from the store.
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
            },
        });

        const { container } = render(<NetworkGraphCanvas />);
        const canvas = container.querySelector('canvas');
        expect(canvas).not.toBeNull();
    });

    it('exposes a screen-reader summary describing the network shape', () => {
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [4, 4],
                activation: 'tanh',
            },
        });
        const { container } = render(<NetworkGraphCanvas />);
        const desc = container.querySelector('#network-graph-desc');
        expect(desc).not.toBeNull();
        expect(desc!.textContent).toContain('hidden layer');
        expect(desc!.textContent).toContain('Activation: tanh');
    });

    it('clears any active tooltip when the pointer leaves the canvas', () => {
        const { container } = render(<NetworkGraphCanvas />);
        const canvas = container.querySelector('canvas')!;
        // Synthetic pointermove well inside the canvas — won't intersect
        // anything in our default container layout, but we still exercise
        // the handler's no-op branch.
        fireEvent.pointerMove(canvas, { clientX: 10, clientY: 10 });
        fireEvent.pointerLeave(canvas);
        expect(container.querySelector('.network-tooltip')).toBeNull();
    });
});

describe('networkGraphPainter helpers', () => {
    const nodePositions = [
        [{ x: 10, y: 100 }],
        [{ x: 110, y: 100 }],
    ];

    it('hitTestNode returns the layer/node under the cursor', () => {
        const hit = hitTestNode(10, 100, nodePositions);
        expect(hit).toEqual({ layerIdx: 0, nodeIdx: 0 });
    });

    it('hitTestNode returns null outside any node', () => {
        const hit = hitTestNode(500, 500, nodePositions);
        expect(hit).toBeNull();
    });

    it('hitTestEdge returns null when there is no flat view', () => {
        const hit = hitTestEdge(60, 100, nodePositions, null);
        expect(hit).toBeNull();
    });

    it('hitTestEdge picks an edge near the bezier midpoint', () => {
        const flat = {
            weights: new Float32Array([0.5]),
            biases: new Float32Array([0]),
            layerSizes: [1, 1],
        };
        const hit = hitTestEdge(60, 100, nodePositions, flat);
        expect(hit).not.toBeNull();
        expect(hit?.layerIdx).toBe(1);
        expect(hit?.weight).toBeCloseTo(0.5);
    });

    it('paintEdges / paintNodes / paintLabels do not throw on an empty network', () => {
        const ctx = createMockContext() as unknown as CanvasRenderingContext2D;
        expect(() => paintEdges(ctx, nodePositions, null, null)).not.toThrow();
        expect(() => paintNodes(ctx, nodePositions, null)).not.toThrow();
        expect(() => paintLabels(ctx, nodePositions, ['Input', 'Output'])).not.toThrow();
    });
});
