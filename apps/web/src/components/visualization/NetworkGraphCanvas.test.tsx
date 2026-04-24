import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { NetworkGraphCanvas } from './NetworkGraphCanvas.tsx';
import {
    resetFrameBuffer,
    updateFrameBuffer,
    getFrameVersions,
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
const mockContexts: ReturnType<typeof createMockContext>[] = [];

function createMockContext() {
    const context = {
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
    mockContexts.push(context);
    return context;
}

describe('NetworkGraphCanvas', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    const originalResizeObserver = window.ResizeObserver;

    beforeEach(() => {
        mockContexts.length = 0;
        resetFrameBuffer();
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            outputGridVersion: 0,
            neuronGridsVersion: 0,
            paramsVersion: 0,
            layerStatsVersion: 0,
            confusionMatrixVersion: 0,
            trainPoints: [],
            testPoints: [],
        });

        const contextsByCanvas = new WeakMap<HTMLCanvasElement, ReturnType<typeof createMockContext>>();
        HTMLCanvasElement.prototype.getContext = vi.fn(function (this: HTMLCanvasElement) {
            let context = contextsByCanvas.get(this);
            if (!context) {
                context = createMockContext();
                contextsByCanvas.set(this, context);
            }
            return context as unknown as CanvasRenderingContext2D;
        }) as unknown as typeof HTMLCanvasElement.prototype.getContext;

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
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
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

    it('repaints graph for weight-only versions but not grid-only versions', () => {
        const layerSizes = [2, 2, 1];

        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]),
                biases: new Float32Array([0.1, -0.1, 0.05]),
                weightLayout: { layerSizes },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });

        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
            },
        });

        render(<NetworkGraphCanvas />);
        const mainContext = mockContexts[0];
        expect(mainContext.clearRect).toHaveBeenCalledTimes(1);

        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.4, -0.6, 0.8, -0.3, 1.0, -0.5]),
                biases: new Float32Array([0.2, -0.2, 0.15]),
                weightLayout: { layerSizes },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });

        expect(mainContext.clearRect).toHaveBeenCalledTimes(2);

        act(() => {
            updateFrameBuffer({
                neuronGrids: new Float32Array(3 * 4),
                neuronGridLayout: { count: 3, gridSize: 2 },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });

        expect(mainContext.clearRect).toHaveBeenCalledTimes(2);
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

    it('shows node details from a keyboard focus target and clears them on blur', () => {
        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]),
                biases: new Float32Array([0.1, -0.1, 0.05]),
                weightLayout: { layerSizes: [2, 2, 1] },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
                activation: 'tanh',
            },
        });

        render(<NetworkGraphCanvas />);

        const nodeTarget = screen.getByRole('button', {
            name: /Hidden 1, Neuron 1.*Bias: 0\.1000.*Activation: tanh/,
        });
        fireEvent.focus(nodeTarget);

        expect(screen.getByText('Hidden 1, Neuron 1')).toBeInTheDocument();
        expect(screen.getByText('Bias: 0.1000')).toBeInTheDocument();
        expect(screen.getByText('Activation: tanh')).toBeInTheDocument();

        fireEvent.blur(nodeTarget);
        expect(screen.queryByText('Hidden 1, Neuron 1')).not.toBeInTheDocument();
    });

    it('shows edge details from a keyboard focus target and clears them on blur', () => {
        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]),
                biases: new Float32Array([0.1, -0.1, 0.05]),
                weightLayout: { layerSizes: [2, 2, 1] },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
            },
        });

        render(<NetworkGraphCanvas />);

        const edgeTarget = screen.getByRole('button', {
            name: /Weight: 0\.3000.*Connection: Input 1 to Hidden 1, Neuron 1/,
        });
        fireEvent.focus(edgeTarget);

        expect(screen.getByText('Weight: 0.3000')).toBeInTheDocument();
        expect(screen.getByText('Layer 1, [0→0]')).toBeInTheDocument();

        fireEvent.blur(edgeTarget);
        expect(screen.queryByText('Weight: 0.3000')).not.toBeInTheDocument();
    });

    it('still shows node details on pointer hover', () => {
        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]),
                biases: new Float32Array([0.1, -0.1, 0.05]),
                weightLayout: { layerSizes: [2, 2, 1] },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
                activation: 'tanh',
            },
        });

        const { container } = render(<NetworkGraphCanvas />);
        const canvas = container.querySelector('canvas')!;
        canvas.getBoundingClientRect = vi.fn(() => ({
            left: 0,
            top: 0,
            right: 1140,
            bottom: 720,
            width: 1140,
            height: 720,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        }));

        fireEvent.pointerMove(canvas, { clientX: 570, clientY: 200 });

        expect(screen.getByText('Hidden 1, Neuron 1')).toBeInTheDocument();
        expect(screen.getByText('Bias: 0.1000')).toBeInTheDocument();
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

    it('paintEdges builds only bucket-level Path2D objects when no edge is hovered', () => {
        const originalPath2D = globalThis.Path2D;
        const path2DMock = vi.fn().mockImplementation(() => ({
            addPath: vi.fn(),
            moveTo: vi.fn(),
            bezierCurveTo: vi.fn(),
        }));
        vi.stubGlobal('Path2D', path2DMock);

        const ctx = createMockContext() as unknown as CanvasRenderingContext2D;
        const flat = {
            weights: new Float32Array([0.25, -0.4, 1, -1.2, 2, -2.4]),
            biases: new Float32Array([0, 0, 0]),
            layerSizes: [2, 2, 1],
        };
        const positions = [
            [{ x: 10, y: 80 }, { x: 10, y: 140 }],
            [{ x: 110, y: 80 }, { x: 110, y: 140 }],
            [{ x: 210, y: 110 }],
        ];

        try {
            paintEdges(ctx, positions, flat, null);

            expect(path2DMock).toHaveBeenCalledTimes(6);
        } finally {
            vi.unstubAllGlobals();
            Object.defineProperty(globalThis, 'Path2D', {
                configurable: true,
                writable: true,
                value: originalPath2D,
            });
        }
    });
});
