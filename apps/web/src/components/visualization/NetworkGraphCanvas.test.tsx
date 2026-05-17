import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import {
    NetworkGraphCanvas,
    classifyNeuronActivity,
    computeChangedEdgeKeys,
    formatArchitectureStory,
    getCapacityLabel,
} from './NetworkGraphCanvas.tsx';
import {
    resetFrameBuffer,
    updateFrameBuffer,
    getFrameVersion,
} from '../../worker/frameBuffer.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    edgeFilterOptions,
    hitTestEdge,
    hitTestNode,
    paintEdges,
    paintLabels,
    paintNodes,
    shouldRenderEdge,
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
        translate: vi.fn(),
        scale: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        createImageData: (w: number, h: number) => ({
            width: w,
            height: h,
            data: new Uint8ClampedArray(w * h * 4),
        }),
        putImageData: vi.fn(),
        drawImage: vi.fn(),
        setLineDash: vi.fn(),
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
            layerStatsVersion: 0,
            trainPoints: [],
            testPoints: [],
            pendingConfigSource: null,
            networkConfigLoading: false,
        });
        useLayoutStore.setState({
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
        usePlaygroundStore.setState({
            data: {
                ...usePlaygroundStore.getState().data,
                dataset: 'gauss',
            },
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [],
                outputSize: 1,
                outputActivation: 'sigmoid',
                activation: 'tanh',
            },
            features: {
                ...usePlaygroundStore.getState().features,
                x: true,
                y: true,
                xSquared: false,
                ySquared: false,
                xy: false,
                sinX: false,
                sinY: false,
                cosX: false,
                cosY: false,
            },
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
        expect(desc!.textContent).toContain('Hidden activation: tanh');
    });

    it('renders architecture story and capacity badge inside the graph', () => {
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [4, 4],
            },
        });

        render(<NetworkGraphCanvas />);

        const summary = screen.getByLabelText('Architecture summary');
        const story = screen.getByText('X₁, X₂ -> [4] -> [4] -> 1 output');
        expect(summary).toContainElement(story);
        expect(screen.getByText('Moderate capacity')).toBeInTheDocument();
        expect(summary.querySelector('.network-graph-summary__hint')).toHaveTextContent('Gaussian blobs');
    });

    it('describes approved multiclass output shapes without scalar copy', () => {
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [4, 4],
                outputSize: 3,
                outputActivation: 'softmax',
                activation: 'tanh',
            },
        });

        const { container } = render(<NetworkGraphCanvas />);

        expect(screen.getByText('X₁, X₂ -> [4] -> [4] -> 3 outputs (softmax)')).toBeInTheDocument();
        expect(container.querySelector('#network-graph-desc')).toHaveTextContent(
            '3 outputs. Hidden activation: tanh. Output activation: softmax.',
        );
    });

    it('shows dataset topology hints only for clear mismatches', () => {
        usePlaygroundStore.setState({
            data: {
                ...usePlaygroundStore.getState().data,
                dataset: 'xor',
            },
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [],
            },
        });

        const { rerender } = render(<NetworkGraphCanvas />);

        expect(screen.getByText('XOR is not linearly separable, so add a hidden layer before training.')).toBeInTheDocument();

        act(() => {
            usePlaygroundStore.setState({
                network: {
                    ...usePlaygroundStore.getState().network,
                    hiddenLayers: [4],
                },
            });
        });
        rerender(<NetworkGraphCanvas />);

        expect(screen.queryByText('XOR is not linearly separable, so add a hidden layer before training.')).not.toBeInTheDocument();
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

    it('attaches graph viewport controls and mode toggle to one toolbar surface', () => {
        const { container } = render(<NetworkGraphCanvas />);

        const toolbar = screen.getByRole('toolbar', { name: 'Network graph toolbar' });
        expect(toolbar).toHaveClass('network-graph-toolbar');
        expect(toolbar?.querySelector('.network-graph-controls')).not.toBeNull();
        expect(toolbar?.querySelector('.network-graph-mode-toggle')).not.toBeNull();

        const zoomLabel = container.querySelector('.network-graph-controls__zoom');
        expect(zoomLabel?.textContent).toBe('100%');

        fireEvent.click(screen.getByRole('button', { name: 'Zoom in graph' }));

        expect(zoomLabel?.textContent).toBe('125%');
        expect(screen.getByRole('button', { name: 'Fit graph to view' })).toBeInTheDocument();

        const activations = screen.getByRole('button', { name: 'Activations' });
        fireEvent.click(activations);

        expect(activations).toHaveAttribute('aria-pressed', 'true');
    });

    it('renders an edge legend and can filter to strong weights', () => {
        const { container } = render(<NetworkGraphCanvas />);

        expect(container.querySelector('.network-graph-legend')).not.toBeNull();
        fireEvent.click(container.querySelector('button[aria-label="Show only strong edges"]')!);

        expect(container.querySelector('button[aria-label="Show only strong edges"]')).toHaveAttribute('aria-pressed', 'true');
    });

    it('renders topology view mode controls', () => {
        render(<NetworkGraphCanvas />);

        const activations = screen.getByRole('button', { name: 'Activations' });
        fireEvent.click(activations);

        expect(activations).toHaveAttribute('aria-pressed', 'true');
    });

    it('uses inline topology buttons to begin network config changes', () => {
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
            },
        });
        render(<NetworkGraphCanvas />);

        fireEvent.click(screen.getByRole('button', { name: 'Add neuron to hidden layer 1' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
        expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual([3]);
    });

    it('renders active network lesson copy and a ghost layer hint', () => {
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [],
            },
        });
        useLayoutStore.setState({
            activeLessonId: 'lesson-xor-hidden-layers',
            activeLessonStepIndex: 1,
        });

        render(<NetworkGraphCanvas />);

        expect(screen.getByText('Two hidden layers let the network combine simple bends into the corners needed for XOR.')).toBeInTheDocument();
        expect(screen.getByText('Add hidden layer here')).toBeInTheDocument();
    });
});

describe('network topology helpers', () => {
    it('formats architecture stories and capacity labels', () => {
        expect(formatArchitectureStory(['x', 'y'], [])).toBe('x, y -> 1 output (linear)');
        expect(formatArchitectureStory(['x', 'y'], [4, 4])).toBe('x, y -> [4] -> [4] -> 1 output');
        expect(formatArchitectureStory(['x', 'y'], [4, 4], 3, 'softmax')).toBe('x, y -> [4] -> [4] -> 3 outputs (softmax)');
        expect(getCapacityLabel([])).toBe('Linear model');
        expect(getCapacityLabel([4])).toBe('Low capacity');
        expect(getCapacityLabel([16, 17])).toBe('Overfit risk');
    });

    it('classifies neuron activity from activation grids', () => {
        expect(classifyNeuronActivity(new Float32Array(20).fill(0), 'relu')).toBe('low');
        expect(classifyNeuronActivity(new Float32Array(20).fill(0.99), 'sigmoid')).toBe('saturated');
        expect(classifyNeuronActivity(new Float32Array([0.1, 0.2, 0.3, 0.4]), 'tanh')).toBeNull();
    });

    it('selects the most changed edges from consecutive weight snapshots', () => {
        const keys = computeChangedEdgeKeys(
            new Float32Array([0, 0, 0, 0]),
            new Float32Array([0.01, 1, 0.02, 0.03]),
            [2, 2],
        );

        expect(keys.has('1:0:1')).toBe(true);
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

    it('hitTestEdge reads weights from later flat-buffer layers', () => {
        const multiLayerNodePositions = [
            [{ x: 10, y: 100 }],
            [{ x: 110, y: 100 }],
            [{ x: 210, y: 100 }],
        ];
        const flat = {
            weights: new Float32Array([0.25, -0.75]),
            biases: new Float32Array([0.1, -0.2]),
            layerSizes: [1, 1, 1],
        };

        const hit = hitTestEdge(160, 100, multiLayerNodePositions, flat);

        expect(hit).toEqual({
            layerIdx: 2,
            nodeIdx: 0,
            prevIdx: 0,
            weight: expect.closeTo(-0.75),
        });
    });

    it('paintEdges / paintNodes / paintLabels do not throw on an empty network', () => {
        const ctx = createMockContext() as unknown as CanvasRenderingContext2D;
        expect(() => paintEdges(ctx, nodePositions, null, null)).not.toThrow();
        expect(() => paintNodes(ctx, nodePositions, null)).not.toThrow();
        expect(() => paintLabels(ctx, nodePositions, ['Input', 'Output'])).not.toThrow();
    });

    it('defines edge filter options used by the graph legend', () => {
        expect(edgeFilterOptions.map((option) => option.id)).toEqual(['all', 'strong', 'positive', 'negative']);
        expect(edgeFilterOptions.map((option) => option.label)).toEqual(['All', 'Strong', 'Positive', 'Negative']);
    });

    it('filters edges by sign and strong magnitude', () => {
        expect(shouldRenderEdge(0.2, 'all')).toBe(true);
        expect(shouldRenderEdge(0.2, 'strong')).toBe(false);
        expect(shouldRenderEdge(1.6, 'strong')).toBe(true);
        expect(shouldRenderEdge(0.7, 'positive')).toBe(true);
        expect(shouldRenderEdge(-0.7, 'positive')).toBe(false);
        expect(shouldRenderEdge(-0.7, 'negative')).toBe(true);
    });
});
