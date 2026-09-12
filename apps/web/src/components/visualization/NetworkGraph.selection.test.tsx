import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { updateCompiledForTest } from '../../test/playgroundStoreTestUtils.ts';
import * as frames from '../../worker/frameBuffer.ts';
import { NetworkGraph } from './NetworkGraph.tsx';
import { NetworkSelectionDeck } from './NetworkSelectionDeck.tsx';
import { useNetworkSelectionController } from './useNetworkSelectionController.ts';

const prepared = PREPARED_PRESETS.find((p) => p.id === 'single-neuron')!.prepared;
let snapshot: frames.FrameBuffer;
function Harness() {
    const controller = useNetworkSelectionController();
    return <><NetworkGraph selectionController={controller} /><NetworkSelectionDeck model={controller.model} onClear={controller.commands.clearSelection} /></>;
}
function context() {
    return { clearRect: vi.fn(), fillRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), bezierCurveTo: vi.fn(), arc: vi.fn(), fill: vi.fn(), stroke: vi.fn(), fillText: vi.fn(), setTransform: vi.fn(), translate: vi.fn(), scale: vi.fn(), save: vi.fn(), restore: vi.fn(), setLineDash: vi.fn(), putImageData: vi.fn(), drawImage: vi.fn(), globalAlpha: 1,
        createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }) } as unknown as CanvasRenderingContext2D;
}
beforeEach(() => {
    frames.resetFrameBuffer();
    usePlaygroundStore.setState({ access: { status: 'ready', prepared } });
    updateCompiledForTest((c) => ({ ...c, network: { ...c.network, hiddenLayers: [2] } }));
    useLayoutStore.setState({ audienceMode: 'lab', advancedToolsOpen: true, activeLessonId: null });
    useTrainingStore.setState({ evidenceGenerationId: 7, paramsVersion: 0, neuronGridsVersion: 0, frameVersion: 0, layerStatsVersion: 0 });
    const model = { generationId: 7, revision: 10, step: 10, epoch: 1 };
    snapshot = { ...frames.getFrameBuffer(), weightLayout: { layerSizes: [2, 2, 1] }, weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]), biases: new Float32Array([0.1, -0.1, 0.05]),
        parameterProvenance: { model, recipeFingerprint: prepared.identities.recipeFingerprint }, neuronGrids: new Float32Array(12).fill(0.25), neuronGridLayout: { count: 3, gridSize: 2 },
        neuronGridsProvenance: { model: { ...model, step: 8, revision: 8 }, dataset: { datasetKey: 'd', generatorVersion: 1, trainCount: 50, testCount: 50 }, objectiveKey: 'o', basis: { kind: 'prediction-grid', pointCount: 4, domain: [-1, 1, -1, 1] } } };
    vi.spyOn(frames, 'getFrameBuffer').mockImplementation(() => snapshot);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => context());
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
function setRenderer(canvas: boolean) {
    act(() => usePlaygroundStore.setState((s) => ({ featuresUI: { ...s.featuresUI, canvasNetworkGraph: canvas } })));
}
for (const canvas of [true, false]) describe(canvas ? 'Canvas selection' : 'SVG selection', () => {
    beforeEach(() => setRenderer(canvas));
    it('selects by pointer and keyboard; retains true artifact steps across new frames', () => {
        render(<Harness />);
        const before = usePlaygroundStore.getState().access;
        const url = window.location.href;
        fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }));
        expect(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ })).toHaveAttribute('aria-pressed', 'true');
        const details = screen.getByRole('region', { name: 'Selected neuron details' });
        expect(details).toHaveTextContent('Hidden 1 · neuron 2');
        expect(details).toHaveTextContent('Weights at step 10; activation grid at step 8');
        expect(details).toHaveTextContent('negative -0.200');
        act(() => { snapshot = { ...snapshot, biases: new Float32Array([0.1, 0.75, 0.05]) }; useTrainingStore.setState({ paramsVersion: 1, neuronGridsVersion: 1 }); });
        expect(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ })).toHaveAttribute('aria-pressed', 'true');
        expect(details).toHaveTextContent('0.750');
        fireEvent.keyDown(screen.getByRole('button', { name: /^Hidden 1, Neuron 1/ }), { key: 'Enter' });
        expect(details).toHaveTextContent('Hidden 1 · neuron 1');
        fireEvent.keyDown(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }), { key: ' ' });
        expect(details).toHaveTextContent('Hidden 1 · neuron 2');
        expect(usePlaygroundStore.getState().access).toBe(before);
        expect(window.location.href).toBe(url);
        fireEvent.keyDown(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }), { key: 'Escape' });
        expect(screen.queryByRole('region', { name: 'Selected neuron details' })).not.toBeInTheDocument();
    });
    it('labels missing grids without manufacturing activation values', () => {
        snapshot = { ...snapshot, neuronGrids: null, neuronGridLayout: null, neuronGridsProvenance: null };
        render(<Harness />);
        const node = screen.getByRole('button', { name: /^Hidden 1, Neuron 1/ });
        expect(node).toHaveAttribute('data-grid-available', 'false');
        expect(node).toHaveAccessibleDescription(/Activation grid not available/);
        fireEvent.click(node);
        const details = screen.getByRole('region', { name: 'Selected neuron details' });
        expect(details).toHaveTextContent('Activation grid not available');
        expect(details).toHaveTextContent('Weights at step 10; activation grid at step not available');
    });
    it('does not clear selection on hover, blur, profile, disclosure, or unrelated live metrics', () => {
        render(<Harness />);
        const node = screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ });
        fireEvent.click(node); fireEvent.focus(node); fireEvent.blur(node); fireEvent.pointerLeave(node);
        act(() => {
            useLayoutStore.setState({ audienceMode: 'beginner', advancedToolsOpen: false });
            useTrainingStore.setState({ frameVersion: 100, outputGridVersion: 100 });
        });
        expect(node).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('region', { name: 'Selected neuron details' })).toHaveTextContent('Hidden 1 · neuron 2');
        fireEvent.click(screen.getByRole('button', { name: 'Clear selection' }));
        expect(node).toHaveAttribute('aria-pressed', 'false');
    });
    it('restores focus to the chosen neuron when the inspector closes', () => {
        render(<Harness />);
        const node = screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ });
        fireEvent.click(node);
        const close = screen.getByRole('button', { name: 'Clear selection' });
        close.focus(); fireEvent.click(close);
        expect(node).toHaveFocus();
    });
    it('hides grids from a different generation in graph and inspector', () => {
        snapshot = { ...snapshot, neuronGridsProvenance: { ...snapshot.neuronGridsProvenance!, model: { ...snapshot.neuronGridsProvenance!.model, generationId: 6 } } };
        const { container } = render(<Harness />);
        expect(container.querySelectorAll('.network-graph-heatmap-slot')).toHaveLength(0);
        fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 1/ }));
        expect(screen.getByRole('region', { name: 'Selected neuron details' })).toHaveTextContent('activation grid at step not available');
    });
    it('keeps all maximum-topology neuron targets at least 44px without overlaps', () => {
        updateCompiledForTest((c) => ({ ...c, features: { x: true, y: true, xSquared: true, ySquared: true, xy: true, sinX: true, sinY: true, cosX: true, cosY: true }, network: { ...c.network, hiddenLayers: [16,16,16,16,16,16] } }));
        const { container } = render(<Harness />);
        fireEvent.click(screen.getByRole('button', { name: 'Zoom out graph' }));
        const targets = [...container.querySelectorAll<HTMLButtonElement>('.network-node-target')];
        expect(targets).toHaveLength(106);
        const rectangles = targets.map((target) => ({ x: parseFloat(target.style.left), y: parseFloat(target.style.top), w: parseFloat(target.style.width), h: parseFloat(target.style.height) }));
        for (let i = 0; i < rectangles.length; i++) {
            const a = rectangles[i]; expect(a.w).toBeGreaterThanOrEqual(44);
            for (const b of rectangles.slice(i + 1)) expect(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y).toBe(true);
        }
    });
    it('exposes zoom/fit, mode, edge filters and architecture summary', () => {
        render(<Harness />);
        expect(screen.getByLabelText('Architecture summary')).toBeInTheDocument();
        const toolbar = screen.getByRole('toolbar', { name: 'Network graph toolbar' });
        fireEvent.click(within(toolbar).getByRole('button', { name: 'Zoom in graph' }));
        fireEvent.click(within(toolbar).getByRole('button', { name: 'Fit graph to view' }));
        fireEvent.click(screen.getByRole('button', { name: 'Activations' }));
        expect(screen.getByRole('button', { name: 'Activations' })).toHaveAttribute('aria-pressed', 'true');
        fireEvent.click(screen.getByRole('button', { name: 'Show negative edges' }));
        expect(screen.getByRole('button', { name: 'Show negative edges' })).toHaveAttribute('aria-pressed', 'true');
    });
});
it('keeps one shared selection when switching Canvas to SVG and back, then clears on generation or architecture', () => {
    setRenderer(true); render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }));
    for (const renderer of [false, true]) {
        setRenderer(renderer);
        expect(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('region', { name: 'Selected neuron details' })).toHaveTextContent('Hidden 1 · neuron 2');
    }
    act(() => useTrainingStore.setState({ evidenceGenerationId: 8 }));
    expect(screen.queryByRole('region', { name: 'Selected neuron details' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 1/ }));
    act(() => updateCompiledForTest((c) => ({ ...c, network: { ...c.network, hiddenLayers: [3] } })));
    expect(screen.queryByRole('region', { name: 'Selected neuron details' })).not.toBeInTheDocument();
});
it('marks the same signed strongest keys in SVG without relying only on color', () => {
    setRenderer(false); const { container } = render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }));
    const highlighted = [...container.querySelectorAll('[data-edge-selected="true"]')];
    expect(highlighted.map((p) => p.getAttribute('data-edge-key')).sort()).toEqual(['1:1:0', '1:1:1', '2:0:1']);
    expect(container.querySelector('[data-edge-key="1:1:1"]')).toHaveAttribute('stroke-dasharray', '6 4');
});

it('the standalone graph also retains its controller when switching renderers', () => {
    setRenderer(true); render(<NetworkGraph />);
    fireEvent.click(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ }));
    setRenderer(false);
    expect(screen.getByRole('button', { name: /^Hidden 1, Neuron 2/ })).toHaveAttribute('aria-pressed', 'true');
});
it('Canvas ignores unrelated output-grid frames instead of repainting the network', () => {
    setRenderer(true); render(<Harness />);
    const calls = vi.mocked(HTMLCanvasElement.prototype.getContext).mock.calls.length;
    act(() => useTrainingStore.setState({ frameVersion: 120, outputGridVersion: 120 }));
    expect(vi.mocked(HTMLCanvasElement.prototype.getContext).mock.calls.length).toBe(calls);
});

it('fits the first real measured viewport, batches resize delivery, and preserves later user zoom', () => {
    let resize!: ResizeObserverCallback;
    const queued: FrameRequestCallback[] = [];
    vi.stubGlobal('ResizeObserver', class { constructor(callback: ResizeObserverCallback) { resize = callback; } observe() {} disconnect() {} });
    vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => { queued.push(callback); return queued.length; }));
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    setRenderer(true); const { container } = render(<Harness />);
    const deliver = (width: number, height: number) => resize([{ contentRect: { width, height } } as ResizeObserverEntry], {} as ResizeObserver);
    act(() => { deliver(330, 300); deliver(320, 300); });
    expect(queued).toHaveLength(1);
    act(() => queued.shift()!(0));
    expect(container.querySelector('.network-graph-controls__zoom')).toHaveTextContent('89%');
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in graph' }));
    expect(container.querySelector('.network-graph-controls__zoom')).toHaveTextContent('111%');
    const paints = vi.mocked(HTMLCanvasElement.prototype.getContext).mock.calls.length;
    act(() => { deliver(320.3, 300.3); queued.shift()!(0); });
    expect(vi.mocked(HTMLCanvasElement.prototype.getContext).mock.calls.length).toBe(paints);
    act(() => { deliver(600, 400); queued.shift()!(0); });
    expect(container.querySelector('.network-graph-controls__zoom')).toHaveTextContent('111%');
});

it('uses the Canvas nonnegative bias stroke for a zero-bias SVG hidden tile', () => {
    snapshot = {...snapshot,biases:new Float32Array([0,-.1,.05])};
    setRenderer(false);
    const {container} = render(<Harness />);
    const nodes = container.querySelectorAll('rect.network-node');
    expect(nodes[2]).toHaveAttribute('stroke','rgba(244, 99, 48, 0.7)');
    expect(nodes[3]).toHaveAttribute('stroke','rgba(59, 130, 246, 0.7)');
});
