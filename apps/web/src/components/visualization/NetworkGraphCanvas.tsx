// ── Canvas Network Graph (AS-5) ──────────────────────────────────────────────
// Replaces the SVG topology view with a single <canvas> for edges, node
// rings, and labels — plus DOM overlays for the per-neuron mini heatmaps,
// the hover tooltip, and a screen-reader-only structural summary.
//
// Contract is intentionally identical to NetworkGraphSVG: takes no props,
// reads everything from the same stores. Toggle is owned by the parent
// `NetworkGraph` switcher.
//
// Why a single canvas? The SVG renderer commits ~650 DOM nodes per frame
// for a 16+16 hidden-layer network and re-runs style recalc on every hover
// className flip. The canvas renderer does one clearRect, ~6 stroke()
// calls (edge buckets), a handful of fill()s, and N fillText for layer
// labels — typically <0.5ms regardless of network size.
//
// Heatmaps stay in the React tree as <HeatmapCanvas> children positioned
// absolutely over the main canvas. Painting heatmaps inside the main
// canvas would require manual createImageData / putImageData per neuron
// every frame, which is fiddly and offers no measurable gain over the
// already-fast HeatmapCanvas component.

import { useCallback, useEffect, useMemo, useRef, useState, memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getActiveFeatures } from '@nn-playground/engine';
import type { ActivationType, DatasetType, LayerStats } from '@nn-playground/engine';
import {
    GRID_SIZE,
    MAX_HIDDEN_LAYERS,
    writeNormalizedHeatmap,
} from '@nn-playground/shared';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { extractNeuronGrid, layerBiasOffset } from '../../worker/frameBufferLayout.ts';
import { getDatasetTopologyHint } from '../../data/datasetInsights.ts';
import { getLessonDefinition } from '../../lessons/lessonRegistry.ts';
import {
    type EdgeRef,
    type EdgeFilter,
    type FlatNetworkView,
    type GraphViewMode,
    type NodeHealth,
    type NodePos,
    type NodeRef,
    NODE_RADIUS,
    edgeRefKey,
    edgeFilterOptions,
    hitTestEdge,
    hitTestNode,
    nodeRefKey,
    paintEdges,
    paintLabels,
    paintNodes,
    shouldRenderEdge,
} from './networkGraphPainter.ts';

// ── Layout constants — match the SVG renderer for visual parity ────────────
const MIN_LAYER_GAP = 120;
const MIN_NODE_GAP = 42;
const PAD_X = 60;
const PAD_Y = 40;
const HEATMAP_SIZE = 24;
const MIN_ZOOM = 0.35;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 1.25;

interface TooltipData {
    x: number;
    y: number;
    text: string[];
}

interface ChangedEdgeDelta {
    key: string;
    delta: number;
}

// ── Persistent source canvas for heatmap upscale (one per process) ─────────
let _sourceCanvas: HTMLCanvasElement | null = null;
let _sourceCtx: CanvasRenderingContext2D | null = null;
let _cachedImageData: ImageData | null = null;
let _cachedGridSize = 0;

function getSourceCanvas(gridSize: number) {
    if (!_sourceCanvas || !_sourceCtx || _cachedGridSize !== gridSize) {
        _sourceCanvas = _sourceCanvas ?? document.createElement('canvas');
        _sourceCanvas.width = gridSize;
        _sourceCanvas.height = gridSize;
        _sourceCtx = _sourceCanvas.getContext('2d')!;
        _cachedImageData = _sourceCtx.createImageData(gridSize, gridSize);
        _cachedGridSize = gridSize;
    }
    return { canvas: _sourceCanvas, ctx: _sourceCtx, imageData: _cachedImageData! };
}

interface HeatmapTileProps {
    grid: ArrayLike<number>;
    gridSize: number;
}

const HeatmapTile = memo(function HeatmapTile({ grid, gridSize }: HeatmapTileProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const src = getSourceCanvas(gridSize);
        writeNormalizedHeatmap(grid, src.imageData, 220);
        src.ctx.putImageData(src.imageData, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.clearRect(0, 0, HEATMAP_SIZE, HEATMAP_SIZE);
        ctx.drawImage(src.canvas, 0, 0, HEATMAP_SIZE, HEATMAP_SIZE);
    }, [grid, gridSize]);

    return (
        <canvas
            ref={canvasRef}
            width={HEATMAP_SIZE}
            height={HEATMAP_SIZE}
            style={{
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                display: 'block',
                pointerEvents: 'none',
            }}
        />
    );
});

interface NeuronGridEntry {
    grid: ArrayLike<number>;
    gridSize: number;
}

interface Viewport {
    zoom: number;
    panX: number;
    panY: number;
}

function clampZoom(zoom: number): number {
    return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
}

function zoomLabel(zoom: number): string {
    return `${Math.round(zoom * 100)}%`;
}

function toFixedLabel(value: number | null | undefined, digits = 4): string {
    return Number.isFinite(value) ? value!.toFixed(digits) : 'n/a';
}

function featureLabel(feature: { label?: string; id?: string }, fallbackIndex: number): string {
    return feature.label ?? feature.id ?? `x${fallbackIndex + 1}`;
}

export function formatArchitectureStory(
    inputLabels: readonly string[],
    hiddenLayers: readonly number[],
    outputSize = 1,
    outputActivation: ActivationType = 'linear',
): string {
    const inputStory = inputLabels.length > 0 ? inputLabels.join(', ') : 'inputs';
    const hiddenStory = hiddenLayers.map((count) => `[${count}]`).join(' -> ');
    const safeOutputSize = Number.isFinite(outputSize) ? Math.max(1, Math.floor(outputSize)) : 1;
    const outputStory = safeOutputSize === 1
        ? hiddenStory ? '1 output' : `1 output (${outputActivation})`
        : `${safeOutputSize} outputs (${outputActivation})`;
    if (!hiddenStory) return `${inputStory} -> ${outputStory}`;
    return `${inputStory} -> ${hiddenStory} -> ${outputStory}`;
}

export function getCapacityLabel(hiddenLayers: readonly number[]): string {
    const totalNeurons = hiddenLayers.reduce((sum, count) => sum + count, 0);
    if (hiddenLayers.length === 0) return 'Linear model';
    if (hiddenLayers.length === 1 && hiddenLayers[0] <= 4) return 'Low capacity';
    if (totalNeurons > 32) return 'Overfit risk';
    return 'Moderate capacity';
}

function formatHiddenLayerSummary(hiddenLayers: readonly number[]): string {
    const layerCount = hiddenLayers.length;
    const layerLabel = `${layerCount} hidden layer${layerCount === 1 ? '' : 's'}`;
    if (layerCount === 0) return layerLabel;
    const neuronLabel = layerCount === 1
        ? `${hiddenLayers[0]} neuron${hiddenLayers[0] === 1 ? '' : 's'}`
        : `${hiddenLayers.join(', ')} neurons`;
    return `${layerLabel} of ${neuronLabel}`;
}

export function classifyNeuronActivity(
    grid: ArrayLike<number>,
    activation: ActivationType,
): NodeHealth | null {
    const len = grid.length;
    if (len === 0) return null;
    let valid = 0;
    let nearZero = 0;
    let nearSaturated = 0;
    for (let i = 0; i < len; i++) {
        const value = grid[i];
        if (!Number.isFinite(value)) continue;
        valid++;
        if (Math.abs(value) <= 0.04) nearZero++;
        const saturationThreshold = activation === 'tanh' ? 0.92 : 0.96;
        if (Math.abs(value) >= saturationThreshold || value >= 0.98) nearSaturated++;
    }
    if (valid === 0) return null;
    if (nearZero / valid > 0.75) return 'low';
    if (['sigmoid', 'tanh', 'relu', 'leakyRelu'].includes(activation) && nearSaturated / valid > 0.75) {
        return 'saturated';
    }
    return null;
}

function getNeuronActivationIntensity(grid: ArrayLike<number>): number {
    if (grid.length === 0) return 0;
    let valid = 0;
    let sum = 0;
    for (let i = 0; i < grid.length; i++) {
        const value = grid[i];
        if (!Number.isFinite(value)) continue;
        valid++;
        sum += Math.min(1, Math.abs(value));
    }
    return valid > 0 ? sum / valid : 0;
}

export function computeChangedEdgeKeys(
    previousWeights: Float32Array | null,
    currentWeights: Float32Array,
    layerSizes: readonly number[],
): Set<string> {
    if (!previousWeights || previousWeights.length !== currentWeights.length) return new Set();
    const deltas: ChangedEdgeDelta[] = [];
    for (let layerIdx = 1; layerIdx < layerSizes.length; layerIdx++) {
        const fanIn = layerSizes[layerIdx - 1];
        const fanOut = layerSizes[layerIdx];
        let base = 0;
        for (let l = 1; l < layerIdx; l++) {
            base += layerSizes[l] * layerSizes[l - 1];
        }
        for (let nodeIdx = 0; nodeIdx < fanOut; nodeIdx++) {
            for (let prevIdx = 0; prevIdx < fanIn; prevIdx++) {
                const offset = base + nodeIdx * fanIn + prevIdx;
                const delta = Math.abs(currentWeights[offset] - previousWeights[offset]);
                if (delta > 0) deltas.push({ key: edgeRefKey(layerIdx, nodeIdx, prevIdx), delta });
            }
        }
    }
    deltas.sort((a, b) => b.delta - a.delta);
    const take = Math.max(1, Math.ceil(deltas.length * 0.05));
    return new Set(deltas.slice(0, take).map((item) => item.key));
}

function getLayerStatsHint(layerStats: readonly LayerStats[] | null): string | null {
    if (!layerStats?.length) return null;
    if (layerStats.some((stats) => stats.meanAbsGradient > 0 && stats.meanAbsGradient < 0.0001)) {
        return 'Some gradients are nearly flat; topology changes may help learning move again.';
    }
    return null;
}

export function NetworkGraphCanvas() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const outputSize = usePlaygroundStore((s) => s.network.outputSize);
    const outputActivation = usePlaygroundStore((s) => s.network.outputActivation);
    const features = usePlaygroundStore((s) => s.features);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const dataset = usePlaygroundStore((s) => s.data.dataset);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const layerStatsVersion = useTrainingStore((s) => s.layerStatsVersion);
    const activeLessonId = useLayoutStore((s) => s.activeLessonId);
    const activeLessonStepIndex = useLayoutStore((s) => s.activeLessonStepIndex);

    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<EdgeRef | null>(null);
    const [hoveredNode, setHoveredNode] = useState<NodeRef | null>(null);
    const [edgeFilter, setEdgeFilter] = useState<EdgeFilter>('all');
    const [viewMode, setViewMode] = useState<GraphViewMode>('weights');
    const [viewport, setViewport] = useState<Viewport>({ zoom: 1, panX: 0, panY: 0 });
    const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
    const previousWeightsRef = useRef<Float32Array | null>(null);

    const activeFeatures = useMemo(() => getActiveFeatures(features), [features]);
    const activeFeatureLabels = useMemo(
        () => activeFeatures.map((feature, index) => featureLabel(feature, index)),
        [activeFeatures],
    );
    const inputSize = activeFeatures.length;
    const architectureStory = useMemo(
        () => formatArchitectureStory(activeFeatureLabels, hiddenLayers, outputSize, outputActivation),
        [activeFeatureLabels, hiddenLayers, outputActivation, outputSize],
    );
    const capacityLabel = useMemo(() => getCapacityLabel(hiddenLayers), [hiddenLayers]);
    const datasetTopologyHint = useMemo(
        () => getDatasetTopologyHint(dataset as DatasetType, hiddenLayers),
        [dataset, hiddenLayers],
    );
    const activeLessonStep = useMemo(() => {
        if (!activeLessonId || activeLessonStepIndex == null) return null;
        const lesson = getLessonDefinition(activeLessonId);
        return lesson?.steps[activeLessonStepIndex] ?? null;
    }, [activeLessonId, activeLessonStepIndex]);
    const networkLessonStep = activeLessonStep?.target === 'network' ? activeLessonStep : null;

    const outputLayerSize = Number.isFinite(outputSize) ? Math.max(1, Math.floor(outputSize)) : 1;
    const layers = useMemo(() => [inputSize, ...hiddenLayers, outputLayerSize], [inputSize, hiddenLayers, outputLayerSize]);
    const layersKey = layers.join(',');
    const maxNodes = Math.max(...layers);

    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [containerSize, setContainerSize] = useState({ width: 800, height: 400 });

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const ro = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) {
                setContainerSize({ width, height });
            }
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const { canvasWidth, canvasHeight, layerGap, nodeGap } = useMemo(() => {
        const w = Math.max(320, containerSize.width);
        const h = Math.max(200, containerSize.height);
        const usableW = Math.max(0, w - PAD_X * 2);
        const usableH = Math.max(0, h - PAD_Y * 2);
        const layerGap = layers.length > 1
            ? Math.max(MIN_LAYER_GAP, usableW / (layers.length - 1))
            : MIN_LAYER_GAP;
        const nodeGap = maxNodes > 1
            ? Math.max(MIN_NODE_GAP, usableH / (maxNodes - 1))
            : MIN_NODE_GAP;
        const canvasWidth = Math.max(w, layers.length * layerGap + PAD_X * 2);
        const canvasHeight = Math.max(h, maxNodes * nodeGap + PAD_Y * 2);
        return { canvasWidth, canvasHeight, layerGap, nodeGap };
    }, [containerSize, layers, maxNodes]);

    const nodePositions: NodePos[][] = useMemo(() => {
        const startX = (canvasWidth - (layers.length - 1) * layerGap) / 2;
        return layers.map((count, layerIdx) => {
            const x = startX + layerIdx * layerGap;
            const totalHeight = (count - 1) * nodeGap;
            const startY = canvasHeight / 2 - totalHeight / 2;
            return Array.from({ length: count }, (_, nodeIdx) => ({
                x,
                y: startY + nodeIdx * nodeGap,
            }));
        });
    }, [layers, canvasWidth, canvasHeight, layerGap, nodeGap]);

    const layerLabels = useMemo(() => {
        return layers.map((_, idx) => {
            if (idx === 0) return 'Input';
            if (idx === layers.length - 1) return 'Output';
            return `Hidden ${idx}`;
        });
    }, [layers]);

    // Flat-buffer view, identical contract to the SVG renderer's `flat`.
    const flat = useMemo<FlatNetworkView | null>(() => {
        void frameVersion;
        const fb = getFrameBuffer();
        if (fb.weights && fb.biases && fb.weightLayout) {
            return {
                weights: fb.weights,
                biases: fb.biases,
                layerSizes: fb.weightLayout.layerSizes,
            };
        }
        if (snapshot?.weights && snapshot.weights.length > 0 && snapshot.biases) {
            const sizes: number[] = [snapshot.weights[0]?.[0]?.length ?? 0];
            let total = 0;
            for (const layer of snapshot.weights) {
                sizes.push(layer.length);
                for (const neuron of layer) total += neuron.length;
            }
            const w = new Float32Array(total);
            let off = 0;
            for (const layer of snapshot.weights) {
                for (const neuron of layer) {
                    w.set(neuron, off);
                    off += neuron.length;
                }
            }
            let btotal = 0;
            for (const layer of snapshot.biases) btotal += layer.length;
            const b = new Float32Array(btotal);
            off = 0;
            for (const layer of snapshot.biases) {
                b.set(layer, off);
                off += layer.length;
            }
            return { weights: w, biases: b, layerSizes: sizes };
        }
        return null;
    }, [frameVersion, snapshot?.weights, snapshot?.biases]);

    // Per-neuron heatmap source data — same derivation as the SVG renderer.
    // Output: array indexed [hidden1, hidden2, ..., output], one entry per
    // non-input neuron in render order.
    const neuronGrids = useMemo<NeuronGridEntry[] | null>(() => {
        void frameVersion;
        const fb = getFrameBuffer();
        if (fb.neuronGrids && fb.neuronGridLayout) {
            const { count, gridSize } = fb.neuronGridLayout;
            const cells = gridSize * gridSize;
            return Array.from({ length: count }, (_, idx) => ({
                grid: extractNeuronGrid(fb.neuronGrids!, idx, cells),
                gridSize,
            }));
        }
        if (!snapshot?.neuronGrids) return null;
        const gridSize = snapshot.gridSize ?? GRID_SIZE;
        const sng = snapshot.neuronGrids;
        if (sng instanceof Float32Array) {
            const count = layers.slice(1).reduce((sum, size) => sum + size, 0);
            const cells = gridSize * gridSize;
            return Array.from({ length: count }, (_, idx) => ({
                grid: extractNeuronGrid(sng, idx, cells),
                gridSize,
            }));
        }
        return sng.map((grid) => ({ grid, gridSize }));
    }, [frameVersion, layers, snapshot?.neuronGrids, snapshot?.gridSize]);

    /** Map (layerIdx, nodeIdx) → index into neuronGrids, or null for input. */
    const getNeuronGridIndex = useCallback(
        (layerIdx: number, nodeIdx: number): number | null => {
            if (!neuronGrids || layerIdx === 0) return null;
            let idx = 0;
            if (layerIdx === layers.length - 1) {
                for (let l = 1; l < layers.length - 1; l++) idx += layers[l];
            } else {
                for (let l = 1; l < layerIdx; l++) idx += layers[l];
            }
            idx += nodeIdx;
            return idx < neuronGrids.length ? idx : null;
        },
        [neuronGrids, layers],
    );

    const nodeHealthByKey = useMemo(() => {
        void frameVersion;
        const health = new Map<string, NodeHealth>();
        if (!neuronGrids) return health;
        for (let layerIdx = 1; layerIdx < layers.length; layerIdx++) {
            for (let nodeIdx = 0; nodeIdx < layers[layerIdx]; nodeIdx++) {
                const gridIndex = getNeuronGridIndex(layerIdx, nodeIdx);
                const entry = gridIndex == null ? null : neuronGrids[gridIndex];
                if (!entry) continue;
                const classification = classifyNeuronActivity(entry.grid, activation);
                if (classification) health.set(nodeRefKey(layerIdx, nodeIdx), classification);
            }
        }
        return health;
    }, [activation, frameVersion, getNeuronGridIndex, layers, neuronGrids]);

    const nodeActivationByKey = useMemo(() => {
        void frameVersion;
        const activations = new Map<string, number>();
        if (!neuronGrids) return activations;
        for (let layerIdx = 1; layerIdx < layers.length; layerIdx++) {
            for (let nodeIdx = 0; nodeIdx < layers[layerIdx]; nodeIdx++) {
                const gridIndex = getNeuronGridIndex(layerIdx, nodeIdx);
                const entry = gridIndex == null ? null : neuronGrids[gridIndex];
                if (!entry) continue;
                activations.set(nodeRefKey(layerIdx, nodeIdx), getNeuronActivationIntensity(entry.grid));
            }
        }
        return activations;
    }, [frameVersion, getNeuronGridIndex, layers, neuronGrids]);

    const layerStats = useMemo<readonly LayerStats[] | null>(() => {
        void layerStatsVersion;
        return getFrameBuffer().layerStats ?? snapshot?.layerStats ?? null;
    }, [layerStatsVersion, snapshot?.layerStats]);
    const layerStatsHint = useMemo(() => getLayerStatsHint(layerStats), [layerStats]);

    const fitGraphToView = useCallback(() => {
        if (containerSize.width <= 0 || containerSize.height <= 0) return;
        const fitZoom = clampZoom(Math.min(
            containerSize.width / canvasWidth,
            containerSize.height / canvasHeight,
            1,
        ));
        setViewport({
            zoom: fitZoom,
            panX: (containerSize.width - canvasWidth * fitZoom) / 2,
            panY: (containerSize.height - canvasHeight * fitZoom) / 2,
        });
    }, [containerSize.width, containerSize.height, canvasWidth, canvasHeight]);

    useEffect(() => {
        fitGraphToView();
    }, [fitGraphToView, layersKey, activeFeatures.length]);

    const zoomGraph = useCallback((direction: 1 | -1) => {
        setViewport((current) => {
            const nextZoom = clampZoom(direction > 0 ? current.zoom * ZOOM_STEP : current.zoom / ZOOM_STEP);
            const centerX = containerSize.width / 2;
            const centerY = containerSize.height / 2;
            const worldX = (centerX - current.panX) / current.zoom;
            const worldY = (centerY - current.panY) / current.zoom;
            return {
                zoom: nextZoom,
                panX: centerX - worldX * nextZoom,
                panY: centerY - worldY * nextZoom,
            };
        });
    }, [containerSize.width, containerSize.height]);

    const screenToWorld = useCallback(
        (screenX: number, screenY: number) => ({
            x: (screenX - viewport.panX) / viewport.zoom,
            y: (screenY - viewport.panY) / viewport.zoom,
        }),
        [viewport],
    );

    /** Tooltip text for a hovered node, mirroring the SVG renderer's format. */
    const buildNodeTooltipLines = useCallback(
        (layerIdx: number, nodeIdx: number): string[] => {
            const isInput = layerIdx === 0;
            const isOutput = layerIdx === layers.length - 1;
            const lines: string[] = [];
            if (isInput) {
                const feat = activeFeatures[nodeIdx];
                lines.push(`Input: ${feat?.label ?? `x${nodeIdx}`}`);
                return lines;
            }
            const bias = flat
                ? flat.biases[layerBiasOffset(flat.layerSizes, layerIdx - 1) + nodeIdx]
                : undefined;
            if (isOutput) {
                lines.push('Output neuron');
                if (bias != null) lines.push(`Bias: ${toFixedLabel(bias)}`);
            } else {
                lines.push(`Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`);
                if (bias != null) lines.push(`Bias: ${toFixedLabel(bias)}`);
                lines.push(`Activation: ${activation}`);
            }
            const health = nodeHealthByKey.get(nodeRefKey(layerIdx, nodeIdx));
            if (health === 'low') lines.push('Low activity: most samples produce near-zero activation here');
            if (health === 'saturated') lines.push('Saturated: most samples push this neuron near its activation limit');
            return lines;
        },
        [layers, activeFeatures, activation, flat, nodeHealthByKey],
    );

    // ── Paint pass ───────────────────────────────────────────────────────────
    // Triggers on frameVersion (new snapshot), layout changes, or hover state.
    // `paintLabels` is a bit redundant on every hover but cheap (~5 fillText
    // calls) and saves us a separate effect.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const logicalW = Math.max(1, containerSize.width);
        const logicalH = Math.max(1, containerSize.height);
        const physicalW = Math.round(logicalW * dpr);
        const physicalH = Math.round(logicalH * dpr);
        if (canvas.width !== physicalW || canvas.height !== physicalH) {
            canvas.width = physicalW;
            canvas.height = physicalH;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, logicalW, logicalH);

        const changedEdgeKeys = flat
            ? computeChangedEdgeKeys(previousWeightsRef.current, flat.weights, flat.layerSizes)
            : new Set<string>();
        previousWeightsRef.current = flat ? new Float32Array(flat.weights) : null;

        ctx.save();
        ctx.translate(viewport.panX, viewport.panY);
        ctx.scale(viewport.zoom, viewport.zoom);
        paintEdges(ctx, nodePositions, flat, hoveredEdge, edgeFilter, {
            viewMode,
            changedEdgeKeys,
            nodeActivationByKey,
        });
        paintNodes(ctx, nodePositions, flat, { nodeHealthByKey });
        paintLabels(ctx, nodePositions, layerLabels);
        ctx.restore();
    }, [
        canvasWidth,
        canvasHeight,
        containerSize.width,
        containerSize.height,
        nodePositions,
        flat,
        hoveredEdge,
        edgeFilter,
        viewMode,
        nodeHealthByKey,
        nodeActivationByKey,
        layerLabels,
        viewport,
        // re-paint on every snapshot even if `flat` reference is stable —
        // weights mutate in place inside the Float32Array.
        frameVersion,
    ]);

    // ── Pointer wiring ──────────────────────────────────────────────────────
    const handlePointerMove = useCallback(
        (event: React.PointerEvent<HTMLCanvasElement>) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const screenX = event.clientX - rect.left;
            const screenY = event.clientY - rect.top;

            if (dragRef.current) {
                const dx = event.clientX - dragRef.current.x;
                const dy = event.clientY - dragRef.current.y;
                dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
                setViewport((current) => ({
                    ...current,
                    panX: current.panX + dx,
                    panY: current.panY + dy,
                }));
                return;
            }

            const { x, y } = screenToWorld(screenX, screenY);

            const node = hitTestNode(x, y, nodePositions);
            if (node) {
                if (
                    !hoveredNode ||
                    hoveredNode.layerIdx !== node.layerIdx ||
                    hoveredNode.nodeIdx !== node.nodeIdx
                ) {
                    setHoveredNode(node);
                }
                if (hoveredEdge) setHoveredEdge(null);
                setTooltip({
                    x: screenX + 12,
                    y: screenY - 8,
                    text: buildNodeTooltipLines(node.layerIdx, node.nodeIdx),
                });
                return;
            }
            if (hoveredNode) setHoveredNode(null);

            const edge = hitTestEdge(x, y, nodePositions, flat);
            if (edge && shouldRenderEdge(edge.weight, edgeFilter)) {
                if (
                    !hoveredEdge ||
                    hoveredEdge.layerIdx !== edge.layerIdx ||
                    hoveredEdge.nodeIdx !== edge.nodeIdx ||
                    hoveredEdge.prevIdx !== edge.prevIdx
                ) {
                    setHoveredEdge(edge);
                }
                setTooltip({
                    x: screenX + 12,
                    y: screenY - 8,
                    text: [
                        `${edge.weight >= 0 ? 'Positive' : 'Negative'} weight: ${toFixedLabel(edge.weight)}`,
                        `Magnitude: ${toFixedLabel(Math.abs(edge.weight))}`,
                        `Layer ${edge.layerIdx}, [${edge.prevIdx}→${edge.nodeIdx}]`,
                    ],
                });
                return;
            }
            if (hoveredEdge) setHoveredEdge(null);
            if (tooltip) setTooltip(null);
        },
        [
            nodePositions,
            flat,
            edgeFilter,
            hoveredEdge,
            hoveredNode,
            tooltip,
            buildNodeTooltipLines,
            screenToWorld,
        ],
    );

    const handlePointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        if (event.button !== 0) return;
        dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
        if (event.currentTarget.setPointerCapture) {
            event.currentTarget.setPointerCapture(event.pointerId);
        }
        setTooltip(null);
    }, []);

    const finishPointerDrag = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        if (dragRef.current?.pointerId === event.pointerId) {
            dragRef.current = null;
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
            }
        }
    }, []);

    const handleWheel = useCallback((event: React.WheelEvent<HTMLCanvasElement>) => {
        event.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const screenX = event.clientX - rect.left;
        const screenY = event.clientY - rect.top;
        setViewport((current) => {
            const nextZoom = clampZoom(event.deltaY < 0 ? current.zoom * ZOOM_STEP : current.zoom / ZOOM_STEP);
            const worldX = (screenX - current.panX) / current.zoom;
            const worldY = (screenY - current.panY) / current.zoom;
            return {
                zoom: nextZoom,
                panX: screenX - worldX * nextZoom,
                panY: screenY - worldY * nextZoom,
            };
        });
    }, []);

    const handlePointerLeave = useCallback(() => {
        dragRef.current = null;
        setHoveredEdge(null);
        setHoveredNode(null);
        setTooltip(null);
    }, []);

    // ── Render ──────────────────────────────────────────────────────────────
    const heatmapTiles: { key: string; x: number; y: number; entry: NeuronGridEntry }[] = [];
    if (neuronGrids) {
        for (let l = 1; l < layers.length; l++) {
            const layer = nodePositions[l];
            for (let i = 0; i < layer.length; i++) {
                const idx = getNeuronGridIndex(l, i);
                if (idx == null) continue;
                const entry = neuronGrids[idx];
                if (!entry) continue;
                heatmapTiles.push({
                    key: `h-${l}-${i}`,
                    x: layer[i].x,
                    y: layer[i].y,
                    entry,
                });
            }
        }
    }

    const ghostLayerX = (() => {
        if (hiddenLayers.length >= MAX_HIDDEN_LAYERS) return null;
        const sourceLayerIdx = hiddenLayers.length;
        const targetLayerIdx = hiddenLayers.length + 1;
        const source = nodePositions[sourceLayerIdx]?.[0];
        const target = nodePositions[targetLayerIdx]?.[0];
        if (!source || !target) return null;
        return ((source.x + target.x) / 2) * viewport.zoom + viewport.panX;
    })();

    const ghostLayerTop = PAD_Y * viewport.zoom + viewport.panY;
    const ghostLayerHeight = Math.max(80, (canvasHeight - PAD_Y * 2) * viewport.zoom);
    const showLessonGhostLayer = networkLessonStep?.id === 'give-model-capacity' && hiddenLayers.length === 0;

    const accessibilitySummary = useMemo(() => {
        const outputLabel = `${outputLayerSize} output${outputLayerSize === 1 ? '' : 's'}`;
        return `Neural network: ${activeFeatures.length} input${activeFeatures.length === 1 ? '' : 's'}, ` +
            formatHiddenLayerSummary(hiddenLayers) +
            `, ${outputLabel}. Hidden activation: ${activation}. Output activation: ${outputActivation}.`;
    }, [activeFeatures.length, hiddenLayers, activation, outputActivation, outputLayerSize]);

    return (
        <div
            ref={containerRef}
            className="network-graph-container"
            style={{ position: 'relative', width: '100%', height: '100%' }}
        >
            <canvas
                ref={canvasRef}
                role="img"
                aria-describedby="network-graph-desc"
                style={{
                    width: '100%',
                    height: '100%',
                    display: 'block',
                    cursor: dragRef.current ? 'grabbing' : hoveredNode || hoveredEdge ? 'pointer' : 'grab',
                }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={finishPointerDrag}
                onPointerCancel={finishPointerDrag}
                onPointerLeave={handlePointerLeave}
                onWheel={handleWheel}
            />

            <div className="network-graph-summary" aria-label="Architecture summary">
                <div className="network-graph-summary__row">
                    <span className="network-graph-summary__story">{architectureStory}</span>
                    <span className="network-graph-summary__badge">{capacityLabel}</span>
                </div>
                {datasetTopologyHint && (
                    <div className="network-graph-summary__hint">{datasetTopologyHint}</div>
                )}
                {layerStatsHint && (
                    <div className="network-graph-summary__hint network-graph-summary__hint--stats">{layerStatsHint}</div>
                )}
            </div>

            {networkLessonStep && (
                <div className="network-graph-lesson-callout" role="note">
                    <span className="network-graph-lesson-callout__label">Lesson</span>
                    <span>{networkLessonStep.body}</span>
                </div>
            )}

            {showLessonGhostLayer && ghostLayerX != null && (
                <div
                    className="network-graph-ghost-layer network-graph-ghost-layer--lesson"
                    style={{
                        left: ghostLayerX,
                        top: ghostLayerTop,
                        height: ghostLayerHeight,
                    }}
                    aria-hidden="true"
                >
                    <span>Add hidden layer here</span>
                </div>
            )}

            <div className="network-graph-toolbar" role="toolbar" aria-label="Network graph toolbar">
                <div className="network-graph-controls" aria-label="Graph view controls">
                    <button
                        type="button"
                        aria-label="Zoom out graph"
                        title="Zoom out"
                        onClick={() => zoomGraph(-1)}
                    >
                        -
                    </button>
                    <span className="network-graph-controls__zoom">{zoomLabel(viewport.zoom)}</span>
                    <button
                        type="button"
                        aria-label="Zoom in graph"
                        title="Zoom in"
                        onClick={() => zoomGraph(1)}
                    >
                        +
                    </button>
                    <button
                        type="button"
                        aria-label="Fit graph to view"
                        title="Fit graph"
                        onClick={fitGraphToView}
                    >
                        Fit
                    </button>
                </div>

                <div className="network-graph-mode-toggle" role="group" aria-label="Topology view mode">
                    {(['weights', 'activations'] as const).map((mode) => (
                        <button
                            key={mode}
                            type="button"
                            className={viewMode === mode ? 'network-graph-mode-toggle__button network-graph-mode-toggle__button--active' : 'network-graph-mode-toggle__button'}
                            aria-pressed={viewMode === mode}
                            onClick={() => setViewMode(mode)}
                        >
                            {mode === 'weights' ? 'Weights' : 'Activations'}
                        </button>
                    ))}
                </div>
            </div>

            <div className="network-graph-legend" aria-label="Edge weight legend">
                <div className="network-graph-legend__scale">
                    <span><i className="network-graph-legend__swatch network-graph-legend__swatch--positive" /> Positive</span>
                    <span><i className="network-graph-legend__swatch network-graph-legend__swatch--negative" /> Negative</span>
                    <span className="network-graph-legend__hint">width = |weight|</span>
                </div>
                <div className="network-graph-legend__filters">
                    {edgeFilterOptions.map((option) => (
                        <button
                            key={option.id}
                            type="button"
                            aria-label={option.id === 'strong' ? 'Show only strong edges' : `Show ${option.label.toLowerCase()} edges`}
                            aria-pressed={edgeFilter === option.id}
                            className={edgeFilter === option.id ? 'network-graph-legend__filter network-graph-legend__filter--active' : 'network-graph-legend__filter'}
                            onClick={() => setEdgeFilter(option.id)}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Heatmap overlays — one per non-input neuron, positioned over
                the corresponding canvas-painted node disc. */}
            {heatmapTiles.map((tile) => {
                const r = NODE_RADIUS - 1.5;
                const scaledR = r * viewport.zoom;
                return (
                    <div
                        key={tile.key}
                        className="network-graph-heatmap-slot"
                        style={{
                            position: 'absolute',
                            left: tile.x * viewport.zoom + viewport.panX - scaledR,
                            top: tile.y * viewport.zoom + viewport.panY - scaledR,
                            width: 2 * scaledR,
                            height: 2 * scaledR,
                            pointerEvents: 'none',
                        }}
                    >
                        <HeatmapTile grid={tile.entry.grid} gridSize={tile.entry.gridSize} />
                    </div>
                );
            })}

            {/* Tooltip — same DOM-overlay style as the SVG renderer */}
            {tooltip && (
                <div
                    className="network-tooltip"
                    style={{ left: tooltip.x, top: tooltip.y }}
                >
                    {tooltip.text.map((line, i) => (
                        <div
                            key={i}
                            className={i === 0 ? 'network-tooltip__title' : 'network-tooltip__detail'}
                        >
                            {line}
                        </div>
                    ))}
                </div>
            )}

            {/* Visually-hidden structural summary for assistive tech. The
                canvas itself can't expose the network shape to screen
                readers. */}
            <p
                id="network-graph-desc"
                style={{
                    position: 'absolute',
                    width: 1,
                    height: 1,
                    padding: 0,
                    margin: -1,
                    overflow: 'hidden',
                    clip: 'rect(0, 0, 0, 0)',
                    whiteSpace: 'nowrap',
                    border: 0,
                }}
            >
                {accessibilitySummary}
            </p>
        </div>
    );
}
