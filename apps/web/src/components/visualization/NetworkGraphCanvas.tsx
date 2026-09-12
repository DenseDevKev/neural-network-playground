// ── Canvas Network Graph (AS-5) ──────────────────────────────────────────────
// Replaces the SVG topology view with a single <canvas> for edges, node
// rings, and labels — plus DOM overlays for the per-neuron mini heatmaps,
// the hover tooltip, and a screen-reader-only structural summary.
//
// Selection is shared with NetworkGraphSVG through an optional external controller;
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
import { readPlotPalette, useThemeStore } from '../../store/theme.ts';
import './networkAtelier.css';
import { NetworkGraphFrame } from './NetworkGraphFrame.tsx';
import { useNetworkSelectionController, type NetworkSelectionController } from './useNetworkSelectionController.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getActiveFeatures } from '@nn-playground/engine';
import type { ActivationType, DatasetType, FeatureFlags, LayerStats } from '@nn-playground/engine';
import { MAX_HIDDEN_LAYERS, writeNormalizedHeatmap } from '@nn-playground/shared';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { extractNeuronGrid, layerBiasOffset, layerWeightOffset } from '../../worker/frameBufferLayout.ts';
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
    deriveNodeGeometry,
    describeGraphNode,
    edgeRefKey,
    edgeColor,
    nodeColor,
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
const MIN_ZOOM = 0.8;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 1.25;
const EMPTY_HIDDEN_LAYERS: number[] = [];
const EMPTY_FEATURES: FeatureFlags = {
    x: false,
    y: false,
    xSquared: false,
    ySquared: false,
    xy: false,
    sinX: false,
    sinY: false,
    cosX: false,
    cosY: false,
};

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
    displaySize: number;
}

export const HeatmapTile = memo(function HeatmapTile({ grid, gridSize, displaySize }: HeatmapTileProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        // Match the backing store to device pixels so tiles stay crisp next
        // to the DPI-aware graph canvas behind them.
        const dpr = window.devicePixelRatio || 1;
        const backingSize = Math.max(1, Math.round(displaySize * dpr));
        if (canvas.width !== backingSize || canvas.height !== backingSize) {
            canvas.width = backingSize;
            canvas.height = backingSize;
        }
        const src = getSourceCanvas(gridSize);
        writeNormalizedHeatmap(grid, src.imageData, 220);
        src.ctx.putImageData(src.imageData, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, displaySize, displaySize);
        ctx.drawImage(src.canvas, 0, 0, displaySize, displaySize);
    }, [grid, gridSize, displaySize]);

    return (
        <canvas
            ref={canvasRef}
            width={HEATMAP_SIZE}
            height={HEATMAP_SIZE}
            style={{
                width: '100%',
                height: '100%',
                borderRadius: '2px',
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

/** Standalone compatibility path; App supplies its one shared controller. */
export function NetworkGraphCanvas({ controller }: { readonly controller?: NetworkSelectionController }) {
    return controller ? <NetworkGraphCanvasView controller={controller} /> : <LocalNetworkGraphCanvas />;
}
function LocalNetworkGraphCanvas() {
    const controller = useNetworkSelectionController();
    return <NetworkGraphCanvasView controller={controller} />;
}
export function NetworkGraphCanvasView({ controller, renderer = 'canvas' }: { readonly controller: NetworkSelectionController; readonly renderer?: 'canvas' | 'svg' }) {
    const theme = useThemeStore((s) => s.resolved);
    const palette = useMemo(() => { void theme; return readPlotPalette(); }, [theme]);
    const selectionOrigin = useRef<HTMLButtonElement | null>(null);
    const previousSelection = useRef(controller.selectedNode);
    useEffect(() => {
        if (previousSelection.current && !controller.selectedNode) selectionOrigin.current?.focus();
        previousSelection.current = controller.selectedNode;
    }, [controller.selectedNode]);
    const compiled = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared.compiled : null
    ));
    const hiddenLayers = compiled?.network.hiddenLayers ?? EMPTY_HIDDEN_LAYERS;
    const outputSize = compiled?.task.outputSize ?? 1;
    const outputActivation = compiled?.task.outputActivation ?? 'sigmoid';
    const features = compiled?.features ?? EMPTY_FEATURES;
    const activation = compiled?.network.activation ?? 'tanh';
    const dataset = compiled?.data.dataset ?? 'circle';
    const paramsVersion = useTrainingStore((s) => s.paramsVersion);
    const neuronGridsVersion = useTrainingStore((s) => s.neuronGridsVersion);
    const generation = useTrainingStore((s) => s.evidenceGenerationId);
    const layerStatsVersion = useTrainingStore((s) => s.layerStatsVersion);
    const activeLessonId = useLayoutStore((s) => s.activeLessonId);
    const activeLessonStepIndex = useLayoutStore((s) => s.activeLessonStepIndex);

    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<EdgeRef | null>(null);
    const [hoveredNode, setHoveredNode] = useState<NodeRef | null>(null);
    const [edgeFilter, setEdgeFilter] = useState<EdgeFilter>('all');
    const [viewMode, setViewMode] = useState<GraphViewMode>('weights');
    const [viewport, setViewport] = useState<Viewport>({ zoom: 1, panX: 0, panY: 0 });
    const dragRef = useRef<{ pointerId: number; x: number; y: number; startX: number; startY: number; moved: boolean } | null>(null);
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
    const svgRef = useRef<SVGSVGElement>(null);
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    // Latest observed size without participating in render identity, so the
    // auto-fit effect does not re-run (and reset user pan/zoom) on resizes.
    const containerSizeRef = useRef(containerSize);
    containerSizeRef.current = containerSize;
    const [isDragging, setIsDragging] = useState(false);

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        let pending: { width: number; height: number } | null = null;
        let ticket: number | null = null;
        const ro = new ResizeObserver((entries) => {
            const { width = 0, height = 0 } = entries[0]?.contentRect ?? {};
            if (width <= 0 || height <= 0) return;
            pending = { width, height };
            if (ticket !== null) return;
            ticket = requestAnimationFrame(() => {
                ticket = null;
                const next = pending;
                if (!next) return;
                setContainerSize((previous) => {
                    if (Math.abs(previous.width - next.width) < 1 && Math.abs(previous.height - next.height) < 1) return previous;
                    containerSizeRef.current = next;
                    return next;
                });
            });
        });
        ro.observe(el);
        return () => { ro.disconnect(); if (ticket !== null) cancelAnimationFrame(ticket); };
    }, []);

    const geometry = useMemo(() => deriveNodeGeometry(containerSize.height, maxNodes), [containerSize.height, maxNodes]);
    const { canvasWidth, canvasHeight, layerGap, nodeGap } = useMemo(() => {
        const w = Math.max(320, containerSize.width);
        const h = Math.max(200, containerSize.height);
        const usableW = Math.max(0, w - PAD_X * 2);
        const usableH = Math.max(0, h - PAD_Y * 2);
        const layerGap = layers.length > 1
            ? Math.max(MIN_LAYER_GAP, usableW / (layers.length - 1))
            : MIN_LAYER_GAP;
        const nodeGap = maxNodes > 1
            ? Math.max(geometry.height + 16, usableH / (maxNodes - 1))
            : MIN_NODE_GAP;
        const canvasWidth = Math.max(w, (layers.length - 1) * layerGap + PAD_X * 2);
        const canvasHeight = Math.max(h, (maxNodes - 1) * nodeGap + PAD_Y * 2);
        return { canvasWidth, canvasHeight, layerGap, nodeGap };
    }, [containerSize, layers, maxNodes, geometry.height]);
    const canvasSizeRef = useRef({ width: canvasWidth, height: canvasHeight });
    canvasSizeRef.current = { width: canvasWidth, height: canvasHeight };

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
        void paramsVersion;
        const fb = getFrameBuffer();
        if (fb.weights && fb.biases && fb.weightLayout && fb.weightLayout.layerSizes.join(',') === layersKey && (!generation || fb.parameterProvenance?.model.generationId === generation)) {
            return {
                weights: fb.weights,
                biases: fb.biases,
                layerSizes: fb.weightLayout.layerSizes,
            };
        }
        return null;
    }, [paramsVersion, layersKey, generation]);

    // Per-neuron heatmap source data — same derivation as the SVG renderer.
    // Output: array indexed [hidden1, hidden2, ..., output], one entry per
    // non-input neuron in render order.
    const neuronGrids = useMemo<NeuronGridEntry[] | null>(() => {
        void neuronGridsVersion;
        const fb = getFrameBuffer();
        if (fb.neuronGrids && fb.neuronGridLayout && fb.neuronGrids.length === fb.neuronGridLayout.count * fb.neuronGridLayout.gridSize ** 2 && fb.neuronGrids.every(Number.isFinite) && fb.neuronGridLayout.count === layers.slice(1).reduce((sum, count) => sum + count, 0) && (!generation || fb.neuronGridsProvenance?.model.generationId === generation)) {
            const { count, gridSize } = fb.neuronGridLayout;
            const cells = gridSize * gridSize;
            return Array.from({ length: count }, (_, idx) => ({
                grid: extractNeuronGrid(fb.neuronGrids!, idx, cells),
                gridSize,
            }));
        }
        return null;
    }, [neuronGridsVersion, generation, layers]);

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
        void neuronGridsVersion;
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
    }, [activation, neuronGridsVersion, getNeuronGridIndex, layers, neuronGrids]);

    const nodeActivationByKey = useMemo(() => {
        void neuronGridsVersion;
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
    }, [neuronGridsVersion, getNeuronGridIndex, layers, neuronGrids]);

    const layerStats = useMemo<readonly LayerStats[] | null>(() => {
        void layerStatsVersion;
        return getFrameBuffer().layerStats ?? null;
    }, [layerStatsVersion]);
    const layerStatsHint = useMemo(() => getLayerStatsHint(layerStats), [layerStats]);

    const fitGraphToView = useCallback(() => {
        const { width, height } = containerSizeRef.current;
        const { width: contentWidth, height: contentHeight } = canvasSizeRef.current;
        if (width <= 0 || height <= 0 || contentWidth <= 0 || contentHeight <= 0) return;
        const fitZoom = clampZoom(Math.min(
            width / contentWidth,
            height / contentHeight,
            1,
        ));
        setViewport({
            zoom: fitZoom,
            panX: (width - contentWidth * fitZoom) / 2,
            panY: (height - contentHeight * fitZoom) / 2,
        });
    }, []);

    // Re-fit only when the topology or feature set changes; plain container
    // resizes preserve the user's pan/zoom.
    useEffect(() => {
        fitGraphToView();
    }, [fitGraphToView, layersKey, activeFeatures.length]);

    // The very first real container measurement defines the initial view.
    // Fitting here (post-render) guarantees canvasSizeRef already reflects
    // the measured size; later resizes must not discard the user's pan/zoom,
    // so the fit is latched after it runs once.
    const didInitialFitRef = useRef(false);
    useEffect(() => {
        if (didInitialFitRef.current) return;
        if (containerSize.width <= 0 || containerSize.height <= 0) return;
        didInitialFitRef.current = true;
        fitGraphToView();
    }, [containerSize, fitGraphToView]);

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
    // Triggers on parameter/grid versions (accepted artifacts), layout changes, or hover state.
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
            highlightedEdgeKeys: controller.model.kind === 'selected' ? controller.model.highlightedEdgeKeys : undefined,
            viewMode,
            changedEdgeKeys,
            nodeActivationByKey,
        });
        paintNodes(ctx, nodePositions, flat, { nodeHealthByKey, geometry, selectedNode: controller.selectedNode, palette });
        paintLabels(ctx, nodePositions, layerLabels, palette);
        ctx.restore();
    }, [
        palette,
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
        paramsVersion,
        neuronGridsVersion,
        controller.model,
        controller.selectedNode,
        geometry,
    ]);

    // ── Pointer wiring ──────────────────────────────────────────────────────
    const handlePointerMove = useCallback(
        (event: React.PointerEvent<HTMLCanvasElement | SVGSVGElement>) => {
            const canvas = canvasRef.current ?? svgRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const screenX = event.clientX - rect.left;
            const screenY = event.clientY - rect.top;

            if (dragRef.current) {
                const dx = event.clientX - dragRef.current.x;
                const dy = event.clientY - dragRef.current.y;
                dragRef.current = { ...dragRef.current, pointerId: event.pointerId, x: event.clientX, y: event.clientY,
                    moved: dragRef.current.moved || Math.hypot(event.clientX - dragRef.current.startX, event.clientY - dragRef.current.startY) > 4 };
                setViewport((current) => ({
                    ...current,
                    panX: current.panX + dx,
                    panY: current.panY + dy,
                }));
                return;
            }

            const { x, y } = screenToWorld(screenX, screenY);

            const node = hitTestNode(x, y, nodePositions, geometry);
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
            geometry,
            flat,
            edgeFilter,
            hoveredEdge,
            hoveredNode,
            tooltip,
            buildNodeTooltipLines,
            screenToWorld,
        ],
    );

    const handlePointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement | SVGSVGElement>) => {
        if (event.button !== 0) return;
        dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, moved: false };
        setIsDragging(true);
        if (event.currentTarget.setPointerCapture) {
            event.currentTarget.setPointerCapture(event.pointerId);
        }
        setTooltip(null);
    }, []);

    const finishPointerDrag = useCallback((event: React.PointerEvent<HTMLCanvasElement | SVGSVGElement>) => {
        if (dragRef.current?.pointerId === event.pointerId) {
            if (event.type !== 'pointercancel' && !dragRef.current.moved) {
                const rect = event.currentTarget.getBoundingClientRect();
                const { x, y } = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
                const node = hitTestNode(x, y, nodePositions, geometry);
                if (node) controller.commands.selectNode(node);
            }
            dragRef.current = null;
            setIsDragging(false);
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
            }
        }
    }, [controller.commands, geometry, nodePositions, screenToWorld]);

    // React attaches delegated wheel listeners as passive, so preventDefault()
    // there is a no-op and zooming would also scroll the workspace. A native
    // non-passive listener is required for the zoom gesture.
    useEffect(() => {
        const canvas = canvasRef.current ?? svgRef.current;
        if (!canvas) return undefined;
        const handleWheel = (event: WheelEvent) => {
            event.preventDefault();
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
        };
        canvas.addEventListener('wheel', handleWheel as EventListener, { passive: false });
        return () => canvas.removeEventListener('wheel', handleWheel as EventListener);
    }, [setViewport]);

    const handlePointerLeave = useCallback(() => {
        dragRef.current = null;
        setIsDragging(false);
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
        <NetworkGraphFrame
            story={architectureStory} capacity={capacityLabel}
            datasetHint={datasetTopologyHint} healthHint={layerStatsHint}
            lesson={networkLessonStep?.body}
            zoom={viewport.zoom} onZoomOut={() => zoomGraph(-1)} onZoomIn={() => zoomGraph(1)} onFit={fitGraphToView}
            viewMode={viewMode} onViewMode={setViewMode} edgeFilter={edgeFilter} onEdgeFilter={setEdgeFilter}
        >
        <div
            ref={containerRef}
            className="network-graph-container"
            style={{ position: 'relative', width: '100%', height: '100%', minWidth: 0, overflow: 'hidden' }}
            onKeyDown={(event) => {
                if (event.key === 'Escape' && controller.selectedNode) {
                    event.preventDefault(); event.stopPropagation(); controller.commands.clearSelection();
                }
            }}
        >
            {renderer === 'canvas' ? <canvas
                ref={canvasRef}
                role="img"
                aria-label="Neural network graph"
                aria-describedby="network-graph-desc"
                style={{
                    width: '100%',
                    height: '100%',
                    display: 'block',
                    cursor: isDragging ? 'grabbing' : hoveredNode || hoveredEdge ? 'pointer' : 'grab',
                }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={finishPointerDrag}
                onPointerCancel={finishPointerDrag}
                onPointerLeave={handlePointerLeave}
            /> : <svg ref={svgRef} role="img" aria-label="Neural network graph" aria-describedby="network-graph-desc"
                viewBox={`0 0 ${Math.max(1, containerSize.width)} ${Math.max(1, containerSize.height)}`}
                style={{ width: '100%', height: '100%', display: 'block', touchAction: 'none' }}
                onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={finishPointerDrag}
                onPointerCancel={finishPointerDrag} onPointerLeave={handlePointerLeave}>
                <g transform={`translate(${viewport.panX} ${viewport.panY}) scale(${viewport.zoom})`}>
                    {flat && nodePositions.flatMap((layer, l) => l === 0 ? [] : layer.flatMap((node, i) => nodePositions[l - 1].map((prev, j) => {
                        const weight = flat.weights[layerWeightOffset(flat.layerSizes, l - 1) + i * flat.layerSizes[l - 1] + j];
                        if (!Number.isFinite(weight) || !shouldRenderEdge(weight, edgeFilter)) return null;
                        const key = edgeRefKey(l, i, j);
                        const selected = controller.model.kind === 'selected' && controller.model.highlightedEdgeKeys.has(key);
                        const intensity = nodeActivationByKey.get(nodeRefKey(l, i)) ?? 0;
                        const cp = (node.x - prev.x) * .45;
                        return <path key={key} role="button" tabIndex={0}
                            aria-label={`Weight: ${toFixedLabel(weight)}. Connection: ${describeGraphNode(l - 1, j, layers.length)} to ${describeGraphNode(l, i, layers.length)}`}
                            onFocus={() => setTooltip({ x: (prev.x + node.x) / 2 * viewport.zoom + viewport.panX, y: (prev.y + node.y) / 2 * viewport.zoom + viewport.panY, text: [`Weight: ${toFixedLabel(weight)}`, `Layer ${l}, [${j}→${i}]`] })}
                            onBlur={() => setTooltip(null)} data-edge-key={key} data-edge-selected={selected}
                            d={`M ${prev.x},${prev.y} C ${prev.x + cp},${prev.y} ${node.x - cp},${node.y} ${node.x},${node.y}`}
                            fill="none" stroke={viewMode === 'activations' && !selected ? `rgba(${weight >= 0 ? '244, 99, 48' : '59, 130, 246'},${.12 + intensity * .72})` : edgeColor(weight, selected)}
                            strokeWidth={selected ? 1.5 + Math.min(3, Math.abs(weight)) * 1.5 : viewMode === 'activations' ? .5 + intensity * 2.2 : Math.max(.5, Math.min(3, Math.abs(weight) * 1.5))}
                            opacity={controller.model.kind === 'selected' && !selected ? .28 : 1} strokeDasharray={selected && weight < 0 ? '6 4' : undefined} />;
                    })))}
                    {nodePositions.flatMap((layer, l) => layer.map((node, i) => {
                        const selected = controller.selectedNode?.layerIdx === l && controller.selectedNode.nodeIdx === i;
                        return <rect key={nodeRefKey(l, i)} className="network-node" x={node.x - geometry.width / 2} y={node.y - geometry.height / 2}
                            width={geometry.width} height={geometry.height} rx={geometry.cornerRadius} fill={palette.surface}
                            stroke={selected ? palette.text : l > 0 && l < layers.length - 1 ? nodeColor(flat?.biases[layerBiasOffset(flat.layerSizes, l - 1) + i] ?? 0) : palette.rule} strokeWidth={selected ? 3 : 1.5} />;
                    }))}
                    {nodePositions.flatMap((layer, l) => layer.map((node, i) => {
                        const health = nodeHealthByKey.get(nodeRefKey(l, i));
                        return health ? <rect key={`health-${l}-${i}`} x={node.x - geometry.width / 2 - 4} y={node.y - geometry.height / 2 - 4}
                            width={geometry.width + 8} height={geometry.height + 8} rx={geometry.cornerRadius + 4} fill="none"
                            stroke={health === 'low' ? 'rgba(239, 68, 68, .78)' : 'rgba(234, 179, 8, .82)'} strokeWidth={2} strokeDasharray={health === 'low' ? '3 3' : '1 3'} /> : null;
                    }))}
                    {nodePositions.map((layer, l) => <text key={l} x={layer[0]?.x ?? 0} y={20} textAnchor="middle" fill={palette.muted} fontSize={10}>{layerLabels[l]}</text>)}
                </g>
            </svg>}

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

            {/* Heatmap overlays — one per non-input neuron, positioned over
                the corresponding canvas-painted node disc. */}
            {heatmapTiles.map((tile) => {
                const r = geometry.width / 2 - 2;
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
                        <HeatmapTile grid={tile.entry.grid} gridSize={tile.entry.gridSize} displaySize={2 * scaledR} />
                    </div>
                );
            })}

            <div className="network-node-targets" role="group" aria-label="Select a neuron">
                {nodePositions.flatMap((layer, layerIdx) => layer.map((node, nodeIdx) => {
                    const selected = controller.selectedNode?.layerIdx === layerIdx && controller.selectedNode.nodeIdx === nodeIdx;
                    const size = Math.max(44, (geometry.width + 2 * geometry.hitPadding) * viewport.zoom);
                    const label = describeGraphNode(layerIdx, nodeIdx, layers.length);
                    const gridAvailable = getNeuronGridIndex(layerIdx, nodeIdx) !== null;
                    const unavailable = layerIdx > 0 && !gridAvailable;
                    return <button key={nodeRefKey(layerIdx, nodeIdx)} type="button"
                        className="network-node-target" aria-label={renderer === 'svg' ? [label, ...buildNodeTooltipLines(layerIdx, nodeIdx).filter((line) => line !== label)].join('. ') : label} aria-pressed={selected}
                        data-grid-available={gridAvailable}
                        aria-description={unavailable ? 'Activation grid not available' : undefined}
                        title={buildNodeTooltipLines(layerIdx, nodeIdx).join('. ')}
                        style={{ position: 'absolute', left: node.x * viewport.zoom + viewport.panX - size / 2,
                            top: node.y * viewport.zoom + viewport.panY - size / 2, width: size, height: size,
                            background: 'transparent', border: 'none', color: palette.muted, borderRadius: geometry.cornerRadius * viewport.zoom, padding: 0 }}
                        onClick={(event) => { selectionOrigin.current = event.currentTarget; controller.commands.selectNode({ layerIdx, nodeIdx }); }}
                        onKeyDown={(event) => {
                            if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault(); event.stopPropagation(); selectionOrigin.current = event.currentTarget; controller.commands.selectNode({ layerIdx, nodeIdx });
                            }
                        }}
                        onFocus={() => {
                            const x = node.x * viewport.zoom + viewport.panX, y = node.y * viewport.zoom + viewport.panY;
                            if (x < size / 2 || x > containerSize.width - size / 2 || y < size / 2 || y > containerSize.height - size / 2) {
                                setViewport((v) => ({ ...v, panX: containerSize.width / 2 - node.x * v.zoom, panY: containerSize.height / 2 - node.y * v.zoom }));
                            }
                            setTooltip({ x, y, text: buildNodeTooltipLines(layerIdx, nodeIdx) });
                        }}
                        onBlur={() => setTooltip(null)}
                        onPointerEnter={() => setTooltip({ x: node.x * viewport.zoom + viewport.panX, y: node.y * viewport.zoom + viewport.panY, text: buildNodeTooltipLines(layerIdx, nodeIdx) })}
                        onPointerLeave={() => setTooltip(null)}
                    >{layerIdx === 0 ? <span aria-hidden="true">{activeFeatureLabels[nodeIdx]}</span> : unavailable && <span aria-hidden="true">—</span>}</button>;
                }))}
            </div>

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
        </NetworkGraphFrame>
    );
}
