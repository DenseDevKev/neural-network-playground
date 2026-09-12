// ── SVG Network Graph ──
// Renders the neural network topology as an interactive SVG with hover tooltips
// and mini heatmaps inside each neuron.
//
// Decomposed into three memoized subcomponents:
//   NetworkLabels  — layer header text; only re-renders on network shape change
//   NetworkEdges   — edge lines coloured by weight; re-renders on weight change or edge hover
//   NetworkNodes   — neuron circles with optional heatmaps; re-renders on bias/heatmap change

import { useMemo, useState, useCallback, useEffect, useRef, memo } from 'react';
import { NetworkGraphFrame } from './NetworkGraphFrame.tsx';
import { useNetworkSelectionController, type NetworkSelectionController } from './useNetworkSelectionController.ts';
import { describeGraphNode, edgeRefKey, shouldRenderEdge, nodeRefKey, type EdgeFilter, type GraphViewMode } from './networkGraphPainter.ts';
import { classifyNeuronActivity, formatArchitectureStory, getCapacityLabel } from './NetworkGraphCanvas.tsx';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getLessonDefinition } from '../../lessons/lessonRegistry.ts';
import { getDatasetTopologyHint } from '../../data/datasetInsights.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { getActiveFeatures, type FeatureFlags } from '@nn-playground/engine';
import { writeNormalizedHeatmap } from '@nn-playground/shared';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import {
    extractNeuronGrid,
    layerBiasOffset,
    layerWeightOffset,
} from '../../worker/frameBufferLayout.ts';

// ── Flat-network view ──────────────────────────────────────────────────────
// Stable reference to the packed weight/bias buffers plus layer sizes. The
// graph never rebuilds nested number[][][] arrays per frame: it indexes the
// flat Float32Array directly using layerWeightOffset / layerBiasOffset.
interface FlatNetworkView {
    weights: Float32Array;
    biases: Float32Array;
    layerSizes: number[];
}

const NODE_RADIUS = 14;
const MIN_LAYER_GAP = 120;
const MIN_NODE_GAP = 42;
const PAD_X = 60;
const PAD_Y = 40;
const HEATMAP_SIZE = 24; // pixels for mini heatmap canvas
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

interface NodePos {
    x: number;
    y: number;
}

interface TooltipData {
    x: number;
    y: number;
    text: string[];
}

interface FocusTargetPosition {
    x: number;
    y: number;
}


function edgeConnectionLabel(layerIdx: number, nodeIdx: number, prevIdx: number, layerCount: number): string {
    return `${describeGraphNode(layerIdx - 1, prevIdx, layerCount)} to ${describeGraphNode(layerIdx, nodeIdx, layerCount)}`;
}

function bezierMidpoint(prev: NodePos, node: NodePos): FocusTargetPosition {
    const cpX = (node.x - prev.x) * 0.45;
    const x0 = prev.x;
    const y0 = prev.y;
    const x1 = prev.x + cpX;
    const y1 = prev.y;
    const x2 = node.x - cpX;
    const y2 = node.y;
    const x3 = node.x;
    const y3 = node.y;
    const t = 0.5;
    const u = 1 - t;
    return {
        x: u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * x3,
        y: u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y3,
    };
}

function toFixedLabel(value: number | null | undefined, digits = 4): string {
    if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
    return value.toFixed(digits);
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function nodeColor(value: number): string {
    const abs = Math.min(Math.abs(value), 2) / 2;
    if (value > 0) return `rgba(129, 236, 255, ${0.4 + abs * 0.6})`;
    return `rgba(188, 135, 254, ${0.4 + abs * 0.6})`;
}

function edgeColor(weight: number): string {
    const abs = Math.min(Math.abs(weight), 3) / 3;
    if (weight > 0) return `rgba(129, 236, 255, ${0.2 + abs * 0.6})`;
    return `rgba(188, 135, 254, ${0.2 + abs * 0.6})`;
}

function edgeWidth(weight: number): number {
    return Math.max(0.5, Math.min(3, Math.abs(weight) * 1.5));
}

// ── Persistent source canvas for heatmap generation ──
// One shared upscale-source canvas is reused across all neurons, every frame.
// Each neuron owns its own display <canvas> inside a <foreignObject>, drawn
// to via drawImage — no PNG encoding (toDataURL) on the hot path.
let _sourceCanvas: HTMLCanvasElement | null = null;
let _sourceCtx: CanvasRenderingContext2D | null = null;
let _cachedImageData: ImageData | null = null;
let _cachedGridSize = 0;

function getSourceCanvas(gridSize: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; imageData: ImageData } {
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

// ── HeatmapCanvas ────────────────────────────────────────────────────────────
// Renders a single neuron's activation grid into a dedicated <canvas> via
// putImageData + drawImage (no toDataURL). Reuses the module-level source
// canvas for the upscale step. The canvas is clipped to a circle via CSS
// border-radius so it fits inside the node.

interface HeatmapCanvasProps {
    grid: ArrayLike<number>;
    gridSize: number;
}

const HeatmapCanvas = memo(function HeatmapCanvas({ grid, gridSize }: HeatmapCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match the backing store to device pixels so tiles stay crisp on
        // high-DPI displays.
        const dpr = window.devicePixelRatio || 1;
        const backingSize = Math.max(HEATMAP_SIZE, Math.round(HEATMAP_SIZE * dpr));
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
        ctx.clearRect(0, 0, HEATMAP_SIZE, HEATMAP_SIZE);
        ctx.drawImage(src.canvas, 0, 0, HEATMAP_SIZE, HEATMAP_SIZE);
    }, [grid, gridSize]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                display: 'block',
            }}
        />
    );
});

// ── NetworkLabels ─────────────────────────────────────────────────────────────
// Only re-renders when the network shape (nodePositions / layerLabels) changes.
// During training with a fixed topology, this never re-renders.

interface NetworkLabelsProps {
    nodePositions: NodePos[][];
    layerLabels: string[];
}

const NetworkLabels = memo(function NetworkLabels({ nodePositions, layerLabels }: NetworkLabelsProps) {
    return (
        <>
            {nodePositions.map((layerNodes, layerIdx) => {
                const x = layerNodes[0]?.x ?? 0;
                return (
                    <text
                        key={`label-${layerIdx}`}
                        x={x}
                        y={20}
                        textAnchor="middle"
                        fill="rgba(255,255,255,0.35)"
                        fontSize="10"
                        fontWeight="600"
                        fontFamily="Inter, sans-serif"
                    >
                        {layerLabels[layerIdx]}
                    </text>
                );
            })}
        </>
    );
});

// ── NetworkEdges ──────────────────────────────────────────────────────────────
// Re-renders on weight change or edge hover change.
// Does NOT re-render when tooltip text changes or nodes are hovered.

interface NetworkEdgesProps {
    nodePositions: NodePos[][];
    // Packed flat-network view (from the frame buffer's Float32Arrays).
    // Kept as a reference prop so the memo skips re-renders when only
    // unrelated state (e.g. tooltip text) changes upstream.
    flat: FlatNetworkView | null;
    hoveredEdge: string | null;
    highlightedEdgeKeys?: ReadonlySet<string>;
    filter: EdgeFilter;
    viewMode: GraphViewMode;
    activations: ReadonlyMap<string, number>;
    onEdgeEnter: (layerIdx: number, nodeIdx: number, prevIdx: number, weight: number, x: number, y: number) => void;
    onEdgeLeave: () => void;
    onEdgeFocus: (layerIdx: number, nodeIdx: number, prevIdx: number, weight: number, x: number, y: number) => void;
}

const NetworkEdges = memo(function NetworkEdges({
    nodePositions,
    flat,
    hoveredEdge,
    highlightedEdgeKeys, filter, viewMode, activations,
    onEdgeEnter,
    onEdgeLeave,
    onEdgeFocus,
}: NetworkEdgesProps) {
    return (
        <>
            {nodePositions.map((layerNodes, layerIdx) => {
                if (layerIdx === 0) return null;
                const prevNodes = nodePositions[layerIdx - 1];
                // Cache the per-layer weight base + fanIn once per layer.
                let base = 0;
                let fanIn = 0;
                if (flat) {
                    base = layerWeightOffset(flat.layerSizes, layerIdx - 1);
                    fanIn = flat.layerSizes[layerIdx - 1];
                }
                return layerNodes.map((node, nodeIdx) =>
                    prevNodes.map((prevNode, prevIdx) => {
                        const weight = flat ? flat.weights[base + nodeIdx * fanIn + prevIdx] : 0;
                        if (!Number.isFinite(weight) || !shouldRenderEdge(weight, filter)) return null;
                        const safeWeight = weight;
                        const selectedKey = edgeRefKey(layerIdx, nodeIdx, prevIdx);
                        const isSelected = highlightedEdgeKeys?.has(selectedKey) ?? false;
                        const intensity = activations.get(nodeRefKey(layerIdx, nodeIdx)) ?? 0;
                        const key = `e-${layerIdx}-${nodeIdx}-${prevIdx}`;
                        const isHovered = hoveredEdge === key;
                        const cpX = (node.x - prevNode.x) * 0.45;
                        const pathD = `M ${prevNode.x},${prevNode.y} C ${prevNode.x + cpX},${prevNode.y} ${node.x - cpX},${node.y} ${node.x},${node.y}`;
                        const midpoint = bezierMidpoint(prevNode, node);
                        const connection = edgeConnectionLabel(layerIdx, nodeIdx, prevIdx, nodePositions.length);
                        const ariaLabel = `Weight: ${toFixedLabel(safeWeight)}. Connection: ${connection}`;

                        return (
                            <g key={key}>
                                {/* Wide transparent hit area */}
                                <path
                                    role="button"
                                    tabIndex={0}
                                    aria-label={ariaLabel}
                                    className="network-edge-hit"
                                    d={pathD}
                                    stroke="transparent"
                                    strokeWidth={12}
                                    fill="none"
                                    style={{ cursor: 'pointer' }}
                                    onMouseEnter={(e) => {
                                        const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                                        onEdgeEnter(layerIdx, nodeIdx, prevIdx, safeWeight, e.clientX - rect.left, e.clientY - rect.top);
                                    }}
                                    onMouseLeave={onEdgeLeave}
                                    onFocus={() => onEdgeFocus(layerIdx, nodeIdx, prevIdx, safeWeight, midpoint.x, midpoint.y)}
                                    onBlur={onEdgeLeave}
                                />
                                {/* Visible edge */}
                                <path
                                    d={pathD}
                                    fill="none"
                                    data-edge-key={selectedKey}
                                    data-edge-selected={isSelected}
                                    stroke={viewMode === 'activations' && !isSelected && !isHovered ? `rgba(249,115,22,${0.12 + intensity * 0.72})` : edgeColor(safeWeight)}
                                    strokeWidth={isSelected ? 1.5 + Math.min(3, Math.abs(safeWeight)) * 1.5 : isHovered ? edgeWidth(safeWeight) * 2 : edgeWidth(safeWeight)}
                                    opacity={highlightedEdgeKeys?.size && !isSelected ? 0.28 : 1}
                                    strokeDasharray={isSelected && safeWeight < 0 ? '6 4' : undefined}
                                    style={{ transition: 'stroke-width 200ms ease, opacity 200ms ease', pointerEvents: 'none' }}
                                />
                                {/* Flow animation */}
                                {Math.abs(safeWeight) > 0.05 && (
                                    <path
                                        d={pathD}
                                        fill="none"
                                        stroke={safeWeight > 0 ? 'rgba(249, 115, 22, 0.8)' : 'rgba(59, 130, 246, 0.8)'}
                                        strokeWidth={1.5}
                                        strokeDasharray="4 12"
                                        className={safeWeight > 0 ? 'network-flow-anim' : 'network-flow-anim-reverse'}
                                        opacity={(highlightedEdgeKeys?.size ? 0.1 : 1) * (0.4 + Math.min(Math.abs(safeWeight), 2) * 0.25)}
                                        style={{ pointerEvents: 'none' }}
                                    />
                                )}
                            </g>
                        );
                    }),
                );
            })}
        </>
    );
});

// ── NetworkNodes ──────────────────────────────────────────────────────────────
// Re-renders on bias / heatmap change.
// Does NOT re-render when edge hover or edge tooltip changes.

interface NeuronGridEntry {
    grid: ArrayLike<number>;
    gridSize: number;
}

interface NetworkNodesProps {
    nodePositions: NodePos[][];
    layers: number[];
    flat: FlatNetworkView | null;
    neuronGrids: NeuronGridEntry[] | null;
    activeFeatures: { label: string }[];
    activation: string;
    controller: NetworkSelectionController;
    onNodeEnter: (x: number, y: number, text: string[]) => void;
    onNodeLeave: () => void;
    onNodeFocus: (x: number, y: number, text: string[]) => void;
}

const NetworkNodes = memo(function NetworkNodes({
    nodePositions,
    layers,
    flat,
    neuronGrids,
    activeFeatures,
    activation,
    controller,
    onNodeEnter,
    onNodeLeave,
    onNodeFocus,
}: NetworkNodesProps) {
    /**
     * Map (layerIdx, nodeIdx) → index into neuronGrids.
     * neuronGrids layout: for each hidden layer in order, then output layer,
     * all neurons concatenated.
     */
    function getNeuronGridIndex(layerIdx: number, nodeIdx: number): number | null {
        if (!neuronGrids || layerIdx === 0) return null;
        let idx = 0;
        for (let l = 1; l < layerIdx; l++) {
            idx += layers[l];
        }
        if (layerIdx === layers.length - 1) {
            // Output layer — skip all hidden layer neurons
            idx = 0;
            for (let l = 1; l < layers.length - 1; l++) {
                idx += layers[l];
            }
        }
        idx += nodeIdx;
        return idx < neuronGrids.length ? idx : null;
    }

    // Read a bias straight from the flat buffer. `layerIdx` is the
    // rendered layer (0 = input); biases live on layers 1..N mapped to
    // flat row `layerIdx - 1`.
    function biasAt(layerIdx: number, nodeIdx: number): number | undefined {
        if (!flat || layerIdx === 0) return undefined;
        const offset = layerBiasOffset(flat.layerSizes, layerIdx - 1);
        return flat.biases[offset + nodeIdx];
    }

    function buildTooltipLines(layerIdx: number, nodeIdx: number): string[] {
        const isInput = layerIdx === 0;
        const isOutput = layerIdx === layers.length - 1;
        const lines: string[] = [];

        if (isInput) {
            const feat = activeFeatures[nodeIdx];
            lines.push(`Input: ${feat?.label ?? `x${nodeIdx}`}`);
        } else if (isOutput) {
            lines.push('Output neuron');
            const bias = biasAt(layerIdx, nodeIdx);
            if (bias != null) {
                const biasText = toFixedLabel(bias);
                if (biasText !== 'N/A') lines.push(`Bias: ${biasText}`);
            }
        } else {
            lines.push(`Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`);
            const bias = biasAt(layerIdx, nodeIdx);
            if (bias != null) {
                const biasText = toFixedLabel(bias);
                if (biasText !== 'N/A') lines.push(`Bias: ${biasText}`);
            }
            lines.push(`Activation: ${activation}`);
        }
        return lines;
    }

    return (
        <>
            {nodePositions.map((layerNodes, layerIdx) =>
                layerNodes.map((node, nodeIdx) => {
                    const bias = biasAt(layerIdx, nodeIdx) ?? 0;
                    const isInput = layerIdx === 0;
                    const isOutput = layerIdx === layers.length - 1;
                    const heatmapIdx = getNeuronGridIndex(layerIdx, nodeIdx);
                    const heatmap = heatmapIdx != null ? neuronGrids?.[heatmapIdx] ?? null : null;
                    const tooltipLines = buildTooltipLines(layerIdx, nodeIdx);
                    const label = describeGraphNode(layerIdx, nodeIdx, layers.length);
                    const ariaLabel = [label, ...tooltipLines.filter((line) => line !== label)].join('. ');
                    const selected = controller.selectedNode?.layerIdx === layerIdx && controller.selectedNode.nodeIdx === nodeIdx;
                    const health = heatmap ? classifyNeuronActivity(heatmap.grid, activation as Parameters<typeof classifyNeuronActivity>[1]) : null;

                    return (
                        <g
                            key={`n-${layerIdx}-${nodeIdx}`}
                            role="button"
                            tabIndex={0}
                            aria-label={ariaLabel}
                            aria-pressed={selected}
                            data-grid-available={heatmap !== null}
                            aria-description={!isInput && !heatmap ? 'Activation grid not available' : undefined}
                            data-node-health={health ?? (heatmap ? 'available' : 'unavailable')}
                            className="network-node-hit"
                            onClick={() => controller.commands.selectNode({ layerIdx, nodeIdx })}
                            onKeyDown={(event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                    event.preventDefault(); event.stopPropagation(); controller.commands.selectNode({ layerIdx, nodeIdx });
                                }
                            }}
                            style={{ cursor: 'pointer' }}
                            onMouseEnter={(e) => {
                                const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                                onNodeEnter(
                                    e.clientX - rect.left,
                                    e.clientY - rect.top,
                                    tooltipLines,
                                );
                            }}
                            onMouseLeave={onNodeLeave}
                            onFocus={() => onNodeFocus(node.x, node.y, tooltipLines)}
                            onBlur={onNodeLeave}
                        >
                            {selected && <circle cx={node.x} cy={node.y} r={NODE_RADIUS + 4} fill="none" stroke="white" strokeWidth={2.5} pointerEvents="none" />}
                            {health && <circle cx={node.x} cy={node.y} r={NODE_RADIUS + 7} fill="none" stroke={health === 'low' ? '#ef4444' : '#eab308'} strokeWidth={2} strokeDasharray={health === 'low' ? '3 3' : '1 3'} pointerEvents="none" />}

                            <circle
                                className="network-node-focus-ring"
                                cx={node.x}
                                cy={node.y}
                                r={NODE_RADIUS + 9}
                                fill="none"
                                stroke="rgba(255, 255, 255, 0)"
                                strokeWidth={2.5}
                                style={{ pointerEvents: 'none' }}
                            />
                            {/* Glow ring */}
                            <circle
                                cx={node.x}
                                cy={node.y}
                                r={NODE_RADIUS + 6}
                                fill="none"
                                stroke={
                                    isOutput
                                        ? 'rgba(124, 92, 252, 0.5)'
                                        : isInput
                                            ? 'rgba(0, 229, 195, 0.5)'
                                            : nodeColor(bias)
                                }
                                strokeWidth={2}
                                opacity={0.6}
                                filter="url(#node-glow)"
                                className={isInput ? 'node-pulse' : ''}
                            />
                            {/* Node background */}
                            <circle
                                cx={node.x}
                                cy={node.y}
                                r={NODE_RADIUS}
                                fill="#1c2030"
                                stroke={
                                    isOutput
                                        ? '#7c5cfc'
                                        : isInput
                                            ? '#00e5c3'
                                            : 'rgba(255,255,255,0.15)'
                                }
                                strokeWidth={1.5}
                                className="network-node"
                            />
                            {/* Mini heatmap (for non-input neurons) */}
                            {!isInput && heatmap && (
                                <foreignObject
                                    x={node.x - NODE_RADIUS + 1.5}
                                    y={node.y - NODE_RADIUS + 1.5}
                                    width={(NODE_RADIUS - 1.5) * 2}
                                    height={(NODE_RADIUS - 1.5) * 2}
                                    style={{ pointerEvents: 'none' }}
                                >
                                    <HeatmapCanvas grid={heatmap.grid} gridSize={heatmap.gridSize} />
                                </foreignObject>
                            )}
                            {/* Missing activation evidence is neutral, never a fabricated grid. */}
                            {!isInput && !heatmap && (
                                <text x={node.x} y={node.y} textAnchor="middle" dominantBaseline="central"
                                    fill="var(--text-muted)" fontSize={12} aria-hidden="true" pointerEvents="none">—</text>
                            )}
                            {/* Border ring on top of heatmap */}
                            {!isInput && heatmap && (
                                <circle
                                    cx={node.x}
                                    cy={node.y}
                                    r={NODE_RADIUS}
                                    fill="none"
                                    stroke={isOutput ? '#7c5cfc' : 'rgba(255,255,255,0.15)'}
                                    strokeWidth={1.5}
                                    style={{ pointerEvents: 'none' }}
                                />
                            )}
                        </g>
                    );
                }),
            )}
        </>
    );
});

// ── NetworkGraphSVG (parent) ─────────────────────────────────────────────────
// Manages tooltip and edge-hover state, passes stable props to subcomponents.
// Tooltip state changes only re-render the tooltip <div> — not the SVG subcomponents.
// Edge hover changes only re-render NetworkEdges.
//
// AS-5 fallback: this is the legacy SVG renderer. The exported `NetworkGraph`
// component in `./NetworkGraph.tsx` picks between this and the canvas
// implementation at runtime via the `featuresUI.canvasNetworkGraph` flag.

export function NetworkGraphSVG({ controller }: { readonly controller?: NetworkSelectionController }) {
    return controller ? <NetworkGraphSVGView controller={controller} /> : <LocalNetworkGraphSVG />;
}
function LocalNetworkGraphSVG() {
    const controller = useNetworkSelectionController();
    return <NetworkGraphSVGView controller={controller} />;
}
function NetworkGraphSVGView({ controller }: { readonly controller: NetworkSelectionController }) {
    const compiled = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared.compiled : null
    ));
    const hiddenLayers = compiled?.network.hiddenLayers ?? EMPTY_HIDDEN_LAYERS;
    const features = compiled?.features ?? EMPTY_FEATURES;
    const activation = compiled?.network.activation ?? 'tanh';
    const outputSize = compiled?.task.outputSize ?? 1;
    const paramsVersion = useTrainingStore((s) => s.paramsVersion);
    const neuronGridsVersion = useTrainingStore((s) => s.neuronGridsVersion);

    const [viewMode, setViewMode] = useState<GraphViewMode>('weights');
    const [filter, setFilter] = useState<EdgeFilter>('all');
    const [viewport, setViewport] = useState({ zoom: 1, x: 0, y: 0 });
    const drag = useRef<{ pointerId: number; x: number; y: number } | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const activeLessonId = useLayoutStore((s) => s.activeLessonId);
    const activeLessonStepIndex = useLayoutStore((s) => s.activeLessonStepIndex);
    const lessonStep = activeLessonId && activeLessonStepIndex !== null ? getLessonDefinition(activeLessonId)?.steps[activeLessonStepIndex] : null;
    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

    const activeFeatures = useMemo(() => getActiveFeatures(features), [features]);
    const inputSize = activeFeatures.length;

    const layers = useMemo(() => {
        return [inputSize, ...hiddenLayers, outputSize];
    }, [inputSize, hiddenLayers, outputSize]);

    const maxNodes = Math.max(...layers);

    const containerRef = useRef<HTMLDivElement>(null);
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

    const { svgWidth, svgHeight, layerGap, nodeGap } = useMemo(() => {
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
        const svgWidth = Math.max(w, layers.length * layerGap + PAD_X * 2);
        const svgHeight = Math.max(h, maxNodes * nodeGap + PAD_Y * 2);
        return { svgWidth, svgHeight, layerGap, nodeGap };
    }, [containerSize, layers, maxNodes]);

    const nodePositions = useMemo(() => {
        const startX = (svgWidth - (layers.length - 1) * layerGap) / 2;
        return layers.map((count, layerIdx) => {
            const x = startX + layerIdx * layerGap;
            const totalHeight = (count - 1) * nodeGap;
            const startY = svgHeight / 2 - totalHeight / 2;
            return Array.from({ length: count }, (_, nodeIdx) => ({
                x,
                y: startY + nodeIdx * nodeGap,
            }));
        });
    }, [layers, svgWidth, svgHeight, layerGap, nodeGap]);

    const layerLabels = useMemo(() => {
        return layers.map((_, idx) => {
            if (idx === 0) return 'Input';
            if (idx === layers.length - 1) return 'Output';
            return `Hidden ${idx}`;
        });
    }, [layers]);

    // Build a flat-buffer view once per frame. Prefer the frame buffer's
    // Float32Arrays (the hot path from the streaming worker); fall back to
    // materialising a packed view from the snapshot's nested arrays only
    // on very first render or in tests that bypass the worker.
    const flat = useMemo<FlatNetworkView | null>(() => {
        // The version selector intentionally drives this mutable frame-buffer read.
        void paramsVersion;
        const frameBuffer = getFrameBuffer();
        if (frameBuffer.weights && frameBuffer.biases && frameBuffer.weightLayout) {
            return {
                weights: frameBuffer.weights,
                biases: frameBuffer.biases,
                layerSizes: frameBuffer.weightLayout.layerSizes,
            };
        }
        return null;
    }, [paramsVersion]);

    // Build per-neuron grid views (no PNG encoding). Each HeatmapCanvas then
    // paints its grid into a real <canvas> via putImageData + drawImage.
    const neuronGrids = useMemo<NeuronGridEntry[] | null>(() => {
        // The version selector intentionally drives this mutable frame-buffer read.
        void neuronGridsVersion;
        const frameBuffer = getFrameBuffer();
        if (frameBuffer.neuronGrids && frameBuffer.neuronGridLayout) {
            const { count, gridSize } = frameBuffer.neuronGridLayout;
            const cells = gridSize * gridSize;
            return Array.from({ length: count }, (_, idx) => ({
                grid: extractNeuronGrid(frameBuffer.neuronGrids!, idx, cells),
                gridSize,
            }));
        }
        return null;
    }, [neuronGridsVersion]);

    const activations = useMemo(() => {
        const values = new Map<string, number>();
        let index = 0;
        for (let layer = 1; layer < layers.length; layer++) {
            for (let node = 0; node < layers[layer]; node++, index++) {
                const grid = neuronGrids?.[index]?.grid;
                if (!grid?.length) continue;
                let magnitude = 0;
                for (let i = 0; i < grid.length; i++) magnitude += Math.abs(grid[i]);
                values.set(nodeRefKey(layer, node), Math.max(0, Math.min(1, magnitude / grid.length)));
            }
        }
        return values;
    }, [layers, neuronGrids]);
    const zoomBy = useCallback((factor: number) => setViewport((v) => {
        const zoom = Math.max(0.35, Math.min(2.5, v.zoom * factor));
        const ratio = zoom / v.zoom;
        return { zoom, x: svgWidth / 2 - (svgWidth / 2 - v.x) * ratio, y: svgHeight / 2 - (svgHeight / 2 - v.y) * ratio };
    }), [svgHeight, svgWidth]);
    const resetViewport = useCallback(() => setViewport({ zoom: 1, x: 0, y: 0 }), []);
    const layerKey = layers.join(':');
    useEffect(() => { resetViewport(); }, [layerKey, resetViewport]);
    useEffect(() => {
        const svg = svgRef.current;
        if (!svg) return;
        const wheel = (event: WheelEvent) => { event.preventDefault(); zoomBy(event.deltaY < 0 ? 1.25 : 0.8); };
        svg.addEventListener('wheel', wheel, { passive: false });
        return () => svg.removeEventListener('wheel', wheel);
    }, [zoomBy]);

    // ── Stable handlers (no deps — all data flows in via arguments or closure over setters) ──

    const handleEdgeEnter = useCallback((
        layerIdx: number,
        nodeIdx: number,
        prevIdx: number,
        weight: number,
        x: number,
        y: number,
    ) => {
        const key = `e-${layerIdx}-${nodeIdx}-${prevIdx}`;
        setHoveredEdge(key);
        setTooltip({
            x,
            y,
            text: [`Weight: ${toFixedLabel(weight)}`, `Layer ${layerIdx}, [${prevIdx}→${nodeIdx}]`],
        });
    }, []);

    const toCssPosition = useCallback(
        ({ x, y }: FocusTargetPosition): FocusTargetPosition => {
            const scale = Math.min(containerSize.width / svgWidth, containerSize.height / svgHeight);
            return {
                x: (containerSize.width - svgWidth * scale) / 2 + (x * viewport.zoom + viewport.x) * scale,
                y: (containerSize.height - svgHeight * scale) / 2 + (y * viewport.zoom + viewport.y) * scale,
            };
        },
        [svgWidth, svgHeight, containerSize.width, containerSize.height, viewport],
    );

    const handleEdgeFocus = useCallback((
        layerIdx: number,
        nodeIdx: number,
        prevIdx: number,
        weight: number,
        x: number,
        y: number,
    ) => {
        const css = toCssPosition({ x, y });
        handleEdgeEnter(layerIdx, nodeIdx, prevIdx, weight, css.x, css.y);
    }, [handleEdgeEnter, toCssPosition]);

    const handleEdgeLeave = useCallback(() => {
        setHoveredEdge(null);
        setTooltip(null);
    }, []);

    const handleNodeEnter = useCallback((x: number, y: number, text: string[]) => {
        setTooltip({ x, y, text });
    }, []);

    const handleNodeFocus = useCallback((x: number, y: number, text: string[]) => {
        setTooltip({ ...toCssPosition({ x, y }), text });
    }, [toCssPosition]);

    const handleNodeLeave = useCallback(() => {
        setTooltip(null);
    }, []);

    return (
        <NetworkGraphFrame
            story={formatArchitectureStory(activeFeatures.map((f) => f.label === 'x' ? 'X₁' : f.label === 'y' ? 'X₂' : f.label), hiddenLayers, outputSize, compiled?.task.outputActivation ?? 'sigmoid')}
            capacity={getCapacityLabel(hiddenLayers)} datasetHint={compiled ? getDatasetTopologyHint(compiled.data.dataset, hiddenLayers) : null}
            lesson={lessonStep?.target === 'network' ? lessonStep.body : undefined}
            zoom={viewport.zoom} onZoomOut={() => zoomBy(0.8)} onZoomIn={() => zoomBy(1.25)} onFit={resetViewport}
            viewMode={viewMode} onViewMode={setViewMode} edgeFilter={filter} onEdgeFilter={setFilter}
        >
        <div ref={containerRef} className="network-graph-container" style={{ position: 'relative', width: '100%', height: '100%', minWidth: 0, overflow: 'hidden' }}
            onKeyDown={(event) => {
                if (event.key === 'Escape' && controller.selectedNode) {
                    event.preventDefault(); event.stopPropagation(); controller.commands.clearSelection();
                }
            }}>
            <svg
                ref={svgRef}
                aria-label="Neural network graph"
                onPointerDown={(event) => {
                    if (event.button !== 0 || (event.target as Element).closest('[role="button"]')) return;
                    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
                    event.currentTarget.setPointerCapture?.(event.pointerId);
                }}
                onPointerMove={(event) => {
                    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
                    const scale = Math.min(containerSize.width / svgWidth, containerSize.height / svgHeight);
                    const dx = (event.clientX - drag.current.x) / scale, dy = (event.clientY - drag.current.y) / scale;
                    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
                    setViewport((v) => ({ ...v, x: v.x + dx, y: v.y + dy }));
                }}
                onPointerUp={(event) => { drag.current = null; if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
                onPointerCancel={() => { drag.current = null; }}
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                preserveAspectRatio="xMidYMid meet"
                style={{ width: '100%', height: '100%', display: 'block' }}
                onMouseLeave={() => { setTooltip(null); setHoveredEdge(null); }}
            >
                {/* Shared defs (glow filter) */}
                <defs>
                    <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                <g transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.zoom})`}>
                {/* Edges — re-renders on weight change or hoveredEdge change */}
                <NetworkEdges
                    nodePositions={nodePositions}
                    flat={flat}
                    hoveredEdge={hoveredEdge}
                    highlightedEdgeKeys={controller.model.kind === 'selected' ? controller.model.highlightedEdgeKeys : undefined}
                    filter={filter} viewMode={viewMode} activations={activations}
                    onEdgeEnter={handleEdgeEnter}
                    onEdgeLeave={handleEdgeLeave}
                    onEdgeFocus={handleEdgeFocus}
                />

                {/* Nodes — re-renders on bias/heatmap change; skips on edge hover */}
                <NetworkNodes
                    nodePositions={nodePositions}
                    layers={layers}
                    flat={flat}
                    neuronGrids={neuronGrids}
                    activeFeatures={activeFeatures}
                    activation={activation}
                    controller={controller}
                    onNodeEnter={handleNodeEnter}
                    onNodeLeave={handleNodeLeave}
                    onNodeFocus={handleNodeFocus}
                />

                {/* Labels — only re-renders when network shape changes */}
                <NetworkLabels
                    nodePositions={nodePositions}
                    layerLabels={layerLabels}
                />
                </g>
            </svg>

            {/* Tooltip overlay */}
            {tooltip && (
                <div className="network-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}>
                    {tooltip.text.map((line, i) => (
                        <div key={i} className={i === 0 ? 'network-tooltip__title' : 'network-tooltip__detail'}>
                            {line}
                        </div>
                    ))}
                </div>
            )}
        </div>
        </NetworkGraphFrame>
    );
}
