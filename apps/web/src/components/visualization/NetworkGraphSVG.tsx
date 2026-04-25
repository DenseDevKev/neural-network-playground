// ── SVG Network Graph ──
// Renders the neural network topology as an interactive SVG with hover tooltips
// and mini heatmaps inside each neuron.
//
// Decomposed into three memoized subcomponents:
//   NetworkLabels  — layer header text; only re-renders on network shape change
//   NetworkEdges   — edge lines coloured by weight; re-renders on weight change or edge hover
//   NetworkNodes   — neuron circles with optional heatmaps; re-renders on bias/heatmap change

import { useMemo, useState, useCallback, useEffect, useRef, memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { getActiveFeatures } from '@nn-playground/engine';
import { GRID_SIZE, writeNormalizedHeatmap } from '@nn-playground/shared';
import {
    extractNeuronGrid,
    getFrameBuffer,
    layerBiasOffset,
    layerWeightOffset,
} from '../../worker/frameBuffer.ts';

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

function describeGraphNode(layerIdx: number, nodeIdx: number, layerCount: number): string {
    if (layerIdx === 0) return `Input ${nodeIdx + 1}`;
    if (layerIdx === layerCount - 1) return 'Output neuron';
    return `Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`;
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
    onEdgeEnter: (layerIdx: number, nodeIdx: number, prevIdx: number, weight: number, x: number, y: number) => void;
    onEdgeLeave: () => void;
    onEdgeFocus: (layerIdx: number, nodeIdx: number, prevIdx: number, weight: number, x: number, y: number) => void;
}

const NetworkEdges = memo(function NetworkEdges({
    nodePositions,
    flat,
    hoveredEdge,
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
                        const safeWeight = Number.isFinite(weight) ? weight : 0;
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
                                    stroke={edgeColor(safeWeight)}
                                    strokeWidth={isHovered ? edgeWidth(safeWeight) * 2 : edgeWidth(safeWeight)}
                                    opacity={isHovered ? 1 : 0.7}
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
                                        opacity={0.4 + Math.min(Math.abs(safeWeight), 2) * 0.25}
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
                    const ariaLabel = tooltipLines.join('. ');

                    return (
                        <g
                            key={`n-${layerIdx}-${nodeIdx}`}
                            role="button"
                            tabIndex={0}
                            aria-label={ariaLabel}
                            className="network-node-hit"
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
                            {/* Fallback inner value indicator (when no heatmap) */}
                            {!isInput && !heatmap && (
                                <circle
                                    cx={node.x}
                                    cy={node.y}
                                    r={NODE_RADIUS - 4}
                                    fill={nodeColor(bias)}
                                    opacity={0.6}
                                    style={{ pointerEvents: 'none' }}
                                />
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

export function NetworkGraphSVG() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const features = usePlaygroundStore((s) => s.features);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const snapshotWeights = useTrainingStore((s) => s.snapshot?.weights);
    const snapshotBiases = useTrainingStore((s) => s.snapshot?.biases);
    const snapshotNeuronGrids = useTrainingStore((s) => s.snapshot?.neuronGrids);
    const snapshotGridSize = useTrainingStore((s) => s.snapshot?.gridSize ?? GRID_SIZE);
    const paramsVersion = useTrainingStore((s) => s.paramsVersion);
    const neuronGridsVersion = useTrainingStore((s) => s.neuronGridsVersion);

    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

    const activeFeatures = useMemo(() => getActiveFeatures(features), [features]);
    const inputSize = activeFeatures.length;

    const layers = useMemo(() => {
        return [inputSize, ...hiddenLayers, 1];
    }, [inputSize, hiddenLayers]);

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
        if (snapshotWeights && snapshotWeights.length > 0 && snapshotBiases) {
            // Snapshot fallback: pack on the fly. Happens rarely — only before
            // the first streamed frame arrives.
            const sizes: number[] = [snapshotWeights[0]?.[0]?.length ?? 0];
            let total = 0;
            for (const layer of snapshotWeights) {
                sizes.push(layer.length);
                for (const neuron of layer) total += neuron.length;
            }
            const w = new Float32Array(total);
            let off = 0;
            for (const layer of snapshotWeights) {
                for (const neuron of layer) {
                    w.set(neuron, off);
                    off += neuron.length;
                }
            }
            let btotal = 0;
            for (const layer of snapshotBiases) btotal += layer.length;
            const b = new Float32Array(btotal);
            off = 0;
            for (const layer of snapshotBiases) {
                b.set(layer, off);
                off += layer.length;
            }
            return { weights: w, biases: b, layerSizes: sizes };
        }
        return null;
    }, [paramsVersion, snapshotWeights, snapshotBiases]);

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
        if (snapshotNeuronGrids instanceof Float32Array) {
            const gridSize = snapshotGridSize;
            const count = layers.slice(1).reduce((sum, size) => sum + size, 0);
            const cells = gridSize * gridSize;
            return Array.from({ length: count }, (_, idx) => ({
                grid: extractNeuronGrid(snapshotNeuronGrids, idx, cells),
                gridSize,
            }));
        }
        if (!snapshotNeuronGrids) return null;
        return snapshotNeuronGrids.map((grid) => ({ grid, gridSize: snapshotGridSize }));
    }, [neuronGridsVersion, layers, snapshotNeuronGrids, snapshotGridSize]);

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
        ({ x, y }: FocusTargetPosition): FocusTargetPosition => ({
            x: (x / svgWidth) * containerSize.width,
            y: (y / svgHeight) * containerSize.height,
        }),
        [svgWidth, svgHeight, containerSize.width, containerSize.height],
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
        <div ref={containerRef} className="network-graph-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
            <svg
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

                {/* Edges — re-renders on weight change or hoveredEdge change */}
                <NetworkEdges
                    nodePositions={nodePositions}
                    flat={flat}
                    hoveredEdge={hoveredEdge}
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
                    onNodeEnter={handleNodeEnter}
                    onNodeLeave={handleNodeLeave}
                    onNodeFocus={handleNodeFocus}
                />

                {/* Labels — only re-renders when network shape changes */}
                <NetworkLabels
                    nodePositions={nodePositions}
                    layerLabels={layerLabels}
                />
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
    );
}
