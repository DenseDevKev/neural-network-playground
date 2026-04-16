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
import { extractNeuronGrid, getFrameBuffer, unflattenBiases, unflattenWeights } from '../../worker/frameBuffer.ts';

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

// ── Persistent canvas pool for heatmap generation ──
// Avoids creating and destroying DOM canvas elements every frame.
let _sourceCanvas: HTMLCanvasElement | null = null;
let _sourceCtx: CanvasRenderingContext2D | null = null;
let _scaledCanvas: HTMLCanvasElement | null = null;
let _scaledCtx: CanvasRenderingContext2D | null = null;
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

function getScaledCanvas(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
    if (!_scaledCanvas || !_scaledCtx) {
        _scaledCanvas = document.createElement('canvas');
        _scaledCanvas.width = HEATMAP_SIZE;
        _scaledCanvas.height = HEATMAP_SIZE;
        _scaledCtx = _scaledCanvas.getContext('2d')!;
        _scaledCtx.imageSmoothingEnabled = true;
        _scaledCtx.imageSmoothingQuality = 'high';
    }
    return { canvas: _scaledCanvas, ctx: _scaledCtx };
}

/**
 * Generate a data URL from a flat grid of neuron activations.
 * Uses shared blue–dark–orange color scale and persistent canvases.
 */
function neuronGridToDataUrl(grid: ArrayLike<number>, gridSize: number): string {
    const { canvas: sourceCanvas, ctx: sourceCtx, imageData } = getSourceCanvas(gridSize);

    writeNormalizedHeatmap(grid, imageData, 220);
    sourceCtx.putImageData(imageData, 0, 0);

    // Scale up to HEATMAP_SIZE with smoothing (reused canvas)
    const { canvas: scaledCanvas, ctx: scaledCtx } = getScaledCanvas();
    scaledCtx.clearRect(0, 0, HEATMAP_SIZE, HEATMAP_SIZE);
    scaledCtx.drawImage(sourceCanvas, 0, 0, HEATMAP_SIZE, HEATMAP_SIZE);

    return scaledCanvas.toDataURL();
}

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
    weights: number[][][] | null | undefined; // snapshot.weights: number[][][]
    hoveredEdge: string | null;
    onEdgeEnter: (layerIdx: number, nodeIdx: number, prevIdx: number, weight: number, x: number, y: number) => void;
    onEdgeLeave: () => void;
}

const NetworkEdges = memo(function NetworkEdges({
    nodePositions,
    weights,
    hoveredEdge,
    onEdgeEnter,
    onEdgeLeave,
}: NetworkEdgesProps) {
    return (
        <>
            {nodePositions.map((layerNodes, layerIdx) => {
                if (layerIdx === 0) return null;
                const prevNodes = nodePositions[layerIdx - 1];
                return layerNodes.map((node, nodeIdx) =>
                    prevNodes.map((prevNode, prevIdx) => {
                        const weight = weights?.[layerIdx - 1]?.[nodeIdx]?.[prevIdx] ?? 0;
                        const key = `e-${layerIdx}-${nodeIdx}-${prevIdx}`;
                        const isHovered = hoveredEdge === key;
                        const cpX = (node.x - prevNode.x) * 0.45;
                        const pathD = `M ${prevNode.x},${prevNode.y} C ${prevNode.x + cpX},${prevNode.y} ${node.x - cpX},${node.y} ${node.x},${node.y}`;

                        return (
                            <g key={key}>
                                {/* Wide transparent hit area */}
                                <path
                                    d={pathD}
                                    stroke="transparent"
                                    strokeWidth={12}
                                    fill="none"
                                    style={{ cursor: 'pointer' }}
                                    onMouseEnter={(e) => {
                                        const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                                        onEdgeEnter(layerIdx, nodeIdx, prevIdx, weight, e.clientX - rect.left, e.clientY - rect.top);
                                    }}
                                    onMouseLeave={onEdgeLeave}
                                />
                                {/* Visible edge */}
                                <path
                                    d={pathD}
                                    fill="none"
                                    stroke={edgeColor(weight)}
                                    strokeWidth={isHovered ? edgeWidth(weight) * 2 : edgeWidth(weight)}
                                    opacity={isHovered ? 1 : 0.7}
                                    style={{ transition: 'stroke-width 200ms ease, opacity 200ms ease', pointerEvents: 'none' }}
                                />
                                {/* Flow animation */}
                                {Math.abs(weight) > 0.05 && (
                                    <path
                                        d={pathD}
                                        fill="none"
                                        stroke={weight > 0 ? 'rgba(249, 115, 22, 0.8)' : 'rgba(59, 130, 246, 0.8)'}
                                        strokeWidth={1.5}
                                        strokeDasharray="4 12"
                                        className={weight > 0 ? 'network-flow-anim' : 'network-flow-anim-reverse'}
                                        opacity={0.4 + Math.min(Math.abs(weight), 2) * 0.25}
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

interface NetworkNodesProps {
    nodePositions: NodePos[][];
    layers: number[];
    biases: number[][] | null | undefined; // snapshot.biases: number[][] (layer × neuron)
    neuronHeatmapUrls: string[] | null;
    activeFeatures: { label: string }[];
    activation: string;
    onNodeEnter: (x: number, y: number, text: string[]) => void;
    onNodeLeave: () => void;
}

const NetworkNodes = memo(function NetworkNodes({
    nodePositions,
    layers,
    biases,
    neuronHeatmapUrls,
    activeFeatures,
    activation,
    onNodeEnter,
    onNodeLeave,
}: NetworkNodesProps) {
    /**
     * Map (layerIdx, nodeIdx) → index into neuronHeatmapUrls.
     * neuronGrids layout: for each hidden layer in order, then output layer,
     * all neurons concatenated.
     */
    function getNeuronGridIndex(layerIdx: number, nodeIdx: number): number | null {
        if (!neuronHeatmapUrls || layerIdx === 0) return null;
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
        return idx < neuronHeatmapUrls.length ? idx : null;
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
            // biases[layerIdx-1] is number[] (one bias per neuron in that layer)
            const bias = biases?.[layerIdx - 1]?.[nodeIdx] as number | undefined;
            if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
        } else {
            lines.push(`Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`);
            const bias = biases?.[layerIdx - 1]?.[nodeIdx] as number | undefined;
            if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
            lines.push(`Activation: ${activation}`);
        }
        return lines;
    }

    return (
        <>
            {nodePositions.map((layerNodes, layerIdx) =>
                layerNodes.map((node, nodeIdx) => {
                    const bias = (biases?.[layerIdx - 1]?.[nodeIdx] as number | undefined) ?? 0;
                    const isInput = layerIdx === 0;
                    const isOutput = layerIdx === layers.length - 1;
                    const heatmapIdx = getNeuronGridIndex(layerIdx, nodeIdx);
                    const heatmapUrl = heatmapIdx != null ? neuronHeatmapUrls?.[heatmapIdx] : null;

                    return (
                        <g
                            key={`n-${layerIdx}-${nodeIdx}`}
                            style={{ cursor: 'pointer' }}
                            onMouseEnter={(e) => {
                                const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                                onNodeEnter(
                                    e.clientX - rect.left,
                                    e.clientY - rect.top,
                                    buildTooltipLines(layerIdx, nodeIdx),
                                );
                            }}
                            onMouseLeave={onNodeLeave}
                        >
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
                            {!isInput && heatmapUrl && (
                                <g>
                                    <defs>
                                        <clipPath id={`clip-${layerIdx}-${nodeIdx}`}>
                                            <circle cx={node.x} cy={node.y} r={NODE_RADIUS - 1.5} />
                                        </clipPath>
                                    </defs>
                                    <image
                                        href={heatmapUrl}
                                        x={node.x - NODE_RADIUS + 1.5}
                                        y={node.y - NODE_RADIUS + 1.5}
                                        width={(NODE_RADIUS - 1.5) * 2}
                                        height={(NODE_RADIUS - 1.5) * 2}
                                        clipPath={`url(#clip-${layerIdx}-${nodeIdx})`}
                                        style={{ pointerEvents: 'none' }}
                                    />
                                </g>
                            )}
                            {/* Fallback inner value indicator (when no heatmap) */}
                            {!isInput && !heatmapUrl && (
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
                            {!isInput && heatmapUrl && (
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

// ── NetworkGraph (parent) ─────────────────────────────────────────────────────
// Manages tooltip and edge-hover state, passes stable props to subcomponents.
// Tooltip state changes only re-render the tooltip <div> — not the SVG subcomponents.
// Edge hover changes only re-render NetworkEdges.

export function NetworkGraph() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const features = usePlaygroundStore((s) => s.features);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);

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

    const weights = useMemo(() => {
        const frameBuffer = getFrameBuffer();
        if (frameBuffer.weights && frameBuffer.weightLayout) {
            return unflattenWeights(frameBuffer.weights, frameBuffer.weightLayout.layerSizes);
        }
        return snapshot?.weights ?? null;
    }, [frameVersion, snapshot?.weights]);

    const biases = useMemo(() => {
        const frameBuffer = getFrameBuffer();
        if (frameBuffer.biases && frameBuffer.weightLayout) {
            return unflattenBiases(frameBuffer.biases, frameBuffer.weightLayout.layerSizes);
        }
        return snapshot?.biases ?? null;
    }, [frameVersion, snapshot?.biases]);

    // Compute mini heatmap data URLs for each non-input neuron
    const neuronHeatmapUrls = useMemo(() => {
        const frameBuffer = getFrameBuffer();
        if (frameBuffer.neuronGrids && frameBuffer.neuronGridLayout) {
            const { count, gridSize } = frameBuffer.neuronGridLayout;
            return Array.from({ length: count }, (_, idx) =>
                neuronGridToDataUrl(extractNeuronGrid(frameBuffer.neuronGrids!, idx, gridSize * gridSize), gridSize),
            );
        }
        if (!snapshot?.neuronGrids) return null;
        const gridSize = snapshot.gridSize ?? GRID_SIZE;
        const snapshotNeuronGrids = snapshot.neuronGrids;
        if (snapshotNeuronGrids instanceof Float32Array) {
            const count = layers.slice(1).reduce((sum, size) => sum + size, 0);
            return Array.from({ length: count }, (_, idx) =>
                neuronGridToDataUrl(extractNeuronGrid(snapshotNeuronGrids, idx, gridSize * gridSize), gridSize),
            );
        }
        return snapshotNeuronGrids.map((grid) => neuronGridToDataUrl(grid, gridSize));
    }, [frameVersion, layers, snapshot?.neuronGrids, snapshot?.gridSize]);

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
            text: [`Weight: ${weight.toFixed(4)}`, `Layer ${layerIdx}, [${prevIdx}→${nodeIdx}]`],
        });
    }, []);

    const handleEdgeLeave = useCallback(() => {
        setHoveredEdge(null);
        setTooltip(null);
    }, []);

    const handleNodeEnter = useCallback((x: number, y: number, text: string[]) => {
        setTooltip({ x, y, text });
    }, []);

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
                    weights={weights}
                    hoveredEdge={hoveredEdge}
                    onEdgeEnter={handleEdgeEnter}
                    onEdgeLeave={handleEdgeLeave}
                />

                {/* Nodes — re-renders on bias/heatmap change; skips on edge hover */}
                <NetworkNodes
                    nodePositions={nodePositions}
                    layers={layers}
                    biases={biases}
                    neuronHeatmapUrls={neuronHeatmapUrls}
                    activeFeatures={activeFeatures}
                    activation={activation}
                    onNodeEnter={handleNodeEnter}
                    onNodeLeave={handleNodeLeave}
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
