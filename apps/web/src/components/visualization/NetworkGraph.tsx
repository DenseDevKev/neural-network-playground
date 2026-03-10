// ── SVG Network Graph ──
// Renders the neural network topology as an interactive SVG with hover tooltips
// and mini heatmaps inside each neuron.

import { useMemo, useState, useCallback } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { getActiveFeatures } from '@nn-playground/engine';
import { GRID_SIZE } from '@nn-playground/shared';

const NODE_RADIUS = 14;
const LAYER_GAP = 120;
const NODE_GAP = 42;
const HEATMAP_SIZE = 24; // pixels for mini heatmap canvas

interface TooltipData {
    x: number;
    y: number;
    text: string[];
}

function nodeColor(value: number): string {
    const abs = Math.min(Math.abs(value), 2) / 2;
    if (value > 0) return `rgba(249, 115, 22, ${0.3 + abs * 0.7})`;
    return `rgba(59, 130, 246, ${0.3 + abs * 0.7})`;
}

function edgeColor(weight: number): string {
    const abs = Math.min(Math.abs(weight), 3) / 3;
    if (weight > 0) return `rgba(249, 115, 22, ${0.15 + abs * 0.6})`;
    return `rgba(59, 130, 246, ${0.15 + abs * 0.6})`;
}

function edgeWidth(weight: number): number {
    return Math.max(0.5, Math.min(3, Math.abs(weight) * 1.5));
}

/**
 * Generate a data URL from a flat grid of neuron activations.
 * Maps activations to blue–dark–orange color scale.
 */
function neuronGridToDataUrl(grid: number[], gridSize: number): string {
    const canvas = document.createElement('canvas');
    canvas.width = gridSize;
    canvas.height = gridSize;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(gridSize, gridSize);

    // Find min/max for normalisation
    let min = Infinity, max = -Infinity;
    for (const v of grid) {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const range = max - min || 1;

    for (let i = 0; i < grid.length; i++) {
        const t = (grid[i] - min) / range; // 0–1
        const idx = i * 4;
        if (t < 0.5) {
            const p = t / 0.5;
            imageData.data[idx] = Math.round(59 * (1 - p) + 28 * p);
            imageData.data[idx + 1] = Math.round(130 * (1 - p) + 32 * p);
            imageData.data[idx + 2] = Math.round(246 * (1 - p) + 48 * p);
        } else {
            const p = (t - 0.5) / 0.5;
            imageData.data[idx] = Math.round(28 * (1 - p) + 249 * p);
            imageData.data[idx + 1] = Math.round(32 * (1 - p) + 115 * p);
            imageData.data[idx + 2] = Math.round(48 * (1 - p) + 22 * p);
        }
        imageData.data[idx + 3] = 220;
    }

    ctx.putImageData(imageData, 0, 0);

    // Scale up to HEATMAP_SIZE with smoothing
    const scaled = document.createElement('canvas');
    scaled.width = HEATMAP_SIZE;
    scaled.height = HEATMAP_SIZE;
    const sctx = scaled.getContext('2d')!;
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(canvas, 0, 0, HEATMAP_SIZE, HEATMAP_SIZE);

    return scaled.toDataURL();
}

export function NetworkGraph() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const features = usePlaygroundStore((s) => s.features);
    const snapshot = usePlaygroundStore((s) => s.snapshot);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

    const activeFeatures = useMemo(() => getActiveFeatures(features), [features]);
    const inputSize = activeFeatures.length;

    const layers = useMemo(() => {
        return [inputSize, ...hiddenLayers, 1];
    }, [inputSize, hiddenLayers]);

    const maxNodes = Math.max(...layers);
    const svgWidth = layers.length * LAYER_GAP + 60;
    const svgHeight = Math.max(200, maxNodes * NODE_GAP + 60);

    const nodePositions = useMemo(() => {
        return layers.map((count, layerIdx) => {
            const x = 50 + layerIdx * LAYER_GAP;
            const totalHeight = (count - 1) * NODE_GAP;
            const startY = svgHeight / 2 - totalHeight / 2;
            return Array.from({ length: count }, (_, nodeIdx) => ({
                x,
                y: startY + nodeIdx * NODE_GAP,
            }));
        });
    }, [layers, svgHeight]);

    const layerLabels = useMemo(() => {
        return layers.map((_, idx) => {
            if (idx === 0) return 'Input';
            if (idx === layers.length - 1) return 'Output';
            return `Hidden ${idx}`;
        });
    }, [layers]);

    // Compute mini heatmap data URLs for each non-input neuron
    const neuronHeatmapUrls = useMemo(() => {
        if (!snapshot?.neuronGrids) return null;
        const gridSize = snapshot.gridSize ?? GRID_SIZE;
        return snapshot.neuronGrids.map((grid) =>
            neuronGridToDataUrl(grid, gridSize),
        );
    }, [snapshot?.neuronGrids, snapshot?.gridSize]);

    // Build a mapping: getNeuronGridIndex(layerIdx, nodeIdx) -> index into neuronHeatmapUrls
    const getNeuronGridIndex = useCallback(
        (layerIdx: number, nodeIdx: number): number | null => {
            if (!neuronHeatmapUrls || layerIdx === 0) return null;
            // neuronGrids are laid out as: for each layer l (0-indexed into weights),
            // for each neuron n in that layer
            let idx = 0;
            for (let l = 1; l < layerIdx; l++) {
                idx += layers[l]; // hidden layer neurons
            }
            if (layerIdx === layers.length - 1) {
                // Output layer — skip all hidden layers
                idx = 0;
                for (let l = 1; l < layers.length - 1; l++) {
                    idx += layers[l];
                }
            }
            idx += nodeIdx;
            return idx < neuronHeatmapUrls.length ? idx : null;
        },
        [neuronHeatmapUrls, layers],
    );

    const handleNodeHover = useCallback(
        (layerIdx: number, nodeIdx: number, screenX: number, screenY: number) => {
            const isInput = layerIdx === 0;
            const isOutput = layerIdx === layers.length - 1;
            const lines: string[] = [];

            if (isInput) {
                const feat = activeFeatures[nodeIdx];
                lines.push(`Input: ${feat?.label ?? `x${nodeIdx}`}`);
            } else if (isOutput) {
                lines.push('Output neuron');
                const bias = snapshot?.biases?.[layerIdx - 1]?.[nodeIdx];
                if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
            } else {
                lines.push(`Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`);
                const bias = snapshot?.biases?.[layerIdx - 1]?.[nodeIdx];
                if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
                lines.push(`Activation: ${activation}`);
            }

            setTooltip({ x: screenX, y: screenY, text: lines });
        },
        [layers, activeFeatures, snapshot, activation],
    );

    const handleEdgeHover = useCallback(
        (
            layerIdx: number,
            nodeIdx: number,
            prevIdx: number,
            weight: number,
            screenX: number,
            screenY: number,
        ) => {
            const key = `e-${layerIdx}-${nodeIdx}-${prevIdx}`;
            setHoveredEdge(key);
            setTooltip({
                x: screenX,
                y: screenY,
                text: [`Weight: ${weight.toFixed(4)}`, `Layer ${layerIdx}, [${prevIdx}→${nodeIdx}]`],
            });
        },
        [],
    );

    return (
        <div className="network-graph-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
            <svg
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                preserveAspectRatio="xMidYMid meet"
                style={{ width: '100%', height: '100%' }}
                onMouseLeave={() => { setTooltip(null); setHoveredEdge(null); }}
            >
                {/* Defs for glow filter and clip paths */}
                <defs>
                    <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                    {/* Clip path for mini heatmaps */}
                    <clipPath id="neuron-clip">
                        <circle cx={HEATMAP_SIZE / 2} cy={HEATMAP_SIZE / 2} r={NODE_RADIUS - 1} />
                    </clipPath>
                </defs>

                {/* Edges */}
                {nodePositions.map((layerNodes, layerIdx) => {
                    if (layerIdx === 0) return null;
                    const prevNodes = nodePositions[layerIdx - 1];
                    return layerNodes.map((node, nodeIdx) =>
                        prevNodes.map((prevNode, prevIdx) => {
                            const weight =
                                snapshot?.weights?.[layerIdx - 1]?.[nodeIdx]?.[prevIdx] ?? 0;
                            const key = `e-${layerIdx}-${nodeIdx}-${prevIdx}`;
                            const isHovered = hoveredEdge === key;
                            return (
                                <g key={key}>
                                    <line
                                        x1={prevNode.x}
                                        y1={prevNode.y}
                                        x2={node.x}
                                        y2={node.y}
                                        stroke="transparent"
                                        strokeWidth={10}
                                        style={{ cursor: 'pointer' }}
                                        onMouseEnter={(e) => {
                                            const rect = (e.target as SVGElement).closest('svg')!.getBoundingClientRect();
                                            handleEdgeHover(layerIdx, nodeIdx, prevIdx, weight, e.clientX - rect.left, e.clientY - rect.top);
                                        }}
                                        onMouseLeave={() => { setHoveredEdge(null); setTooltip(null); }}
                                    />
                                    <line
                                        x1={prevNode.x}
                                        y1={prevNode.y}
                                        x2={node.x}
                                        y2={node.y}
                                        stroke={edgeColor(weight)}
                                        strokeWidth={isHovered ? edgeWidth(weight) * 2 : edgeWidth(weight)}
                                        opacity={isHovered ? 1 : 0.7}
                                        style={{ transition: 'stroke-width 100ms, opacity 100ms', pointerEvents: 'none' }}
                                    />
                                </g>
                            );
                        }),
                    );
                })}

                {/* Nodes */}
                {nodePositions.map((layerNodes, layerIdx) =>
                    layerNodes.map((node, nodeIdx) => {
                        const bias = snapshot?.biases?.[layerIdx - 1]?.[nodeIdx] ?? 0;
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
                                    handleNodeHover(layerIdx, nodeIdx, e.clientX - rect.left, e.clientY - rect.top);
                                }}
                                onMouseLeave={() => setTooltip(null)}
                            >
                                {/* Glow */}
                                <circle
                                    cx={node.x}
                                    cy={node.y}
                                    r={NODE_RADIUS + 4}
                                    fill="none"
                                    stroke={
                                        isOutput
                                            ? 'rgba(124, 92, 252, 0.3)'
                                            : isInput
                                                ? 'rgba(0, 229, 195, 0.2)'
                                                : nodeColor(bias)
                                    }
                                    strokeWidth={2}
                                    opacity={0.5}
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
                                {/* Node border ring (redraw on top of heatmap) */}
                                {!isInput && heatmapUrl && (
                                    <circle
                                        cx={node.x}
                                        cy={node.y}
                                        r={NODE_RADIUS}
                                        fill="none"
                                        stroke={
                                            isOutput
                                                ? '#7c5cfc'
                                                : 'rgba(255,255,255,0.15)'
                                        }
                                        strokeWidth={1.5}
                                        style={{ pointerEvents: 'none' }}
                                    />
                                )}
                            </g>
                        );
                    }),
                )}

                {/* Layer labels */}
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
