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
import { getActiveFeatures } from '@nn-playground/engine';
import { GRID_SIZE, writeNormalizedHeatmap } from '@nn-playground/shared';
import {
    extractNeuronGrid,
    getFrameBuffer,
    layerBiasOffset,
} from '../../worker/frameBuffer.ts';
import {
    type EdgeRef,
    type FlatNetworkView,
    type NodePos,
    type NodeRef,
    NODE_RADIUS,
    hitTestEdge,
    hitTestNode,
    paintEdges,
    paintLabels,
    paintNodes,
} from './networkGraphPainter.ts';

// ── Layout constants — match the SVG renderer for visual parity ────────────
const MIN_LAYER_GAP = 120;
const MIN_NODE_GAP = 42;
const PAD_X = 60;
const PAD_Y = 40;
const HEATMAP_SIZE = 24;

interface TooltipData {
    x: number;
    y: number;
    text: string[];
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

export function NetworkGraphCanvas() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const features = usePlaygroundStore((s) => s.features);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);

    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<EdgeRef | null>(null);
    const [hoveredNode, setHoveredNode] = useState<NodeRef | null>(null);

    const activeFeatures = useMemo(() => getActiveFeatures(features), [features]);
    const inputSize = activeFeatures.length;

    const layers = useMemo(() => [inputSize, ...hiddenLayers, 1], [inputSize, hiddenLayers]);
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
                if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
            } else {
                lines.push(`Hidden ${layerIdx}, Neuron ${nodeIdx + 1}`);
                if (bias != null) lines.push(`Bias: ${bias.toFixed(4)}`);
                lines.push(`Activation: ${activation}`);
            }
            return lines;
        },
        [layers, activeFeatures, activation, flat],
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
        const physicalW = Math.round(canvasWidth * dpr);
        const physicalH = Math.round(canvasHeight * dpr);
        if (canvas.width !== physicalW || canvas.height !== physicalH) {
            canvas.width = physicalW;
            canvas.height = physicalH;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);

        paintEdges(ctx, nodePositions, flat, hoveredEdge);
        paintNodes(ctx, nodePositions, flat);
        paintLabels(ctx, nodePositions, layerLabels);
    }, [
        canvasWidth,
        canvasHeight,
        nodePositions,
        flat,
        hoveredEdge,
        layerLabels,
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
            // Map client coordinates back into canvas-logical units (the
            // canvas is rendered to fit its container via CSS `width: 100%`,
            // so the on-screen size and the logical drawing size differ).
            const sx = canvasWidth / rect.width;
            const sy = canvasHeight / rect.height;
            const x = (event.clientX - rect.left) * sx;
            const y = (event.clientY - rect.top) * sy;

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
                    x: event.clientX - rect.left + 12,
                    y: event.clientY - rect.top - 8,
                    text: buildNodeTooltipLines(node.layerIdx, node.nodeIdx),
                });
                return;
            }
            if (hoveredNode) setHoveredNode(null);

            const edge = hitTestEdge(x, y, nodePositions, flat);
            if (edge) {
                if (
                    !hoveredEdge ||
                    hoveredEdge.layerIdx !== edge.layerIdx ||
                    hoveredEdge.nodeIdx !== edge.nodeIdx ||
                    hoveredEdge.prevIdx !== edge.prevIdx
                ) {
                    setHoveredEdge(edge);
                }
                setTooltip({
                    x: event.clientX - rect.left + 12,
                    y: event.clientY - rect.top - 8,
                    text: [
                        `Weight: ${edge.weight.toFixed(4)}`,
                        `Layer ${edge.layerIdx}, [${edge.prevIdx}→${edge.nodeIdx}]`,
                    ],
                });
                return;
            }
            if (hoveredEdge) setHoveredEdge(null);
            if (tooltip) setTooltip(null);
        },
        [
            canvasWidth,
            canvasHeight,
            nodePositions,
            flat,
            hoveredEdge,
            hoveredNode,
            tooltip,
            buildNodeTooltipLines,
        ],
    );

    const handlePointerLeave = useCallback(() => {
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

    const accessibilitySummary = useMemo(() => {
        return `Neural network: ${activeFeatures.length} input${activeFeatures.length === 1 ? '' : 's'}, ` +
            `${hiddenLayers.length} hidden layer${hiddenLayers.length === 1 ? '' : 's'}` +
            (hiddenLayers.length > 0 ? ` of ${hiddenLayers.join(', ')} neuron${hiddenLayers.length === 1 ? '' : 's'}` : '') +
            `, 1 output. Activation: ${activation}.`;
    }, [activeFeatures.length, hiddenLayers, activation]);

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
                    cursor: hoveredNode || hoveredEdge ? 'pointer' : 'default',
                }}
                onPointerMove={handlePointerMove}
                onPointerLeave={handlePointerLeave}
            />

            {/* Heatmap overlays — one per non-input neuron, positioned over
                the corresponding canvas-painted node disc. */}
            {heatmapTiles.map((tile) => {
                const r = NODE_RADIUS - 1.5;
                // Convert canvas-logical coordinates back into CSS pixels:
                // the canvas itself is rendered with `width: 100%`, so we
                // scale node positions by container/canvas ratio. The
                // ResizeObserver keeps containerSize in sync with the
                // visible width, so canvasWidth / containerSize.width is
                // typically 1; the math is here for completeness.
                const left = (tile.x / canvasWidth) * containerSize.width - r;
                const top = (tile.y / canvasHeight) * containerSize.height - r;
                return (
                    <div
                        key={tile.key}
                        className="network-graph-heatmap-slot"
                        style={{
                            position: 'absolute',
                            left,
                            top,
                            width: 2 * r,
                            height: 2 * r,
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
