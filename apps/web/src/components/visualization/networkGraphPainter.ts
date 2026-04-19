// ── Network graph painters + hit-tests (AS-5) ───────────────────────────────
// Pure functions used by NetworkGraphCanvas. Kept out of the React component
// so they can be unit-tested independently and so the component file stays
// focused on layout and event wiring.
//
// Edge / node colours mirror the SVG renderer so toggling between the two
// implementations doesn't change perceived appearance.

import { layerWeightOffset, layerBiasOffset } from '../../worker/frameBuffer.ts';

export interface NodePos {
    x: number;
    y: number;
}

export interface FlatNetworkView {
    weights: Float32Array;
    biases: Float32Array;
    layerSizes: number[];
}

export interface EdgeRef {
    layerIdx: number; // destination layer (1..N)
    nodeIdx: number; // destination neuron index within layer
    prevIdx: number; // source neuron index within layer-1
    weight: number;
}

export interface NodeRef {
    layerIdx: number;
    nodeIdx: number;
}

// ── Visual constants (must match the SVG renderer for parity) ───────────────

export const NODE_RADIUS = 14;
export const EDGE_HIT_THRESHOLD = 6; // pixels, perpendicular distance for hover
export const NODE_HIT_PADDING = 4; // grow hit area a bit beyond the visible disc
const LABEL_FONT = '600 10px Inter, sans-serif';
const LABEL_FILL = 'rgba(255,255,255,0.35)';
const LABEL_Y = 20;

// ── Colour helpers — identical formulas to the SVG component ───────────────

function nodeColor(value: number): string {
    const abs = Math.min(Math.abs(value), 2) / 2;
    if (value > 0) return `rgba(129, 236, 255, ${0.4 + abs * 0.6})`;
    return `rgba(188, 135, 254, ${0.4 + abs * 0.6})`;
}

function edgeColor(weight: number, isHovered: boolean): string {
    const abs = Math.min(Math.abs(weight), 3) / 3;
    const baseAlpha = 0.2 + abs * 0.6;
    const alpha = isHovered ? 1 : baseAlpha * 0.7 + 0.3 * baseAlpha; // emphasise on hover
    if (weight > 0) return `rgba(129, 236, 255, ${alpha.toFixed(3)})`;
    return `rgba(188, 135, 254, ${alpha.toFixed(3)})`;
}

function edgeWidth(weight: number, isHovered: boolean): number {
    const w = Math.max(0.5, Math.min(3, Math.abs(weight) * 1.5));
    return isHovered ? w * 2 : w;
}

// ── Bezier helpers ─────────────────────────────────────────────────────────

/** Cubic bezier control-x offset (matches the SVG `C` path command). */
function cpX(prev: NodePos, node: NodePos): number {
    return (node.x - prev.x) * 0.45;
}

/** Sample N points along the cubic bezier between two nodes. Used by the
 *  edge hit-test below — perpendicular-distance against a line segment is
 *  too crude for curved edges, but a 12-point polyline approximation is
 *  fast and indistinguishable from the curve at the threshold we use. */
function sampleBezier(
    prev: NodePos,
    node: NodePos,
    samples: number,
    out: Float32Array,
): void {
    const dx = cpX(prev, node);
    const x0 = prev.x, y0 = prev.y;
    const x1 = prev.x + dx, y1 = prev.y;
    const x2 = node.x - dx, y2 = node.y;
    const x3 = node.x, y3 = node.y;
    for (let i = 0; i < samples; i++) {
        const t = i / (samples - 1);
        const u = 1 - t;
        const b0 = u * u * u;
        const b1 = 3 * u * u * t;
        const b2 = 3 * u * t * t;
        const b3 = t * t * t;
        out[i * 2] = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3;
        out[i * 2 + 1] = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3;
    }
}

// ── Painters ────────────────────────────────────────────────────────────────

/**
 * Stroke every edge with a single Path2D per (sign × magnitude) bucket so
 * we issue ~6 strokeStyle changes per frame instead of one per edge.
 */
export function paintEdges(
    ctx: CanvasRenderingContext2D,
    nodePositions: NodePos[][],
    flat: FlatNetworkView | null,
    hovered: EdgeRef | null,
): void {
    if (!flat) return;

    // Bucket edges by sign × magnitude band. 3 bands per sign = 6 total.
    // Each bucket gets a single beginPath / strokeStyle / lineWidth /
    // stroke triplet — the cheapest possible Canvas2D edge render.
    const bandsPos: Path2D[] = [new Path2D(), new Path2D(), new Path2D()];
    const bandsNeg: Path2D[] = [new Path2D(), new Path2D(), new Path2D()];
    // We want a representative weight per band so the colour mid-tone
    // matches the underlying weight magnitude. Track running max-magnitude
    // per band and recolor at stroke time.
    const bandMagsPos = [0, 0, 0];
    const bandMagsNeg = [0, 0, 0];

    const addEdge = (prev: NodePos, node: NodePos, weight: number) => {
        const dx = cpX(prev, node);
        const path = new Path2D();
        path.moveTo(prev.x, prev.y);
        path.bezierCurveTo(
            prev.x + dx, prev.y,
            node.x - dx, node.y,
            node.x, node.y,
        );
        const abs = Math.abs(weight);
        const band = abs < 0.5 ? 0 : abs < 1.5 ? 1 : 2;
        const arr = weight >= 0 ? bandsPos : bandsNeg;
        const mags = weight >= 0 ? bandMagsPos : bandMagsNeg;
        arr[band].addPath(path);
        if (abs > mags[band]) mags[band] = abs;
    };

    for (let layerIdx = 1; layerIdx < nodePositions.length; layerIdx++) {
        const prevNodes = nodePositions[layerIdx - 1];
        const layerNodes = nodePositions[layerIdx];
        const base = layerWeightOffset(flat.layerSizes, layerIdx - 1);
        const fanIn = flat.layerSizes[layerIdx - 1];
        for (let nodeIdx = 0; nodeIdx < layerNodes.length; nodeIdx++) {
            for (let prevIdx = 0; prevIdx < prevNodes.length; prevIdx++) {
                const w = flat.weights[base + nodeIdx * fanIn + prevIdx];
                addEdge(prevNodes[prevIdx], layerNodes[nodeIdx], w);
            }
        }
    }

    // Stroke each non-empty band with the colour matching its weight band.
    // Band representative weights: 0.25, 1.0, 2.5 — chosen mid-band.
    const REP = [0.25, 1.0, 2.5];
    for (let i = 0; i < 3; i++) {
        if (bandMagsPos[i] > 0) {
            ctx.strokeStyle = edgeColor(REP[i], false);
            ctx.lineWidth = edgeWidth(REP[i], false);
            ctx.stroke(bandsPos[i]);
        }
        if (bandMagsNeg[i] > 0) {
            ctx.strokeStyle = edgeColor(-REP[i], false);
            ctx.lineWidth = edgeWidth(-REP[i], false);
            ctx.stroke(bandsNeg[i]);
        }
    }

    // Hovered edge gets its own stroke on top with the exact weight, so the
    // highlight is visually correct (not just a band approximation).
    if (hovered) {
        const prevLayer = nodePositions[hovered.layerIdx - 1];
        const layer = nodePositions[hovered.layerIdx];
        const prev = prevLayer?.[hovered.prevIdx];
        const node = layer?.[hovered.nodeIdx];
        if (prev && node) {
            const path = new Path2D();
            const dx = cpX(prev, node);
            path.moveTo(prev.x, prev.y);
            path.bezierCurveTo(
                prev.x + dx, prev.y,
                node.x - dx, node.y,
                node.x, node.y,
            );
            ctx.strokeStyle = edgeColor(hovered.weight, true);
            ctx.lineWidth = edgeWidth(hovered.weight, true);
            ctx.stroke(path);
        }
    }
}

/**
 * Paint the discrete node rings (background fill + coloured stroke). We
 * leave the inside of each non-input node empty — the React layer drops a
 * mini-heatmap canvas on top via absolute positioning, which is far simpler
 * than image-data-blitting inside the main canvas paint.
 */
export function paintNodes(
    ctx: CanvasRenderingContext2D,
    nodePositions: NodePos[][],
    flat: FlatNetworkView | null,
): void {
    const layerCount = nodePositions.length;

    // Background fill — every node, single style.
    ctx.fillStyle = '#1c2030';
    const fillPath = new Path2D();
    for (let l = 0; l < layerCount; l++) {
        for (const n of nodePositions[l]) {
            fillPath.moveTo(n.x + NODE_RADIUS, n.y);
            fillPath.arc(n.x, n.y, NODE_RADIUS, 0, Math.PI * 2);
        }
    }
    ctx.fill(fillPath);

    // Strokes — bucket by category. Input = teal, output = purple, hidden
    // = bias-tinted. We split hidden into pos/neg buckets to keep at most
    // five distinct strokeStyle values per frame regardless of network size.
    const inputStroke = new Path2D();
    const outputStroke = new Path2D();
    const hiddenPos = new Path2D();
    const hiddenNeg = new Path2D();

    for (let l = 0; l < layerCount; l++) {
        const isInput = l === 0;
        const isOutput = l === layerCount - 1;
        const target = isInput ? inputStroke : isOutput ? outputStroke : null;
        const layer = nodePositions[l];

        for (let i = 0; i < layer.length; i++) {
            const n = layer[i];
            const subPath = new Path2D();
            subPath.moveTo(n.x + NODE_RADIUS, n.y);
            subPath.arc(n.x, n.y, NODE_RADIUS, 0, Math.PI * 2);

            if (target) {
                target.addPath(subPath);
            } else {
                // Hidden layer — bias-tinted; bucket by sign.
                const bias = flat
                    ? flat.biases[layerBiasOffset(flat.layerSizes, l - 1) + i]
                    : 0;
                (bias >= 0 ? hiddenPos : hiddenNeg).addPath(subPath);
            }
        }
    }

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#00e5c3'; ctx.stroke(inputStroke);
    ctx.strokeStyle = '#7c5cfc'; ctx.stroke(outputStroke);
    ctx.strokeStyle = nodeColor(1); ctx.stroke(hiddenPos);
    ctx.strokeStyle = nodeColor(-1); ctx.stroke(hiddenNeg);
}

/**
 * Layer header text (Input / Hidden N / Output). Plain fillText calls — one
 * per layer is cheap and there's no batching worth doing for ~5 strings.
 */
export function paintLabels(
    ctx: CanvasRenderingContext2D,
    nodePositions: NodePos[][],
    layerLabels: string[],
): void {
    ctx.font = LABEL_FONT;
    ctx.fillStyle = LABEL_FILL;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    for (let l = 0; l < nodePositions.length; l++) {
        const x = nodePositions[l][0]?.x ?? 0;
        ctx.fillText(layerLabels[l] ?? '', x, LABEL_Y);
    }
}

// ── Hit-tests ───────────────────────────────────────────────────────────────

/**
 * Find the node whose disc the cursor is currently inside.
 * Returns null when no node is under the cursor.
 */
export function hitTestNode(
    x: number,
    y: number,
    nodePositions: NodePos[][],
): NodeRef | null {
    const r = NODE_RADIUS + NODE_HIT_PADDING;
    const r2 = r * r;
    for (let l = 0; l < nodePositions.length; l++) {
        const layer = nodePositions[l];
        for (let i = 0; i < layer.length; i++) {
            const n = layer[i];
            const dx = n.x - x;
            const dy = n.y - y;
            if (dx * dx + dy * dy <= r2) {
                return { layerIdx: l, nodeIdx: i };
            }
        }
    }
    return null;
}

/**
 * Find the edge closest to the cursor within EDGE_HIT_THRESHOLD pixels.
 * Edges are sampled along the bezier; a per-segment perpendicular distance
 * gives a tight enough match for hover at the threshold we use.
 */
export function hitTestEdge(
    x: number,
    y: number,
    nodePositions: NodePos[][],
    flat: FlatNetworkView | null,
): EdgeRef | null {
    if (!flat) return null;

    const SAMPLES = 12;
    const samples = new Float32Array(SAMPLES * 2);
    const threshold2 = EDGE_HIT_THRESHOLD * EDGE_HIT_THRESHOLD;
    let best: { ref: EdgeRef; dist2: number } | null = null;

    for (let layerIdx = 1; layerIdx < nodePositions.length; layerIdx++) {
        const prevNodes = nodePositions[layerIdx - 1];
        const layerNodes = nodePositions[layerIdx];
        const base = layerWeightOffset(flat.layerSizes, layerIdx - 1);
        const fanIn = flat.layerSizes[layerIdx - 1];

        for (let nodeIdx = 0; nodeIdx < layerNodes.length; nodeIdx++) {
            const node = layerNodes[nodeIdx];
            for (let prevIdx = 0; prevIdx < prevNodes.length; prevIdx++) {
                const prev = prevNodes[prevIdx];
                sampleBezier(prev, node, SAMPLES, samples);
                // Per-segment distance against (x,y).
                for (let s = 0; s < SAMPLES - 1; s++) {
                    const ax = samples[s * 2];
                    const ay = samples[s * 2 + 1];
                    const bx = samples[(s + 1) * 2];
                    const by = samples[(s + 1) * 2 + 1];
                    const d2 = pointSegmentDist2(x, y, ax, ay, bx, by);
                    if (d2 < threshold2 && (!best || d2 < best.dist2)) {
                        const weight = flat.weights[base + nodeIdx * fanIn + prevIdx];
                        best = {
                            ref: { layerIdx, nodeIdx, prevIdx, weight },
                            dist2: d2,
                        };
                    }
                }
            }
        }
    }
    return best ? best.ref : null;
}

function pointSegmentDist2(
    px: number, py: number,
    ax: number, ay: number,
    bx: number, by: number,
): number {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) {
        const ex = px - ax;
        const ey = py - ay;
        return ex * ex + ey * ey;
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    const cx = ax + t * dx;
    const cy = ay + t * dy;
    const ex = px - cx;
    const ey = py - cy;
    return ex * ex + ey * ey;
}
