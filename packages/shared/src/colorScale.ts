// ── Color Scale ──
// Shared blue → dark → orange color scale used by all visualizations.
// Centralizes the color math that was previously duplicated in
// DecisionBoundary.tsx and NetworkGraph.tsx.

/** RGB tuple [r, g, b], each 0–255. */
export type RGB = [number, number, number];

// ── Palette anchors ──
// These match the CSS theme colors for class 0 (blue) and class 1 (orange).
export const COLOR_BLUE: RGB = [59, 130, 246];   // #3b82f6
export const COLOR_ORANGE: RGB = [249, 115, 22]; // #f97316
export const COLOR_DARK: RGB = [60, 64, 84];     // midpoint for heatmaps (lighter than BG so 0.5 is visible)

/** Hex strings for use in CSS / SVG. */
export const HEX_BLUE = '#3b82f6';
export const HEX_ORANGE = '#f97316';

/**
 * Map a value in [0, 1] to the blue → dark → orange color scale.
 *
 * When `discretize` is true, values are snapped to 0 (blue) or 1 (orange).
 *
 * @param v         Value in [0, 1] (clamped internally)
 * @param discretize If true, snap to nearest class color
 * @returns RGB tuple [r, g, b]
 */
export function valueToColor(v: number, discretize = false): RGB {
    const clamped = Math.max(0, Math.min(1, v));
    const t = discretize ? (clamped >= 0.5 ? 1 : 0) : clamped;

    if (t < 0.5) {
        const p = t / 0.5;
        return [
            Math.round(COLOR_BLUE[0] * (1 - p) + COLOR_DARK[0] * p),
            Math.round(COLOR_BLUE[1] * (1 - p) + COLOR_DARK[1] * p),
            Math.round(COLOR_BLUE[2] * (1 - p) + COLOR_DARK[2] * p),
        ];
    }
    const p = (t - 0.5) / 0.5;
    return [
        Math.round(COLOR_DARK[0] * (1 - p) + COLOR_ORANGE[0] * p),
        Math.round(COLOR_DARK[1] * (1 - p) + COLOR_ORANGE[1] * p),
        Math.round(COLOR_DARK[2] * (1 - p) + COLOR_ORANGE[2] * p),
    ];
}

/**
 * Write the color scale into an ImageData pixel buffer for a grid of values.
 *
 * @param values    Flat array of predictions (one per grid cell)
 * @param imageData Target ImageData (must have values.length pixels)
 * @param alpha     Alpha channel value (0–255)
 * @param discretize If true, snap to class colors
 */
export function writeGridToImageData(
    values: ArrayLike<number>,
    imageData: ImageData,
    alpha = 200,
    discretize = false,
): void {
    for (let i = 0; i < values.length; i++) {
        const [r, g, b] = valueToColor(values[i], discretize);
        const idx = i * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = alpha;
    }
}

/**
 * Write a normalized heatmap into an ImageData buffer.
 * Normalizes values to [0, 1] based on their min/max, then applies the color scale.
 *
 * @param values    Raw activation values (any range)
 * @param imageData Target ImageData
 * @param alpha     Alpha channel value
 */
export function writeNormalizedHeatmap(
    values: ArrayLike<number>,
    imageData: ImageData,
    alpha = 220,
): void {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < values.length; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }
    const range = max - min || 1;

    for (let i = 0; i < values.length; i++) {
        const t = (values[i] - min) / range;
        const [r, g, b] = valueToColor(t);
        const idx = i * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = alpha;
    }
}
