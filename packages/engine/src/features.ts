// ── Feature transforms ──
// Converts raw (x, y) inputs into the selected feature vector.

import type { FeatureFlags } from './types.js';

export type FeatureId = keyof FeatureFlags;

export interface FeatureSpec {
    id: FeatureId;
    label: string;
    fn: (x: number, y: number) => number;
}

export const ALL_FEATURES: FeatureSpec[] = [
    { id: 'x', label: 'X₁', fn: (x) => x },
    { id: 'y', label: 'X₂', fn: (_x, y) => y },
    { id: 'xSquared', label: 'X₁²', fn: (x) => x * x },
    { id: 'ySquared', label: 'X₂²', fn: (_x, y) => y * y },
    { id: 'xy', label: 'X₁X₂', fn: (x, y) => x * y },
    { id: 'sinX', label: 'sin(X₁)', fn: (x) => Math.sin(x) },
    { id: 'sinY', label: 'sin(X₂)', fn: (_x, y) => Math.sin(y) },
    { id: 'cosX', label: 'cos(X₁)', fn: (x) => Math.cos(x) },
    { id: 'cosY', label: 'cos(X₂)', fn: (_x, y) => Math.cos(y) },
];

/** Get the list of active feature specs given flags. */
export function getActiveFeatures(flags: FeatureFlags): FeatureSpec[] {
    return ALL_FEATURES.filter((f) => flags[f.id]);
}

/** Transform a raw (x, y) point into the active feature vector. */
export function transformPoint(
    x: number,
    y: number,
    activeFeatures: FeatureSpec[],
): number[] {
    return activeFeatures.map((f) => f.fn(x, y));
}

/** Transform an array of raw points into feature matrix. */
export function transformDataset(
    points: { x: number; y: number }[],
    activeFeatures: FeatureSpec[],
): number[][] {
    return points.map((p) => transformPoint(p.x, p.y, activeFeatures));
}

/** Count active features. */
export function countActiveFeatures(flags: FeatureFlags): number {
    return ALL_FEATURES.filter((f) => flags[f.id]).length;
}

/** Default features: just x and y enabled. */
export function defaultFeatureFlags(): FeatureFlags {
    return {
        x: true,
        y: true,
        xSquared: false,
        ySquared: false,
        xy: false,
        sinX: false,
        sinY: false,
        cosX: false,
        cosY: false,
    };
}
