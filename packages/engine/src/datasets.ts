// ── Dataset generators ──
// All generators produce points in [-1, 1]² with deterministic seeding.

import type { DataPoint, DataSplit, DatasetType } from './types.js';
import { PRNG } from './prng.js';

const DEFAULT_NUM_SAMPLES = 300;

/** Generate a full dataset and split into train/test. */
export function generateDataset(
    type: DatasetType,
    numSamples: number = DEFAULT_NUM_SAMPLES,
    noise: number = 0,
    trainRatio: number = 0.5,
    seed: number = 42,
): DataSplit {
    const rng = new PRNG(seed);
    const half = Math.floor(numSamples / 2);

    let points: DataPoint[];
    switch (type) {
        case 'circle': points = genCircle(half, noise, rng); break;
        case 'xor': points = genXor(half, noise, rng); break;
        case 'gauss': points = genGauss(half, noise, rng); break;
        case 'spiral': points = genSpiral(half, noise, rng); break;
        case 'moons': points = genMoons(half, noise, rng); break;
        case 'checkerboard': points = genCheckerboard(half, noise, rng); break;
        case 'rings': points = genRings(half, noise, rng); break;
        case 'heart': points = genHeart(half, noise, rng); break;
        case 'reg-plane': points = genRegPlane(numSamples, noise, rng); break;
        case 'reg-gauss': points = genRegGauss(numSamples, noise, rng); break;
        default: points = genCircle(half, noise, rng);
    }

    rng.shuffle(points);
    const splitIdx = Math.floor(points.length * trainRatio);
    return {
        train: points.slice(0, splitIdx),
        test: points.slice(splitIdx),
    };
}

// ── Classification datasets ──

function genCircle(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    const radius = 0.5;
    for (let i = 0; i < pointsPerClass; i++) {
        // Inner circle (label 0)
        const r0 = rng.range(0, radius * 0.5);
        const a0 = rng.range(0, 2 * Math.PI);
        points.push({
            x: r0 * Math.cos(a0) + rng.gaussian(0, noise * 0.01),
            y: r0 * Math.sin(a0) + rng.gaussian(0, noise * 0.01),
            label: 0,
        });
        // Outer ring (label 1)
        const r1 = rng.range(radius * 0.7, radius);
        const a1 = rng.range(0, 2 * Math.PI);
        points.push({
            x: r1 * Math.cos(a1) + rng.gaussian(0, noise * 0.01),
            y: r1 * Math.sin(a1) + rng.gaussian(0, noise * 0.01),
            label: 1,
        });
    }
    return points;
}

function genXor(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < pointsPerClass; i++) {
        // Class 0: top-left and bottom-right
        const x0 = rng.range(-1, 0) + rng.gaussian(0, noise * 0.01);
        const y0 = rng.range(0, 1) + rng.gaussian(0, noise * 0.01);
        points.push({ x: x0, y: y0, label: 0 });

        const x1 = rng.range(0, 1) + rng.gaussian(0, noise * 0.01);
        const y1 = rng.range(-1, 0) + rng.gaussian(0, noise * 0.01);
        points.push({ x: x1, y: y1, label: 0 });
    }
    for (let i = 0; i < pointsPerClass; i++) {
        // Class 1: top-right and bottom-left
        const x0 = rng.range(0, 1) + rng.gaussian(0, noise * 0.01);
        const y0 = rng.range(0, 1) + rng.gaussian(0, noise * 0.01);
        points.push({ x: x0, y: y0, label: 1 });

        const x1 = rng.range(-1, 0) + rng.gaussian(0, noise * 0.01);
        const y1 = rng.range(-1, 0) + rng.gaussian(0, noise * 0.01);
        points.push({ x: x1, y: y1, label: 1 });
    }
    return points;
}

function genGauss(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    const scale = 0.3 + noise * 0.005;
    for (let i = 0; i < pointsPerClass; i++) {
        points.push({
            x: rng.gaussian(-0.3, scale),
            y: rng.gaussian(-0.3, scale),
            label: 0,
        });
        points.push({
            x: rng.gaussian(0.3, scale),
            y: rng.gaussian(0.3, scale),
            label: 1,
        });
    }
    return points;
}

function genSpiral(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < pointsPerClass; i++) {
        for (let cls = 0; cls < 2; cls++) {
            const r = (i / pointsPerClass) * 0.8;
            const t = (cls * Math.PI) + (i / pointsPerClass) * 3 * Math.PI + rng.gaussian(0, noise * 0.04);
            points.push({
                x: r * Math.sin(t),
                y: r * Math.cos(t),
                label: cls,
            });
        }
    }
    return points;
}

function genMoons(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < pointsPerClass; i++) {
        // Upper moon (label 0)
        const angle0 = Math.PI * (i / pointsPerClass);
        points.push({
            x: Math.cos(angle0) * 0.5 + rng.gaussian(0, noise * 0.01),
            y: Math.sin(angle0) * 0.5 + rng.gaussian(0, noise * 0.01),
            label: 0,
        });
        // Lower moon (label 1)
        const angle1 = Math.PI * (i / pointsPerClass);
        points.push({
            x: 0.5 - Math.cos(angle1) * 0.5 + rng.gaussian(0, noise * 0.01),
            y: -Math.sin(angle1) * 0.5 + 0.3 + rng.gaussian(0, noise * 0.01),
            label: 1,
        });
    }
    return points;
}

function genCheckerboard(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    const n = pointsPerClass * 2;
    for (let i = 0; i < n; i++) {
        const x = rng.range(-1, 1) + rng.gaussian(0, noise * 0.005);
        const y = rng.range(-1, 1) + rng.gaussian(0, noise * 0.005);
        // 2×2 checkerboard
        const cx = x >= 0 ? 1 : 0;
        const cy = y >= 0 ? 1 : 0;
        const label = (cx + cy) % 2;
        points.push({ x, y, label });
    }
    return points;
}

function genRings(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    const rings = 3;
    const pointsPerRing = Math.floor((pointsPerClass * 2) / rings);
    for (let ring = 0; ring < rings; ring++) {
        const rMin = (ring * 0.9) / rings;
        const rMax = ((ring + 1) * 0.9) / rings;
        const label = ring % 2;
        for (let i = 0; i < pointsPerRing; i++) {
            const r = rng.range(rMin, rMax);
            const angle = rng.range(0, 2 * Math.PI);
            points.push({
                x: r * Math.cos(angle) + rng.gaussian(0, noise * 0.008),
                y: r * Math.sin(angle) + rng.gaussian(0, noise * 0.008),
                label,
            });
        }
    }
    return points;
}

function genHeart(pointsPerClass: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    const n = pointsPerClass * 2;
    for (let i = 0; i < n; i++) {
        const x = rng.range(-1, 1) + rng.gaussian(0, noise * 0.005);
        const y = rng.range(-1, 1) + rng.gaussian(0, noise * 0.005);
        // Heart curve boundary: (x² + y² - 1)³ - x²y³ < 0
        const x2 = x * x;
        const y2 = y * y;
        const inner = x2 + y2 - 0.6; // scaled down
        const val = inner * inner * inner - x2 * y2 * y;
        const label = val < 0 ? 1 : 0;
        points.push({ x, y, label });
    }
    return points;
}

// ── Regression datasets ──

function genRegPlane(n: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const x = rng.range(-1, 1);
        const y = rng.range(-1, 1);
        points.push({
            x,
            y,
            label: x + y + rng.gaussian(0, noise * 0.02),
        });
    }
    return points;
}

function genRegGauss(n: number, noise: number, rng: PRNG): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const x = rng.range(-1, 1);
        const y = rng.range(-1, 1);
        const v =
            Math.exp(-((x - 0.3) ** 2 + (y - 0.3) ** 2) / 0.2) +
            Math.exp(-((x + 0.3) ** 2 + (y + 0.3) ** 2) / 0.2);
        points.push({
            x,
            y,
            label: v + rng.gaussian(0, noise * 0.02),
        });
    }
    return points;
}

/** Get default problem type for a dataset. */
export function getDefaultProblemType(dataset: DatasetType): 'classification' | 'regression' {
    return dataset.startsWith('reg-') ? 'regression' : 'classification';
}
