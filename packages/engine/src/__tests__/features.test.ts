import { describe, it, expect } from 'vitest';
import {
    ALL_FEATURES,
    getActiveFeatures,
    transformPoint,
    transformDataset,
    countActiveFeatures,
    defaultFeatureFlags,
} from '../features.js';
import type { FeatureFlags } from '../types.js';

describe('ALL_FEATURES', () => {
    it('has 9 features', () => {
        expect(ALL_FEATURES).toHaveLength(9);
    });

    it('each feature has id, label, fn', () => {
        for (const f of ALL_FEATURES) {
            expect(typeof f.id).toBe('string');
            expect(typeof f.label).toBe('string');
            expect(typeof f.fn).toBe('function');
        }
    });
});

describe('feature computations', () => {
    const x = 0.6;
    const y = -0.4;

    const featureMap = Object.fromEntries(ALL_FEATURES.map((f) => [f.id, f]));

    it('x feature returns x', () => {
        expect(featureMap.x.fn(x, y)).toBe(x);
    });

    it('y feature returns y', () => {
        expect(featureMap.y.fn(x, y)).toBe(y);
    });

    it('xSquared returns x²', () => {
        expect(featureMap.xSquared.fn(x, y)).toBeCloseTo(x * x, 8);
    });

    it('ySquared returns y²', () => {
        expect(featureMap.ySquared.fn(x, y)).toBeCloseTo(y * y, 8);
    });

    it('xy returns x*y', () => {
        expect(featureMap.xy.fn(x, y)).toBeCloseTo(x * y, 8);
    });

    it('sinX returns sin(x)', () => {
        expect(featureMap.sinX.fn(x, y)).toBeCloseTo(Math.sin(x), 8);
    });

    it('sinY returns sin(y)', () => {
        expect(featureMap.sinY.fn(x, y)).toBeCloseTo(Math.sin(y), 8);
    });

    it('cosX returns cos(x)', () => {
        expect(featureMap.cosX.fn(x, y)).toBeCloseTo(Math.cos(x), 8);
    });

    it('cosY returns cos(y)', () => {
        expect(featureMap.cosY.fn(x, y)).toBeCloseTo(Math.cos(y), 8);
    });
});

describe('getActiveFeatures', () => {
    it('returns only enabled features', () => {
        const flags: FeatureFlags = {
            x: true, y: false, xSquared: true, ySquared: false,
            xy: false, sinX: false, sinY: false, cosX: false, cosY: false,
        };
        const active = getActiveFeatures(flags);
        expect(active).toHaveLength(2);
        expect(active.map((f) => f.id)).toEqual(['x', 'xSquared']);
    });

    it('returns all features when all enabled', () => {
        const allOn: FeatureFlags = {
            x: true, y: true, xSquared: true, ySquared: true,
            xy: true, sinX: true, sinY: true, cosX: true, cosY: true,
        };
        expect(getActiveFeatures(allOn)).toHaveLength(9);
    });

    it('returns empty array when none enabled', () => {
        const noneOn: FeatureFlags = {
            x: false, y: false, xSquared: false, ySquared: false,
            xy: false, sinX: false, sinY: false, cosX: false, cosY: false,
        };
        expect(getActiveFeatures(noneOn)).toHaveLength(0);
    });
});

describe('transformPoint', () => {
    it('transforms (x, y) into the correct feature vector', () => {
        const flags: FeatureFlags = {
            x: true, y: true, xSquared: false, ySquared: false,
            xy: true, sinX: false, sinY: false, cosX: false, cosY: false,
        };
        const active = getActiveFeatures(flags);
        const result = transformPoint(0.5, -0.3, active);
        expect(result).toHaveLength(3);
        expect(result[0]).toBe(0.5);     // x
        expect(result[1]).toBe(-0.3);    // y
        expect(result[2]).toBeCloseTo(-0.15, 8); // xy
    });
});

describe('transformDataset', () => {
    it('transforms an array of points', () => {
        const active = getActiveFeatures(defaultFeatureFlags());
        const points = [
            { x: 1, y: 0 },
            { x: 0, y: -1 },
        ];
        const transformed = transformDataset(points, active);
        expect(transformed).toHaveLength(2);
        expect(transformed[0]).toEqual([1, 0]);
        expect(transformed[1]).toEqual([0, -1]);
    });
});

describe('countActiveFeatures', () => {
    it('counts correct number', () => {
        expect(countActiveFeatures(defaultFeatureFlags())).toBe(2);
    });

    it('counts all when all enabled', () => {
        const allOn: FeatureFlags = {
            x: true, y: true, xSquared: true, ySquared: true,
            xy: true, sinX: true, sinY: true, cosX: true, cosY: true,
        };
        expect(countActiveFeatures(allOn)).toBe(9);
    });

    it('counts 0 when none enabled', () => {
        const noneOn: FeatureFlags = {
            x: false, y: false, xSquared: false, ySquared: false,
            xy: false, sinX: false, sinY: false, cosX: false, cosY: false,
        };
        expect(countActiveFeatures(noneOn)).toBe(0);
    });
});

describe('defaultFeatureFlags', () => {
    it('has x and y enabled, rest disabled', () => {
        const flags = defaultFeatureFlags();
        expect(flags.x).toBe(true);
        expect(flags.y).toBe(true);
        expect(flags.xSquared).toBe(false);
        expect(flags.ySquared).toBe(false);
        expect(flags.xy).toBe(false);
        expect(flags.sinX).toBe(false);
        expect(flags.sinY).toBe(false);
        expect(flags.cosX).toBe(false);
        expect(flags.cosY).toBe(false);
    });
});
