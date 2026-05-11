import { describe, it, expect } from 'vitest';
import { computeLearningRate, sanitizeLRSchedule, validateLRSchedule } from '../schedules.js';

describe('computeLearningRate', () => {
    it('returns base lr when schedule is omitted', () => {
        expect(computeLearningRate(0.1, 0)).toBe(0.1);
        expect(computeLearningRate(0.1, 1000)).toBe(0.1);
    });

    it('constant schedule never decays', () => {
        const s = { type: 'constant' as const };
        expect(computeLearningRate(0.1, 0, s)).toBe(0.1);
        expect(computeLearningRate(0.1, 999, s)).toBe(0.1);
    });

    it('step schedule decays by gamma every stepSize', () => {
        const s = { type: 'step' as const, stepSize: 10, gamma: 0.5 };
        expect(computeLearningRate(1.0, 0, s)).toBe(1.0);     // no decays yet
        expect(computeLearningRate(1.0, 9, s)).toBe(1.0);
        expect(computeLearningRate(1.0, 10, s)).toBe(0.5);    // 1 decay
        expect(computeLearningRate(1.0, 25, s)).toBe(0.25);   // 2 decays
        expect(computeLearningRate(1.0, 30, s)).toBe(0.125);  // 3 decays
    });

    it('step schedule with stepSize <= 0 is a no-op', () => {
        const s = { type: 'step' as const, stepSize: 0, gamma: 0.5 };
        expect(computeLearningRate(0.1, 50, s)).toBe(0.1);
    });

    it('cosine schedule starts at base and ends at minLr', () => {
        const s = { type: 'cosine' as const, totalSteps: 100, minLr: 0.01 };
        expect(computeLearningRate(0.1, 0, s)).toBeCloseTo(0.1, 8);
        expect(computeLearningRate(0.1, 100, s)).toBeCloseTo(0.01, 8);
        // Past totalSteps, stays at minLr.
        expect(computeLearningRate(0.1, 200, s)).toBeCloseTo(0.01, 8);
    });

    it('cosine schedule is monotonically decreasing between start and end', () => {
        const s = { type: 'cosine' as const, totalSteps: 100, minLr: 0 };
        let prev = computeLearningRate(1.0, 0, s);
        for (let step = 10; step <= 100; step += 10) {
            const cur = computeLearningRate(1.0, step, s);
            expect(cur).toBeLessThanOrEqual(prev);
            prev = cur;
        }
    });
});

describe('learning-rate schedule sanitization', () => {
    it('normalizes invalid schedule values into safe runtime values', () => {
        expect(sanitizeLRSchedule({ type: 'constant' })).toBeUndefined();
        expect(sanitizeLRSchedule({ type: 'step', stepSize: 0, gamma: 2 } as any)).toEqual({
            type: 'step',
            stepSize: 1,
            gamma: 0.5,
        });
        expect(sanitizeLRSchedule(
            { type: 'cosine', totalSteps: 0, minLr: 0.2 } as any,
            { baseLearningRate: 0.03 },
        )).toEqual({
            type: 'cosine',
            totalSteps: 1,
            minLr: 0.03,
        });
    });

    it('rejects unsafe schedule values before training updates', () => {
        expect(() => validateLRSchedule({ type: 'step', stepSize: 0, gamma: 0.5 } as any)).toThrow(RangeError);
        expect(() => validateLRSchedule({ type: 'step', stepSize: 10, gamma: 1 } as any)).toThrow(RangeError);
        expect(() => validateLRSchedule({ type: 'cosine', totalSteps: 0, minLr: 0 } as any)).toThrow(RangeError);
        expect(() => validateLRSchedule({ type: 'cosine', totalSteps: 10, minLr: -1 } as any)).toThrow(RangeError);
    });
});
