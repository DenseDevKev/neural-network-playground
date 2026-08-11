import { describe, expect, it, vi } from 'vitest';
import {
    assertPerformanceBudgets,
    measureMedianMsPerIteration,
    median,
} from './performanceStatistics.js';

describe('median', () => {
    it('returns the middle value after numeric sorting', () => {
        expect(median([9, 1, 5, 3, 7])).toBe(5);
    });

    it('averages the two middle values without mutating the input', () => {
        const values = [10, 2, 8, 4];

        expect(median(values)).toBe(6);
        expect(values).toEqual([10, 2, 8, 4]);
    });

    it('rejects empty and non-finite samples', () => {
        expect(() => median([])).toThrow(/at least one value/);
        expect(() => median([1, Number.NaN])).toThrow(/finite/);
        expect(() => median([1, Number.POSITIVE_INFINITY])).toThrow(/finite/);
    });
});

describe('measureMedianMsPerIteration', () => {
    it('warms the operation, normalizes measured rounds, and returns their median', () => {
        const run = vi.fn();
        const timestamps = [0, 20, 20, 60, 60, 120, 120, 200, 200, 300];

        const measured = measureMedianMsPerIteration(run, {
            warmupIterations: 2,
            measuredRounds: 5,
            iterationsPerRound: 2,
            now: () => {
                const timestamp = timestamps.shift();
                if (timestamp === undefined) throw new Error('unexpected clock read');
                return timestamp;
            },
        });

        expect(measured).toBe(30);
        expect(run).toHaveBeenCalledTimes(12);
        expect(timestamps).toEqual([]);
    });

    it('rejects invalid sampling options before running the operation', () => {
        const run = vi.fn();

        expect(() => measureMedianMsPerIteration(run, {
            warmupIterations: -1,
            measuredRounds: 5,
            iterationsPerRound: 1,
        })).toThrow(/warmupIterations/);
        expect(() => measureMedianMsPerIteration(run, {
            warmupIterations: 1,
            measuredRounds: 4,
            iterationsPerRound: 1,
        })).toThrow(/measuredRounds/);
        expect(() => measureMedianMsPerIteration(run, {
            warmupIterations: 1,
            measuredRounds: 5,
            iterationsPerRound: 0,
        })).toThrow(/iterationsPerRound/);
        expect(() => measureMedianMsPerIteration(run, {
            warmupIterations: 1,
            measuredRounds: 5,
            iterationsPerRound: 1.5,
        })).toThrow(/iterationsPerRound/);
        expect(run).not.toHaveBeenCalled();
    });

    it('rejects negative and non-finite elapsed measurements', () => {
        expect(() => measureMedianMsPerIteration(() => undefined, {
            warmupIterations: 0,
            measuredRounds: 5,
            iterationsPerRound: 1,
            now: vi.fn()
                .mockReturnValueOnce(10)
                .mockReturnValueOnce(9),
        })).toThrow(/finite non-negative elapsed time/);

        expect(() => measureMedianMsPerIteration(() => undefined, {
            warmupIterations: 0,
            measuredRounds: 5,
            iterationsPerRound: 1,
            now: vi.fn()
                .mockReturnValueOnce(0)
                .mockReturnValueOnce(Number.POSITIVE_INFINITY),
        })).toThrow(/finite non-negative elapsed time/);
    });
});

describe('assertPerformanceBudgets', () => {
    it('reports every over-budget path in one failure', () => {
        expect(() => assertPerformanceBudgets([
            { name: 'predictGrid', measured: 12, limit: 11 },
            { name: 'withinBudget', measured: 8, limit: 10 },
            { name: 'predictGridInto', measured: 13, limit: 10 },
        ])).toThrow(/predictGrid[\s\S]*predictGridInto/);
    });

    it('treats non-finite measurements as failures and accepts finite results at the limit', () => {
        expect(() => assertPerformanceBudgets([
            { name: 'notMeasured', measured: Number.NaN, limit: 10 },
            { name: 'unboundedMeasurement', measured: Number.POSITIVE_INFINITY, limit: 10 },
        ])).toThrow(/notMeasured[\s\S]*unboundedMeasurement/);

        expect(() => assertPerformanceBudgets([
            { name: 'predictGrid', measured: 11, limit: 11 },
        ])).not.toThrow();
    });

    it('rejects vacuous and duplicate result sets', () => {
        expect(() => assertPerformanceBudgets([])).toThrow(/at least one result/);
        expect(() => assertPerformanceBudgets([
            { name: 'predictGrid', measured: 10, limit: 11 },
            { name: 'predictGrid', measured: 9, limit: 11 },
        ])).toThrow(/duplicate.*predictGrid/i);
    });
});
