import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DEMAND,
    isMainToWorkerCommand,
} from '../index.js';

describe('isMainToWorkerCommand', () => {
    it('accepts updateDemand commands with valid demand values', () => {
        expect(isMainToWorkerCommand({
            type: 'updateDemand',
            demand: {
                ...DEFAULT_DEMAND,
                needLayerStats: true,
                testEvalInterval: 1,
                trainEvalInterval: 2,
                gridInterval: 3,
            },
        })).toBe(true);
    });

    it('rejects updateDemand commands with malformed demand payloads', () => {
        for (const demand of [null, true, 3, 'demand', []]) {
            expect(isMainToWorkerCommand({ type: 'updateDemand', demand })).toBe(false);
        }
    });

    it('rejects updateDemand commands with missing boolean flags', () => {
        const { needDecisionBoundary: _missing, ...missingDemand } = DEFAULT_DEMAND;

        expect(isMainToWorkerCommand({
            type: 'updateDemand',
            demand: missingDemand,
        })).toBe(false);
    });

    it('rejects updateDemand commands with non-boolean flags', () => {
        expect(isMainToWorkerCommand({
            type: 'updateDemand',
            demand: {
                ...DEFAULT_DEMAND,
                needDecisionBoundary: 1,
            },
        })).toBe(false);
    });

    it('rejects updateDemand commands with zero, negative, non-finite, or non-integer intervals', () => {
        for (const field of ['testEvalInterval', 'trainEvalInterval', 'gridInterval'] as const) {
            for (const interval of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, 1.5]) {
                expect(isMainToWorkerCommand({
                    type: 'updateDemand',
                    demand: {
                        ...DEFAULT_DEMAND,
                        [field]: interval,
                    },
                })).toBe(false);
            }
        }
    });
});
