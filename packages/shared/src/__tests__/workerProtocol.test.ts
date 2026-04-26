import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DEMAND,
    isMainToWorkerCommand,
    isWorkerToMainMessage,
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

describe('isWorkerToMainMessage', () => {
    it('accepts status messages without a pause reason for backward compatibility', () => {
        expect(isWorkerToMainMessage({
            type: 'status',
            runId: 1,
            status: 'paused',
        })).toBe(true);
    });

    it('accepts status messages with a valid pause reason', () => {
        expect(isWorkerToMainMessage({
            type: 'status',
            runId: 1,
            status: 'paused',
            pauseReason: 'diverged',
        })).toBe(true);
    });

    it('accepts status messages with a null pause reason', () => {
        expect(isWorkerToMainMessage({
            type: 'status',
            runId: 1,
            status: 'running',
            pauseReason: null,
        })).toBe(true);
    });

    it('rejects status messages with a malformed pause reason', () => {
        expect(isWorkerToMainMessage({
            type: 'status',
            runId: 1,
            status: 'paused',
            pauseReason: 'because',
        })).toBe(false);
    });
});
