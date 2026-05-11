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
                needActivationHistograms: true,
                testEvalInterval: 1,
                trainEvalInterval: 2,
                gridInterval: 3,
                activationHistogramInterval: 4,
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
                needActivationHistograms: 1,
            },
        })).toBe(false);
    });

    it('rejects updateDemand commands with zero, negative, non-finite, or non-integer intervals', () => {
        for (const field of [
            'testEvalInterval',
            'trainEvalInterval',
            'gridInterval',
            'activationHistogramInterval',
        ] as const) {
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
    it('accepts bounded activation histogram payloads on snapshot messages', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 0,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                gridSize: 2,
            },
            historyPoint: {
                step: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
            },
            activationHistogramBins: new Float32Array([2, 1, 0, 1]),
            activationHistogramLayout: {
                binCount: 2,
                layers: [
                    {
                        layerIndex: 0,
                        binCount: 2,
                        binStart: 0,
                        binWidth: 0.5,
                        minActivation: 0,
                        maxActivation: 1,
                        totalCount: 3,
                        nearZeroCount: 1,
                        saturatedCount: 0,
                    },
                ],
            },
            activationHistogramVersion: 1,
        })).toBe(true);
    });

    it('rejects malformed activation histogram snapshot payloads', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 0,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                gridSize: 2,
            },
            historyPoint: {
                step: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
            },
            activationHistogramBins: [2, 1],
            activationHistogramLayout: null,
            activationHistogramVersion: '1',
        })).toBe(false);
    });

    it('rejects activation histogram layouts with invalid numbers or too few bins', () => {
        const validSnapshot = {
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 0,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                gridSize: 2,
            },
            historyPoint: {
                step: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
            },
            activationHistogramBins: new Float32Array([2, 1]),
            activationHistogramLayout: {
                binCount: 2,
                layers: [
                    {
                        layerIndex: 0,
                        binCount: 2,
                        binStart: 0,
                        binWidth: 0.5,
                        minActivation: 0,
                        maxActivation: 1,
                        totalCount: 3,
                        nearZeroCount: 1,
                        saturatedCount: 0,
                    },
                ],
            },
            activationHistogramVersion: 1,
        };

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            activationHistogramLayout: {
                ...validSnapshot.activationHistogramLayout,
                layers: [
                    {
                        ...validSnapshot.activationHistogramLayout.layers[0],
                        binWidth: Number.NaN,
                    },
                ],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            activationHistogramBins: new Float32Array([2]),
        })).toBe(false);
    });

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
