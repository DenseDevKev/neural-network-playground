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
    it('accepts scalar-only live arena snapshot summaries', () => {
        expect(isWorkerToMainMessage({
            type: 'arenaSnapshot',
            runId: 2,
            snapshotId: 1,
            summaries: [
                {
                    side: 'A',
                    label: 'Model A',
                    status: 'running',
                    step: 5,
                    epoch: 0,
                    trainLoss: 0.42,
                    testLoss: 0.51,
                    trainAccuracy: 0.7,
                    testAccuracy: 0.65,
                },
                {
                    side: 'B',
                    label: 'Model B',
                    status: 'paused',
                    pauseReason: 'manual',
                    step: 5,
                    epoch: 0,
                    trainLoss: 0.5,
                    testLoss: 0.58,
                },
            ],
        })).toBe(true);
    });

    it('rejects malformed live arena snapshot summaries', () => {
        expect(isWorkerToMainMessage({
            type: 'arenaSnapshot',
            runId: 2,
            snapshotId: 1,
            summaries: [
                {
                    side: 'left',
                    label: '',
                    status: 'running',
                    step: -1,
                    epoch: 0,
                    trainLoss: Number.NaN,
                    testLoss: 0.51,
                },
            ],
        })).toBe(false);
    });

    it('accepts lightweight checkpoint timeline metadata on snapshot messages', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 20,
                epoch: 2,
                trainLoss: 0.4,
                testLoss: 0.5,
                gridSize: 2,
            },
            historyPoint: {
                step: 20,
                trainLoss: 0.4,
                testLoss: 0.5,
            },
            checkpointTimeline: {
                checkpoints: [
                    {
                        id: 1,
                        step: 10,
                        epoch: 1,
                        trainLoss: 0.45,
                        testLoss: 0.55,
                        label: 'Step 10',
                    },
                    {
                        id: 2,
                        step: 20,
                        epoch: 2,
                        trainLoss: 0.4,
                        testLoss: 0.5,
                        trainAccuracy: 0.75,
                        testAccuracy: 0.7,
                        label: 'Step 20',
                    },
                ],
                maxCheckpoints: 8,
                evictedCount: 1,
                liveCheckpointId: 2,
                restoredCheckpointId: null,
            },
        })).toBe(true);
    });

    it('rejects malformed checkpoint timeline metadata on snapshot messages', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 20,
                epoch: 2,
                trainLoss: 0.4,
                testLoss: 0.5,
                gridSize: 2,
            },
            historyPoint: {
                step: 20,
                trainLoss: 0.4,
                testLoss: 0.5,
            },
            checkpointTimeline: {
                checkpoints: [
                    {
                        id: 1,
                        step: -1,
                        epoch: 1,
                        trainLoss: 0.45,
                        testLoss: 0.55,
                        label: 'Step -1',
                    },
                ],
                maxCheckpoints: 0,
                evictedCount: Number.NaN,
                liveCheckpointId: '1',
                restoredCheckpointId: null,
            },
        })).toBe(false);
    });

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

    it('accepts omitted activation histogram payloads represented as undefined optional fields', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 10,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                gridSize: 40,
            },
            historyPoint: {
                step: 10,
                trainLoss: 0.5,
                testLoss: 0.6,
            },
            activationHistogramBins: undefined,
            activationHistogramLayout: undefined,
            activationHistogramVersion: undefined,
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

    it('accepts bounded multiclass boundary payloads on snapshot messages', () => {
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
            outputGrid: new Float32Array(0),
            neuronGrids: new Float32Array(0),
            multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
            multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
            multiclassBoundaryVersion: 1,
        })).toBe(true);
    });

    it('rejects malformed multiclass boundary payloads', () => {
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
            multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
            multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
            multiclassBoundaryVersion: 1,
        };

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfidenceGrid: undefined,
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassClassGrid: new Float32Array([0, 1, 2, 1]),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassClassGrid: new Uint8Array([0, 1, 2]),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassClassGrid: new Uint8Array([0, 1, 3, 1]),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfidenceGrid: new Float32Array([0.7, Number.NaN, 0.9, 0.5]),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfidenceGrid: new Float32Array([0.7, 1.1, 0.9, 0.5]),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassBoundaryLayout: {
                gridSize: 3,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 4,
                classLabels: [0, 1, 2],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 2, 1],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassBoundaryVersion: -1,
        })).toBe(false);
    });

    it('accepts bounded multiclass confusion matrix payloads on snapshot messages', () => {
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
            multiclassConfusionMatrix: {
                classCount: 3,
                classLabels: [0, 1, 2],
                counts: [
                    2, 1, 0,
                    0, 3, 1,
                    1, 0, 4,
                ],
            },
            multiclassConfusionMatrixVersion: 1,
        })).toBe(true);
    });

    it('rejects malformed multiclass confusion matrix payloads', () => {
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
            multiclassConfusionMatrix: {
                classCount: 3,
                classLabels: [0, 1, 2],
                counts: [
                    2, 1, 0,
                    0, 3, 1,
                    1, 0, 4,
                ],
            },
            multiclassConfusionMatrixVersion: 1,
        };

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrixVersion: undefined,
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: undefined,
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                classCount: 4,
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                classLabels: [0, 2, 1],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                classLabels: [0, 1],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                counts: [2, 1, 0, 0, 3, 1, 1, 0],
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                counts: Array(9),
            },
        })).toBe(false);

        for (const badCount of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, '1']) {
            expect(isWorkerToMainMessage({
                ...validSnapshot,
                multiclassConfusionMatrix: {
                    ...validSnapshot.multiclassConfusionMatrix,
                    counts: [
                        badCount, 1, 0,
                        0, 3, 1,
                        1, 0, 4,
                    ],
                },
            })).toBe(false);
        }

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrix: {
                ...validSnapshot.multiclassConfusionMatrix,
                counts: new Uint32Array([2, 1, 0, 0, 3, 1, 1, 0, 4]),
            },
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            multiclassConfusionMatrixVersion: -1,
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...validSnapshot,
            confusionMatrix: { tn: 1, fp: 0, fn: 0, tp: 1 },
            confusionMatrixVersion: 1,
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
