import { describe, expect, it } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    DEFAULT_DEMAND,
    WORKER_PROTOCOL_VERSION,
    isCaptureCheckpointRequestV2,
    isRestoreCheckpointRequestV2,
    isCaptureRunRequestV2,
    isForceEvaluationRequestV2,
    isMainToWorkerCommand,
    isWorkerEvidenceMessageV2,
    isWorkerExperimentRequestV2,
    isWorkerProtocolErrorMessageV2,
    isWorkerToMainMessage,
    parseWorkerEvidenceMessageV2,
    parseWorkerExperimentRequestV2,
    parseMainToWorkerRequestV2,
    parseCheckpointTimelineV2,
    parseWorkerToMainMessageV2,
    prepareExperimentDocument,
} from '../index.js';

const STRICT_DATASET = {
    generatorVersion: 2,
    datasetKey: 'dataset-strict',
    trainCount: 200,
    testCount: 100,
} as const;

function strictProvenance(
    basis: Record<string, unknown>,
    revision = 2,
) {
    return {
        model: { generationId: 1, revision, step: 2, epoch: 0 },
        dataset: STRICT_DATASET,
        objectiveKey: 'objective-strict',
        basis,
    };
}

function validStrictSnapshot() {
    return {
        type: 'snapshot',
        protocolVersion: 2,
        runId: 1,
        snapshotId: 1,
        model: { generationId: 1, revision: 2, step: 2, epoch: 0 },
        scalars: {
            step: 2,
            epoch: 0,
            trainLoss: 0.4,
            testLoss: 0.5,
            gridSize: 2,
        },
        outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
        weights: new Float32Array([0.1, 0.2]),
        biases: new Float32Array([0.1]),
        weightLayout: { layerSizes: [2, 1] },
        layerStats: [{
            meanActivation: 0.2,
            activationStd: 0.1,
            meanAbsWeight: 0.3,
            meanAbsGradient: 0.05,
        }],
        layerStatsGradientRevision: 1,
        artifacts: {
            decisionBoundary: strictProvenance({
                kind: 'prediction-grid',
                pointCount: 4,
                domain: [-1, 1, -1, 1],
            }),
            activationStatistics: strictProvenance({
                kind: 'bounded-sample',
                split: 'train',
                sampleCount: 128,
                populationCount: 200,
            }),
        },
    } as const;
}

function validStrictHistogramSnapshot() {
    const valid = validStrictSnapshot();
    return {
        ...valid,
        activationHistogramBins: Float32Array.from([
            128,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]),
        activationHistogramLayout: {
            binCount: 12,
            layers: [{
                layerIndex: 0,
                binCount: 12,
                binStart: -1,
                binWidth: 1 / 6,
                minActivation: -0.5,
                maxActivation: 0.75,
                totalCount: 128,
                nearZeroCount: 1,
                saturatedCount: 2,
            }],
        },
        activationHistogramVersion: 1,
        artifacts: {
            ...valid.artifacts,
            activationHistogram: strictProvenance({
                kind: 'bounded-sample',
                split: 'train',
                sampleCount: 128,
                populationCount: 200,
            }),
        },
    };
}

function validStrictConfusionSnapshot() {
    const valid = validStrictSnapshot();
    return {
        ...valid,
        confusionMatrix: { tp: 100, tn: 0, fp: 0, fn: 0 },
        confusionMatrixEvaluationId: 7,
        confusionMatrixVersion: 1,
        artifacts: {
            ...valid.artifacts,
            confusionMatrix: strictProvenance({
                kind: 'full-split',
                split: 'test',
                sampleCount: 100,
                populationCount: 100,
            }),
        },
    };
}

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
    it('accepts a strict snapshot whose artifact payloads have matching provenance', () => {
        expect(isWorkerToMainMessage(validStrictSnapshot())).toBe(true);
    });

    it('accepts legacy snapshot payloads without artifact provenance', () => {
        expect(isWorkerToMainMessage({
            type: 'snapshot',
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 2,
                epoch: 0,
                trainLoss: 0.4,
                testLoss: 0.5,
                gridSize: 2,
            },
            outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            historyPoint: { step: 2, trainLoss: 0.4, testLoss: 0.5 },
        })).toBe(true);
    });

    it('rejects strict artifact payload/provenance mismatches in either direction', () => {
        const valid = validStrictSnapshot();
        expect(isWorkerToMainMessage({
            ...valid,
            artifacts: {
                activationStatistics: valid.artifacts.activationStatistics,
            },
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            outputGrid: undefined,
        })).toBe(false);
    });

    it('rejects a strict snapshot that retains an own legacy history key', () => {
        expect(isWorkerToMainMessage({
            ...validStrictSnapshot(),
            historyPoint: undefined,
        })).toBe(false);
    });

    it.each([
        ['step', -1],
        ['epoch', 1.5],
        ['trainLoss', Number.NaN],
        ['testLoss', Number.POSITIVE_INFINITY],
        ['trainAccuracy', Number.NaN],
        ['testAccuracy', Number.NEGATIVE_INFINITY],
        ['gridSize', 0],
        ['testMetricsStale', 'yes'],
    ])('rejects strict snapshots with an invalid %s scalar', (key, value) => {
        const valid = validStrictSnapshot();
        expect(isWorkerToMainMessage({
            ...valid,
            scalars: {
                ...valid.scalars,
                [key]: value,
            },
        })).toBe(false);
    });

    it.each([
        ['generation', {
            model: { generationId: 2, revision: 2, step: 2, epoch: 0 },
        }],
        ['dataset', {
            dataset: { ...STRICT_DATASET, datasetKey: 'another-dataset' },
        }],
        ['objective', {
            objectiveKey: 'another-objective',
        }],
    ] as const)('rejects mixed strict artifact %s identity', (_label, change) => {
        const valid = validStrictSnapshot();
        expect(isWorkerToMainMessage({
            ...valid,
            artifacts: {
                ...valid.artifacts,
                decisionBoundary: {
                    ...valid.artifacts.decisionBoundary,
                    ...change,
                },
            },
        })).toBe(false);
    });

    it('rejects a layer-statistics gradient revision newer than the activation model', () => {
        expect(isWorkerToMainMessage({
            ...validStrictSnapshot(),
            layerStatsGradientRevision: 3,
        })).toBe(false);
    });

    it('binds strict confusion payloads to one positive paired evaluation ID', () => {
        const valid = validStrictConfusionSnapshot();
        expect(isWorkerToMainMessage(valid)).toBe(true);
        expect(isWorkerToMainMessage({
            ...valid,
            confusionMatrixEvaluationId: undefined,
        })).toBe(false);
        for (const confusionMatrixEvaluationId of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
            expect(isWorkerToMainMessage({
                ...valid,
                confusionMatrixEvaluationId,
            })).toBe(false);
        }
        expect(isWorkerToMainMessage({
            ...validStrictSnapshot(),
            confusionMatrixEvaluationId: 1,
        })).toBe(false);
    });

    it('rejects strict histogram fractions, wrong totals, bad order, and invalid ranges', () => {
        const valid = validStrictHistogramSnapshot();
        expect(isWorkerToMainMessage(valid)).toBe(true);

        const invalid = [
            {
                ...valid,
                activationHistogramBins: Float32Array.from([
                    127.5,
                    0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
            },
            {
                ...valid,
                activationHistogramBins: Float32Array.from([
                    127,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
            },
            {
                ...valid,
                activationHistogramLayout: {
                    ...valid.activationHistogramLayout,
                    layers: [{
                        ...valid.activationHistogramLayout.layers[0],
                        layerIndex: 1,
                    }],
                },
            },
            {
                ...valid,
                activationHistogramLayout: {
                    ...valid.activationHistogramLayout,
                    layers: [{
                        ...valid.activationHistogramLayout.layers[0],
                        minActivation: 1,
                        maxActivation: -1,
                    }],
                },
            },
            {
                ...valid,
                activationHistogramLayout: {
                    ...valid.activationHistogramLayout,
                    layers: [{
                        ...valid.activationHistogramLayout.layers[0],
                        totalCount: 127,
                    }],
                },
            },
            {
                ...valid,
                activationHistogramBins: new Float32Array(13),
            },
        ];
        for (const message of invalid) {
            expect(isWorkerToMainMessage(message)).toBe(false);
        }
    });

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

describe('version 2 worker protocol', () => {
    async function preparedRequest() {
        const prepared = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        if (!prepared.ok) throw new Error('default experiment must prepare');
        return {
            type: 'initialize-experiment',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: 1,
            document: prepared.value.document,
            claimedIdentities: prepared.value.identities,
        } as const;
    }

    const model = { generationId: 1, revision: 4, step: 4, epoch: 0 } as const;
    const dataset = {
        generatorVersion: 2,
        datasetKey: 'd2.1.' + 'A'.repeat(43),
        trainCount: 2,
        testCount: 2,
    } as const;
    const provenance = {
        model,
        dataset,
        objectiveKey: 'o2.1.' + 'B'.repeat(43),
        basis: {
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: 2,
            populationCount: 2,
        },
    } as const;
    const evaluation = {
        evaluationId: 1,
        trigger: 'cadence',
        model,
        dataset,
        objectiveKey: 'o2.1.' + 'B'.repeat(43),
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 2, populationCount: 2 },
            values: { dataLoss: 0.25 },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 2, populationCount: 2 },
            values: { dataLoss: 0.5 },
        },
        objective: { regularizationPenalty: 0.1, trainTotalObjective: 0.35 },
    } as const;
    const liveSignal = {
        model,
        dataset,
        objectiveKey: 'o2.1.' + 'B'.repeat(43),
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 2,
            throughStep: 4,
        },
        dataLoss: 0.3,
    } as const;

    it('accepts only an exact version-2 document plus all claimed identities', async () => {
        const request = await preparedRequest();
        expect(WORKER_PROTOCOL_VERSION).toBe(2);
        expect(isWorkerExperimentRequestV2(request)).toBe(true);

        const { objectiveKey: _missing, ...missingIdentity } = request.claimedIdentities;
        expect(isWorkerExperimentRequestV2({
            ...request,
            claimedIdentities: missingIdentity,
        })).toBe(false);
        expect(isWorkerExperimentRequestV2({ ...request, protocolVersion: 1 })).toBe(false);
        expect(isWorkerExperimentRequestV2({ ...request, unexpected: true })).toBe(false);
        expect(isWorkerExperimentRequestV2({
            ...request,
            document: { ...request.document, unexpected: true },
        })).toBe(false);

        const accessor = { ...request } as Record<string, unknown>;
        Object.defineProperty(accessor, 'type', {
            enumerable: true,
            get: () => 'initialize-experiment',
        });
        expect(isWorkerExperimentRequestV2(accessor)).toBe(false);

        const customPrototype = { ...request };
        Object.setPrototypeOf(customPrototype, { inherited: true });
        expect(isWorkerExperimentRequestV2(customPrototype)).toBe(false);

        const mutableRequest: any = structuredClone(request);
        const parsed = parseWorkerExperimentRequestV2(mutableRequest);
        mutableRequest.document.view.showTestData = true;
        mutableRequest.claimedIdentities.datasetKey = 'd2.1.' + 'Z'.repeat(43);
        expect(parsed.document.view.showTestData).toBe(false);
        expect(parsed.claimedIdentities.datasetKey).not.toContain('ZZZZ');
        expect(Object.isFrozen(parsed)).toBe(true);
        expect(Object.isFrozen(parsed.document)).toBe(true);
        expect(() => parseWorkerExperimentRequestV2({ ...request, protocolVersion: 1 }))
            .toThrow('worker experiment request');
    });

    it('validates live, paired, and artifact provenance on evidence messages', () => {
        const message = {
            type: 'evidence',
            protocolVersion: 2,
            liveSignal,
            latestEvaluation: evaluation,
            artifacts: {
                activationStatistics: provenance,
                predictionTrace: {
                    ...provenance,
                    basis: {
                        kind: 'bounded-sample',
                        split: 'test',
                        sampleCount: 1,
                        populationCount: 2,
                    },
                },
            },
        } as const;
        expect(isWorkerEvidenceMessageV2(message)).toBe(true);
        expect(isWorkerToMainMessage(message)).toBe(true);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            artifacts: {
                activationStatistics: {
                    ...provenance,
                    basis: { ...provenance.basis, sampleCount: 3 },
                },
            },
        })).toBe(false);
        expect(isWorkerEvidenceMessageV2({
            type: 'evidence',
            protocolVersion: 2,
        })).toBe(false);
        expect(isWorkerEvidenceMessageV2({ ...message, accuracy: 0.5 })).toBe(false);

        const parsed = parseWorkerEvidenceMessageV2(message);
        (message.liveSignal as any).dataLoss = 9;
        expect(parsed.liveSignal?.dataLoss).toBe(0.3);
        expect(Object.isFrozen(parsed)).toBe(true);
        expect(Object.isFrozen(parsed.liveSignal)).toBe(true);
    });

    it('rejects evidence bundles that mix generations, datasets, or objectives', () => {
        const message = {
            type: 'evidence',
            protocolVersion: 2,
            liveSignal,
            latestEvaluation: evaluation,
            artifacts: { activationStatistics: provenance },
        } as const;

        expect(isWorkerEvidenceMessageV2({
            ...message,
            latestEvaluation: {
                ...evaluation,
                dataset: { ...evaluation.dataset, datasetKey: 'd2.1.' + 'C'.repeat(43) },
            },
        })).toBe(false);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            artifacts: {
                activationStatistics: {
                    ...provenance,
                    model: { ...provenance.model, generationId: 2 },
                },
            },
        })).toBe(false);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            liveSignal: { ...liveSignal, objectiveKey: 'o2.1.' + 'C'.repeat(43) },
        })).toBe(false);
    });

    it('enforces the deterministic bounded training basis for activation statistics', () => {
        const message = {
            type: 'evidence',
            protocolVersion: 2,
            artifacts: { activationStatistics: provenance },
        } as const;
        expect(isWorkerEvidenceMessageV2(message)).toBe(true);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            artifacts: {
                activationStatistics: {
                    ...provenance,
                    basis: { ...provenance.basis, split: 'test' },
                },
            },
        })).toBe(false);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            artifacts: {
                activationStatistics: {
                    ...provenance,
                    basis: { ...provenance.basis, sampleCount: 1 },
                },
            },
        })).toBe(false);
    });

    it('accepts strict structured errors and rejects unknown sources or fields', () => {
        const message = {
            type: 'worker-error',
            protocolVersion: 2,
            requestId: 7,
            generationId: 2,
            code: 'identity-mismatch',
            path: 'claimedIdentities.datasetKey',
            message: 'Dataset identity does not match the prepared document.',
            source: 'preparation',
        } as const;
        expect(isWorkerProtocolErrorMessageV2(message)).toBe(true);
        expect(isWorkerToMainMessage(message)).toBe(true);
        expect(isWorkerProtocolErrorMessageV2({ ...message, source: 'somewhere' })).toBe(false);
        expect(isWorkerProtocolErrorMessageV2({ ...message, detail: 'hidden' })).toBe(false);
        expect(isWorkerProtocolErrorMessageV2({ ...message, message: '' })).toBe(false);
        const { requestId: _requestId, ...missingRequestId } = message;
        expect(isWorkerProtocolErrorMessageV2(missingRequestId)).toBe(false);
        expect(isWorkerProtocolErrorMessageV2({
            ...message,
            requestId: null,
            generationId: null,
        })).toBe(true);
        const parsed = parseWorkerToMainMessageV2(message);
        expect(parsed).toEqual(message);
        expect(Object.isFrozen(parsed)).toBe(true);
    });

    it('uses explicit strict requests for forced evaluation, run capture, and checkpoint capture', () => {
        expect(isForceEvaluationRequestV2({
            type: 'force-evaluation',
            protocolVersion: 2,
            requestId: 3,
            trigger: 'pause',
        })).toBe(true);
        expect(isCaptureRunRequestV2({
            type: 'capture-run',
            protocolVersion: 2,
            requestId: 4,
            id: '00000000-0000-0000-0000-000000000004',
            createdAt: '2026-07-11T12:00:00.000Z',
            updatedAt: '2026-07-11T12:00:00.000Z',
            title: 'Saved evidence',
        })).toBe(true);
        expect(isCaptureCheckpointRequestV2({
            type: 'capture-checkpoint',
            protocolVersion: 2,
            requestId: 5,
        })).toBe(true);
        expect(isRestoreCheckpointRequestV2({
            type: 'restore-checkpoint',
            protocolVersion: 2,
            requestId: 6,
            checkpointId: 3,
        })).toBe(true);

        expect(isForceEvaluationRequestV2({
            type: 'force-evaluation',
            protocolVersion: 2,
            requestId: 3,
            trigger: 'cadence',
        })).toBe(false);

        const parsed = parseMainToWorkerRequestV2({
            type: 'force-evaluation',
            protocolVersion: 2,
            requestId: 3,
            trigger: 'pause',
        });
        expect(parsed).toMatchObject({ type: 'force-evaluation', trigger: 'pause' });
        expect(Object.isFrozen(parsed)).toBe(true);
        expect(isCaptureRunRequestV2({
            type: 'capture-run',
            protocolVersion: 2,
            requestId: 0,
            id: '00000000-0000-0000-0000-000000000004',
            createdAt: '2026-07-11T12:00:00.000Z',
            updatedAt: '2026-07-11T12:00:00.000Z',
        })).toBe(false);
        expect(isCaptureRunRequestV2({
            type: 'capture-run',
            protocolVersion: 2,
            requestId: 4,
        })).toBe(false);
        expect(isCaptureCheckpointRequestV2({
            type: 'capture-checkpoint',
            protocolVersion: 2,
            requestId: 5,
            label: 'silently ignored',
        })).toBe(false);
        expect(parseMainToWorkerRequestV2({
            type: 'restore-checkpoint',
            protocolVersion: 2,
            requestId: 6,
            checkpointId: 3,
        })).toEqual({
            type: 'restore-checkpoint',
            protocolVersion: 2,
            requestId: 6,
            checkpointId: 3,
        });
        expect(isRestoreCheckpointRequestV2({
            type: 'restore-checkpoint',
            protocolVersion: 2,
            requestId: 6,
            checkpointId: 0,
        })).toBe(false);
        expect(isRestoreCheckpointRequestV2({
            type: 'restore-checkpoint',
            protocolVersion: 2,
            requestId: 6,
            checkpointId: 3,
            payload: { weights: [] },
        })).toBe(false);
    });

    it('strictly parses bounded unique checkpoint metadata without checkpoint payloads', () => {
        const timeline = {
            checkpoints: [
                {
                    id: 1,
                    step: 0,
                    epoch: 0,
                    trainLoss: 0.5,
                    testLoss: 0.6,
                    label: 'Step 0',
                },
                {
                    id: 2,
                    step: 4,
                    epoch: 1,
                    trainLoss: 0.4,
                    testLoss: 0.5,
                    trainAccuracy: 0.75,
                    testAccuracy: 0.5,
                    label: 'Step 4',
                },
            ],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: 2,
            restoredCheckpointId: 1,
        };
        const parsed = parseCheckpointTimelineV2(timeline);
        expect(parsed).toEqual(timeline);
        expect(Object.isFrozen(parsed)).toBe(true);
        expect(JSON.stringify(parsed)).not.toMatch(/weights|biases|optimizer|cursor/i);

        expect(() => parseCheckpointTimelineV2({
            ...timeline,
            checkpoints: [...timeline.checkpoints, { ...timeline.checkpoints[0] }],
        })).toThrow(/unique|duplicate/i);
        expect(() => parseCheckpointTimelineV2({
            ...timeline,
            liveCheckpointId: 99,
        })).toThrow(/liveCheckpointId|present/i);
        expect(() => parseCheckpointTimelineV2({
            ...timeline,
            restoredCheckpointId: 99,
        })).toThrow(/restoredCheckpointId|present/i);
        expect(() => parseCheckpointTimelineV2({
            ...timeline,
            checkpoints: Array.from({ length: 9 }, (_, index) => ({
                ...timeline.checkpoints[0],
                id: index + 1,
            })),
        })).toThrow(/8|bounded/i);
        expect(() => parseCheckpointTimelineV2({ ...timeline, payload: {} })).toThrow(/exactly/i);
    });
});
