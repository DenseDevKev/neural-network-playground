import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DEMAND,
    DEFAULT_EXPERIMENT_DOCUMENT,
    GRID_SIZE,
    MAX_HIDDEN_LAYERS,
    MAX_NEURONS_PER_LAYER,
    WORKER_PROTOCOL_VERSION,
    isCaptureCheckpointRequestV2,
    isCaptureRunRequestV2,
    isForceEvaluationRequestV2,
    isMainToWorkerCommand,
    isRestoreCheckpointRequestV2,
    isWorkerEvidenceMessageV2,
    isWorkerExperimentRequestV2,
    isWorkerProtocolErrorMessageV2,
    isWorkerToMainMessage,
    normalizeVisualizationDemand,
    parseCheckpointTimelineV2,
    parseMainToWorkerRequestV2,
    parseWorkerEvidenceMessageV2,
    parseWorkerExperimentRequestV2,
    parseWorkerToMainMessageV2,
    prepareExperimentDocument,
} from '../index.js';

const STRICT_DATASET = {
    generatorVersion: 2,
    datasetKey: 'dataset-strict',
    trainCount: 200,
    testCount: 100,
} as const;

function strictProvenance(basis: Record<string, unknown>, revision = 2) {
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
        protocolVersion: WORKER_PROTOCOL_VERSION,
        runId: 1,
        snapshotId: 1,
        model: { generationId: 1, revision: 2, step: 2, epoch: 0 },
        recipeFingerprint: `r2.1.${'A'.repeat(43)}`,
        scalars: { step: 2, epoch: 0, gridSize: 2 },
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
        checkpointTimeline: {
            checkpoints: [],
            maxCheckpoints: 8,
            evictedCount: 0,
            liveCheckpointId: null,
            restoredCheckpointId: null,
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

describe('strict V2 streaming protocol', () => {
    it('keeps visualization demand independent from scientific evaluation cadence', () => {
        expect(DEFAULT_DEMAND).toEqual({
            needDecisionBoundary: true,
            needNeuronGrids: true,
            needLayerStats: false,
            needActivationHistograms: false,
            needConfusionMatrix: true,
            gridInterval: 2,
            activationHistogramInterval: 5,
        });
        expect(normalizeVisualizationDemand(DEFAULT_DEMAND)).toEqual(DEFAULT_DEMAND);
        for (const extra of [
            { trainEvalInterval: 5 },
            { testEvalInterval: 10 },
            { unexpected: true },
        ]) {
            expect(normalizeVisualizationDemand({ ...DEFAULT_DEMAND, ...extra })).toBeNull();
        }
    });

    it('requires exact versioned streaming commands', () => {
        expect(isMainToWorkerCommand({
            type: 'updateDemand',
            protocolVersion: 2,
            demand: { ...DEFAULT_DEMAND, needLayerStats: true },
        })).toBe(true);
        expect(isMainToWorkerCommand({
            type: 'startTraining',
            protocolVersion: 2,
            stepsPerFrame: 10,
        })).toBe(true);
        expect(isMainToWorkerCommand({
            type: 'stopTraining',
            protocolVersion: 2,
        })).toBe(true);
        expect(isMainToWorkerCommand({ type: 'stopTraining' })).toBe(false);
        expect(isMainToWorkerCommand({
            type: 'stopTraining',
            protocolVersion: 2,
            unexpected: true,
        })).toBe(false);
        expect(isMainToWorkerCommand({
            type: 'startTraining',
            protocolVersion: 2,
            stepsPerFrame: Number.NaN,
        })).toBe(false);
        expect(isMainToWorkerCommand({
            type: 'updateDemand',
            protocolVersion: 2,
            demand: { ...DEFAULT_DEMAND, trainEvalInterval: 5 },
        })).toBe(false);
    });

    it('accepts strict snapshot frames with matching artifact provenance', () => {
        expect(isWorkerToMainMessage(validStrictSnapshot())).toBe(true);
        expect(isWorkerToMainMessage(validStrictHistogramSnapshot())).toBe(true);
    });

    it('rejects unversioned, future, unknown-field, and scalar-evidence frames', () => {
        const valid = validStrictSnapshot();
        const { protocolVersion: _version, ...unversioned } = valid;
        expect(isWorkerToMainMessage(unversioned)).toBe(false);
        expect(isWorkerToMainMessage({ ...valid, protocolVersion: 3 })).toBe(false);
        expect(isWorkerToMainMessage({ ...valid, historyPoint: undefined })).toBe(false);
        for (const scalar of [
            { trainLoss: 0.4 },
            { testLoss: 0.5 },
            { trainAccuracy: 0.75 },
            { testAccuracy: 0.7 },
            { testMetricsStale: false },
        ]) {
            expect(isWorkerToMainMessage({
                ...valid,
                scalars: { ...valid.scalars, ...scalar },
            })).toBe(false);
        }
        expect(isWorkerToMainMessage({
            type: 'arenaSnapshot',
            protocolVersion: 2,
            runId: 1,
            snapshotId: 1,
            summaries: [],
        })).toBe(false);
    });

    it('requires model and checkpoint identity on every snapshot', () => {
        const valid = validStrictSnapshot();
        const { model: _model, ...withoutModel } = valid;
        const { checkpointTimeline: _timeline, ...withoutTimeline } = valid;
        const { recipeFingerprint: _fingerprint, ...withoutFingerprint } = valid;
        expect(isWorkerToMainMessage(withoutModel)).toBe(false);
        expect(isWorkerToMainMessage(withoutTimeline)).toBe(false);
        expect(isWorkerToMainMessage(withoutFingerprint)).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            model: { ...valid.model, step: 1 },
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            checkpointTimeline: { ...valid.checkpointTimeline, maxCheckpoints: 7 },
        })).toBe(false);
    });

    it('binds strict payloads to matching artifact identity and basis', () => {
        const valid = validStrictSnapshot();
        expect(isWorkerToMainMessage({
            ...valid,
            artifacts: {
                activationStatistics: valid.artifacts.activationStatistics,
            },
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            artifacts: {
                ...valid.artifacts,
                decisionBoundary: {
                    ...valid.artifacts.decisionBoundary,
                    model: { ...valid.model, generationId: 2 },
                },
            },
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            layerStatsGradientRevision: 3,
        })).toBe(false);
    });

    it('validates histogram payload shapes and totals', () => {
        const valid = validStrictHistogramSnapshot();
        expect(isWorkerToMainMessage({
            ...valid,
            activationHistogramBins: Float32Array.from([
                127,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]),
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            activationHistogramLayout: {
                ...valid.activationHistogramLayout,
                layers: [{ ...valid.activationHistogramLayout.layers[0], totalCount: 127 }],
            },
        })).toBe(false);
    });

    it('binds confusion payloads to an exact paired evaluation ID', () => {
        const valid = validStrictSnapshot();
        const confusion = {
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
        expect(isWorkerToMainMessage(confusion)).toBe(true);
        expect(isWorkerToMainMessage({
            ...confusion,
            confusionMatrixEvaluationId: 0,
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...confusion,
            confusionMatrix: { tp: 99, tn: 0, fp: 0, fn: 0 },
        })).toBe(false);
    });

    it('requires protocol version 2 for status and shared stream frames', () => {
        const checkpointTimeline = validStrictSnapshot().checkpointTimeline;
        expect(isWorkerToMainMessage({
            type: 'status',
            protocolVersion: 2,
            runId: 1,
            status: 'paused',
            pauseReason: 'diverged',
            checkpointTimeline,
        })).toBe(true);
        expect(isWorkerToMainMessage({
            type: 'status',
            protocolVersion: 2,
            runId: 1,
            status: 'paused',
            checkpointTimeline: { ...checkpointTimeline, maxCheckpoints: 7 },
        })).toBe(false);
        expect(isWorkerToMainMessage({
            type: 'status',
            runId: 1,
            status: 'paused',
        })).toBe(false);
        expect(isWorkerToMainMessage({
            type: 'status',
            protocolVersion: 2,
            runId: 1,
            status: 'paused',
            unexpected: true,
        })).toBe(false);
    });

    it('accepts only exact, resource-bounded SharedArrayBuffer handshakes', () => {
        const gridSize = 2;
        const neuronCount = 1;
        const valid = {
            type: 'sharedBuffers',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            runId: 1,
            control: new SharedArrayBuffer(8 * Int32Array.BYTES_PER_ELEMENT),
            outputGrid: new SharedArrayBuffer(
                gridSize * gridSize * Float32Array.BYTES_PER_ELEMENT,
            ),
            neuronGrids: new SharedArrayBuffer(
                neuronCount * gridSize * gridSize * Float32Array.BYTES_PER_ELEMENT,
            ),
            gridSize,
            neuronGridLayout: { count: neuronCount, gridSize },
        } as const;
        expect(isWorkerToMainMessage(valid)).toBe(true);

        const regularBuffer = new ArrayBuffer(valid.control.byteLength);
        expect(isWorkerToMainMessage({ ...valid, control: regularBuffer })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            outputGrid: new ArrayBuffer(valid.outputGrid.byteLength),
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            neuronGrids: new ArrayBuffer(valid.neuronGrids.byteLength),
        })).toBe(false);

        expect(isWorkerToMainMessage({
            ...valid,
            control: new SharedArrayBuffer(valid.control.byteLength + 4),
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            outputGrid: new SharedArrayBuffer(valid.outputGrid.byteLength + 4),
        })).toBe(false);
        expect(isWorkerToMainMessage({
            ...valid,
            neuronGrids: new SharedArrayBuffer(valid.neuronGrids.byteLength + 4),
        })).toBe(false);

        for (const invalidGridSize of [0, 1.5, GRID_SIZE + 1, Number.MAX_SAFE_INTEGER + 1]) {
            expect(isWorkerToMainMessage({ ...valid, gridSize: invalidGridSize })).toBe(false);
        }
        expect(isWorkerToMainMessage({
            ...valid,
            neuronGridLayout: { count: neuronCount, gridSize: gridSize + 1 },
        })).toBe(false);

        const maximumNeuronCount = MAX_HIDDEN_LAYERS * MAX_NEURONS_PER_LAYER + 3;
        for (const invalidNeuronCount of [
            0,
            1.5,
            maximumNeuronCount + 1,
            Number.MAX_SAFE_INTEGER + 1,
        ]) {
            expect(isWorkerToMainMessage({
                ...valid,
                neuronGridLayout: { count: invalidNeuronCount, gridSize },
            })).toBe(false);
        }
    });

    it('returns false instead of invoking hostile accessors or leaking proxy errors', () => {
        const throwingAccessor = {};
        Object.defineProperty(throwingAccessor, 'type', {
            enumerable: true,
            get: () => { throw new Error('hostile type getter'); },
        });
        const throwingProxy = new Proxy({}, {
            getPrototypeOf: () => { throw new Error('hostile prototype trap'); },
            ownKeys: () => { throw new Error('hostile ownKeys trap'); },
            get: () => { throw new Error('hostile get trap'); },
        });

        expect(() => isWorkerToMainMessage(throwingAccessor)).not.toThrow();
        expect(isWorkerToMainMessage(throwingAccessor)).toBe(false);
        expect(() => isWorkerToMainMessage(throwingProxy)).not.toThrow();
        expect(isWorkerToMainMessage(throwingProxy)).toBe(false);
    });
});

describe('version 2 worker protocol boundaries', () => {
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
        expect(isWorkerExperimentRequestV2(request)).toBe(true);
        expect(isWorkerExperimentRequestV2({ ...request, protocolVersion: 1 })).toBe(false);
        expect(isWorkerExperimentRequestV2({ ...request, unexpected: true })).toBe(false);
        const { objectiveKey: _missing, ...missingIdentity } = request.claimedIdentities;
        expect(isWorkerExperimentRequestV2({
            ...request,
            claimedIdentities: missingIdentity,
        })).toBe(false);

        const mutableRequest: any = structuredClone(request);
        const parsed = parseWorkerExperimentRequestV2(mutableRequest);
        mutableRequest.document.view.showTestData = true;
        expect(parsed.document.view.showTestData).toBe(false);
        expect(Object.isFrozen(parsed.document)).toBe(true);
    });

    it('validates live, paired, and artifact provenance on evidence messages', () => {
        const message = {
            type: 'evidence',
            protocolVersion: 2,
            liveSignal,
            latestEvaluation: evaluation,
            artifacts: { activationStatistics: provenance },
        } as const;
        expect(isWorkerEvidenceMessageV2(message)).toBe(true);
        expect(isWorkerToMainMessage(message)).toBe(true);
        expect(isWorkerEvidenceMessageV2({ ...message, accuracy: 0.5 })).toBe(false);
        expect(isWorkerEvidenceMessageV2({
            ...message,
            latestEvaluation: {
                ...evaluation,
                dataset: { ...dataset, datasetKey: 'd2.1.' + 'C'.repeat(43) },
            },
        })).toBe(false);

        const parsed = parseWorkerEvidenceMessageV2(message);
        (message.liveSignal as any).dataLoss = 9;
        expect(parsed.liveSignal?.dataLoss).toBe(0.3);
        expect(Object.isFrozen(parsed)).toBe(true);
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
        expect(Object.isFrozen(parseWorkerToMainMessageV2(message))).toBe(true);
    });

    it('uses explicit strict requests for forced evaluation, capture, and restore', () => {
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
        expect(Object.isFrozen(parseMainToWorkerRequestV2({
            type: 'force-evaluation',
            protocolVersion: 2,
            requestId: 3,
            trigger: 'pause',
        }))).toBe(true);
    });

    it('strictly parses bounded unique checkpoint metadata without payloads', () => {
        const timeline = {
            checkpoints: [
                {
                    id: 1,
                    step: 0,
                    epoch: 0,
                    trainDataLoss: 0.5,
                    testDataLoss: 0.6,
                    label: 'Step 0',
                },
                {
                    id: 2,
                    step: 4,
                    epoch: 1,
                    trainDataLoss: 0.4,
                    testDataLoss: 0.5,
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
            liveCheckpointId: 99,
        })).toThrow(/liveCheckpointId|present/i);
        expect(() => parseCheckpointTimelineV2({
            ...timeline,
            checkpoints: [{
                ...timeline.checkpoints[0],
                trainDataLoss: undefined,
                trainLoss: 0.5,
            }],
            liveCheckpointId: 1,
            restoredCheckpointId: 1,
        })).toThrow(/malformed|checkpoint/i);
        expect(() => parseCheckpointTimelineV2({ ...timeline, payload: {} })).toThrow(/exactly/i);
    });
});
