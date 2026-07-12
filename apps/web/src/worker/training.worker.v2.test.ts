import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    Network,
    PRNG,
    softmax,
} from '@nn-playground/engine';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    PREPARED_PRESETS,
    WORKER_PROTOCOL_VERSION,
    isWorkerEvidenceMessageV2,
    isWorkerToMainMessage,
    prepareExperimentDocument,
} from '@nn-playground/shared';
import type {
    PreparedExperimentDocumentV2,
    SchemaResult,
    WorkerEvidenceMessageV2,
    WorkerExperimentRequestV2,
    WorkerSnapshotMessage,
    SessionCheckpointV2,
} from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';

vi.mock('comlink', () => ({
    expose: vi.fn(),
}));

import {
    getV2AllocationCountForTests,
    setGpuInitializationForTests,
    setGpuPredictorForTests,
    setRuntimeStopConditionsForTests,
    setV2OutputOverflowForTests,
    setV2PrepareForTests,
    getV2CheckpointForTests,
    replaceV2CheckpointForTests,
    workerApi,
} from './training.worker.ts';
import { EvaluationRuntime } from './evaluationRuntime.ts';

let nextRequestId = 10_000;

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((resolver) => {
        resolve = resolver;
    });
    return { promise, resolve };
}

function createDeferredGpuReadback(staleValue = 9_999) {
    const completion = deferred<void>();
    let output: Float32Array | null = null;
    let neurons: Float32Array | null = null;
    const predictor = {
        updateWeights: vi.fn(),
        predictGridInto: vi.fn((destination: Float32Array) => {
            output = destination;
            return completion.promise;
        }),
        predictGridWithNeuronsInto: vi.fn((
            outputDestination: Float32Array,
            neuronDestination: Float32Array,
        ) => {
            output = outputDestination;
            neurons = neuronDestination;
            return completion.promise;
        }),
        dispose: vi.fn(),
    };
    return {
        predictor,
        completion: completion.promise,
        release(): void {
            output?.fill(staleValue);
            neurons?.fill(staleValue);
            completion.resolve();
        },
        staleValue,
    };
}

async function flushMicrotasks(turns = 8): Promise<void> {
    for (let index = 0; index < turns; index++) await Promise.resolve();
}

function withFreshId(
    request: WorkerExperimentRequestV2,
): WorkerExperimentRequestV2 {
    return { ...request, requestId: nextRequestId++ };
}

function requestForPrepared(
    prepared: (typeof PREPARED_PRESETS)[number]['prepared'],
): WorkerExperimentRequestV2 {
    return {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: nextRequestId++,
        document: prepared.document,
        claimedIdentities: prepared.identities,
    };
}

function createCapturingPort(): {
    messages: unknown[];
    dispatch: (data: unknown) => void;
    port: MessagePort;
} {
    let listener: ((event: MessageEvent<unknown>) => void) | null = null;
    const messages: unknown[] = [];
    const port = {
        addEventListener: vi.fn((_type: string, callback: (event: MessageEvent<unknown>) => void) => {
            listener = callback;
        }),
        start: vi.fn(),
        postMessage: vi.fn((message: unknown) => messages.push(message)),
    } as unknown as MessagePort;
    return {
        messages,
        dispatch(data: unknown): void {
            if (!listener) throw new Error('stream listener was not registered');
            listener({ data } as MessageEvent<unknown>);
        },
        port,
    };
}

function evidenceMessages(messages: unknown[]): WorkerEvidenceMessageV2[] {
    return messages.filter((message): message is WorkerEvidenceMessageV2 => (
        isWorkerEvidenceMessageV2(message)
    ));
}

function snapshotMessages(messages: unknown[]): WorkerSnapshotMessage[] {
    return messages.filter((message): message is WorkerSnapshotMessage => (
        typeof message === 'object'
        && message !== null
        && (message as { type?: unknown }).type === 'snapshot'
    ));
}

describe('training worker scientific-trust V2 boundary', () => {
    beforeEach(() => {
        vi.useRealTimers();
    });

    afterEach(() => {
        setGpuPredictorForTests(null);
        setGpuInitializationForTests();
        setRuntimeStopConditionsForTests();
        vi.restoreAllMocks();
        vi.unstubAllGlobals();
        vi.useRealTimers();
    });

    it('commits a prepared experiment and returns a validated initial pair', async () => {
        const fixtures = await createScientificTrustFixtures();
        workerApi.updateDemand(DEFAULT_DEMAND);
        const result = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));

        expect(result.runId).toBeGreaterThan(0);
        expect(result.evidence.latestEvaluation).toMatchObject({
            trigger: 'initial',
            evaluationId: 1,
            model: {
                generationId: result.runId,
                revision: 0,
                step: 0,
                epoch: 0,
            },
            dataset: {
                generatorVersion: 2,
                datasetKey: fixtures.prepared.identities.datasetKey,
            },
            objectiveKey: fixtures.prepared.identities.objectiveKey,
        });
        expect(result.evidence.latestEvaluation?.train.basis.sampleCount).toBe(
            result.evidence.latestEvaluation?.dataset.trainCount,
        );
        expect(result.evidence.latestEvaluation?.test.basis.sampleCount).toBe(
            result.evidence.latestEvaluation?.dataset.testCount,
        );
        const pair = result.evidence.latestEvaluation!;
        const predictionBasis = {
            kind: 'prediction-grid',
            pointCount: result.snapshot.gridSize * result.snapshot.gridSize,
            domain: [-1, 1, -1, 1],
        } as const;
        expect(result.snapshot.outputGrid).toHaveLength(predictionBasis.pointCount);
        expect(result.snapshot.neuronGrids).toBeInstanceOf(Float32Array);
        expect(result.artifacts?.decisionBoundary).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: predictionBasis,
        });
        expect(result.artifacts?.neuronGrids).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: predictionBasis,
        });
        expect(result.snapshot.historyPoint).toBeUndefined();
        expect(Object.prototype.hasOwnProperty.call(result.snapshot, 'historyPoint')).toBe(false);
        expect(isWorkerEvidenceMessageV2(result.evidence)).toBe(true);
        expect(isWorkerToMainMessage(result.evidence)).toBe(true);
    });

    it('seeds exactly one metadata-only step-zero checkpoint from the initial pair', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);

        const result = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const timeline = workerApi.getCheckpointTimeline();

        expect(result.evidence.latestEvaluation).toMatchObject({
            evaluationId: 1,
            trigger: 'initial',
        });
        expect(result.checkpointTimeline).toEqual(timeline);
        expect(timeline.checkpoints).toHaveLength(1);
        expect(timeline.checkpoints[0]).toMatchObject({ id: 1, step: 0, epoch: 0 });
        expect(evidenceMessages(capture.messages).filter(
            (message) => message.latestEvaluation?.trigger === 'checkpoint',
        )).toEqual([]);
        expect(JSON.stringify(timeline)).not.toMatch(/weights|biases|optimizer|cursor/i);
    });

    it('forces explicit and every-five-step checkpoint pairs into a bounded ring', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const explicit = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });

        expect(explicit.runId).toBe(initialized.runId);
        expect(explicit.evidence.latestEvaluation).toMatchObject({
            trigger: 'checkpoint',
            model: { generationId: initialized.runId, step: 0 },
        });
        expect(explicit.checkpointTimeline.checkpoints).toHaveLength(2);

        for (let index = 0; index < 9; index++) {
            await workerApi.stepExperimentV2(5);
        }

        const timeline = workerApi.getCheckpointTimeline();
        expect(timeline.checkpoints).toHaveLength(8);
        expect(timeline.evictedCount).toBeGreaterThan(0);
        expect(timeline.checkpoints.at(-1)?.step).toBe(45);
        expect(workerApi.getMetricHistoryV2().evaluationHistory.some(
            (evaluation) => evaluation.trigger === 'checkpoint' && evaluation.model.step === 45,
        )).toBe(true);
    });

    it('leaves explicit checkpoint evidence atomic when envelope validation fails', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        capture.messages.length = 0;
        const historyBefore = workerApi.getMetricHistoryV2();
        const timelineBefore = workerApi.getCheckpointTimeline();
        const captureSessionState = Network.prototype.captureSessionState;
        vi.spyOn(Network.prototype, 'captureSessionState').mockImplementationOnce(function (
            this: Network,
            ...args: Parameters<Network['captureSessionState']>
        ) {
            const session = captureSessionState.apply(this, args);
            Object.defineProperty(session.network.layers[0].weights, 'unexpected', {
                configurable: true,
                value: true,
            });
            return session;
        });

        await expect(workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        })).rejects.toThrow(/dense|own|extra/i);

        expect(workerApi.getMetricHistoryV2()).toEqual(historyBefore);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
        expect(evidenceMessages(capture.messages)).toEqual([]);

        const recovered = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        expect(recovered.evidence.latestEvaluation).toMatchObject({
            evaluationId: 2,
            trigger: 'checkpoint',
        });
        expect(recovered.checkpointTimeline.checkpoints).toHaveLength(2);
    });

    it('keeps a failed periodic capture out of paired history and continues the FIFO', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        capture.messages.length = 0;
        const evaluationsBefore = workerApi.getMetricHistoryV2().evaluationHistory;
        const timelineBefore = workerApi.getCheckpointTimeline();
        vi.spyOn(Network.prototype, 'captureSessionState').mockImplementationOnce(() => {
            throw new Error('periodic capture failed');
        });

        await expect(workerApi.stepExperimentV2(5)).rejects.toThrow('periodic capture failed');

        expect(workerApi.getMetricHistoryV2().trendHistory).toHaveLength(5);
        expect(workerApi.getMetricHistoryV2().evaluationHistory).toEqual(evaluationsBefore);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
        expect(evidenceMessages(capture.messages)).toEqual([]);

        const recovered = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        expect(recovered.evidence.latestEvaluation).toMatchObject({
            evaluationId: 2,
            trigger: 'checkpoint',
            model: { step: 5 },
        });
    });

    it('keeps coincident cadence and checkpoint pairs uncommitted when step-50 capture fails', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        for (let index = 0; index < 4; index++) await workerApi.stepExperimentV2(10);
        await workerApi.stepExperimentV2(5);
        const before = workerApi.getMetricHistoryV2();
        const timelineBefore = workerApi.getCheckpointTimeline();
        const nextEvaluationId = before.evaluationHistory.at(-1)!.evaluationId + 1;
        capture.messages.length = 0;
        vi.spyOn(Network.prototype, 'captureSessionState').mockImplementationOnce(() => {
            throw new Error('step-50 capture failed');
        });

        await expect(workerApi.stepExperimentV2(5)).rejects.toThrow('step-50 capture failed');

        const failed = workerApi.getMetricHistoryV2();
        expect(failed.trendHistory).toHaveLength(before.trendHistory.length + 5);
        expect(failed.evaluationHistory).toEqual(before.evaluationHistory);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
        expect(evidenceMessages(capture.messages)).toEqual([]);

        const recovered = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        expect(recovered.evidence.latestEvaluation).toMatchObject({
            evaluationId: nextEvaluationId,
            trigger: 'checkpoint',
            model: { step: 50 },
        });
    });

    it('commits coincident runtime pairs before publishing either pair', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        for (let index = 0; index < 4; index++) await workerApi.stepExperimentV2(10);
        await workerApi.stepExperimentV2(5);
        const historyBefore = workerApi.getMetricHistoryV2();
        const timelineBefore = workerApi.getCheckpointTimeline();
        capture.messages.length = 0;
        const prepareCheckpoint = EvaluationRuntime.prototype.prepareCheckpointEvaluation;
        let observedCheckpointCommit = false;
        vi.spyOn(
            EvaluationRuntime.prototype,
            'prepareCheckpointEvaluation',
        ).mockImplementation(function (
            this: EvaluationRuntime,
            ...args: Parameters<EvaluationRuntime['prepareCheckpointEvaluation']>
        ) {
            const prepared = prepareCheckpoint.apply(this, args);
            if (args[0] === undefined) return prepared;
            return Object.freeze({
                evaluation: prepared.evaluation,
                commit: () => {
                    observedCheckpointCommit = true;
                    expect(workerApi.getMetricHistoryV2().evaluationHistory).toEqual(
                        historyBefore.evaluationHistory,
                    );
                    expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
                    expect(evidenceMessages(capture.messages)).toEqual([]);
                    return prepared.commit();
                },
            });
        });

        const completed = await workerApi.stepExperimentV2(5);

        expect(observedCheckpointCommit).toBe(true);
        expect(completed.evidence.latestEvaluation).toMatchObject({
            trigger: 'manual-step',
            model: { step: 50 },
        });
        expect(evidenceMessages(capture.messages).map(
            (message) => message.latestEvaluation?.trigger,
        )).toEqual(['cadence', 'checkpoint', 'manual-step']);
    });

    it('restores captured parameters, optimizer, cursor, and history in the same generation', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(1);
        const captured = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        const checkpointId = captured.checkpointTimeline.liveCheckpointId!;
        const privateCheckpoint = getV2CheckpointForTests(checkpointId);
        await workerApi.stepExperimentV2(2);

        const restored = await workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId,
        });

        expect(restored.runId).toBe(initialized.runId);
        expect(restored.evidence.liveSignal).toBeUndefined();
        expect(restored.evidence.latestEvaluation).toMatchObject({
            trigger: 'restore',
            model: {
                generationId: initialized.runId,
                revision: 4,
                step: privateCheckpoint.model.step,
                epoch: privateCheckpoint.cursor.epoch,
            },
        });
        expect(restored.snapshot.weights).toEqual(captured.snapshot.weights);
        expect(restored.snapshot.biases).toEqual(captured.snapshot.biases);
        expect(restored.checkpointTimeline.restoredCheckpointId).toBe(checkpointId);
        expect(workerApi.getMetricHistoryV2().trendHistory).toEqual([]);
        expect(workerApi.getMetricHistoryV2().evaluationHistory).toEqual([
            restored.evidence.latestEvaluation,
        ]);

        const trainSpy = vi.spyOn(Network.prototype, 'trainBatchIndexedV2');
        await workerApi.stepExperimentV2(1);
        const call = trainSpy.mock.calls.at(-1)!;
        expect(Array.from(call[2])).toEqual(Array.from(privateCheckpoint.cursor.shuffledIndices));
        expect(call[3]).toBe(privateCheckpoint.cursor.batchStart);
    });

    it('prepares a private checkpoint on one candidate before atomically replacing live state', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(2);
        const baselineCapture = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        const baselineId = baselineCapture.checkpointTimeline.liveCheckpointId!;
        const baseline = getV2CheckpointForTests(baselineId);

        const malformed = getV2CheckpointForTests(1) as SessionCheckpointV2 & Record<string, unknown>;
        malformed.extra = true;
        replaceV2CheckpointForTests(1, malformed);
        const timelineBeforeMalformed = workerApi.getCheckpointTimeline();
        await expect(workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId: 1,
        })).rejects.toThrow(/exactly/i);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBeforeMalformed);

        const overflowing = structuredClone(baseline);
        overflowing.network.layers.forEach((layer) => layer.weights.fill(Number.MAX_VALUE));
        replaceV2CheckpointForTests(baselineId, overflowing);
        const timelineBeforeOverflow = workerApi.getCheckpointTimeline();
        await expect(workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId: baselineId,
        })).rejects.toThrow();
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBeforeOverflow);

        replaceV2CheckpointForTests(baselineId, baseline);
        const evaluateObjective = Network.prototype.evaluateObjective;
        const evaluatedNetworks = new Set<Network>();
        vi.spyOn(Network.prototype, 'evaluateObjective').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['evaluateObjective']>
        ) {
            evaluatedNetworks.add(this);
            if (evaluatedNetworks.size > 1) {
                throw new Error('restore evaluated a second network after preparation');
            }
            return evaluateObjective.apply(this, args);
        });
        const restored = await workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId: baselineId,
        });
        expect(evaluatedNetworks.size).toBe(1);
        expect(restored.evidence.latestEvaluation?.trigger).toBe('restore');
        expect(restored.checkpointTimeline.restoredCheckpointId).toBe(baselineId);

        const afterCapture = await workerApi.captureCheckpointV2({
            type: 'capture-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
        });
        const after = getV2CheckpointForTests(afterCapture.checkpointTimeline.liveCheckpointId!);
        expect(after.network).toEqual(baseline.network);
        expect(after.optimizer).toEqual(baseline.optimizer);
        expect(after.cursor).toEqual(baseline.cursor);
        expect(after.model).toEqual(restored.evidence.latestEvaluation?.model);
    });

    it('rejects a restore when its detached candidate mutates during evaluation', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const historyBefore = workerApi.getMetricHistoryV2();
        const timelineBefore = workerApi.getCheckpointTimeline();
        const evaluateObjective = Network.prototype.evaluateObjective;
        let mutatedCandidate = false;
        vi.spyOn(Network.prototype, 'evaluateObjective').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['evaluateObjective']>
        ) {
            const objective = evaluateObjective.apply(this, args);
            if (!mutatedCandidate) {
                mutatedCandidate = true;
                this.setBias(0, 0, this.getBias(0, 0));
            }
            return objective;
        });

        await expect(workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId: 1,
        })).rejects.toThrow(/changed during evaluation/i);
        expect(mutatedCandidate).toBe(true);
        expect(workerApi.getMetricHistoryV2()).toEqual(historyBefore);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
    });

    it('rejects finite checkpoint parameters that disagree with captured evaluation values', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const forged = getV2CheckpointForTests(1);
        forged.network.layers[0].weights[0] += 0.5;
        replaceV2CheckpointForTests(1, forged);
        const historyBefore = workerApi.getMetricHistoryV2();
        const timelineBefore = workerApi.getCheckpointTimeline();

        await expect(workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId: 1,
        })).rejects.toThrow(/evaluation.*parameters|parameters.*evaluation/i);
        expect(workerApi.getMetricHistoryV2()).toEqual(historyBefore);
        expect(workerApi.getCheckpointTimeline()).toEqual(timelineBefore);
    });

    it('captures the natural epoch-boundary sentinel and refuses preview without mutation', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(10);
        const shuffleSpy = vi.spyOn(PRNG.prototype, 'shuffle');
        const previewSpy = vi.spyOn(Network.prototype, 'explainBackpropStep');

        const completed = await workerApi.stepExperimentV2(5);
        const checkpointId = completed.checkpointTimeline.liveCheckpointId!;
        const boundary = getV2CheckpointForTests(checkpointId);

        expect(boundary.model).toMatchObject({ step: 15, epoch: 1 });
        expect(boundary.evaluation.model).toEqual(boundary.model);
        expect(boundary.cursor.epoch).toBe(1);
        expect(boundary.cursor.batchStart).toBe(boundary.evaluation.dataset.trainCount);
        expect(shuffleSpy).not.toHaveBeenCalled();
        expect(() => workerApi.getBackpropExplanation()).toThrow(/epoch shuffle boundary/i);
        expect(previewSpy).not.toHaveBeenCalled();
        expect(shuffleSpy).not.toHaveBeenCalled();
        expect(getV2CheckpointForTests(checkpointId)).toEqual(boundary);
    });

    it('restores a natural boundary and shuffles once without double-incrementing epoch', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(10);
        const completed = await workerApi.stepExperimentV2(5);
        const checkpointId = completed.checkpointTimeline.liveCheckpointId!;
        const boundary = getV2CheckpointForTests(checkpointId);
        expect(boundary.cursor.batchStart).toBe(boundary.evaluation.dataset.trainCount);
        const revisionBeforeRestore = completed.evidence.latestEvaluation!.model.revision;

        const restored = await workerApi.restoreCheckpointV2({
            type: 'restore-checkpoint',
            protocolVersion: WORKER_PROTOCOL_VERSION,
            requestId: nextRequestId++,
            checkpointId,
        });
        expect(restored.evidence.latestEvaluation?.model).toMatchObject({
            revision: revisionBeforeRestore + 1,
            step: 15,
            epoch: 1,
        });

        const shuffleSpy = vi.spyOn(PRNG.prototype, 'shuffle');
        const trainSpy = vi.spyOn(Network.prototype, 'trainBatchIndexedV2');
        const stepped = await workerApi.stepExperimentV2(1);
        expect(shuffleSpy).toHaveBeenCalledTimes(1);
        expect(trainSpy.mock.calls.at(-1)?.[3]).toBe(0);
        expect(stepped.evidence.liveSignal?.model.epoch).toBe(1);
    });

    it.each(['recipe', 'datasetKey', 'objectiveKey'] as const)(
        'rejects forged %s evidence before V2 allocation or current-run mutation',
        async (kind) => {
            const fixtures = await createScientificTrustFixtures();
            const current = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            const beforeAllocations = getV2AllocationCountForTests();
            const forged = withFreshId(fixtures.forged[kind]);

            await expect(workerApi.initializeExperimentV2(forged)).rejects.toMatchObject({
                name: 'ExperimentTransactionError',
            });

            expect(getV2AllocationCountForTests()).toBe(beforeAllocations);
            expect(workerApi.getMetricHistoryV2().evaluationHistory[0]?.model.generationId)
                .toBe(current.runId);
        },
    );

    it.each([
        ['canonicalRecipeKey', 'not-the-canonical-recipe'],
        ['recipeFingerprint', `r2.1.${'A'.repeat(43)}`],
    ] as const)(
        'rejects a forged %s before allocation',
        async (key, forgedValue) => {
            const fixtures = await createScientificTrustFixtures();
            const beforeAllocations = getV2AllocationCountForTests();
            const forged = withFreshId({
                ...fixtures.request,
                claimedIdentities: {
                    ...fixtures.request.claimedIdentities,
                    [key]: forgedValue,
                },
            });

            await expect(workerApi.initializeExperimentV2(forged)).rejects.toMatchObject({
                name: 'ExperimentTransactionError',
            });
            expect(getV2AllocationCountForTests()).toBe(beforeAllocations);
        },
    );

    it('rejects a stale request without replacing the active generation', async () => {
        const fixtures = await createScientificTrustFixtures();
        const request = withFreshId(fixtures.request);
        const current = await workerApi.initializeExperimentV2(request);
        const beforeAllocations = getV2AllocationCountForTests();

        await expect(workerApi.initializeExperimentV2(request)).rejects.toMatchObject({
            code: 'stale-request',
            requestId: request.requestId,
        });
        expect(getV2AllocationCountForTests()).toBe(beforeAllocations);
        expect(workerApi.getMetricHistoryV2().evaluationHistory.at(-1)?.model.generationId)
            .toBe(current.runId);
    });

    it('records every live batch and consumes the step-50 cadence inside one burst', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 49 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        let history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory).toHaveLength(49);
        expect(history.evaluationHistory.map((point) => point.trigger)).toEqual([
            'initial',
            ...Array.from({ length: 9 }, () => 'checkpoint' as const),
        ]);

        capture.dispatch({ type: 'updateSpeed', stepsPerFrame: 1 });
        capture.dispatch({ type: 'frameAck' });
        await vi.advanceTimersByTimeAsync(20);

        history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory).toHaveLength(50);
        expect(history.evaluationHistory.map((point) => point.trigger)).toEqual([
            'initial',
            ...Array.from({ length: 9 }, () => 'checkpoint' as const),
            'cadence',
            'checkpoint',
        ]);
        expect(history.evaluationHistory.find(
            (point) => point.trigger === 'cadence',
        )?.model).toEqual(history.trendHistory[49]?.model);
        const beforePause = evidenceMessages(capture.messages);
        expect(beforePause.filter((message) => message.latestEvaluation?.trigger === 'cadence'))
            .toHaveLength(1);
        expect(beforePause.filter((message) => message.liveSignal)).toHaveLength(2);
        expect(beforePause.filter((message) => message.latestEvaluation?.trigger === 'checkpoint'))
            .toHaveLength(10);
        expect(beforePause.length).toBeLessThanOrEqual(15);
        expect(beforePause.every(isWorkerToMainMessage)).toBe(true);

        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
        history = workerApi.getMetricHistoryV2();
        const pause = history.evaluationHistory.at(-1)!;
        expect(pause.trigger).toBe('pause');
        expect(pause.model).toEqual(history.trendHistory.at(-1)?.model);
        expect(evidenceMessages(capture.messages).at(-1)?.latestEvaluation).toEqual(pause);
    });

    it('forces current manual-step evidence and resets to a fresh generation', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initial = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const stepped = await workerApi.stepExperimentV2(1);

        expect(stepped.runId).toBe(initial.runId);
        expect(stepped.evidence.latestEvaluation).toMatchObject({
            trigger: 'manual-step',
            model: { generationId: initial.runId, step: 1 },
        });
        expect(stepped.evidence.liveSignal?.model)
            .toEqual(stepped.evidence.latestEvaluation?.model);
        expect(stepped.snapshot.historyPoint).toBeUndefined();
        expect(Object.prototype.hasOwnProperty.call(stepped.snapshot, 'historyPoint')).toBe(false);

        const reset = await workerApi.resetExperimentV2();
        expect(reset.runId).toBe(initial.runId + 1);
        expect(reset.evidence.latestEvaluation).toMatchObject({
            trigger: 'initial',
            evaluationId: 1,
            model: {
                generationId: reset.runId,
                revision: 0,
                step: 0,
                epoch: 0,
            },
        });
        expect(reset.snapshot.historyPoint).toBeUndefined();
        expect(Object.prototype.hasOwnProperty.call(reset.snapshot, 'historyPoint')).toBe(false);
        expect(workerApi.getMetricHistoryV2().trendHistory).toEqual([]);
    });

    it('returns exact direct artifact provenance and omits cadence-reused binary grids', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: true,
            needLayerStats: true,
            needActivationHistograms: true,
            needConfusionMatrix: true,
            gridInterval: 2,
            activationHistogramInterval: 2,
        });

        const produced = await workerApi.stepExperimentV2(1);
        const pair = produced.evidence.latestEvaluation!;
        const predictionBasis = {
            kind: 'prediction-grid',
            pointCount: produced.snapshot.gridSize * produced.snapshot.gridSize,
            domain: [-1, 1, -1, 1],
        } as const;
        expect(produced.snapshot.outputGrid).toHaveLength(predictionBasis.pointCount);
        expect(produced.snapshot.neuronGrids).toBeInstanceOf(Float32Array);
        expect(produced.snapshot.activationHistograms?.bins).toBeInstanceOf(Float32Array);
        expect(produced.snapshot.layerStats?.length).toBeGreaterThan(0);
        expect(produced.snapshot.testMetrics.confusionMatrix)
            .toEqual(pair.test.values.confusionMatrix);
        expect(produced.artifacts?.decisionBoundary).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: predictionBasis,
        });
        expect(produced.artifacts?.neuronGrids).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: predictionBasis,
        });
        expect(produced.artifacts?.activationStatistics).toMatchObject({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: {
                kind: 'bounded-sample',
                split: 'train',
                sampleCount: Math.min(128, pair.dataset.trainCount),
                populationCount: pair.dataset.trainCount,
            },
        });
        expect(produced.artifacts?.activationHistogram).toEqual(
            produced.artifacts?.activationStatistics,
        );
        expect(produced.artifacts?.confusionMatrix).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: pair.test.basis,
        });
        expect(Number.isSafeInteger(produced.layerStatsGradientRevision)).toBe(true);
        expect(produced.layerStatsGradientRevision).toBeGreaterThanOrEqual(0);
        expect(produced.layerStatsGradientRevision).toBeLessThanOrEqual(pair.model.revision);

        const reused = await workerApi.stepExperimentV2(1);
        expect(reused.snapshot.outputGrid).toHaveLength(0);
        expect(reused.snapshot.neuronGrids).toBeUndefined();
        expect(reused.snapshot.activationHistograms).toBeUndefined();
        expect(reused.artifacts?.decisionBoundary).toBeUndefined();
        expect(reused.artifacts?.neuronGrids).toBeUndefined();
        expect(reused.artifacts?.activationHistogram).toBeUndefined();
        expect(reused.artifacts?.activationStatistics).toBeDefined();
        expect(reused.artifacts?.confusionMatrix).toBeDefined();
    });

    it('returns current direct provenance for multiclass boundary and confusion artifacts', async () => {
        const multiclassPrepared = PREPARED_PRESETS.find(
            (entry) => entry.id === 'three-class-clusters',
        )?.prepared;
        if (!multiclassPrepared) throw new Error('missing three-class recipe');
        await workerApi.initializeExperimentV2(requestForPrepared(multiclassPrepared));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: true,
            needConfusionMatrix: true,
            gridInterval: 1,
        });

        const produced = await workerApi.stepExperimentV2(1);
        const pair = produced.evidence.latestEvaluation!;
        expect(produced.snapshot.outputGrid).toHaveLength(0);
        expect(produced.snapshot.multiclassBoundary?.classGrid).toHaveLength(
            produced.snapshot.gridSize * produced.snapshot.gridSize,
        );
        expect(produced.snapshot.multiclassBoundary?.confidenceGrid).toHaveLength(
            produced.snapshot.gridSize * produced.snapshot.gridSize,
        );
        expect(produced.snapshot.neuronGrids).toBeUndefined();
        expect(produced.snapshot.testMetrics.multiclassConfusionMatrix)
            .toEqual(pair.test.values.confusionMatrix);
        expect(produced.artifacts?.decisionBoundary).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: {
                kind: 'prediction-grid',
                pointCount: produced.snapshot.gridSize * produced.snapshot.gridSize,
                domain: [-1, 1, -1, 1],
            },
        });
        expect(produced.artifacts?.neuronGrids).toBeUndefined();
        expect(produced.artifacts?.confusionMatrix).toEqual({
            model: pair.model,
            dataset: pair.dataset,
            objectiveKey: pair.objectiveKey,
            basis: pair.test.basis,
        });
    });

    it('discards a deferred GPU readback after a training mutation and recomputes on CPU', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('crossOriginIsolated', false);
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            gridInterval: 1,
        });
        workerApi.setWebGpuEnabled(true);
        const gpu = createDeferredGpuReadback();
        setGpuPredictorForTests(gpu.predictor);
        const cpu = vi.spyOn(Network.prototype, 'predictGridWithNeuronsInto');

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        vi.advanceTimersByTime(20);
        await flushMicrotasks();
        expect(gpu.predictor.predictGridWithNeuronsInto).toHaveBeenCalledTimes(1);

        const current = await workerApi.stepExperimentV2(1);
        cpu.mockClear();
        gpu.release();
        await gpu.completion;
        await flushMicrotasks();

        const snapshot = snapshotMessages(capture.messages).at(-1);
        expect(cpu).toHaveBeenCalledTimes(1);
        expect(snapshot?.artifacts?.decisionBoundary?.model).toEqual(
            current.evidence.latestEvaluation?.model,
        );
        expect(snapshot?.outputGrid).toBeInstanceOf(Float32Array);
        expect(snapshot?.outputGrid?.some((value) => value === gpu.staleValue)).toBe(false);
        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it('discards a deferred output-only GPU readback after neuron demand changes', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('crossOriginIsolated', false);
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needNeuronGrids: false,
            gridInterval: 1,
        });
        workerApi.setWebGpuEnabled(true);
        const gpu = createDeferredGpuReadback();
        setGpuPredictorForTests(gpu.predictor);
        const cpu = vi.spyOn(Network.prototype, 'predictGridWithNeuronsInto');

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        vi.advanceTimersByTime(20);
        await flushMicrotasks();
        expect(gpu.predictor.predictGridInto).toHaveBeenCalledTimes(1);
        expect(gpu.predictor.predictGridWithNeuronsInto).not.toHaveBeenCalled();

        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needNeuronGrids: true,
            gridInterval: 1,
        });
        gpu.release();
        await gpu.completion;
        await flushMicrotasks();

        const snapshot = snapshotMessages(capture.messages).at(-1);
        expect(cpu).toHaveBeenCalledTimes(1);
        expect(snapshot?.outputGrid?.some((value) => value === gpu.staleValue)).toBe(false);
        expect(snapshot?.neuronGrids).toBeInstanceOf(Float32Array);
        expect(snapshot?.artifacts?.decisionBoundary).toBeDefined();
        expect(snapshot?.artifacts?.neuronGrids).toBeDefined();
        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it('does not install a GPU predictor when the toggle changes during device detection', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('crossOriginIsolated', false);
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({ ...DEFAULT_DEMAND, gridInterval: 1 });
        workerApi.setWebGpuEnabled(true);
        const detection = deferred<GPUDevice | null>();
        const create = vi.fn();
        setGpuInitializationForTests({
            detect: () => detection.promise,
            create,
        });
        const cpu = vi.spyOn(Network.prototype, 'predictGridWithNeuronsInto');

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        vi.advanceTimersByTime(20);
        await flushMicrotasks();
        workerApi.setWebGpuEnabled(false);
        detection.resolve({} as GPUDevice);
        await detection.promise;
        await flushMicrotasks();

        expect(create).not.toHaveBeenCalled();
        expect(cpu).toHaveBeenCalledTimes(1);
        const snapshot = snapshotMessages(capture.messages).at(-1);
        expect(snapshot?.artifacts?.decisionBoundary).toBeDefined();
        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it('discards an old-generation GPU readback without disturbing new readiness', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('crossOriginIsolated', false);
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            gridInterval: 1,
        });
        workerApi.setWebGpuEnabled(true);
        const gpu = createDeferredGpuReadback();
        setGpuPredictorForTests(gpu.predictor);
        const cpu = vi.spyOn(Network.prototype, 'predictGridWithNeuronsInto');

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        vi.advanceTimersByTime(20);
        await flushMicrotasks();
        expect(gpu.predictor.predictGridWithNeuronsInto).toHaveBeenCalledTimes(1);

        const snapshotsBeforeReset = snapshotMessages(capture.messages).length;
        const reset = await workerApi.resetExperimentV2();
        cpu.mockClear();
        gpu.release();
        await gpu.completion;
        await flushMicrotasks();

        expect(snapshotMessages(capture.messages)).toHaveLength(snapshotsBeforeReset);
        expect(cpu).not.toHaveBeenCalled();
        workerApi.setWebGpuEnabled(false);
        workerApi.updateDemand({ ...DEFAULT_DEMAND, gridInterval: 1 });
        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);
        const [snapshot] = snapshotMessages(capture.messages);
        expect(snapshot?.runId).toBe(reset.runId);
        expect(snapshot?.artifacts?.decisionBoundary?.model.generationId).toBe(reset.runId);
        expect(cpu).toHaveBeenCalledTimes(1);
        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it('publishes complete binary and multiclass task metrics without legacy loss math', async () => {
        const fixtures = await createScientificTrustFixtures();
        const binary = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const binaryPair = binary.evidence.latestEvaluation!;

        for (const side of [binaryPair.train, binaryPair.test]) {
            expect(side.values.accuracy).toBeGreaterThanOrEqual(0);
            expect(side.values.accuracy).toBeLessThanOrEqual(1);
            const matrix = side.values.confusionMatrix as { tp: number; tn: number; fp: number; fn: number };
            expect(matrix.tp + matrix.tn + matrix.fp + matrix.fn).toBe(side.basis.sampleCount);
            expect(Number.isFinite(side.values.dataLoss)).toBe(true);
        }

        const multiclassPrepared = PREPARED_PRESETS.find(
            (entry) => entry.id === 'three-class-clusters',
        )?.prepared;
        if (!multiclassPrepared) throw new Error('missing three-class recipe');
        const multiclass = await workerApi.initializeExperimentV2(
            requestForPrepared(multiclassPrepared),
        );
        const multiclassPair = multiclass.evidence.latestEvaluation!;

        for (const side of [multiclassPair.train, multiclassPair.test]) {
            const matrix = side.values.confusionMatrix as {
                classCount: number;
                counts: readonly number[];
            };
            expect(matrix.classCount).toBe(3);
            expect(matrix.counts).toHaveLength(9);
            expect(matrix.counts.reduce((sum, count) => sum + count, 0))
                .toBe(side.basis.sampleCount);
            expect(Number.isFinite(side.values.dataLoss)).toBe(true);
        }
    });

    it('keeps every published history value finite', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(7);

        const history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory.every((point) => Number.isFinite(point.dataLoss))).toBe(true);
        expect(history.evaluationHistory.every((point) => (
            Number.isFinite(point.train.values.dataLoss)
            && Number.isFinite(point.test.values.dataLoss)
            && Number.isFinite(point.objective.regularizationPenalty)
            && Number.isFinite(point.objective.trainTotalObjective)
        ))).toBe(true);
    });

    it('captures one worker-owned save pair with bounded same-generation histories', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(3);

        const record = await workerApi.captureRunArtifact({
            id: '00000000-0000-0000-0000-000000000009',
            createdAt: '2026-07-11T12:00:00.000Z',
            updatedAt: '2026-07-11T12:00:00.000Z',
            title: 'Worker evidence',
        });

        expect(record).toMatchObject({
            kind: 'nn-playground-run',
            schemaVersion: 2,
            recipe: fixtures.prepared.document.recipe,
            recipeFingerprint: fixtures.prepared.identities.recipeFingerprint,
            snapshot: {
                model: { generationId: initialized.runId, revision: 3, step: 3 },
                evaluation: {
                    trigger: 'save',
                    model: { generationId: initialized.runId, revision: 3, step: 3 },
                    dataset: { datasetKey: fixtures.prepared.identities.datasetKey },
                    objectiveKey: fixtures.prepared.identities.objectiveKey,
                },
            },
        });
        expect(record.snapshot.evaluationHistory.at(-1)).toEqual(record.snapshot.evaluation);
        expect(record.snapshot.trendHistory.length).toBeLessThanOrEqual(512);
        expect(record.snapshot.evaluationHistory.length).toBeLessThanOrEqual(256);
        expect(record.snapshot.trendHistory.every(
            (point) => point.model.generationId === initialized.runId,
        )).toBe(true);
        expect(record.snapshot.evaluationHistory.every(
            (point) => point.model.generationId === initialized.runId,
        )).toBe(true);
        expect(JSON.stringify(record)).not.toMatch(/weights|biases|parameters/i);
    });

    it('snapshots capture evidence before asynchronous validation can interleave a later step', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        await workerApi.stepExperimentV2(1);

        const pending = workerApi.captureRunArtifact({
            id: '00000000-0000-0000-0000-000000000010',
            createdAt: '2026-07-11T12:00:00.000Z',
            updatedAt: '2026-07-11T12:00:00.000Z',
        });
        const later = workerApi.stepExperimentV2(1);
        const record = await pending;
        const laterResult = await later;

        expect(record.snapshot.model.revision).toBe(1);
        expect(laterResult.evidence.latestEvaluation?.model.revision).toBe(2);
        expect(record.snapshot.evaluationHistory.at(-1)).toEqual(record.snapshot.evaluation);
    });

    it('queues capture behind an earlier delayed initialization and snapshots its generation', async () => {
        const fixtures = await createScientificTrustFixtures();
        const previous = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const preparation = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const prepare = vi.fn(() => preparation.promise);
        setV2PrepareForTests(prepare);

        try {
            const initialization = workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            await vi.waitFor(() => expect(prepare).toHaveBeenCalledTimes(1));
            let captureSettled = false;
            const capture = workerApi.captureRunArtifact({
                id: '00000000-0000-0000-0000-000000000011',
                createdAt: '2026-07-11T12:00:00.000Z',
                updatedAt: '2026-07-11T12:00:00.000Z',
            }).finally(() => { captureSettled = true; });

            await flushMicrotasks();
            expect(captureSettled).toBe(false);
            preparation.resolve({ ok: true, value: fixtures.prepared });

            const initialized = await initialization;
            const record = await capture;
            expect(initialized.runId).toBe(previous.runId + 1);
            expect(record.snapshot.model.generationId).toBe(initialized.runId);
            expect(record.snapshot.evaluation.model.generationId).toBe(initialized.runId);
        } finally {
            setV2PrepareForTests();
        }
    });

    it('queues a V2 stream pause behind an earlier delayed initialization', async () => {
        const fixtures = await createScientificTrustFixtures();
        const stream = createCapturingPort();
        workerApi.setStreamPort(stream.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const preparation = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const prepare = vi.fn(() => preparation.promise);
        setV2PrepareForTests(prepare);

        try {
            const initialization = workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            await vi.waitFor(() => expect(prepare).toHaveBeenCalledTimes(1));
            stream.dispatch({ type: 'stopTraining' });
            await flushMicrotasks();
            expect(stream.messages).not.toContainEqual(expect.objectContaining({
                type: 'status',
                status: 'paused',
            }));

            preparation.resolve({ ok: true, value: fixtures.prepared });
            const initialized = await initialization;
            await flushMicrotasks();
            expect(stream.messages).toContainEqual(expect.objectContaining({
                type: 'status',
                status: 'paused',
                runId: initialized.runId,
            }));
        } finally {
            setV2PrepareForTests();
        }
    });

    it('continues the V2 mutation queue after a queued mutation rejects', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));

        const rejected = workerApi.forceEvaluationV2('cadence' as never);
        const stepped = workerApi.stepExperimentV2(1);

        await expect(rejected).rejects.toThrow(/unsupported forced v2 evaluation trigger/i);
        await expect(stepped).resolves.toMatchObject({
            runId: initialized.runId,
            evidence: { latestEvaluation: { model: { revision: 1 } } },
        });
    });

    it('rejects malformed capture metadata before changing worker history', async () => {
        const fixtures = await createScientificTrustFixtures();
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const before = workerApi.getMetricHistoryV2();

        await expect(workerApi.captureRunArtifact({
            id: 'not-a-uuid',
            createdAt: '2026-07-11T12:00:00.000Z',
            updatedAt: '2026-07-11T12:00:00.000Z',
        })).rejects.toThrow(/capture run artifact request/i);

        expect(workerApi.getMetricHistoryV2()).toEqual(before);
    });

    it('forces and publishes an exact current pair before a comparison stop condition', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        setRuntimeStopConditionsForTests([
            { kind: 'target', metric: 'trainObjective', threshold: 1_000_000 },
        ]);

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        const history = workerApi.getMetricHistoryV2();
        const live = history.trendHistory.at(-1)!;
        const pair = history.evaluationHistory.at(-1)!;
        expect(pair).toMatchObject({
            trigger: 'stop-condition',
            model: live.model,
        });
        expect(pair.model.generationId).toBe(initialized.runId);
        expect(evidenceMessages(capture.messages).some(
            (message) => message.latestEvaluation?.evaluationId === pair.evaluationId,
        )).toBe(true);
        expect(capture.messages).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'target-loss-reached',
        }));
    });

    it('reuses an exact step-50 cadence pair instead of publishing a duplicate stop pair', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        setRuntimeStopConditionsForTests([
            { kind: 'target', metric: 'testDataLoss', threshold: -1 },
        ]);

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 49 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);
        capture.dispatch({ type: 'updateSpeed', stepsPerFrame: 1 });
        capture.dispatch({ type: 'frameAck' });
        await vi.advanceTimersByTimeAsync(20);

        const step50Pairs = workerApi.getMetricHistoryV2().evaluationHistory.filter(
            (evaluation) => evaluation.model.step === 50,
        );
        const step50Evidence = evidenceMessages(capture.messages).filter(
            (message) => message.latestEvaluation?.model.step === 50,
        );
        expect(step50Pairs.map((evaluation) => evaluation.trigger)).toEqual([
            'cadence',
            'checkpoint',
        ]);
        expect(step50Evidence).toHaveLength(2);

        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it('stops non-finite batch objectives as structured divergence without evidence or checkpoints', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const beforeHistory = workerApi.getMetricHistoryV2();
        const beforeMessages = capture.messages.length;
        const original = Network.prototype.trainBatchIndexedV2;
        vi.spyOn(Network.prototype, 'trainBatchIndexedV2').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['trainBatchIndexedV2']>
        ) {
            const result = original.apply(this, args);
            return {
                ...result,
                objective: {
                    ...result.objective,
                    dataLoss: Number.NaN,
                    totalObjective: Number.NaN,
                },
            };
        });

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        expect(workerApi.getMetricHistoryV2()).toEqual(beforeHistory);
        expect(workerApi.getCheckpointTimeline().checkpoints).toEqual(
            expect.arrayContaining([expect.objectContaining({ id: 1, step: 0 })]),
        );
        const emitted = capture.messages.slice(beforeMessages);
        expect(evidenceMessages(emitted)).toEqual([]);
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'worker-error',
            code: 'runtime-failure',
            source: 'training',
            path: '$.objective.dataLoss',
        }));
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'diverged',
        }));
    });

    it('translates a real engine training overflow without publishing evidence or checkpoints', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const beforeHistory = workerApi.getMetricHistoryV2();
        const beforeMessages = capture.messages.length;
        setV2OutputOverflowForTests();

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        expect(workerApi.getMetricHistoryV2()).toEqual(beforeHistory);
        expect(workerApi.getCheckpointTimeline().checkpoints).toEqual(
            expect.arrayContaining([expect.objectContaining({ id: 1, step: 0 })]),
        );
        const emitted = capture.messages.slice(beforeMessages);
        expect(evidenceMessages(emitted)).toEqual([]);
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'worker-error',
            source: 'training',
            path: '$.training.logits[0]',
        }));
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'diverged',
        }));
    });

    it('translates a real pause evaluation overflow into terminal divergence', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const beforeHistory = workerApi.getMetricHistoryV2();
        const beforeMessages = capture.messages.length;
        vi.spyOn(Network.prototype, 'evaluateObjective').mockImplementation(() => {
            softmax([Number.POSITIVE_INFINITY]);
            throw new Error('unreachable');
        });

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();

        expect(workerApi.getMetricHistoryV2()).toEqual(beforeHistory);
        const emitted = capture.messages.slice(beforeMessages);
        expect(evidenceMessages(emitted)).toEqual([]);
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'worker-error',
            source: 'evaluation',
            code: 'evaluation-failed',
            path: '$.evaluation.train.logits[0]',
        }));
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'diverged',
        }));
    });

    it('stops non-finite paired evaluation as structured divergence before evidence publication', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const beforeHistory = workerApi.getMetricHistoryV2();
        const beforeMessages = capture.messages.length;
        const original = Network.prototype.evaluateObjective;
        vi.spyOn(Network.prototype, 'evaluateObjective').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['evaluateObjective']>
        ) {
            const result = original.apply(this, args);
            return {
                ...result,
                dataLoss: Number.POSITIVE_INFINITY,
                totalObjective: Number.POSITIVE_INFINITY,
            };
        });

        await expect(workerApi.forceEvaluationV2('stop-condition'))
            .rejects.toThrow('terminal divergence');

        expect(workerApi.getMetricHistoryV2()).toEqual(beforeHistory);
        expect(workerApi.getCheckpointTimeline().checkpoints).toEqual(
            expect.arrayContaining([expect.objectContaining({ id: 1, step: 0 })]),
        );
        const emitted = capture.messages.slice(beforeMessages);
        expect(evidenceMessages(emitted)).toEqual([]);
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'worker-error',
            code: 'evaluation-failed',
            source: 'evaluation',
            path: '$.evaluation.train.values.dataLoss',
        }));
        expect(emitted).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'diverged',
        }));
    });

    it('rejects non-finite V2 compatibility scalars before posting a snapshot', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            trainEvalInterval: 1,
            testEvalInterval: 1,
        });
        const original = Network.prototype.evaluate;
        vi.spyOn(Network.prototype, 'evaluate').mockImplementation(function (
            this: Network,
            ...args: Parameters<Network['evaluate']>
        ) {
            return { ...original.apply(this, args), loss: Number.NaN };
        });

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        expect(capture.messages.some((message) => (
            typeof message === 'object'
            && message !== null
            && (message as { type?: unknown }).type === 'snapshot'
        ))).toBe(false);
        expect(capture.messages).toContainEqual(expect.objectContaining({
            type: 'worker-error',
            source: 'artifact',
            code: 'artifact-failed',
            path: '$.snapshot.trainLoss',
        }));
        expect(capture.messages).toContainEqual(expect.objectContaining({
            type: 'status',
            status: 'paused',
            pauseReason: 'diverged',
        }));
        expect(workerApi.getMetricHistoryV2().trendHistory).toHaveLength(1);
        expect(evidenceMessages(capture.messages).every((message) => (
            message.liveSignal === undefined || Number.isFinite(message.liveSignal.dataLoss)
        ))).toBe(true);
    });

    it('publishes strict artifact provenance and deterministic bounded layer statistics', async () => {
        vi.useFakeTimers();
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        workerApi.updateDemand({
            ...DEFAULT_DEMAND,
            needDecisionBoundary: true,
            needNeuronGrids: false,
            needLayerStats: true,
            needActivationHistograms: true,
            needConfusionMatrix: true,
            gridInterval: 1,
            activationHistogramInterval: 1,
        });
        const grid = vi.spyOn(Network.prototype, 'predictGridInto');
        const statistics = vi.spyOn(Network.prototype, 'computeLayerStatistics');

        capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
        await flushMicrotasks();
        await vi.advanceTimersByTimeAsync(20);

        const snapshot = capture.messages.find((message): message is WorkerSnapshotMessage => (
            typeof message === 'object'
            && message !== null
            && (message as { type?: unknown }).type === 'snapshot'
        ));
        const populationCount = initialized.evidence.latestEvaluation!.dataset.trainCount;
        const sampleCount = Math.min(128, populationCount);
        expect(snapshot).toBeDefined();
        expect(snapshot?.protocolVersion).toBe(2);
        expect(snapshot?.historyPoint).toBeUndefined();
        expect(Object.prototype.hasOwnProperty.call(snapshot, 'historyPoint')).toBe(false);
        expect(snapshot?.layerStats?.length).toBeGreaterThan(0);
        expect(snapshot?.artifacts?.activationStatistics).toMatchObject({
            model: {
                generationId: initialized.runId,
                revision: 1,
                step: 1,
            },
            dataset: { trainCount: populationCount },
            objectiveKey: fixtures.prepared.identities.objectiveKey,
            basis: {
                kind: 'bounded-sample',
                split: 'train',
                sampleCount,
                populationCount,
            },
        });
        expect(snapshot?.artifacts?.decisionBoundary?.basis).toEqual({
            kind: 'prediction-grid',
            pointCount: snapshot!.scalars.gridSize ** 2,
            domain: [-1, 1, -1, 1],
        });
        expect(snapshot?.artifacts?.activationHistogram?.basis).toEqual({
            kind: 'bounded-sample',
            split: 'train',
            sampleCount,
            populationCount,
        });
        expect(snapshot?.artifacts?.confusionMatrix?.basis).toEqual({
            kind: 'full-split',
            split: 'test',
            sampleCount: initialized.evidence.latestEvaluation!.dataset.testCount,
            populationCount: initialized.evidence.latestEvaluation!.dataset.testCount,
        });
        expect(snapshot?.confusionMatrix).toEqual(
            initialized.evidence.latestEvaluation!.test.values.confusionMatrix,
        );
        expect(snapshot?.artifacts?.confusionMatrix?.model).toEqual(
            initialized.evidence.latestEvaluation!.model,
        );
        expect(snapshot?.confusionMatrixEvaluationId).toBe(
            initialized.evidence.latestEvaluation!.evaluationId,
        );
        expect(snapshot?.layerStatsGradientRevision).toBe(1);
        expect(statistics).toHaveBeenCalledTimes(1);
        expect(statistics.mock.calls[0]?.[0]).toHaveLength(populationCount);
        expect(statistics.mock.calls[0]?.[1]).toBe(128);
        expect(grid).toHaveBeenCalled();
        expect(grid.mock.invocationCallOrder[0]).toBeLessThan(
            statistics.mock.invocationCallOrder[0],
        );

        capture.dispatch({ type: 'frameAck' });
        await vi.advanceTimersByTimeAsync(20);
        const snapshots = capture.messages.filter((message): message is WorkerSnapshotMessage => (
            typeof message === 'object'
            && message !== null
            && (message as { type?: unknown }).type === 'snapshot'
        ));
        expect(snapshots).toHaveLength(2);
        expect(snapshots[1]?.confusionMatrix).toBeUndefined();
        expect(snapshots[1]?.multiclassConfusionMatrix).toBeUndefined();
        expect(snapshots[1]?.artifacts?.confusionMatrix).toBeUndefined();

        capture.dispatch({ type: 'stopTraining' });
        await flushMicrotasks();
    });

    it.each(['sampleCount', 'populationCount'] as const)(
        'rejects mismatched layer-statistics %s before publishing provenance',
        async (field) => {
            vi.useFakeTimers();
            const fixtures = await createScientificTrustFixtures();
            const capture = createCapturingPort();
            workerApi.setStreamPort(capture.port);
            await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            workerApi.updateDemand({
                ...DEFAULT_DEMAND,
                needDecisionBoundary: false,
                needNeuronGrids: false,
                needLayerStats: true,
                needActivationHistograms: false,
                needConfusionMatrix: false,
            });
            const original = Network.prototype.computeLayerStatistics;
            vi.spyOn(Network.prototype, 'computeLayerStatistics').mockImplementation(function (
                this: Network,
                inputs: Parameters<Network['computeLayerStatistics']>[0],
                maxSamples?: number,
            ) {
                const result = original.call(this, inputs, maxSamples);
                return { ...result, [field]: result[field] + 1 };
            });

            capture.dispatch({ type: 'startTraining', stepsPerFrame: 1 });
            await flushMicrotasks();
            await vi.advanceTimersByTimeAsync(20);

            expect(snapshotMessages(capture.messages)).toHaveLength(0);
            expect(capture.messages).toContainEqual(expect.objectContaining({
                type: 'worker-error',
                source: 'runtime',
                code: 'runtime-failure',
                message: 'Layer statistics sample basis does not match the training population.',
            }));
        },
    );

    it('wraps the shuffle seed for the maximum allowed uint32 data seed', async () => {
        const fixtures = await createScientificTrustFixtures();
        const prepared = await prepareExperimentDocument({
            ...fixtures.prepared.document,
            recipe: {
                ...fixtures.prepared.document.recipe,
                data: {
                    ...fixtures.prepared.document.recipe.data,
                    seed: 0xffff_ffff,
                },
            },
        });
        if (!prepared.ok) throw new Error('maximum uint32 seed must prepare');

        const initialized = await workerApi.initializeExperimentV2(
            requestForPrepared(prepared.value),
        );
        const stepped = await workerApi.stepExperimentV2(1);

        expect(initialized.evidence.latestEvaluation?.model.revision).toBe(0);
        expect(stepped.evidence.liveSignal?.model).toMatchObject({
            generationId: initialized.runId,
            revision: 1,
            step: 1,
        });
    });

    it('rejects legacy model mutators without changing active V2 provenance', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const before = workerApi.getMetricHistoryV2();

        expect(() => workerApi.updateConfig(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA },
            { ...DEFAULT_FEATURES },
            false,
        )).toThrow('unavailable while a V2 experiment is active');
        expect(() => workerApi.step(1)).toThrow('unavailable while a V2 experiment is active');
        expect(() => workerApi.reset()).toThrow('unavailable while a V2 experiment is active');
        expect(() => workerApi.restoreCheckpoint(1))
            .toThrow('unavailable while a V2 experiment is active');

        expect(workerApi.getMetricHistoryV2()).toEqual(before);
        expect(workerApi.getCheckpointTimeline().checkpoints).toEqual(
            expect.arrayContaining([expect.objectContaining({ id: 1, step: 0 })]),
        );
        const validStep = await workerApi.stepExperimentV2(1);
        expect(validStep.runId).toBe(initialized.runId);
        expect(validStep.evidence.liveSignal?.model).toMatchObject({
            generationId: initialized.runId,
            revision: 1,
            step: 1,
        });
        expect(workerApi.getMetricHistoryV2().trendHistory).toHaveLength(1);
    });

    it('streams structured current-generation errors without failed evidence', async () => {
        const fixtures = await createScientificTrustFixtures();
        const capture = createCapturingPort();
        workerApi.setStreamPort(capture.port);
        const initialized = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const before = workerApi.getMetricHistoryV2();

        await expect(workerApi.forceEvaluationV2('cadence' as never))
            .rejects.toThrow('Unsupported forced V2 evaluation trigger');

        expect(workerApi.getMetricHistoryV2()).toEqual(before);
        expect(capture.messages.at(-1)).toMatchObject({
            type: 'worker-error',
            protocolVersion: 2,
            requestId: null,
            generationId: initialized.runId,
            code: 'evaluation-failed',
            source: 'evaluation',
        });

        await expect(workerApi.initializeExperimentV2(
            withFreshId(fixtures.forged.datasetKey),
        )).rejects.toMatchObject({ code: 'identity-mismatch' });
        expect(capture.messages.at(-1)).toMatchObject({
            type: 'worker-error',
            generationId: null,
            code: 'identity-mismatch',
            source: 'preparation',
        });
        expect(workerApi.getMetricHistoryV2()).toEqual(before);
    });

    it.each([11, 1_000_000])(
        'rejects %i manual steps before mutation or message flooding',
        async (iterations) => {
            const fixtures = await createScientificTrustFixtures();
            const capture = createCapturingPort();
            workerApi.setStreamPort(capture.port);
            await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            const before = workerApi.getMetricHistoryV2();
            const beforeMessages = capture.messages.length;

            await expect(workerApi.stepExperimentV2(iterations)).rejects.toThrow('from 1 to 10');

            expect(workerApi.getMetricHistoryV2()).toEqual(before);
            expect(capture.messages.length - beforeMessages).toBeLessThanOrEqual(1);
        },
    );

    it('orders a V2 step after an earlier delayed initialization', async () => {
        const fixtures = await createScientificTrustFixtures();
        const active = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const preparation = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const prepare = vi.fn(() => preparation.promise);
        setV2PrepareForTests(prepare);
        const allocationsBefore = getV2AllocationCountForTests();

        try {
            const pending = workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            await vi.waitFor(() => expect(prepare).toHaveBeenCalledTimes(1));
            const stepped = workerApi.stepExperimentV2(1);
            preparation.resolve({ ok: true, value: fixtures.prepared });

            const initialized = await pending;
            const steppedResult = await stepped;
            expect(getV2AllocationCountForTests()).toBe(allocationsBefore + 1);
            expect(initialized.runId).toBe(active.runId + 1);
            expect(steppedResult.runId).toBe(initialized.runId);
            expect(workerApi.getMetricHistoryV2().trendHistory.at(-1)?.model).toEqual(
                steppedResult.evidence.liveSignal?.model,
            );
            expect(workerApi.getMetricHistoryV2().evaluationHistory.at(-1)?.model.generationId)
                .toBe(initialized.runId);
        } finally {
            setV2PrepareForTests();
        }
    });

    it('prevents delayed V2 initialization from overwriting a newer legacy step', async () => {
        const fixtures = await createScientificTrustFixtures();
        const legacy = workerApi.initialize(
            { ...DEFAULT_NETWORK },
            { ...DEFAULT_TRAINING },
            { ...DEFAULT_DATA },
            { ...DEFAULT_FEATURES },
        );
        const preparation = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const prepare = vi.fn(() => preparation.promise);
        setV2PrepareForTests(prepare);
        const allocationsBefore = getV2AllocationCountForTests();

        try {
            const pending = workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            await vi.waitFor(() => expect(prepare).toHaveBeenCalledTimes(1));
            const stepped = workerApi.step(1);
            preparation.resolve({ ok: true, value: fixtures.prepared });

            await expect(pending).rejects.toMatchObject({ code: 'stale-request' });
            expect(getV2AllocationCountForTests()).toBe(allocationsBefore);
            expect(stepped.step).toBe(1);
            expect(() => workerApi.getMetricHistoryV2()).toThrow('V2 experiment is not initialized');
            expect(legacy.runId).toBeGreaterThan(0);
        } finally {
            setV2PrepareForTests();
        }
    });
});
