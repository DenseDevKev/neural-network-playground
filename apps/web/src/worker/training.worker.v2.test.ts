import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_DATA,
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
} from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';

vi.mock('comlink', () => ({
    expose: vi.fn(),
}));

import {
    getV2AllocationCountForTests,
    setV2PrepareForTests,
    workerApi,
} from './training.worker.ts';

let nextRequestId = 10_000;

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((resolver) => {
        resolve = resolver;
    });
    return { promise, resolve };
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

describe('training worker scientific-trust V2 boundary', () => {
    beforeEach(() => {
        vi.useRealTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('commits a prepared experiment and returns a validated initial pair', async () => {
        const fixtures = await createScientificTrustFixtures();
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
        expect(isWorkerEvidenceMessageV2(result.evidence)).toBe(true);
        expect(isWorkerToMainMessage(result.evidence)).toBe(true);
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
        await vi.advanceTimersByTimeAsync(20);

        let history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory).toHaveLength(49);
        expect(history.evaluationHistory.map((point) => point.trigger)).toEqual(['initial']);

        capture.dispatch({ type: 'updateSpeed', stepsPerFrame: 1 });
        capture.dispatch({ type: 'frameAck' });
        await vi.advanceTimersByTimeAsync(20);

        history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory).toHaveLength(50);
        expect(history.evaluationHistory.map((point) => point.trigger)).toEqual([
            'initial',
            'cadence',
        ]);
        expect(history.evaluationHistory[1]?.model).toEqual(history.trendHistory[49]?.model);
        const beforePause = evidenceMessages(capture.messages);
        expect(beforePause.filter((message) => message.latestEvaluation?.trigger === 'cadence'))
            .toHaveLength(1);
        expect(beforePause.filter((message) => message.liveSignal)).toHaveLength(2);
        expect(beforePause.length).toBeLessThanOrEqual(4);
        expect(beforePause.every(isWorkerToMainMessage)).toBe(true);

        capture.dispatch({ type: 'stopTraining' });
        history = workerApi.getMetricHistoryV2();
        const pause = history.evaluationHistory.at(-1)!;
        expect(pause.trigger).toBe('pause');
        expect(pause.model).toEqual(history.trendHistory.at(-1)?.model);
        expect(evidenceMessages(capture.messages).at(-1)?.latestEvaluation).toEqual(pause);
    });

    it('forces current manual-step evidence and resets to a fresh generation', async () => {
        const fixtures = await createScientificTrustFixtures();
        const initial = await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
        const stepped = workerApi.stepExperimentV2(1);

        expect(stepped.runId).toBe(initial.runId);
        expect(stepped.evidence.latestEvaluation).toMatchObject({
            trigger: 'manual-step',
            model: { generationId: initial.runId, step: 1 },
        });
        expect(stepped.evidence.liveSignal?.model)
            .toEqual(stepped.evidence.latestEvaluation?.model);

        const reset = workerApi.resetExperimentV2();
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
        expect(workerApi.getMetricHistoryV2().trendHistory).toEqual([]);
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
        workerApi.stepExperimentV2(7);

        const history = workerApi.getMetricHistoryV2();
        expect(history.trendHistory.every((point) => Number.isFinite(point.dataLoss))).toBe(true);
        expect(history.evaluationHistory.every((point) => (
            Number.isFinite(point.train.values.dataLoss)
            && Number.isFinite(point.test.values.dataLoss)
            && Number.isFinite(point.objective.regularizationPenalty)
            && Number.isFinite(point.objective.trainTotalObjective)
        ))).toBe(true);
    });

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
        const stepped = workerApi.stepExperimentV2(1);

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
        expect(workerApi.getCheckpointTimeline().checkpoints).toEqual([]);
        const validStep = workerApi.stepExperimentV2(1);
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

        expect(() => workerApi.forceEvaluationV2('cadence' as never))
            .toThrow('Unsupported forced V2 evaluation trigger');

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
        'rejects %i synchronous manual steps before mutation or message flooding',
        async (iterations) => {
            const fixtures = await createScientificTrustFixtures();
            const capture = createCapturingPort();
            workerApi.setStreamPort(capture.port);
            await workerApi.initializeExperimentV2(withFreshId(fixtures.request));
            const before = workerApi.getMetricHistoryV2();
            const beforeMessages = capture.messages.length;

            expect(() => workerApi.stepExperimentV2(iterations)).toThrow('from 1 to 10');

            expect(workerApi.getMetricHistoryV2()).toEqual(before);
            expect(capture.messages.length - beforeMessages).toBeLessThanOrEqual(1);
        },
    );

    it('prevents delayed initialization from overwriting a newer V2 step', async () => {
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

            await expect(pending).rejects.toMatchObject({
                code: 'stale-request',
            });
            expect(getV2AllocationCountForTests()).toBe(allocationsBefore);
            expect(stepped.runId).toBe(active.runId);
            expect(workerApi.getMetricHistoryV2().trendHistory.at(-1)?.model).toEqual(
                stepped.evidence.liveSignal?.model,
            );
            expect(workerApi.getMetricHistoryV2().evaluationHistory.at(-1)?.model.generationId)
                .toBe(active.runId);
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
