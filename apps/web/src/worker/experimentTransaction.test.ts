import { describe, expect, it, vi } from 'vitest';
import type {
    PreparedExperimentDocumentV2,
    SchemaResult,
    WorkerExperimentRequestV2,
} from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';
import {
    ExperimentRequestGate,
    ExperimentTransactionError,
} from './experimentTransaction.ts';

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((resolver) => {
        resolve = resolver;
    });
    return { promise, resolve };
}

function withRequestId(
    request: WorkerExperimentRequestV2,
    requestId: number,
): WorkerExperimentRequestV2 {
    return { ...request, requestId };
}

describe('ExperimentRequestGate', () => {
    it('classifies a recognizable future-version initialize request separately', async () => {
        const fixtures = await createScientificTrustFixtures();
        const gate = new ExperimentRequestGate();
        const commit = vi.fn();

        await expect(gate.run({
            ...fixtures.request,
            protocolVersion: 3,
        }, commit)).rejects.toMatchObject({
            code: 'unsupported-protocol-version',
            path: '$.protocolVersion',
            requestId: fixtures.request.requestId,
        });
        expect(commit).not.toHaveBeenCalled();
    });

    it('keeps truly malformed initialize requests classified as malformed', async () => {
        const fixtures = await createScientificTrustFixtures();
        const gate = new ExperimentRequestGate();
        const commit = vi.fn();
        const { claimedIdentities: _missing, ...malformed } = fixtures.request;

        await expect(gate.run(malformed, commit)).rejects.toMatchObject({
            code: 'malformed-request',
            path: '$',
            requestId: null,
        });
        expect(commit).not.toHaveBeenCalled();
    });

    it('prepares and checks every identity before committing allocation work', async () => {
        const fixtures = await createScientificTrustFixtures();
        const order: string[] = [];
        const prepare = vi.fn(async () => {
            order.push('prepare');
            return { ok: true, value: fixtures.prepared } as const;
        });
        const commit = vi.fn(() => {
            order.push('commit');
            return 'allocated';
        });
        const gate = new ExperimentRequestGate({ prepare });

        await expect(gate.run(fixtures.request, commit)).resolves.toEqual({
            requestId: fixtures.request.requestId,
            prepared: fixtures.prepared,
            value: 'allocated',
        });
        expect(order).toEqual(['prepare', 'commit']);
        expect(commit).toHaveBeenCalledWith(fixtures.prepared);
    });

    it.each(['recipe', 'datasetKey', 'objectiveKey'] as const)(
        'rejects a forged %s before commit',
        async (kind) => {
            const fixtures = await createScientificTrustFixtures();
            const commit = vi.fn();
            const gate = new ExperimentRequestGate();

            await expect(gate.run(fixtures.forged[kind], commit)).rejects.toBeInstanceOf(
                ExperimentTransactionError,
            );
            expect(commit).not.toHaveBeenCalled();
        },
    );

    it('rejects a forged recipe fingerprint before commit', async () => {
        const fixtures = await createScientificTrustFixtures();
        const commit = vi.fn();
        const gate = new ExperimentRequestGate();
        const forged: WorkerExperimentRequestV2 = {
            ...fixtures.request,
            claimedIdentities: {
                ...fixtures.request.claimedIdentities,
                recipeFingerprint: `r2.1.${'A'.repeat(43)}`,
            },
        };

        await expect(gate.run(forged, commit)).rejects.toMatchObject({
            code: 'identity-mismatch',
            path: '$.claimedIdentities.recipeFingerprint',
        });
        expect(commit).not.toHaveBeenCalled();
    });

    it('lets the newest request win when asynchronous preparation resolves out of order', async () => {
        const fixtures = await createScientificTrustFixtures();
        const first = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const second = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const prepare = vi
            .fn()
            .mockImplementationOnce(() => first.promise)
            .mockImplementationOnce(() => second.promise);
        const gate = new ExperimentRequestGate({ prepare });
        const commitFirst = vi.fn();
        const commitSecond = vi.fn(() => 'second');

        const pendingFirst = gate.run(withRequestId(fixtures.request, 10), commitFirst);
        const pendingSecond = gate.run(withRequestId(fixtures.request, 11), commitSecond);
        second.resolve({ ok: true, value: fixtures.prepared });
        await expect(pendingSecond).resolves.toMatchObject({ value: 'second' });
        first.resolve({ ok: true, value: fixtures.prepared });
        await expect(pendingFirst).rejects.toMatchObject({
            code: 'stale-request',
            requestId: 10,
        });
        expect(commitFirst).not.toHaveBeenCalled();
        expect(commitSecond).toHaveBeenCalledTimes(1);
    });

    it('does not commit a preparation failure or reuse a request id', async () => {
        const fixtures = await createScientificTrustFixtures();
        const prepare = vi.fn(async () => ({
            ok: false,
            issues: [{ code: 'invalid-field', path: '$.recipe', message: 'invalid' }],
        }) satisfies SchemaResult<PreparedExperimentDocumentV2>);
        const gate = new ExperimentRequestGate({ prepare });
        const commit = vi.fn();

        await expect(gate.run(fixtures.request, commit)).rejects.toMatchObject({
            code: 'invalid-experiment',
            path: '$.recipe',
        });
        await expect(gate.run(fixtures.request, commit)).rejects.toMatchObject({
            code: 'stale-request',
        });
        expect(commit).not.toHaveBeenCalled();
    });
});
