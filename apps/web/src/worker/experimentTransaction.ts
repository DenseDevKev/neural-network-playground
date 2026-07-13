import {
    parseWorkerExperimentRequestV2,
    prepareExperimentDocument,
    WORKER_PROTOCOL_VERSION,
} from '@nn-playground/shared';
import type {
    PreparedExperimentDocumentV2,
    SchemaResult,
    WorkerExperimentRequestV2,
    WorkerProtocolErrorCodeV2,
} from '@nn-playground/shared';

type PrepareExperiment = (
    value: unknown,
) => Promise<SchemaResult<PreparedExperimentDocumentV2>>;

export class ExperimentTransactionError extends Error {
    readonly name = 'ExperimentTransactionError';

    constructor(
        readonly code: Extract<
            WorkerProtocolErrorCodeV2,
            | 'malformed-request'
            | 'unsupported-protocol-version'
            | 'invalid-experiment'
            | 'identity-mismatch'
            | 'stale-request'
        >,
        readonly path: string,
        message: string,
        readonly requestId: number | null,
    ) {
        super(message);
    }
}

export interface ExperimentRequestGateDependencies {
    readonly prepare?: PrepareExperiment;
}

export interface ExperimentTransactionResult<T> {
    readonly requestId: number;
    readonly prepared: PreparedExperimentDocumentV2;
    readonly value: T;
}

const IDENTITY_KEYS = [
    'canonicalRecipeKey',
    'recipeFingerprint',
    'datasetKey',
    'objectiveKey',
] as const;

function parseRequest(value: unknown): WorkerExperimentRequestV2 {
    try {
        return parseWorkerExperimentRequestV2(value);
    } catch (error) {
        const unsupported = recognizableFutureInitializeRequest(value);
        if (unsupported !== null) {
            throw new ExperimentTransactionError(
                'unsupported-protocol-version',
                '$.protocolVersion',
                `Worker protocol version ${unsupported.protocolVersion} is not supported.`,
                unsupported.requestId,
            );
        }
        throw new ExperimentTransactionError(
            'malformed-request',
            '$',
            error instanceof Error ? error.message : 'Invalid version-2 worker request.',
            null,
        );
    }
}

function recognizableFutureInitializeRequest(value: unknown): {
    readonly protocolVersion: number;
    readonly requestId: number;
} | null {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
    try {
        const prototype = Object.getPrototypeOf(value);
        if (prototype !== Object.prototype && prototype !== null) return null;
        const record = value as Record<string, unknown>;
        for (const key of ['type', 'protocolVersion', 'requestId']) {
            const descriptor = Object.getOwnPropertyDescriptor(record, key);
            if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
                return null;
            }
        }
        const protocolVersion = record['protocolVersion'];
        const requestId = record['requestId'];
        if (record['type'] !== 'initialize-experiment'
            || !Number.isSafeInteger(protocolVersion)
            || (protocolVersion as number) <= WORKER_PROTOCOL_VERSION
            || !Number.isSafeInteger(requestId)
            || (requestId as number) <= 0) {
            return null;
        }
        return {
            protocolVersion: protocolVersion as number,
            requestId: requestId as number,
        };
    } catch {
        return null;
    }
}

/**
 * Serializes the asynchronous experiment-preparation boundary. The commit
 * callback is deliberately synchronous: once it begins it is the single
 * allocation/mutation transaction for the winning request.
 */
export class ExperimentRequestGate {
    private readonly prepare: PrepareExperiment;
    private latestRequestId = 0;

    constructor(dependencies: ExperimentRequestGateDependencies = {}) {
        this.prepare = dependencies.prepare ?? prepareExperimentDocument;
    }

    async run<T>(
        value: unknown,
        commit: (prepared: PreparedExperimentDocumentV2) => T,
        onReserved?: (request: WorkerExperimentRequestV2) => void,
    ): Promise<ExperimentTransactionResult<T>> {
        const request = parseRequest(value);
        if (request.requestId <= this.latestRequestId) {
            throw new ExperimentTransactionError(
                'stale-request',
                '$.requestId',
                `Request ${request.requestId} is not newer than ${this.latestRequestId}.`,
                request.requestId,
            );
        }

        // Reserve the sequence before hashing. A later request immediately
        // makes this one stale, even if the older digest finishes last.
        this.latestRequestId = request.requestId;
        onReserved?.(request);
        const preparedResult = await this.prepare(request.document);
        if (request.requestId !== this.latestRequestId) {
            throw new ExperimentTransactionError(
                'stale-request',
                '$.requestId',
                `Request ${request.requestId} was superseded during preparation.`,
                request.requestId,
            );
        }
        if (!preparedResult.ok) {
            const issue = preparedResult.issues[0];
            throw new ExperimentTransactionError(
                'invalid-experiment',
                issue?.path ?? '$',
                issue?.message ?? 'Experiment preparation failed.',
                request.requestId,
            );
        }

        const prepared = preparedResult.value;
        for (const key of IDENTITY_KEYS) {
            if (request.claimedIdentities[key] !== prepared.identities[key]) {
                throw new ExperimentTransactionError(
                    'identity-mismatch',
                    `$.claimedIdentities.${key}`,
                    `Claimed ${key} does not match the prepared experiment.`,
                    request.requestId,
                );
            }
        }

        // No await is allowed between the final stale check and commit.
        if (request.requestId !== this.latestRequestId) {
            throw new ExperimentTransactionError(
                'stale-request',
                '$.requestId',
                `Request ${request.requestId} was superseded before commit.`,
                request.requestId,
            );
        }
        const committed = commit(prepared);
        return {
            requestId: request.requestId,
            prepared,
            value: committed,
        };
    }

    getLatestRequestId(): number {
        return this.latestRequestId;
    }
}
