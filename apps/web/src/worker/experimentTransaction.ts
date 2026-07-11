import {
    parseWorkerExperimentRequestV2,
    prepareExperimentDocument,
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
            'malformed-request' | 'invalid-experiment' | 'identity-mismatch' | 'stale-request'
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
        throw new ExperimentTransactionError(
            'malformed-request',
            '$',
            error instanceof Error ? error.message : 'Invalid version-2 worker request.',
            null,
        );
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
