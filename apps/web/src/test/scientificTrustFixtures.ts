import { getDatasetContract } from '@nn-playground/engine';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    WORKER_PROTOCOL_VERSION,
    isWorkerEvidenceMessageV2,
    isWorkerExperimentRequestV2,
    prepareExperimentDocument,
    validateExperimentDocument,
} from '@nn-playground/shared';
import type {
    LiveTrainingSignal,
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerEvidenceMessageV2,
    WorkerExperimentRequestV2,
} from '@nn-playground/shared';

export interface ScientificTrustFixtures {
    readonly prepared: PreparedExperimentDocumentV2;
    readonly request: WorkerExperimentRequestV2;
    readonly liveSignal: LiveTrainingSignal;
    readonly evaluation: PairedEvaluation;
    readonly evidence: WorkerEvidenceMessageV2;
    readonly forged: {
        readonly recipe: WorkerExperimentRequestV2;
        readonly datasetKey: WorkerExperimentRequestV2;
        readonly objectiveKey: WorkerExperimentRequestV2;
    };
}

function alternateIdentity(prefix: 'd2.1.' | 'o2.1.', current: string): string {
    const first = `${prefix}${'A'.repeat(43)}`;
    return first === current ? `${prefix}${'B'.repeat(43)}` : first;
}

/** Canonical, deterministic protocol fixtures shared by Task 8 worker tests. */
export async function createScientificTrustFixtures(): Promise<ScientificTrustFixtures> {
    const preparedResult = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
    if (!preparedResult.ok) {
        throw new Error('The default scientific-trust fixture must prepare successfully.');
    }
    const prepared = preparedResult.value;
    const request: WorkerExperimentRequestV2 = {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: 1,
        document: prepared.document,
        claimedIdentities: prepared.identities,
    };
    if (!isWorkerExperimentRequestV2(request)) {
        throw new Error('The default scientific-trust request must satisfy protocol V2.');
    }

    const sampleCount = prepared.document.recipe.data.sampleCount;
    const trainCount = Math.floor(
        sampleCount * prepared.document.recipe.data.trainFraction,
    );
    const testCount = sampleCount - trainCount;
    const model = {
        generationId: 1,
        revision: 0,
        step: 0,
        epoch: 0,
    } as const;
    const dataset = {
        generatorVersion: getDatasetContract(
            prepared.document.recipe.task.dataset,
        ).generatorVersion,
        datasetKey: prepared.identities.datasetKey,
        trainCount,
        testCount,
    } as const;
    const objectiveKey = prepared.identities.objectiveKey;
    const liveSignal: LiveTrainingSignal = {
        model,
        dataset,
        objectiveKey,
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: Math.min(prepared.document.recipe.training.batchSize, trainCount),
            throughStep: 0,
        },
        dataLoss: 0.6931471805599453,
    };
    const evaluation: PairedEvaluation = {
        evaluationId: 1,
        trigger: 'initial',
        model,
        dataset,
        objectiveKey,
        train: {
            basis: {
                kind: 'full-split',
                split: 'train',
                sampleCount: trainCount,
                populationCount: trainCount,
            },
            values: { dataLoss: 0.6931471805599453 },
        },
        test: {
            basis: {
                kind: 'full-split',
                split: 'test',
                sampleCount: testCount,
                populationCount: testCount,
            },
            values: { dataLoss: 0.6931471805599453 },
        },
        objective: {
            regularizationPenalty: 0,
            trainTotalObjective: 0.6931471805599453,
        },
    };
    const evidence: WorkerEvidenceMessageV2 = {
        type: 'evidence',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        liveSignal,
        latestEvaluation: evaluation,
        artifacts: {
            activationStatistics: {
                model,
                dataset,
                objectiveKey,
                basis: {
                    kind: 'bounded-sample',
                    split: 'train',
                    sampleCount: Math.min(128, trainCount),
                    populationCount: trainCount,
                },
            },
        },
    };
    if (!isWorkerEvidenceMessageV2(evidence)) {
        throw new Error('The default scientific-trust evidence must satisfy protocol V2.');
    }

    const changedRecipeResult = validateExperimentDocument({
        ...prepared.document,
        recipe: {
            ...prepared.document.recipe,
            data: {
                ...prepared.document.recipe.data,
                noise: prepared.document.recipe.data.noise + 1,
            },
        },
    });
    if (!changedRecipeResult.ok) {
        throw new Error('The forged-recipe fixture must remain schema-valid.');
    }

    return {
        prepared,
        request,
        liveSignal,
        evaluation,
        evidence,
        forged: {
            recipe: { ...request, document: changedRecipeResult.value },
            datasetKey: {
                ...request,
                claimedIdentities: {
                    ...request.claimedIdentities,
                    datasetKey: alternateIdentity(
                        'd2.1.',
                        request.claimedIdentities.datasetKey,
                    ),
                },
            },
            objectiveKey: {
                ...request,
                claimedIdentities: {
                    ...request.claimedIdentities,
                    objectiveKey: alternateIdentity(
                        'o2.1.',
                        request.claimedIdentities.objectiveKey,
                    ),
                },
            },
        },
    };
}
