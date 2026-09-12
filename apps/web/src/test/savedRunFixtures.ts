import type { ExperimentRunRecordV2, PreparedExperimentDocumentV2 } from '@nn-playground/shared';

/** Test-only record factory. Each call owns its recipe and evidence objects. */
export function makeSavedRunRecord(
    prepared: PreparedExperimentDocumentV2,
    id: string = '00000000-0000-0000-0000-000000000001',
    title = 'Saved evidence',
    testDataLoss = 0.5,
): ExperimentRunRecordV2 {
    const sampleCount = prepared.document.recipe.data.sampleCount;
    const trainCount = Math.floor(sampleCount * prepared.document.recipe.data.trainFraction);
    const testCount = sampleCount - trainCount;
    const model = { generationId: Number(id.at(-1)) || 1, revision: 4, step: 4, epoch: 0 };
    const dataset = {
        generatorVersion: 2,
        datasetKey: prepared.identities.datasetKey,
        trainCount,
        testCount,
    };
    const evaluation = {
        evaluationId: 2,
        trigger: 'save' as const,
        model,
        dataset,
        objectiveKey: prepared.identities.objectiveKey,
        train: {
            basis: { kind: 'full-split' as const, split: 'train' as const, sampleCount: trainCount, populationCount: trainCount },
            values: { dataLoss: 0.4 },
        },
        test: {
            basis: { kind: 'full-split' as const, split: 'test' as const, sampleCount: testCount, populationCount: testCount },
            values: { dataLoss: testDataLoss },
        },
        objective: { regularizationPenalty: 0, trainTotalObjective: 0.4 },
    };
    return {
        kind: 'nn-playground-run',
        schemaVersion: 2,
        id,
        createdAt: '2026-07-11T12:00:00.000Z',
        updatedAt: '2026-07-11T12:00:00.000Z',
        title,
        recipe: structuredClone(prepared.document.recipe),
        recipeFingerprint: prepared.identities.recipeFingerprint,
        snapshot: {
            model,
            evaluation,
            trendHistory: [],
            evaluationHistory: [evaluation],
        },
    };
}
