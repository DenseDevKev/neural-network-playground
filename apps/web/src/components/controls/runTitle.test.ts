import { describe, expect, it } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import type { ExperimentRunRecordV2, PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { createDefaultRunTitle } from './runTitle.ts';

function preset(id: (typeof PREPARED_PRESETS)[number]['id']): PreparedExperimentDocumentV2 {
    const entry = PREPARED_PRESETS.find((candidate) => candidate.id === id);
    if (!entry) throw new Error(`Missing preset ${id}`);
    return entry.prepared;
}

function snapshotAt(
    prepared: PreparedExperimentDocumentV2,
    step: number,
): ExperimentRunRecordV2['snapshot'] {
    const sampleCount = prepared.document.recipe.data.sampleCount;
    const trainCount = Math.floor(sampleCount * prepared.document.recipe.data.trainFraction);
    const model = { generationId: 1, revision: step, step, epoch: 0 };
    const evaluation = {
        evaluationId: 1,
        trigger: 'save' as const,
        model,
        dataset: {
            generatorVersion: 2,
            datasetKey: prepared.identities.datasetKey,
            trainCount,
            testCount: sampleCount - trainCount,
        },
        objectiveKey: prepared.identities.objectiveKey,
        train: {
            basis: { kind: 'full-split' as const, split: 'train' as const, sampleCount: trainCount, populationCount: trainCount },
            values: { dataLoss: 0.4 },
        },
        test: {
            basis: { kind: 'full-split' as const, split: 'test' as const, sampleCount: sampleCount - trainCount, populationCount: sampleCount - trainCount },
            values: { dataLoss: 0.5 },
        },
        objective: { regularizationPenalty: 0, trainTotalObjective: 0.4 },
    };
    return { model, evaluation, trendHistory: [], evaluationHistory: [evaluation] };
}

describe('createDefaultRunTitle', () => {
    it('names a saved run from its dataset, complete architecture, and captured step', () => {
        const prepared = preset('xor-hidden');

        expect(createDefaultRunTitle(DEFAULT_EXPERIMENT_DOCUMENT.recipe, snapshotAt(prepared, 400)))
            .toBe('Circle · 2-4-4-1 · step 400');
    });

    it('derives the output width from the multiclass recipe contract', () => {
        const prepared = preset('three-class-clusters');

        expect(createDefaultRunTitle(prepared.document.recipe, snapshotAt(prepared, 12)))
            .toBe('Three-class clusters · 2-6-6-3 · step 12');
    });
});
