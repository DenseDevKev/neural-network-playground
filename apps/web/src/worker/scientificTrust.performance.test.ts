import { beforeAll, describe, expect, it, vi } from 'vitest';
import {
    PREPARED_PRESETS,
    WORKER_PROTOCOL_VERSION,
    prepareExperimentDocument,
} from '@nn-playground/shared';
import type {
    PairedEvaluation,
    PreparedExperimentDocumentV2,
    WorkerExperimentRequestV2,
} from '@nn-playground/shared';

vi.mock('comlink', () => ({ expose: vi.fn() }));

import { workerApi } from './training.worker.ts';

const WARMUP_RUNS = 5;
const MEASURED_RUNS = 20;
const FORCED_PAIR_BUDGET_MS = 250;
const SAVE_CAPTURE_BUDGET_MS = 500;
const MAXIMUM_HIDDEN_LAYERS = [16, 16, 16, 16, 16, 16] as const;
const ALL_PUBLIC_FEATURES = [
    'x',
    'y',
    'xSquared',
    'ySquared',
    'xy',
    'sinX',
    'sinY',
    'cosX',
    'cosY',
] as const;

let nextRequestId = 90_000;
let maximumPrepared: PreparedExperimentDocumentV2;

function median(values: readonly number[]): number {
    if (values.length === 0) throw new RangeError('median requires at least one value');
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle];
}

function requestForMaximumRecipe(): WorkerExperimentRequestV2 {
    return {
        type: 'initialize-experiment',
        protocolVersion: WORKER_PROTOCOL_VERSION,
        requestId: nextRequestId++,
        document: maximumPrepared.document,
        claimedIdentities: maximumPrepared.identities,
    };
}

function expectMaximumRecipeContract(): void {
    const { recipe } = maximumPrepared.document;
    expect(recipe.data.sampleCount).toBe(1_000);
    expect(recipe.inputs.featureIds).toEqual(ALL_PUBLIC_FEATURES);
    expect(recipe.model.hiddenLayers).toEqual(MAXIMUM_HIDDEN_LAYERS);
    expect(recipe.training.optimizer).toMatchObject({ kind: 'adam' });
    expect(recipe.training.gradientClipping).toEqual({
        kind: 'global-norm',
        maximumNorm: 1,
        scope: 'total-objective-gradient',
    });
    expect(recipe.objective.penalty).toEqual({
        kind: 'l2',
        coefficient: 0.01,
        applyTo: 'weights',
    });
}

function expectScientificDiagnostics(evaluation: PairedEvaluation): void {
    expect(evaluation.model.generationId).toBeGreaterThan(0);
    expect(evaluation.dataset.trainCount + evaluation.dataset.testCount).toBe(1_000);
    expect(evaluation.train.basis).toMatchObject({
        kind: 'full-split',
        split: 'train',
        sampleCount: evaluation.dataset.trainCount,
        populationCount: evaluation.dataset.trainCount,
    });
    expect(evaluation.test.basis).toMatchObject({
        kind: 'full-split',
        split: 'test',
        sampleCount: evaluation.dataset.testCount,
        populationCount: evaluation.dataset.testCount,
    });
    expect(evaluation.objective.regularizationPenalty).toBeGreaterThan(0);
    expect(evaluation.objective.trainTotalObjective).toBeCloseTo(
        evaluation.train.values.dataLoss + evaluation.objective.regularizationPenalty,
        12,
    );
}

async function measure(operation: () => Promise<void>): Promise<readonly number[]> {
    for (let index = 0; index < WARMUP_RUNS; index++) await operation();
    const samples: number[] = [];
    for (let index = 0; index < MEASURED_RUNS; index++) {
        const startedAt = performance.now();
        await operation();
        samples.push(performance.now() - startedAt);
    }
    return samples;
}

beforeAll(async () => {
    const multiclass = PREPARED_PRESETS.find(
        (entry) => entry.id === 'three-class-clusters',
    );
    if (!multiclass) throw new Error('missing built-in three-class recipe');
    const result = await prepareExperimentDocument({
        ...multiclass.prepared.document,
        recipe: {
            ...multiclass.prepared.document.recipe,
            data: {
                ...multiclass.prepared.document.recipe.data,
                sampleCount: 1_000,
            },
            inputs: { featureIds: ALL_PUBLIC_FEATURES },
            model: {
                ...multiclass.prepared.document.recipe.model,
                hiddenLayers: MAXIMUM_HIDDEN_LAYERS,
            },
            training: {
                ...multiclass.prepared.document.recipe.training,
                optimizer: {
                    kind: 'adam',
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
                gradientClipping: {
                    kind: 'global-norm',
                    maximumNorm: 1,
                    scope: 'total-objective-gradient',
                },
            },
            objective: {
                ...multiclass.prepared.document.recipe.objective,
                penalty: { kind: 'l2', coefficient: 0.01, applyTo: 'weights' },
            },
        },
    });
    if (!result.ok) {
        throw new Error(result.issues.map((issue) => `${issue.path}: ${issue.message}`).join('; '));
    }
    maximumPrepared = result.value;
});

describe('scientific-trust worker performance gates', () => {
    it('keeps 20 warmed forced paired evaluations within the maximum-recipe budget', { timeout: 60_000 }, async () => {
        expectMaximumRecipeContract();
        await workerApi.initializeExperimentV2(requestForMaximumRecipe());
        let latest: PairedEvaluation | undefined;
        const samples = await measure(async () => {
            latest = (await workerApi.forceEvaluationV2('pause')).evidence.latestEvaluation;
        });
        if (!latest) throw new Error('forced evaluation did not publish evidence');
        expectScientificDiagnostics(latest);

        const measuredMedian = median(samples);
        console.log(`Scientific trust forced pair samples (ms): ${samples.map((value) => value.toFixed(4)).join(', ')}`);
        console.log(`Scientific trust forced pair median: ${measuredMedian.toFixed(4)}ms`);
        expect(samples).toHaveLength(MEASURED_RUNS);
        expect(measuredMedian).toBeLessThanOrEqual(FORCED_PAIR_BUDGET_MS);
    });

    it('keeps 20 warmed worker-owned save captures within the maximum-recipe budget', { timeout: 60_000 }, async () => {
        expectMaximumRecipeContract();
        await workerApi.initializeExperimentV2(requestForMaximumRecipe());
        let captureIndex = 0;
        let latestEvaluation: PairedEvaluation | undefined;
        let latestHistorySizes: Readonly<{ trend: number; evaluations: number }> | undefined;
        const samples = await measure(async () => {
            const suffix = String(captureIndex++).padStart(12, '0');
            const record = await workerApi.captureRunArtifact({
                id: `00000000-0000-4000-8000-${suffix}`,
                createdAt: '2026-07-12T12:00:00.000Z',
                updatedAt: '2026-07-12T12:00:00.000Z',
                title: `Maximum recipe capture ${captureIndex}`,
            });
            latestEvaluation = record.snapshot.evaluation;
            latestHistorySizes = {
                trend: record.snapshot.trendHistory.length,
                evaluations: record.snapshot.evaluationHistory.length,
            };
        });
        if (!latestEvaluation || !latestHistorySizes) {
            throw new Error('save capture did not publish evidence');
        }
        expectScientificDiagnostics(latestEvaluation);
        expect(latestHistorySizes.trend).toBeLessThanOrEqual(512);
        expect(latestHistorySizes.evaluations).toBeLessThanOrEqual(256);

        const measuredMedian = median(samples);
        console.log(`Scientific trust save capture samples (ms): ${samples.map((value) => value.toFixed(4)).join(', ')}`);
        console.log(`Scientific trust save capture median: ${measuredMedian.toFixed(4)}ms`);
        expect(samples).toHaveLength(MEASURED_RUNS);
        expect(measuredMedian).toBeLessThanOrEqual(SAVE_CAPTURE_BUDGET_MS);
    });
});
