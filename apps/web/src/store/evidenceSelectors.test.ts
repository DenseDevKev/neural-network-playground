import { describe, expect, it } from 'vitest';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';
import { selectScientificEvidence } from './evidenceSelectors.ts';

function model(step: number, revision = step, generationId = 7) {
    return { generationId, revision, step, epoch: 12 } as const;
}

function dataset() {
    return {
        generatorVersion: 1,
        datasetKey: 'dataset-v2',
        trainCount: 210,
        testCount: 90,
    } as const;
}

function live(step: number): LiveTrainingSignal {
    return {
        model: model(step),
        dataset: dataset(),
        objectiveKey: 'objective-v2',
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 10,
            throughStep: step,
        },
        dataLoss: 0.24,
    };
}

function evaluation(step: number, generationId = 7): PairedEvaluation {
    const evaluationModel = model(step, step, generationId);
    return {
        evaluationId: 31,
        trigger: 'cadence',
        model: evaluationModel,
        dataset: dataset(),
        objectiveKey: 'objective-v2',
        train: {
            basis: {
                kind: 'full-split',
                split: 'train',
                sampleCount: 210,
                populationCount: 210,
            },
            values: { dataLoss: 0.2, accuracy: 0.9 },
        },
        test: {
            basis: {
                kind: 'full-split',
                split: 'test',
                sampleCount: 90,
                populationCount: 90,
            },
            values: { dataLoss: 0.35, accuracy: 0.8 },
        },
        objective: {
            regularizationPenalty: 0.03,
            trainTotalObjective: 0.23,
        },
    };
}

describe('selectScientificEvidence', () => {
    it('keeps a newer batch trend separate from one paired full evaluation', () => {
        const selected = selectScientificEvidence({
            latestLiveSignal: live(1240),
            latestEvaluation: evaluation(1230),
        });

        expect(selected).toMatchObject({
            currentModel: model(1240),
            batchTrend: {
                step: 1240,
                epoch: 12,
                dataLoss: 0.24,
                latestBatchSize: 10,
            },
            fullEvaluation: {
                evaluationId: 31,
                step: 1230,
                epoch: 12,
                trigger: 'cadence',
                trainDataLoss: 0.2,
                testDataLoss: 0.35,
                trainingObjective: 0.23,
                regularizationPenalty: 0.03,
                trainAccuracy: 0.9,
                testAccuracy: 0.8,
                trainSampleCount: 210,
                testSampleCount: 90,
            },
            evaluationAgeSteps: 10,
        });
        expect(selected.generalizationGap).toBeCloseTo(0.15);
    });

    it('never computes age across generations and never invents an unpaired gap', () => {
        expect(selectScientificEvidence({
            latestLiveSignal: live(1240),
            latestEvaluation: evaluation(1230, 6),
        })).toMatchObject({
            currentModel: model(1240),
            evaluationAgeSteps: null,
        });
        expect(selectScientificEvidence({
            latestLiveSignal: live(1240),
            latestEvaluation: evaluation(1230, 6),
        }).generalizationGap).toBeCloseTo(0.15);

        expect(selectScientificEvidence({
            latestLiveSignal: live(1240),
            latestEvaluation: null,
        })).toMatchObject({
            fullEvaluation: null,
            evaluationAgeSteps: null,
            generalizationGap: null,
        });
    });
});
