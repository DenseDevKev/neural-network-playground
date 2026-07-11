import { describe, expect, expectTypeOf, it } from 'vitest';
import {
    DEFAULT_EVALUATION_POLICY,
    assertArtifactProvenance,
    assertLiveTrainingSignal,
    assertPairedEvaluation,
    parseArtifactProvenance,
    parseLiveTrainingSignal,
    parsePairedEvaluation,
    type ArtifactProvenance,
    type LiveTrainingSignal,
    type PairedEvaluation,
} from '../metricProvenance.js';

const model = {
    generationId: 1,
    revision: 7,
    step: 50,
    epoch: 2,
} as const;

const dataset = {
    generatorVersion: 2,
    datasetKey: 'd2.1.example',
    trainCount: 210,
    testCount: 90,
} as const;

function pairedEvaluation(): PairedEvaluation {
    return {
        evaluationId: 3,
        trigger: 'cadence',
        model,
        dataset,
        objectiveKey: 'o2.1.example',
        train: {
            basis: {
                kind: 'full-split',
                split: 'train',
                sampleCount: 210,
                populationCount: 210,
            },
            values: { dataLoss: 0.25, accuracy: 0.9 },
        },
        test: {
            basis: {
                kind: 'full-split',
                split: 'test',
                sampleCount: 90,
                populationCount: 90,
            },
            values: { dataLoss: 0.3, accuracy: 0.85 },
        },
        objective: {
            regularizationPenalty: 0.02,
            trainTotalObjective: 0.27,
        },
    };
}

function liveSignal(): LiveTrainingSignal {
    return {
        model,
        dataset,
        objectiveKey: 'o2.1.example',
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 10,
            throughStep: 50,
        },
        dataLoss: 0.28,
    };
}

describe('metric provenance contract', () => {
    it('locks the version-2 evaluation cadence and forced triggers', () => {
        expect(DEFAULT_EVALUATION_POLICY).toEqual({
            everySteps: 50,
            forceOnPause: true,
            forceOnManualStep: true,
            forceOnCheckpoint: true,
            forceOnSave: true,
        });
        expect(Object.isFrozen(DEFAULT_EVALUATION_POLICY)).toBe(true);
    });

    it('keeps live signals free of paired/full-objective claims at the type level', () => {
        expectTypeOf<LiveTrainingSignal>().not.toHaveProperty('accuracy');
        expectTypeOf<LiveTrainingSignal>().not.toHaveProperty('confusionMatrix');
        expectTypeOf<LiveTrainingSignal>().not.toHaveProperty('generalizationGap');
        expectTypeOf<LiveTrainingSignal>().not.toHaveProperty('totalObjective');
    });

    it('accepts an exact same-revision paired evaluation', () => {
        const pair = pairedEvaluation();

        expect(() => assertPairedEvaluation(pair)).not.toThrow();
    });

    it('returns detached deeply frozen provenance snapshots', () => {
        const pair = pairedEvaluation();
        const signal = liveSignal();
        const artifact: ArtifactProvenance = {
            model,
            dataset,
            objectiveKey: 'o2.1.example',
            basis: {
                kind: 'bounded-sample',
                split: 'train',
                sampleCount: 128,
                populationCount: 210,
            },
        };

        const parsedPair = parsePairedEvaluation(pair);
        const parsedSignal = parseLiveTrainingSignal(signal);
        const parsedArtifact = parseArtifactProvenance(artifact);
        (pair as any).model.generationId = 9;
        (signal as any).dataset.datasetKey = 'changed';
        (artifact as any).basis.sampleCount = 1;

        expect(parsedPair.model.generationId).toBe(1);
        expect(parsedSignal.dataset.datasetKey).toBe('d2.1.example');
        expect(parsedArtifact.basis).toMatchObject({ sampleCount: 128 });
        expect(Object.isFrozen(parsedPair)).toBe(true);
        expect(Object.isFrozen(parsedPair.model)).toBe(true);
        expect(Object.isFrozen(parsedArtifact.basis)).toBe(true);
    });

    it.each([
        ['evaluationId', (pair: any) => { pair.evaluationId = 0; }],
        ['model.generationId', (pair: any) => { pair.model.generationId = 0; }],
        ['dataset.trainCount', (pair: any) => { pair.dataset.trainCount = 0; }],
        ['train.basis.sampleCount', (pair: any) => { pair.train.basis.sampleCount = 209; }],
        ['test.basis.populationCount', (pair: any) => { pair.test.basis.populationCount = 89; }],
        ['train.values.dataLoss', (pair: any) => { pair.train.values.dataLoss = Number.NaN; }],
        ['train.values.accuracy', (pair: any) => { pair.train.values.accuracy = 1.01; }],
        ['objective.trainTotalObjective', (pair: any) => { pair.objective.trainTotalObjective = 0.4; }],
        ['objectiveKey', (pair: any) => { pair.objectiveKey = ''; }],
    ])('rejects invalid %s provenance', (path, mutate) => {
        const pair: any = structuredClone(pairedEvaluation());
        mutate(pair);

        expect(() => assertPairedEvaluation(pair)).toThrow(path);
    });

    it('validates binary and multiclass confusion counts against the exact split', () => {
        const binary: any = pairedEvaluation();
        binary.test.values.confusionMatrix = { tp: 30, tn: 30, fp: 15, fn: 15 };
        binary.test.values.accuracy = 2 / 3;
        expect(() => assertPairedEvaluation(binary)).not.toThrow();
        binary.test.values.confusionMatrix.fp = 14;
        expect(() => assertPairedEvaluation(binary)).toThrow('test.values.confusionMatrix');

        const multiclass: any = pairedEvaluation();
        multiclass.test.values.confusionMatrix = {
            classCount: 3,
            classLabels: [0, 1, 2],
            counts: [10, 10, 10, 10, 10, 10, 10, 10, 10],
        };
        multiclass.test.values.accuracy = 1 / 3;
        expect(() => assertPairedEvaluation(multiclass)).not.toThrow();
        multiclass.test.values.confusionMatrix.counts[8] = -1;
        expect(() => assertPairedEvaluation(multiclass)).toThrow('test.values.confusionMatrix');
    });

    it('rejects confusion/accuracy contradictions and objective-sum overflow', () => {
        const contradictory: any = pairedEvaluation();
        contradictory.test.values.accuracy = 1;
        contradictory.test.values.confusionMatrix = { tp: 0, tn: 0, fp: 45, fn: 45 };
        expect(() => assertPairedEvaluation(contradictory)).toThrow('test.values.accuracy');

        const overflow: any = pairedEvaluation();
        overflow.train.values.dataLoss = Number.MAX_VALUE;
        overflow.objective.regularizationPenalty = Number.MAX_VALUE;
        overflow.objective.trainTotalObjective = Number.MAX_VALUE;
        expect(() => assertPairedEvaluation(overflow)).toThrow(
            'objective.trainTotalObjective',
        );
    });

    it('accepts exact live EMA provenance', () => {
        expect(() => assertLiveTrainingSignal(liveSignal())).not.toThrow();
    });

    it.each([
        ['basis.alpha', (signal: any) => { signal.basis.alpha = 0; }],
        ['basis.latestBatchSize', (signal: any) => { signal.basis.latestBatchSize = 211; }],
        ['basis.throughStep', (signal: any) => { signal.basis.throughStep = 49; }],
        ['dataLoss', (signal: any) => { signal.dataLoss = Number.POSITIVE_INFINITY; }],
        ['accuracy', (signal: any) => { signal.accuracy = 0.8; }],
    ])('rejects invalid or extra live claim %s', (path, mutate) => {
        const signal: any = structuredClone(liveSignal());
        mutate(signal);

        expect(() => assertLiveTrainingSignal(signal)).toThrow(path);
    });

    it('rejects accessors, hidden fields, and custom record or array prototypes', () => {
        const accessor: any = structuredClone(liveSignal());
        Object.defineProperty(accessor, 'dataLoss', {
            enumerable: true,
            get: () => 0.2,
        });
        expect(() => assertLiveTrainingSignal(accessor)).toThrow('dataLoss');

        const hidden: any = structuredClone(pairedEvaluation());
        Object.defineProperty(hidden.objective, 'regularizationPenalty', {
            value: 0.02,
            enumerable: false,
        });
        expect(() => assertPairedEvaluation(hidden)).toThrow(
            'objective.regularizationPenalty',
        );

        const customRecord: any = structuredClone(liveSignal());
        Object.setPrototypeOf(customRecord, { inherited: true });
        expect(() => assertLiveTrainingSignal(customRecord)).toThrow('plain records');

        const customArray: any = structuredClone(pairedEvaluation());
        customArray.test.values.confusionMatrix = {
            classCount: 3,
            classLabels: [0, 1, 2],
            counts: [10, 10, 10, 10, 10, 10, 10, 10, 10],
        };
        Object.setPrototypeOf(customArray.test.values.confusionMatrix.counts, { inherited: true });
        expect(() => assertPairedEvaluation(customArray)).toThrow(
            'test.values.confusionMatrix.counts',
        );
    });

    it('accepts each bounded artifact basis', () => {
        const artifacts: ArtifactProvenance[] = [
            {
                model,
                dataset,
                objectiveKey: 'o2.1.example',
                basis: {
                    kind: 'full-split',
                    split: 'test',
                    sampleCount: 90,
                    populationCount: 90,
                },
            },
            {
                model,
                dataset,
                objectiveKey: 'o2.1.example',
                basis: {
                    kind: 'bounded-sample',
                    split: 'train',
                    sampleCount: 128,
                    populationCount: 210,
                },
            },
            {
                model,
                dataset,
                objectiveKey: 'o2.1.example',
                basis: {
                    kind: 'prediction-grid',
                    pointCount: 1600,
                    domain: [-6, 6, -6, 6],
                },
            },
            {
                model,
                dataset,
                objectiveKey: 'o2.1.example',
                basis: {
                    kind: 'parameter-grid',
                    sampleCount: 10,
                    parameterPositions: 441,
                },
            },
        ];

        for (const artifact of artifacts) {
            expect(() => assertArtifactProvenance(artifact)).not.toThrow();
        }
    });

    it.each([
        ['basis.sampleCount', { kind: 'bounded-sample', split: 'train', sampleCount: 211, populationCount: 210 }],
        ['basis.domain', { kind: 'prediction-grid', pointCount: 40, domain: [6, -6, -6, 6] }],
        ['basis.parameterPositions', { kind: 'parameter-grid', sampleCount: 10, parameterPositions: 0 }],
    ])('rejects invalid artifact %s', (path, basis) => {
        const artifact = { model, dataset, objectiveKey: 'o2.1.example', basis };

        expect(() => assertArtifactProvenance(artifact)).toThrow(path);
    });
});
