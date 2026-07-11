import { describe, expect, it } from 'vitest';
import {
    PREPARED_PRESETS,
    prepareExperimentDocument,
    type ExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
} from '@nn-playground/shared';
import { projectPreparedExperiment } from './legacyProjection.ts';

async function prepare(document: ExperimentDocumentV2): Promise<PreparedExperimentDocumentV2> {
    const result = await prepareExperimentDocument(document);
    if (!result.ok) throw new Error(JSON.stringify(result.issues));
    return result.value;
}

describe('legacy prepared-experiment projection', () => {
    it('maps binary, multiclass, and regression task contracts without deriving output fields itself', () => {
        const binary = projectPreparedExperiment(PREPARED_PRESETS[0].prepared);
        const multiclass = projectPreparedExperiment(
            PREPARED_PRESETS.find((entry) => entry.id === 'three-class-clusters')!.prepared,
        );
        const regression = projectPreparedExperiment(
            PREPARED_PRESETS.find((entry) => entry.id === 'regression-plane')!.prepared,
        );

        expect(binary.data.problemType).toBe('classification');
        expect(binary.network).toMatchObject({ outputSize: 1, outputActivation: 'sigmoid' });
        expect(binary.training.lossType).toBe('crossEntropy');
        expect(multiclass.data.problemType).toBe('classification');
        expect(multiclass.network).toMatchObject({ outputSize: 3, outputActivation: 'softmax' });
        expect(multiclass.training.lossType).toBe('categoricalCrossEntropy');
        expect(regression.data.problemType).toBe('regression');
        expect(regression.network).toMatchObject({ outputSize: 1, outputActivation: 'linear' });
        expect(regression.training.lossType).toBe('mse');
    });

    it('maps every active V2 optimizer, schedule, clipping, Huber, and penalty field exactly', async () => {
        const base = PREPARED_PRESETS.find((entry) => entry.id === 'regression-plane')!.prepared.document;
        const prepared = await prepare({
            ...base,
            recipe: {
                ...base.recipe,
                training: {
                    ...base.recipe.training,
                    learningRate: 0.02,
                    schedule: { kind: 'step', interval: 17, gamma: 0.6 },
                    optimizer: { kind: 'sgd-momentum', momentum: 0.77 },
                    gradientClipping: {
                        kind: 'global-norm',
                        maximumNorm: 0.45,
                        scope: 'total-objective-gradient',
                    },
                },
                objective: {
                    dataLoss: { kind: 'huber', delta: 0.75 },
                    penalty: { kind: 'l2', coefficient: 0.03, applyTo: 'weights' },
                    reduction: 'mean-per-sample',
                },
            },
        });

        expect(projectPreparedExperiment(prepared).training).toEqual({
            learningRate: 0.02,
            batchSize: 10,
            lossType: 'huber',
            optimizer: 'sgdMomentum',
            momentum: 0.77,
            regularization: 'l2',
            regularizationRate: 0.03,
            gradientClip: 0.45,
            huberDelta: 0.75,
            lrSchedule: { type: 'step', stepSize: 17, gamma: 0.6 },
        });

        const cosineAdam = await prepare({
            ...base,
            recipe: {
                ...base.recipe,
                training: {
                    ...base.recipe.training,
                    schedule: { kind: 'cosine', totalSteps: 321, minimumRate: 0.0002 },
                    optimizer: { kind: 'adam', beta1: 0.8, beta2: 0.98, epsilon: 1e-7 },
                },
            },
        });
        expect(projectPreparedExperiment(cosineAdam).training).toMatchObject({
            optimizer: 'adam',
            adamBeta1: 0.8,
            adamBeta2: 0.98,
            adamEps: 1e-7,
            lrSchedule: { type: 'cosine', totalSteps: 321, minLr: 0.0002 },
        });
    });

    it('returns detached compatibility objects and never exposes compiled objective functions', () => {
        const prepared = PREPARED_PRESETS[0].prepared;
        const projection = projectPreparedExperiment(prepared);

        expect(projection.network).not.toBe(prepared.compiled.network);
        expect(projection.network.hiddenLayers).not.toBe(prepared.compiled.network.hiddenLayers);
        expect(projection.training).not.toHaveProperty('objective');
        expect(projection.data).toEqual({
            dataset: prepared.document.recipe.task.dataset,
            problemType: 'classification',
            trainTestRatio: prepared.document.recipe.data.trainFraction,
            noise: prepared.document.recipe.data.noise,
            numSamples: prepared.document.recipe.data.sampleCount,
            seed: prepared.document.recipe.data.seed,
        });
        expect(projection.ui).toEqual(prepared.document.view);
    });

    it('recursively freezes selector-exposed compatibility projections', () => {
        const projection = projectPreparedExperiment(PREPARED_PRESETS[0].prepared);

        expect(Object.isFrozen(projection)).toBe(true);
        expect(Object.isFrozen(projection.network)).toBe(true);
        expect(Object.isFrozen(projection.network.hiddenLayers)).toBe(true);
        expect(Object.isFrozen(projection.training)).toBe(true);
        expect(Object.isFrozen(projection.training.lrSchedule ?? projection.training)).toBe(true);
        expect(Object.isFrozen(projection.data)).toBe(true);
        expect(Object.isFrozen(projection.features)).toBe(true);
        expect(Object.isFrozen(projection.ui)).toBe(true);
        expect(() => projection.network.hiddenLayers.push(99)).toThrow(TypeError);
        expect(projection.network.hiddenLayers).toEqual(
            PREPARED_PRESETS[0].prepared.document.recipe.model.hiddenLayers,
        );
    });
});
