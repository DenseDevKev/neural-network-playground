import { describe, expect, it } from 'vitest';
import {
    compileExperimentRecipe,
    Network,
    type CompilableExperimentRecipe,
} from '../index.js';
// @ts-expect-error -- Vitest's raw loader exposes this source as a string.
import trainingContractSource from '../trainingContract.ts?raw';

const commonRecipe = {
    data: {
        sampleCount: 300,
        trainFraction: 0.5,
        noise: 0,
        seed: 42,
    },
    inputs: {
        featureIds: ['x', 'y'],
    },
    model: {
        hiddenLayers: [4, 4],
        hiddenActivation: 'tanh',
        initialization: 'xavier',
        seed: 73,
    },
    training: {
        batchSize: 10,
        learningRate: 0.03,
        schedule: { kind: 'constant' },
        optimizer: { kind: 'sgd' },
        gradientClipping: { kind: 'none' },
    },
} as const;

function binaryRecipe(): CompilableExperimentRecipe {
    return {
        ...commonRecipe,
        task: { kind: 'binary-classification', dataset: 'xor' },
        objective: {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        },
    };
}

describe('compileExperimentRecipe', () => {
    it('derives the complete binary network, data, target, and objective contract', () => {
        const recipe = binaryRecipe();

        const compiled = compileExperimentRecipe(recipe);

        expect(compiled.network).toEqual({
            inputSize: 2,
            hiddenLayers: [4, 4],
            outputSize: 1,
            activation: 'tanh',
            outputActivation: 'sigmoid',
            weightInit: 'xavier',
            seed: 73,
        });
        expect(compiled.data).toEqual({
            dataset: 'xor',
            sampleCount: 300,
            trainFraction: 0.5,
            noise: 0,
            seed: 42,
        });
        expect(compiled.task).toEqual({
            kind: 'binary-classification',
            dataset: 'xor',
            outputSize: 1,
            outputActivation: 'sigmoid',
            target: { kind: 'scalar', values: [0, 1] },
        });
        expect(compiled.features).toEqual({
            x: true,
            y: true,
            xSquared: false,
            ySquared: false,
            xy: false,
            sinX: false,
            sinY: false,
            cosX: false,
            cosY: false,
        });
        expect(compiled.training).toMatchObject({
            learningRate: 0.03,
            batchSize: 10,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: { kind: 'none' },
        });
        expect(compiled.objective.spec).toEqual(recipe.objective);
        expect(compiled.training.objective).toBe(compiled.objective);
    });

    it('derives three softmax outputs and a length-three one-hot target', () => {
        const recipe: CompilableExperimentRecipe = {
            ...commonRecipe,
            task: {
                kind: 'multiclass-classification',
                dataset: 'three-class-clusters',
            },
            objective: {
                dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
                penalty: { kind: 'l2', coefficient: 0.1, applyTo: 'weights' },
                reduction: 'mean-per-sample',
            },
        };

        const compiled = compileExperimentRecipe(recipe);

        expect(compiled.network.outputSize).toBe(3);
        expect(compiled.network.outputActivation).toBe('softmax');
        expect(compiled.task).toEqual({
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
            outputSize: 3,
            outputActivation: 'softmax',
            target: { kind: 'one-hot', length: 3 },
        });
        expect(compiled.training.objective).toBe(compiled.objective);
        expect(compiled.objective.spec).toEqual(recipe.objective);
    });

    it.each([
        { dataLoss: { kind: 'mean-squared-error' } as const },
        { dataLoss: { kind: 'huber', delta: 0.75 } as const },
    ])('derives one linear output and a finite scalar target for $dataLoss.kind', ({ dataLoss }) => {
        const recipe: CompilableExperimentRecipe = {
            ...commonRecipe,
            task: { kind: 'regression', dataset: 'reg-plane' },
            objective: {
                dataLoss,
                penalty: { kind: 'none' },
                reduction: 'mean-per-sample',
            },
        };

        const compiled = compileExperimentRecipe(recipe);

        expect(compiled.network.outputSize).toBe(1);
        expect(compiled.network.outputActivation).toBe('linear');
        expect(compiled.task).toEqual({
            kind: 'regression',
            dataset: 'reg-plane',
            outputSize: 1,
            outputActivation: 'linear',
            target: { kind: 'scalar', finite: true },
        });
        expect(compiled.training.objective).toBe(compiled.objective);
        expect(compiled.objective.spec).toEqual(recipe.objective);
    });

    it('derives FeatureFlags from the ordered unique feature IDs without changing input size', () => {
        const recipe: CompilableExperimentRecipe = {
            ...binaryRecipe(),
            inputs: { featureIds: ['cosY', 'xSquared', 'xy', 'sinX'] },
        };

        const compiled = compileExperimentRecipe(recipe);

        expect(compiled.network.inputSize).toBe(4);
        expect(compiled.features).toEqual({
            x: false,
            y: false,
            xSquared: true,
            ySquared: false,
            xy: true,
            sinX: true,
            sinY: false,
            cosX: false,
            cosY: true,
        });
    });

    it.each([
        {
            schedule: { kind: 'step', interval: 17, gamma: 0.375 } as const,
            optimizer: { kind: 'sgd-momentum', momentum: 0.625 } as const,
        },
        {
            schedule: { kind: 'cosine', totalSteps: 913, minimumRate: 0.00017 } as const,
            optimizer: {
                kind: 'adam',
                beta1: 0.81,
                beta2: 0.972,
                epsilon: 3e-7,
            } as const,
        },
    ])('retains the canonical $schedule.kind schedule and $optimizer.kind optimizer exactly', ({
        schedule,
        optimizer,
    }) => {
        const recipe: CompilableExperimentRecipe = {
            ...binaryRecipe(),
            training: {
                batchSize: 37,
                learningRate: 0.012345,
                schedule,
                optimizer,
                gradientClipping: {
                    kind: 'global-norm',
                    maximumNorm: 0.875,
                    scope: 'total-objective-gradient',
                },
            },
        };

        const compiled = compileExperimentRecipe(recipe);

        expect(compiled.training).toMatchObject({
            batchSize: 37,
            learningRate: 0.012345,
            schedule,
            optimizer,
            gradientClipping: {
                kind: 'global-norm',
                maximumNorm: 0.875,
                scope: 'total-objective-gradient',
            },
        });
        expect(compiled.training.schedule).not.toBe(schedule);
        expect(compiled.training.optimizer).not.toBe(optimizer);
        expect(compiled.training.gradientClipping).not.toBe(recipe.training.gradientClipping);
    });

    it('runs one compiled objective through Network training, evaluation, and trace evidence', () => {
        const recipe: CompilableExperimentRecipe = {
            data: {
                sampleCount: 20,
                trainFraction: 0.5,
                noise: 0,
                seed: 19,
            },
            inputs: { featureIds: ['x'] },
            model: {
                hiddenLayers: [],
                hiddenActivation: 'linear',
                initialization: 'zeros',
                seed: 23,
            },
            training: {
                batchSize: 2,
                learningRate: 0.1,
                schedule: { kind: 'constant' },
                optimizer: { kind: 'sgd' },
                gradientClipping: { kind: 'none' },
            },
            task: { kind: 'regression', dataset: 'reg-plane' },
            objective: {
                dataLoss: { kind: 'mean-squared-error' },
                penalty: { kind: 'none' },
                reduction: 'mean-per-sample',
            },
        };
        const compiled = compileExperimentRecipe(recipe);
        const network = new Network(compiled.network);
        const inputs = [[1], [2]];
        const targets = [[1], [2]];

        const trainingResult = network.trainBatchV2(inputs, targets, compiled.training);
        const dataLoss = network.evaluateDataLoss(inputs, targets, compiled.objective);
        const objective = network.evaluateObjective(inputs, targets, compiled.objective);
        const trace = network.tracePredictionV2(inputs[0], targets[0], compiled.objective);

        expect(compiled.training.objective).toBe(compiled.objective);
        expect(trainingResult.objective.dataLoss).toBeCloseTo(2.5, 12);
        expect(trainingResult.objective.regularizationPenalty).toBe(0);
        expect(trainingResult.objective.totalObjective).toBeCloseTo(2.5, 12);
        expect(dataLoss).toBeCloseTo(0.265, 12);
        expect(objective.dataLoss).toBeCloseTo(0.265, 12);
        expect(objective.regularizationPenalty).toBe(0);
        expect(objective.totalObjective).toBeCloseTo(0.265, 12);
        expect(trace.output[0]).toBeCloseTo(0.8, 12);
        expect(trace.sampleDataLoss).toBeCloseTo(0.04, 12);
        expect(trace.regularizationPenalty).toBe(0);
    });

    it.each([
        {
            task: { kind: 'binary-classification', dataset: 'xor' } as const,
            dataLoss: { kind: 'mean-squared-error' } as const,
        },
        {
            task: {
                kind: 'multiclass-classification',
                dataset: 'three-class-clusters',
            } as const,
            dataLoss: { kind: 'binary-cross-entropy-with-logits' } as const,
        },
        {
            task: { kind: 'regression', dataset: 'reg-gauss' } as const,
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' } as const,
        },
    ])('rejects $task.kind with incompatible $dataLoss.kind', ({ task, dataLoss }) => {
        const recipe = {
            ...commonRecipe,
            task,
            objective: {
                dataLoss,
                penalty: { kind: 'none' },
                reduction: 'mean-per-sample',
            },
        } as CompilableExperimentRecipe;

        expect(() => compileExperimentRecipe(recipe)).toThrow(/objective.*task|task.*objective/iu);
    });

    it('is engine-owned and contains no shared, Network, PRNG, or dataset allocation dependency', () => {
        expect(trainingContractSource).not.toMatch(/@nn-playground\/shared/iu);
        expect(trainingContractSource).not.toMatch(/from ['"]\.\/network/iu);
        expect(trainingContractSource).not.toMatch(/from ['"]\.\/prng/iu);
        expect(trainingContractSource).not.toMatch(/from ['"]\.\/datasets/iu);
        expect(trainingContractSource).not.toMatch(/new (?:Network|PRNG)\b/iu);
        expect(trainingContractSource).not.toMatch(/generateDataset/iu);
    });
});
