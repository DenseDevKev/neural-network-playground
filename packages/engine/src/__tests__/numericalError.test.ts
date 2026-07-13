import { describe, expect, it } from 'vitest';
import {
    batchLoss,
    binaryCrossEntropyWithLogits,
    categoricalCrossEntropy,
    categoricalCrossEntropyWithLogits,
    compileObjective,
    meanSquaredError,
    huberLoss,
    Network,
    NonFiniteNumericalError,
    softmax,
    type CompiledTrainingContractV2,
} from '../index.js';

function captureNonFinite(action: () => unknown): NonFiniteNumericalError {
    try {
        action();
    } catch (error) {
        expect(error).toBeInstanceOf(NonFiniteNumericalError);
        return error as NonFiniteNumericalError;
    }
    throw new Error('expected a NonFiniteNumericalError');
}

describe('NonFiniteNumericalError', () => {
    it('is an exported RangeError with structured path and value fields', () => {
        const error = new NonFiniteNumericalError('logits[2]', Number.NaN);

        expect(error).toBeInstanceOf(RangeError);
        expect(error.name).toBe('NonFiniteNumericalError');
        expect(error.path).toBe('logits[2]');
        expect(Number.isNaN(error.value)).toBe(true);
    });

    it('identifies the exact non-finite softmax input', () => {
        const error = captureNonFinite(() => softmax([0, Number.NaN, 1]));

        expect(error.path).toBe('logits[1]');
        expect(Number.isNaN(error.value)).toBe(true);
    });

    it('leaves a finite-but-invalid zero softmax normalizer as a plain RangeError', () => {
        let reads = 0;
        const changingLogit = {
            length: 1,
            get 0(): number {
                reads++;
                return reads === 1 ? 0 : Number.NEGATIVE_INFINITY;
            },
        };

        let thrown: unknown;
        try {
            softmax(changingLogit);
        } catch (error) {
            thrown = error;
        }

        expect(thrown).toBeInstanceOf(RangeError);
        expect(thrown).not.toBeInstanceOf(NonFiniteNumericalError);
    });

    it('identifies non-finite categorical loss inputs', () => {
        const error = captureNonFinite(() => categoricalCrossEntropy(
            [Number.POSITIVE_INFINITY],
            [1],
        ));

        expect(error.path).toBe('probabilities[0]');
        expect(error.value).toBe(Number.POSITIVE_INFINITY);
    });

    it('classifies finite-input objective overflow with the computed value', () => {
        const mseError = captureNonFinite(() => meanSquaredError(
            [Number.MAX_VALUE],
            [-Number.MAX_VALUE],
        ));
        const categoricalError = captureNonFinite(() => categoricalCrossEntropyWithLogits(
            [Number.MAX_VALUE, -Number.MAX_VALUE],
            [0, 1],
        ));

        expect(mseError).toMatchObject({
            path: 'meanSquaredError.sum',
            value: Number.POSITIVE_INFINITY,
        });
        expect(categoricalError).toMatchObject({
            path: 'categoricalCrossEntropyWithLogits.loss',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('classifies non-finite objective targets and Huber overflow without message matching', () => {
        const targetError = captureNonFinite(() => binaryCrossEntropyWithLogits(0, Number.NaN));
        const huberError = captureNonFinite(() => huberLoss(
            [Number.MAX_VALUE],
            [-Number.MAX_VALUE],
            1,
        ));

        expect(targetError.path).toBe('target');
        expect(Number.isNaN(targetError.value)).toBe(true);
        expect(huberError).toMatchObject({
            path: 'huberLoss.sum',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('classifies a custom scalar loss overflow before batch aggregation publishes it', () => {
        const error = captureNonFinite(() => batchLoss(
            { loss: () => Number.POSITIVE_INFINITY, dloss: () => 0 },
            [1],
            [0],
        ));

        expect(error).toMatchObject({
            path: 'batchLoss[0]',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('surfaces a real forward-pass overflow from V2 training without advancing state', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        });
        network.setWeight(0, 0, 0, Number.MAX_VALUE);
        const objective = compileObjective({
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        }, network.config);
        const training = {
            learningRate: 0.1,
            batchSize: 1,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: { kind: 'none' },
            objective,
        } satisfies CompiledTrainingContractV2;
        const before = network.createCheckpoint();

        const error = captureNonFinite(() => network.trainBatchV2(
            [[Number.MAX_VALUE]],
            [[0]],
            training,
        ));

        expect(error).toMatchObject({
            path: 'logits[0]',
            value: Number.POSITIVE_INFINITY,
        });
        expect(network.getStep()).toBe(0);
        expect(network.getRevision()).toBe(1);
        expect(network.createCheckpoint()).toEqual(before);
    });

    it('keeps non-finite network contract fields structurally identifiable', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        });
        const objective = compileObjective({
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        }, network.config);
        const invalidTraining = {
            learningRate: 0.1,
            batchSize: Number.NaN,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: { kind: 'none' },
            objective,
        } satisfies CompiledTrainingContractV2;
        const invalidCheckpoint = network.createCheckpoint();
        invalidCheckpoint.currentStep = Number.POSITIVE_INFINITY;

        const trainingError = captureNonFinite(() => network.trainBatchV2(
            [[0]],
            [[0]],
            invalidTraining,
        ));
        const checkpointError = captureNonFinite(() => network.restoreCheckpoint(invalidCheckpoint));

        expect(trainingError).toMatchObject({ path: 'batchSize' });
        expect(Number.isNaN(trainingError.value)).toBe(true);
        expect(checkpointError).toMatchObject({
            path: 'checkpoint.currentStep',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('classifies a non-finite multiclass boundary confidence from network softmax overflow', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 3,
            activation: 'linear',
            outputActivation: 'softmax',
            weightInit: 'zeros',
            seed: 1,
        });
        network.setWeight(0, 0, 0, Number.MAX_VALUE);

        const error = captureNonFinite(() => network.predictMulticlassBoundaryInto(
            [[Number.MAX_VALUE]],
            new Uint8Array(1),
            new Float64Array(1),
        ));

        expect(error.path).toBe('multiclassBoundary.confidence[0]');
        expect(Number.isNaN(error.value)).toBe(true);
    });

    it('rejects finite parameters that overflow the Float32 wire format', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        });
        network.setWeight(0, 0, 0, 1e100);
        const weightError = captureNonFinite(() => network.getWeightsFlat());
        network.setWeight(0, 0, 0, 0);
        network.setBias(0, 0, 1e100);
        const biasError = captureNonFinite(() => network.getBiasesFlat());

        expect(weightError).toMatchObject({
            path: 'weights[0][0]',
            value: Number.POSITIVE_INFINITY,
        });
        expect(biasError).toMatchObject({
            path: 'biases[0][0]',
            value: Number.POSITIVE_INFINITY,
        });
    });

    it('preserves huge finite grids in Float64 targets and rejects Float32 overflow', () => {
        const network = new Network({
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        });
        network.setWeight(0, 0, 0, 1e100);

        const output64 = new Float64Array(1);
        const neurons64 = new Float64Array(1);
        network.predictGridInto([[1]], output64);
        network.predictGridWithNeuronsInto([[1]], output64, neurons64);
        expect(output64[0]).toBe(1e100);
        expect(neurons64[0]).toBe(1e100);

        const outputError = captureNonFinite(() => network.predictGridInto(
            [[1]],
            new Float32Array(1),
        ));
        const neuronError = captureNonFinite(() => network.predictGridWithNeuronsInto(
            [[1]],
            new Float64Array(1),
            new Float32Array(1),
        ));
        expect(outputError).toMatchObject({
            path: 'predictionGrid.output[0]',
            value: Number.POSITIVE_INFINITY,
        });
        expect(neuronError).toMatchObject({
            path: 'predictionGrid.neurons[0][0]',
            value: Number.POSITIVE_INFINITY,
        });
    });
});
