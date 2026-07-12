import { describe, expect, it } from 'vitest';
import {
    compileObjective,
    Network,
    validateNetworkSessionStateV2,
    type CompiledTrainingContractV2,
    type ExpectedNetworkSessionShape,
    type NetworkConfig,
    type NetworkSessionStateV2,
    type OptimizerSpecV2,
} from '../index.js';

const networkConfig: NetworkConfig = {
    inputSize: 2,
    hiddenLayers: [2],
    outputSize: 1,
    activation: 'tanh',
    outputActivation: 'linear',
    weightInit: 'xavier',
    seed: 17,
};

const adam: OptimizerSpecV2 = {
    kind: 'adam',
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};

const momentum: OptimizerSpecV2 = {
    kind: 'sgd-momentum',
    momentum: 0.9,
};

const sgd: OptimizerSpecV2 = { kind: 'sgd' };

const expectedShape: ExpectedNetworkSessionShape = {
    layerSizes: [2, 2, 1],
    maximumBytes: 262_144,
};

function makeTraining(optimizer: OptimizerSpecV2): CompiledTrainingContractV2 {
    return {
        learningRate: 0.05,
        batchSize: 2,
        schedule: { kind: 'constant' },
        optimizer,
        gradientClipping: { kind: 'none' },
        objective: compileObjective({
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        }, networkConfig),
    };
}

function cloneSessionState(state: NetworkSessionStateV2): NetworkSessionStateV2 {
    const network = {
        layers: state.network.layers.map((layer) => ({
            inputSize: layer.inputSize,
            outputSize: layer.outputSize,
            weights: new Float64Array(layer.weights),
            biases: new Float64Array(layer.biases),
        })),
    };
    switch (state.optimizer.kind) {
        case 'sgd':
            return {
                network,
                optimizer: { ...state.optimizer },
            };
        case 'sgd-momentum':
            return {
                network,
                optimizer: {
                    ...state.optimizer,
                    weightVelocity: state.optimizer.weightVelocity.map((buffer) => new Float64Array(buffer)),
                    biasVelocity: state.optimizer.biasVelocity.map((buffer) => new Float64Array(buffer)),
                },
            };
        case 'adam':
            return {
                network,
                optimizer: {
                    ...state.optimizer,
                    firstWeightMoment: state.optimizer.firstWeightMoment.map((buffer) => new Float64Array(buffer)),
                    firstBiasMoment: state.optimizer.firstBiasMoment.map((buffer) => new Float64Array(buffer)),
                    secondWeightMoment: state.optimizer.secondWeightMoment.map((buffer) => new Float64Array(buffer)),
                    secondBiasMoment: state.optimizer.secondBiasMoment.map((buffer) => new Float64Array(buffer)),
                },
            };
    }
}

function expectBuffersToBeZero(buffers: readonly Float64Array[], expectedLengths: readonly number[]): void {
    expect(buffers.map((buffer) => buffer.length)).toEqual(expectedLengths);
    for (const buffer of buffers) {
        expect(Array.from(buffer).every((value) => value === 0)).toBe(true);
    }
}

function makeTrainedAdamNetwork(): Network {
    const network = new Network(networkConfig);
    network.trainBatchV2([[1, -1], [0.5, 0.25]], [[0.75], [-0.5]], makeTraining(adam));
    network.forward([0.25, -0.5]);
    network.backward([0.5], 'mse');
    return network;
}

function expectRejectedWithoutMutation(
    network: Network,
    candidate: unknown,
    trainingStep = 12,
    expectedOptimizer: OptimizerSpecV2 = adam,
): void {
    const before = {
        session: network.captureSessionState(expectedOptimizer),
        trainingStep: network.getStep(),
        revision: network.getRevision(),
        weightGrads: network.getWeightGrads(),
        biasGrads: network.getBiasGrads(),
        recentGradient: network.getRecentGradientSnapshot(),
    };

    let restoreError: unknown;
    try {
        network.restoreSessionState(candidate, expectedOptimizer, trainingStep);
    } catch (error) {
        restoreError = error;
    }

    expect(network.captureSessionState(expectedOptimizer)).toEqual(before.session);
    expect(network.getStep()).toBe(before.trainingStep);
    expect(network.getRevision()).toBe(before.revision);
    expect(network.getWeightGrads()).toEqual(before.weightGrads);
    expect(network.getBiasGrads()).toEqual(before.biasGrads);
    expect(network.getRecentGradientSnapshot()).toEqual(before.recentGradient);
    expect(restoreError).toBeInstanceOf(RangeError);
}

describe('Network V2 session state', () => {
    it('round-trips Adam parameters, moments, optimizer step, and supplied training step without aliasing', () => {
        const source = new Network(networkConfig);
        source.trainBatchV2([[1, -1], [0.5, 0.25]], [[0.75], [-0.5]], makeTraining(adam));
        source.trainBatchV2([[-0.25, 0.75], [1.5, 0.5]], [[0.25], [1]], makeTraining(adam));
        const captured = source.captureSessionState(adam);
        const expected = source.captureSessionState(adam);
        const legacyCheckpoint = source.createCheckpoint();

        expect(captured.network.layers.map((layer) => layer.weights)).toEqual(legacyCheckpoint.weights);
        expect(captured.network.layers.map((layer) => layer.biases)).toEqual(legacyCheckpoint.biases);
        expect(captured.optimizer.optimizerStep).toBe(legacyCheckpoint.optimizerStep);
        if (captured.optimizer.kind !== 'adam') throw new Error('expected Adam state');
        expect(captured.optimizer.firstWeightMoment).toEqual(legacyCheckpoint.mWeights);
        expect(captured.optimizer.firstBiasMoment).toEqual(legacyCheckpoint.mBiases);
        expect(captured.optimizer.secondWeightMoment).toEqual(legacyCheckpoint.vWeights);
        expect(captured.optimizer.secondBiasMoment).toEqual(legacyCheckpoint.vBiases);
        expect(captured.optimizer.firstWeightMoment.some((buffer) => buffer.some((value) => value !== 0)))
            .toBe(true);

        const restored = new Network({ ...networkConfig, seed: 999 });
        restored.restoreSessionState(captured, adam, 37);

        expect(restored.captureSessionState(adam)).toEqual(expected);
        expect(restored.getStep()).toBe(37);
        expect(restored.getRevision()).toBe(1);

        captured.network.layers[0].weights[0] = 999;
        if (captured.optimizer.kind === 'adam') {
            captured.optimizer.firstWeightMoment[0][0] = 999;
            captured.optimizer.secondBiasMoment[0][0] = 999;
        }
        expect(restored.captureSessionState(adam)).toEqual(expected);

        const independentCapture = restored.captureSessionState(adam);
        independentCapture.network.layers[0].biases[0] = 999;
        expect(restored.captureSessionState(adam)).toEqual(expected);
    });

    it('builds a detached restore candidate at an explicit future revision', () => {
        const source = makeTrainedAdamNetwork();
        const captured = source.captureSessionState(adam);
        const candidate = new Network({ ...networkConfig, seed: 999 });

        candidate.restoreSessionState(captured, adam, 19, 41);

        expect(candidate.getStep()).toBe(19);
        expect(candidate.getRevision()).toBe(41);
        expect(candidate.captureSessionState(adam)).toEqual(captured);
    });

    it.each([
        ['Adam', adam],
        ['momentum', momentum],
    ] as const)('captures correctly shaped all-zero %s state before the first update', (_name, optimizer) => {
        const network = new Network(networkConfig);
        const captured = network.captureSessionState(optimizer);

        expect(captured.optimizer.optimizerStep).toBe(0);
        if (captured.optimizer.kind === 'adam') {
            expectBuffersToBeZero(captured.optimizer.firstWeightMoment, [4, 2]);
            expectBuffersToBeZero(captured.optimizer.firstBiasMoment, [2, 1]);
            expectBuffersToBeZero(captured.optimizer.secondWeightMoment, [4, 2]);
            expectBuffersToBeZero(captured.optimizer.secondBiasMoment, [2, 1]);
        } else if (captured.optimizer.kind === 'sgd-momentum') {
            expectBuffersToBeZero(captured.optimizer.weightVelocity, [4, 2]);
            expectBuffersToBeZero(captured.optimizer.biasVelocity, [2, 1]);
        } else {
            throw new Error('expected stateful optimizer capture');
        }
    });

    it('returns a validated deep clone that cannot alias its source or the live Network', () => {
        const network = new Network(networkConfig);
        network.trainBatchV2([[1, -1]], [[0.75]], makeTraining(adam));
        const captured = network.captureSessionState(adam);
        const capturedBeforeMutation = cloneSessionState(captured);
        const validated = validateNetworkSessionStateV2(captured, expectedShape, adam);

        validated.network.layers[0].weights[0] = 123;
        if (validated.optimizer.kind !== 'adam') throw new Error('expected Adam state');
        validated.optimizer.firstBiasMoment[0][0] = 456;

        expect(captured).toEqual(capturedBeforeMutation);
        expect(network.captureSessionState(adam)).toEqual(capturedBeforeMutation);

        captured.network.layers[0].biases[0] = 789;
        if (captured.optimizer.kind !== 'adam') throw new Error('expected Adam state');
        captured.optimizer.secondWeightMoment[0][0] = 789;
        expect(validated.network.layers[0].biases[0]).not.toBe(789);
        expect(validated.optimizer.secondWeightMoment[0][0]).not.toBe(789);
    });

    it('rejects deceptive typed-array accessors before an earlier live layer can mutate', () => {
        class DeceptiveFloat64Array extends Float64Array {
            get length(): number {
                return 2;
            }

            get byteLength(): number {
                return 16;
            }
        }

        const network = new Network(networkConfig);
        network.trainBatchV2([[1, -1]], [[0.75]], makeTraining(sgd));
        const candidate = cloneSessionState(network.captureSessionState(sgd));
        candidate.network.layers[0].weights[0] += 100;
        const deceptiveLaterWeights = new DeceptiveFloat64Array(3);
        deceptiveLaterWeights.set(candidate.network.layers[1].weights);
        candidate.network.layers[1].weights = deceptiveLaterWeights;

        let validationError: unknown;
        try {
            validateNetworkSessionStateV2(candidate, expectedShape, sgd);
        } catch (error) {
            validationError = error;
        }

        expectRejectedWithoutMutation(network, candidate, 12, sgd);
        expect(validationError).toBeInstanceOf(RangeError);
    });

    it('rejects a sparse later network layer before an earlier live layer can mutate', () => {
        const network = new Network(networkConfig);
        network.trainBatchV2([[1, -1]], [[0.75]], makeTraining(sgd));
        const candidate = cloneSessionState(network.captureSessionState(sgd));
        candidate.network.layers[0].weights[0] += 100;
        candidate.network.layers.length = 1;
        candidate.network.layers.length = 2;

        let validationError: unknown;
        try {
            validateNetworkSessionStateV2(candidate, expectedShape, sgd);
        } catch (error) {
            validationError = error;
        }

        expectRejectedWithoutMutation(network, candidate, 12, sgd);
        expect(validationError).toBeInstanceOf(RangeError);
    });

    it('rejects a sparse later Adam buffer with an inherited fallback before any mutation', () => {
        const network = makeTrainedAdamNetwork();
        const candidate = cloneSessionState(network.captureSessionState(adam));
        candidate.network.layers[0].weights[0] += 100;
        if (candidate.optimizer.kind !== 'adam') throw new Error('expected Adam state');
        const inheritedLaterBuffer = candidate.optimizer.secondBiasMoment[1];
        candidate.optimizer.secondBiasMoment.length = 1;
        candidate.optimizer.secondBiasMoment.length = 2;
        const sparseBufferPrototype = Object.create(Array.prototype) as Float64Array[];
        sparseBufferPrototype[1] = inheritedLaterBuffer;
        Object.setPrototypeOf(candidate.optimizer.secondBiasMoment, sparseBufferPrototype);

        let validationError: unknown;
        try {
            validateNetworkSessionStateV2(candidate, expectedShape, adam);
        } catch (error) {
            validationError = error;
        }

        expectRejectedWithoutMutation(network, candidate);
        expect(validationError).toBeInstanceOf(RangeError);
    });

    it.each([
        ['wrong layer count', (state: NetworkSessionStateV2) => {
            state.network.layers.pop();
        }],
        ['wrong declared input dimension', (state: NetworkSessionStateV2) => {
            state.network.layers[0].inputSize++;
        }],
        ['wrong declared output dimension', (state: NetworkSessionStateV2) => {
            state.network.layers[0].outputSize++;
        }],
        ['wrong weight length', (state: NetworkSessionStateV2) => {
            state.network.layers[0].weights = new Float64Array(3);
        }],
        ['wrong bias length', (state: NetworkSessionStateV2) => {
            state.network.layers[0].biases = new Float64Array(1);
        }],
        ['non-finite parameter', (state: NetworkSessionStateV2) => {
            state.network.layers[0].weights[0] = Number.NaN;
        }],
        ['wrong moment layer count', (state: NetworkSessionStateV2) => {
            if (state.optimizer.kind !== 'adam') throw new Error('expected Adam state');
            state.optimizer.firstWeightMoment.pop();
        }],
        ['wrong moment length', (state: NetworkSessionStateV2) => {
            if (state.optimizer.kind !== 'adam') throw new Error('expected Adam state');
            state.optimizer.secondBiasMoment[0] = new Float64Array(1);
        }],
        ['non-finite moment', (state: NetworkSessionStateV2) => {
            if (state.optimizer.kind !== 'adam') throw new Error('expected Adam state');
            state.optimizer.firstWeightMoment[0][0] = Number.POSITIVE_INFINITY;
        }],
        ['optimizer mismatch', (state: NetworkSessionStateV2) => {
            state.optimizer = { kind: 'sgd', optimizerStep: state.optimizer.optimizerStep };
        }],
        ['negative optimizer step', (state: NetworkSessionStateV2) => {
            state.optimizer.optimizerStep = -1;
        }],
        ['fractional optimizer step', (state: NetworkSessionStateV2) => {
            state.optimizer.optimizerStep = 1.5;
        }],
    ] as const)('rejects %s without mutating any live state', (_name, mutate) => {
        const network = makeTrainedAdamNetwork();
        const candidate = cloneSessionState(network.captureSessionState(adam));
        mutate(candidate);

        expectRejectedWithoutMutation(network, candidate);
    });

    it.each([
        ['negative', -1],
        ['fractional', 1.5],
    ] as const)('rejects a %s supplied training step without mutating any live state', (_name, trainingStep) => {
        const network = makeTrainedAdamNetwork();
        const candidate = network.captureSessionState(adam);

        expectRejectedWithoutMutation(network, candidate, trainingStep);
    });

    it('rejects an over-256-KiB payload without mutating the live Network', () => {
        const largeConfig: NetworkConfig = {
            inputSize: 128,
            hiddenLayers: [],
            outputSize: 128,
            activation: 'linear',
            outputActivation: 'linear',
            weightInit: 'zeros',
            seed: 1,
        };
        const network = new Network(largeConfig);
        const parameterCount = 128 * 128;
        const candidate: NetworkSessionStateV2 = {
            network: {
                layers: [{
                    inputSize: 128,
                    outputSize: 128,
                    weights: new Float64Array(parameterCount),
                    biases: new Float64Array(128),
                }],
            },
            optimizer: {
                kind: 'adam',
                optimizerStep: 0,
                firstWeightMoment: [new Float64Array(parameterCount)],
                firstBiasMoment: [new Float64Array(128)],
                secondWeightMoment: [new Float64Array(parameterCount)],
                secondBiasMoment: [new Float64Array(128)],
            },
        };
        const before = {
            weights: network.getWeights(),
            biases: network.getBiases(),
            step: network.getStep(),
            revision: network.getRevision(),
            weightGrads: network.getWeightGrads(),
            biasGrads: network.getBiasGrads(),
            recentGradient: network.getRecentGradientSnapshot(),
        };

        expect(() => network.restoreSessionState(candidate, adam, 10)).toThrow(RangeError);

        expect(network.getWeights()).toEqual(before.weights);
        expect(network.getBiases()).toEqual(before.biases);
        expect(network.getStep()).toBe(before.step);
        expect(network.getRevision()).toBe(before.revision);
        expect(network.getWeightGrads()).toEqual(before.weightGrads);
        expect(network.getBiasGrads()).toEqual(before.biasGrads);
        expect(network.getRecentGradientSnapshot()).toEqual(before.recentGradient);
    });

    it('rejects capture with a different optimizer kind after optimizer state exists', () => {
        const network = new Network(networkConfig);
        network.trainBatchV2([[1, -1]], [[0.75]], makeTraining(adam));
        const before = network.captureSessionState(adam);

        expect(() => network.captureSessionState(momentum)).toThrow(RangeError);
        expect(network.captureSessionState(adam)).toEqual(before);
    });

    it('restores once, clears accumulated and recent gradients, and increments revision exactly once', () => {
        const source = new Network(networkConfig);
        source.trainBatchV2([[1, -1]], [[0.75]], makeTraining(adam));
        const captured = source.captureSessionState(adam);

        const target = new Network({ ...networkConfig, seed: 42 });
        target.trainBatchV2([[-1, 0.5]], [[-0.25]], makeTraining(adam));
        target.forward([0.5, 0.5]);
        target.backward([1], 'mse');
        const revisionBeforeRestore = target.getRevision();

        target.restoreSessionState(captured, adam, 19);

        expect(target.getRevision()).toBe(revisionBeforeRestore + 1);
        expect(target.getStep()).toBe(19);
        expect(target.captureSessionState(adam)).toEqual(captured);
        expect(target.getWeightGrads().flat(2).every((value) => value === 0)).toBe(true);
        expect(target.getBiasGrads().flat().every((value) => value === 0)).toBe(true);
        expect(target.getRecentGradientSnapshot().weightGradients.flat(2).every((value) => value === 0)).toBe(true);
        expect(target.getRecentGradientSnapshot().biasGradients.flat().every((value) => value === 0)).toBe(true);
    });

    it('captures and restores SGD without optimizer arrays while preserving its optimizer step', () => {
        const source = new Network(networkConfig);
        source.trainBatchV2([[1, -1]], [[0.75]], makeTraining(sgd));
        source.trainBatchV2([[0.5, 0.25]], [[-0.5]], makeTraining(sgd));
        const captured = source.captureSessionState(sgd);

        expect(captured.optimizer).toEqual({ kind: 'sgd', optimizerStep: 2 });
        expect(Object.keys(captured.optimizer).sort()).toEqual(['kind', 'optimizerStep']);

        const target = new Network({ ...networkConfig, seed: 999 });
        target.restoreSessionState(captured, sgd, 23);

        expect(target.captureSessionState(sgd)).toEqual(captured);
        expect(target.getStep()).toBe(23);
    });
});
