import { runInNewContext } from 'node:vm';
import { describe, expect, it, vi } from 'vitest';
import {
    SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES,
    validateSessionCheckpointV2,
    type DatasetRevision,
    type SessionCheckpointV2,
    type SessionCheckpointValidationContext,
} from '../index.js';

const MODEL = {
    generationId: 5,
    revision: 3,
    step: 9,
    epoch: 2,
} as const;

const DATASET: DatasetRevision = {
    generatorVersion: 2,
    datasetKey: 'd2.1.dataset',
    trainCount: 4,
    testCount: 2,
};

function context(
    overrides: Partial<SessionCheckpointValidationContext> = {},
): SessionCheckpointValidationContext {
    return {
        recipeFingerprint: 'r2.1.recipe',
        objectiveKey: 'o2.1.objective',
        dataset: DATASET,
        layerSizes: [2, 3, 1],
        optimizer: { kind: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
        ...overrides,
    };
}

function checkpoint(): SessionCheckpointV2 {
    return {
        kind: 'nn-playground-session-checkpoint',
        schemaVersion: 2,
        recipeFingerprint: 'r2.1.recipe',
        objectiveKey: 'o2.1.objective',
        datasetKey: DATASET.datasetKey,
        model: { ...MODEL },
        evaluation: {
            evaluationId: 11,
            trigger: 'checkpoint',
            model: { ...MODEL },
            dataset: { ...DATASET },
            objectiveKey: 'o2.1.objective',
            train: {
                basis: {
                    kind: 'full-split',
                    split: 'train',
                    sampleCount: DATASET.trainCount,
                    populationCount: DATASET.trainCount,
                },
                values: { dataLoss: 0.25 },
            },
            test: {
                basis: {
                    kind: 'full-split',
                    split: 'test',
                    sampleCount: DATASET.testCount,
                    populationCount: DATASET.testCount,
                },
                values: { dataLoss: 0.3 },
            },
            objective: {
                regularizationPenalty: 0.05,
                trainTotalObjective: 0.3,
            },
        },
        network: {
            layers: [
                {
                    inputSize: 2,
                    outputSize: 3,
                    weights: new Float64Array([1, 2, 3, 4, 5, 6]),
                    biases: new Float64Array([0.1, 0.2, 0.3]),
                },
                {
                    inputSize: 3,
                    outputSize: 1,
                    weights: new Float64Array([7, 8, 9]),
                    biases: new Float64Array([0.4]),
                },
            ],
        },
        optimizer: {
            kind: 'adam',
            optimizerStep: 9,
            firstWeightMoment: [new Float64Array(6), new Float64Array(3)],
            firstBiasMoment: [new Float64Array(3), new Float64Array(1)],
            secondWeightMoment: [new Float64Array(6), new Float64Array(3)],
            secondBiasMoment: [new Float64Array(3), new Float64Array(1)],
        },
        cursor: {
            epoch: 2,
            batchStart: 2,
            shuffledIndices: new Uint32Array([2, 0, 3, 1]),
        },
        trajectoryGuarantee: 'parameters-and-optimizer-only',
    };
}

function cloneCheckpoint(source: SessionCheckpointV2 = checkpoint()): SessionCheckpointV2 {
    return structuredClone(source);
}

describe('validateSessionCheckpointV2', () => {
    it('returns a fully detached clone that survives structuredClone round trips', () => {
        const source = checkpoint();
        const parsed = validateSessionCheckpointV2(structuredClone(source), context());

        expect(parsed).toEqual(source);
        expect(parsed).not.toBe(source);
        expect(parsed.network.layers[0].weights).not.toBe(source.network.layers[0].weights);
        expect(parsed.optimizer.kind).toBe('adam');
        if (parsed.optimizer.kind === 'adam') {
            expect(parsed.optimizer.firstWeightMoment[0]).not.toBe(
                source.optimizer.kind === 'adam' ? source.optimizer.firstWeightMoment[0] : null,
            );
        }
        expect(parsed.cursor.shuffledIndices).not.toBe(source.cursor.shuffledIndices);

        source.network.layers[0].weights[0] = 999;
        source.cursor.shuffledIndices[0] = 0;
        expect(parsed.network.layers[0].weights[0]).toBe(1);
        expect([...parsed.cursor.shuffledIndices]).toEqual([2, 0, 3, 1]);
    });

    it.each([
        ['top-level unknown field', (value: SessionCheckpointV2 & Record<string, unknown>) => {
            value.extra = true;
        }],
        ['network unknown field', (value: SessionCheckpointV2) => {
            (value.network as unknown as Record<string, unknown>).extra = true;
        }],
        ['optimizer unknown field', (value: SessionCheckpointV2) => {
            (value.optimizer as unknown as Record<string, unknown>).extra = true;
        }],
        ['cursor unknown field', (value: SessionCheckpointV2) => {
            (value.cursor as unknown as Record<string, unknown>).extra = true;
        }],
    ])('rejects %s', (_name, mutate) => {
        const value = cloneCheckpoint() as SessionCheckpointV2 & Record<string, unknown>;
        mutate(value);
        expect(() => validateSessionCheckpointV2(value, context())).toThrow(/exactly|unknown/i);
    });

    it.each([
        ['recipe fingerprint', (value: SessionCheckpointV2) => { (value as { recipeFingerprint: string }).recipeFingerprint = 'wrong'; }],
        ['objective key', (value: SessionCheckpointV2) => { (value as { objectiveKey: string }).objectiveKey = 'wrong'; }],
        ['dataset key', (value: SessionCheckpointV2) => { (value as { datasetKey: string }).datasetKey = 'wrong'; }],
        ['evaluation objective', (value: SessionCheckpointV2) => {
            (value.evaluation as { objectiveKey: string }).objectiveKey = 'wrong';
        }],
        ['evaluation dataset revision', (value: SessionCheckpointV2) => {
            (value.evaluation.dataset as { generatorVersion: number }).generatorVersion++;
        }],
        ['model identity', (value: SessionCheckpointV2) => {
            (value.evaluation.model as { revision: number }).revision++;
        }],
        ['checkpoint trigger', (value: SessionCheckpointV2) => {
            (value.evaluation as { trigger: string }).trigger = 'save';
        }],
        ['trajectory guarantee', (value: SessionCheckpointV2) => {
            (value as { trajectoryGuarantee: string }).trajectoryGuarantee = 'exact-resume';
        }],
    ])('rejects mismatched %s', (_name, mutate) => {
        const value = cloneCheckpoint();
        mutate(value);
        expect(() => validateSessionCheckpointV2(value, context())).toThrow();
    });

    it('accepts the initialization-only seed pair but rejects unrelated triggers', () => {
        const initial = checkpoint();
        (initial.evaluation as { trigger: string }).trigger = 'initial';
        (initial.model as { revision: number; step: number; epoch: number }).revision = 0;
        (initial.model as { revision: number; step: number; epoch: number }).step = 0;
        (initial.model as { revision: number; step: number; epoch: number }).epoch = 0;
        (initial.evaluation as { evaluationId: number }).evaluationId = 1;
        (initial.evaluation.model as { revision: number; step: number; epoch: number }).revision = 0;
        (initial.evaluation.model as { revision: number; step: number; epoch: number }).step = 0;
        (initial.evaluation.model as { revision: number; step: number; epoch: number }).epoch = 0;
        (initial.optimizer as { optimizerStep: number }).optimizerStep = 0;
        (initial.cursor as { epoch: number; batchStart: number }).epoch = 0;
        (initial.cursor as { epoch: number; batchStart: number }).batchStart = 0;
        expect(validateSessionCheckpointV2(initial, context()).evaluation.trigger).toBe('initial');

        const lateInitial = checkpoint();
        (lateInitial.evaluation as { trigger: string }).trigger = 'initial';
        expect(() => validateSessionCheckpointV2(lateInitial, context())).toThrow(/initial/i);

        const unrelated = checkpoint();
        (unrelated.evaluation as { trigger: string }).trigger = 'manual-step';
        expect(() => validateSessionCheckpointV2(unrelated, context())).toThrow(/trigger/i);
    });

    it.each([
        ['wrong layer count', (value: SessionCheckpointV2) => { value.network.layers.pop(); }],
        ['wrong declared size', (value: SessionCheckpointV2) => {
            (value.network.layers[0] as { outputSize: number }).outputSize = 4;
        }],
        ['wrong parameter shape', (value: SessionCheckpointV2) => {
            (value.network.layers[0] as { weights: Float64Array }).weights = new Float64Array(5);
        }],
        ['non-finite parameter', (value: SessionCheckpointV2) => {
            value.network.layers[0].weights[0] = Number.NaN;
        }],
        ['wrong optimizer kind', (value: SessionCheckpointV2) => {
            (value as unknown as { optimizer: { kind: string; optimizerStep: number } }).optimizer = {
                kind: 'sgd',
                optimizerStep: 9,
            };
        }],
        ['wrong optimizer shape', (value: SessionCheckpointV2) => {
            if (value.optimizer.kind === 'adam') {
                value.optimizer.firstWeightMoment[0] = new Float64Array(5);
            }
        }],
        ['non-finite optimizer value', (value: SessionCheckpointV2) => {
            if (value.optimizer.kind === 'adam') {
                value.optimizer.secondBiasMoment[0][0] = Number.POSITIVE_INFINITY;
            }
        }],
        ['negative optimizer step', (value: SessionCheckpointV2) => {
            (value.optimizer as { optimizerStep: number }).optimizerStep = -1;
        }],
        ['optimizer/model step mismatch', (value: SessionCheckpointV2) => {
            (value.optimizer as { optimizerStep: number }).optimizerStep = value.model.step - 1;
        }],
    ])('rejects %s', (_name, mutate) => {
        const value = cloneCheckpoint();
        mutate(value);
        expect(() => validateSessionCheckpointV2(value, context())).toThrow();
    });

    it.each([
        ['negative epoch', (value: SessionCheckpointV2) => { (value.cursor as { epoch: number }).epoch = -1; }],
        ['model/cursor epoch mismatch', (value: SessionCheckpointV2) => { (value.cursor as { epoch: number }).epoch = 1; }],
        ['cursor past training set', (value: SessionCheckpointV2) => { (value.cursor as { batchStart: number }).batchStart = 5; }],
        ['short permutation', (value: SessionCheckpointV2) => {
            (value.cursor as { shuffledIndices: Uint32Array }).shuffledIndices = new Uint32Array([0, 1, 2]);
        }],
        ['duplicate permutation entry', (value: SessionCheckpointV2) => {
            (value.cursor as { shuffledIndices: Uint32Array }).shuffledIndices = new Uint32Array([0, 1, 1, 3]);
        }],
        ['out-of-range permutation entry', (value: SessionCheckpointV2) => {
            (value.cursor as { shuffledIndices: Uint32Array }).shuffledIndices = new Uint32Array([0, 1, 2, 4]);
        }],
    ])('rejects %s', (_name, mutate) => {
        const value = cloneCheckpoint();
        mutate(value);
        expect(() => validateSessionCheckpointV2(value, context())).toThrow();
    });

    it('rejects repeated and non-tight typed-array backing buffers', () => {
        const repeated = checkpoint();
        if (repeated.optimizer.kind !== 'adam') throw new Error('fixture must use Adam');
        repeated.optimizer.firstWeightMoment[0] = repeated.network.layers[0].weights;
        expect(() => validateSessionCheckpointV2(repeated, context())).toThrow(/backing|alias/i);

        const subview = checkpoint();
        const backing = new Float64Array(7);
        subview.network.layers[0].weights = new Float64Array(backing.buffer, 8, 6);
        expect(() => validateSessionCheckpointV2(subview, context())).toThrow(/tight|backing|subview/i);
    });

    it('rejects exotic or accessor-backed container arrays', () => {
        const exotic = checkpoint();
        Object.setPrototypeOf(exotic.network.layers, null);
        expect(() => validateSessionCheckpointV2(exotic, context())).toThrow(/plain|prototype|dense/i);

        const accessor = checkpoint();
        if (accessor.optimizer.kind !== 'adam') throw new Error('fixture must use Adam');
        const moments = accessor.optimizer.firstWeightMoment;
        Object.defineProperty(moments, 0, {
            enumerable: true,
            configurable: true,
            get: () => new Float64Array(6),
        });
        expect(() => validateSessionCheckpointV2(accessor, context())).toThrow(/data|accessor|dense/i);
    });

    it('rejects SharedArrayBuffer-backed views', () => {
        if (typeof SharedArrayBuffer === 'undefined') return;
        const value = checkpoint();
        value.network.layers[0].weights = new Float64Array(new SharedArrayBuffer(6 * 8));
        expect(() => validateSessionCheckpointV2(value, context())).toThrow(/shared/i);
    });

    it('rejects SharedArrayBuffer backing even when its toStringTag is forged', () => {
        if (typeof SharedArrayBuffer === 'undefined') return;
        const value = checkpoint();
        const backing = new SharedArrayBuffer(6 * 8);
        Object.defineProperty(backing, Symbol.toStringTag, {
            configurable: true,
            value: 'ArrayBuffer',
        });
        value.network.layers[0].weights = new Float64Array(backing);

        expect(() => validateSessionCheckpointV2(value, context())).toThrow(/shared/i);
    });

    it.each([
        ['buffer', new ArrayBuffer(6 * 8)],
        ['byteOffset', 0],
        ['byteLength', 6 * 8],
        ['length', 6],
    ] as const)('rejects a typed array with an own %s accessor', (property, result) => {
        const value = checkpoint();
        const get = vi.fn(() => result);
        Object.defineProperty(value.network.layers[0].weights, property, {
            configurable: true,
            get,
        });

        expect(() => validateSessionCheckpointV2(value, context())).toThrow(/dense|own|extra/i);
        expect(get).not.toHaveBeenCalled();
    });

    it.each([
        ['named', (weights: Float64Array & { note?: string }) => { weights.note = 'unexpected'; }],
        ['symbol', (weights: Float64Array) => {
            Object.defineProperty(weights, Symbol('unexpected'), {
                configurable: true,
                value: true,
            });
        }],
    ] as const)('rejects an extra %s own key on a typed array', (_kind, mutate) => {
        const value = checkpoint();
        mutate(value.network.layers[0].weights);

        expect(() => validateSessionCheckpointV2(value, context())).toThrow(/dense|own|extra/i);
    });

    it('accepts valid typed arrays created in another realm', () => {
        const source = checkpoint();
        const value: SessionCheckpointV2 = {
            ...source,
            network: {
                layers: source.network.layers.map((layer, index) => index === 0
                    ? {
                        ...layer,
                        weights: runInNewContext(
                            'new Float64Array([1, 2, 3, 4, 5, 6])',
                        ) as Float64Array,
                    }
                    : layer),
            },
            cursor: {
                ...source.cursor,
                shuffledIndices: runInNewContext(
                    'new Uint32Array([2, 0, 3, 1])',
                ) as Uint32Array,
            },
        };

        const parsed = validateSessionCheckpointV2(value, context());
        expect(parsed.network.layers[0].weights).toEqual(
            new Float64Array([1, 2, 3, 4, 5, 6]),
        );
        expect(parsed.cursor.shuffledIndices).toEqual(new Uint32Array([2, 0, 3, 1]));
    });

    it('clones valid views without consulting a replaceable iterator', () => {
        const descriptor = Object.getOwnPropertyDescriptor(
            Float64Array.prototype,
            Symbol.iterator,
        );
        Object.defineProperty(Float64Array.prototype, Symbol.iterator, {
            configurable: true,
            value: () => {
                throw new Error('replaceable iterator must not be used');
            },
        });
        try {
            expect(() => validateSessionCheckpointV2(checkpoint(), context())).not.toThrow();
        } finally {
            if (descriptor === undefined) {
                Reflect.deleteProperty(Float64Array.prototype, Symbol.iterator);
            } else {
                Object.defineProperty(Float64Array.prototype, Symbol.iterator, descriptor);
            }
        }
    });

    it('accepts exactly 256 KiB of typed arrays and rejects the next element', () => {
        const makeBounded = (trainCount: number): {
            value: SessionCheckpointV2;
            expected: SessionCheckpointValidationContext;
        } => {
            const dataset = { ...DATASET, trainCount, testCount: 1 };
            const model = { ...MODEL, epoch: 0, step: 0, revision: 0 };
            const outputSize = 127;
            const inputSize = 255;
            const value: SessionCheckpointV2 = {
                ...checkpoint(),
                model,
                evaluation: {
                    ...checkpoint().evaluation,
                    evaluationId: 1,
                    model,
                    dataset,
                    train: {
                        basis: { kind: 'full-split', split: 'train', sampleCount: trainCount, populationCount: trainCount },
                        values: { dataLoss: 0 },
                    },
                    test: {
                        basis: { kind: 'full-split', split: 'test', sampleCount: 1, populationCount: 1 },
                        values: { dataLoss: 0 },
                    },
                    objective: { regularizationPenalty: 0, trainTotalObjective: 0 },
                },
                network: {
                    layers: [{
                        inputSize,
                        outputSize,
                        weights: new Float64Array(inputSize * outputSize),
                        biases: new Float64Array(outputSize),
                    }],
                },
                optimizer: { kind: 'sgd', optimizerStep: 0 },
                cursor: {
                    epoch: 0,
                    batchStart: 0,
                    shuffledIndices: Uint32Array.from({ length: trainCount }, (_, index) => index),
                },
            };
            return {
                value,
                expected: context({
                    dataset,
                    layerSizes: [inputSize, outputSize],
                    optimizer: { kind: 'sgd' },
                }),
            };
        };

        const exact = makeBounded(512);
        expect(
            exact.value.network.layers[0].weights.byteLength
            + exact.value.network.layers[0].biases.byteLength
            + exact.value.cursor.shuffledIndices.byteLength,
        ).toBe(SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES);
        expect(() => validateSessionCheckpointV2(exact.value, exact.expected)).not.toThrow();

        const oversized = makeBounded(513);
        expect(() => validateSessionCheckpointV2(oversized.value, oversized.expected)).toThrow(/262144|256 KiB|bytes/i);
    });

    it.each([
        [{ kind: 'sgd' } as const, { kind: 'sgd', optimizerStep: MODEL.step } as const],
        [
            { kind: 'sgd-momentum', momentum: 0.9 } as const,
            {
                kind: 'sgd-momentum',
                optimizerStep: MODEL.step,
                weightVelocity: [new Float64Array(6), new Float64Array(3)],
                biasVelocity: [new Float64Array(3), new Float64Array(1)],
            } as const,
        ],
    ])('accepts and detaches %s optimizer checkpoints', (optimizer, optimizerState) => {
        const value = checkpoint();
        (value as unknown as { optimizer: typeof optimizerState }).optimizer = optimizerState;
        expect(validateSessionCheckpointV2(value, context({ optimizer })).optimizer).toEqual(optimizerState);
    });
});
