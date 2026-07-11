import { describe, expect, it } from 'vitest';
import {
    compileValidatedExperiment,
    validateExperimentDocument,
} from '../experimentSchema.js';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    EXPERIMENT_SCHEMA_VERSION,
    validateExperimentDocument as validateFromPublicBarrel,
} from '../index.js';
// @ts-expect-error -- Vitest's raw loader exposes this source as a string.
import experimentSchemaSource from '../experimentSchema.ts?raw';
// @ts-expect-error -- Vitest's raw loader exposes this source as a string.
import publicBarrelSource from '../index.ts?raw';
import {
    VALID_BINARY_DOCUMENT,
    VALID_HUBER_DOCUMENT,
    VALID_MULTICLASS_DOCUMENT,
    VALID_REGRESSION_DOCUMENT,
    VALID_TASK_DOCUMENTS,
} from './fixtures/experimentV2.js';

type MutableDocument = any;

function cloneDocument(source: unknown = VALID_BINARY_DOCUMENT): MutableDocument {
    return structuredClone(source);
}

function expectIssue(
    candidate: unknown,
    code: string,
    path: string,
): void {
    const result = validateExperimentDocument(candidate);
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error(`Expected ${code} at ${path}`);
    expect(result.issues).toEqual(expect.arrayContaining([
        expect.objectContaining({ code, path }),
    ]));
}

describe('version-2 experiment schema', () => {
    it('brands an exact deeply frozen plain-data snapshot', () => {
        const candidate = structuredClone(VALID_BINARY_DOCUMENT);

        const result = validateExperimentDocument(candidate);

        expect(result).toEqual({ ok: true, value: candidate });
        if (result.ok) {
            expect(result.value).not.toBe(candidate);
            expect(result.value.recipe).not.toBe(candidate.recipe);
            expect(result.value).toEqual(candidate);
            expect(Object.isFrozen(result.value)).toBe(true);
            expect(Object.isFrozen(result.value.recipe)).toBe(true);
            expect(Object.isFrozen(result.value.recipe.inputs.featureIds)).toBe(true);
            expect(() => {
                (result.value.recipe.data as { seed: number }).seed = 43;
            }).toThrow();
        }
    });

    it('rejects hidden, symbol, accessor, sparse, and exotic object state safely', () => {
        const hidden = cloneDocument();
        Object.defineProperty(hidden.view, 'hidden', { value: true, enumerable: false });
        expectIssue(hidden, 'invalid-field', 'view.hidden');

        const symbolState = cloneDocument();
        symbolState.recipe[Symbol('hidden')] = true;
        expectIssue(symbolState, 'invalid-field', 'recipe');

        let getterCalls = 0;
        const accessor = cloneDocument();
        Object.defineProperty(accessor.recipe.data, 'seed', {
            enumerable: true,
            get: () => {
                getterCalls += 1;
                return 42;
            },
        });
        expectIssue(accessor, 'invalid-field', 'recipe.data.seed');
        expect(getterCalls).toBe(0);

        const sparse = cloneDocument();
        sparse.recipe.inputs.featureIds = new Array(2);
        sparse.recipe.inputs.featureIds[1] = 'y';
        expectIssue(sparse, 'invalid-field', 'recipe.inputs.featureIds[0]');

        const exotic = cloneDocument();
        Object.setPrototypeOf(exotic.recipe.data, { inherited: true });
        expectIssue(exotic, 'invalid-field', 'recipe.data');
    });

    it('collects unknown fields instead of silently dropping them', () => {
        const candidate = structuredClone(VALID_BINARY_DOCUMENT) as any;
        candidate.recipe.model.outputActivation = 'linear';

        const result = validateExperimentDocument(candidate);

        expect(result).toEqual({
            ok: false,
            issues: expect.arrayContaining([
                expect.objectContaining({
                    code: 'unknown-field',
                    path: 'recipe.model.outputActivation',
                }),
            ]),
        });
    });

    it('reports a task and dataset mismatch alongside unrelated strictness issues', () => {
        const candidate = structuredClone(VALID_BINARY_DOCUMENT) as any;
        candidate.recipe.model.outputActivation = 'linear';
        candidate.recipe.task = { kind: 'regression', dataset: 'circle' };

        const result = validateExperimentDocument(candidate);

        expect(result).toEqual({
            ok: false,
            issues: expect.arrayContaining([
                expect.objectContaining({
                    code: 'unknown-field',
                    path: 'recipe.model.outputActivation',
                }),
                expect.objectContaining({
                    code: 'incompatible-task',
                    path: 'recipe.task.dataset',
                }),
            ]),
        });
    });
});

interface NumericBoundaryCase {
    name: string;
    path: string;
    make?: () => MutableDocument;
    set: (candidate: MutableDocument, value: number) => void;
    accepted: readonly number[];
    rejected: readonly { value: number; code: 'invalid-field' | 'out-of-range' }[];
}

function sampleBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.batchSize = 1;
    return candidate;
}

function splitBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.data.sampleCount = 10;
    candidate.recipe.training.batchSize = 1;
    return candidate;
}

function batchBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.data.sampleCount = 1_000;
    candidate.recipe.data.trainFraction = 0.9;
    return candidate;
}

function momentumBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.optimizer = { kind: 'sgd-momentum', momentum: 0.5 };
    return candidate;
}

function adamBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.optimizer = {
        kind: 'adam',
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    return candidate;
}

function penaltyBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.objective.penalty = {
        kind: 'l1',
        coefficient: 0.1,
        applyTo: 'weights',
    };
    return candidate;
}

function clippingBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.gradientClipping = {
        kind: 'global-norm',
        maximumNorm: 1,
        scope: 'total-objective-gradient',
    };
    return candidate;
}

function stepBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.schedule = { kind: 'step', interval: 10, gamma: 0.5 };
    return candidate;
}

function cosineBoundaryDocument(): MutableDocument {
    const candidate = cloneDocument();
    candidate.recipe.training.schedule = {
        kind: 'cosine',
        totalSteps: 100,
        minimumRate: 0.001,
    };
    return candidate;
}

const NUMERIC_BOUNDARIES: readonly NumericBoundaryCase[] = [
    {
        name: 'sample count',
        path: 'recipe.data.sampleCount',
        make: sampleBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.data.sampleCount = value; },
        accepted: [2, 1_000],
        rejected: [
            { value: 1, code: 'out-of-range' },
            { value: 1_001, code: 'out-of-range' },
            { value: 2.5, code: 'invalid-field' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'train fraction',
        path: 'recipe.data.trainFraction',
        make: splitBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.data.trainFraction = value; },
        accepted: [0.1, 0.9],
        rejected: [
            { value: 0.099_999, code: 'out-of-range' },
            { value: 0.900_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'noise',
        path: 'recipe.data.noise',
        set: (candidate, value) => { candidate.recipe.data.noise = value; },
        accepted: [0, 100],
        rejected: [
            { value: -Number.EPSILON, code: 'out-of-range' },
            { value: 100.000_001, code: 'out-of-range' },
            { value: Number.POSITIVE_INFINITY, code: 'invalid-field' },
        ],
    },
    ...(['data', 'model'] as const).map((seedOwner): NumericBoundaryCase => ({
        name: `${seedOwner} seed`,
        path: `recipe.${seedOwner}.seed`,
        set: (candidate, value) => { candidate.recipe[seedOwner].seed = value; },
        accepted: [0, 4_294_967_295],
        rejected: [
            { value: -1, code: 'out-of-range' },
            { value: 4_294_967_296, code: 'out-of-range' },
            { value: 42.5, code: 'invalid-field' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    })),
    {
        name: 'learning rate',
        path: 'recipe.training.learningRate',
        set: (candidate, value) => { candidate.recipe.training.learningRate = value; },
        accepted: [Number.MIN_VALUE, 10],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 10.000_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'batch size',
        path: 'recipe.training.batchSize',
        make: batchBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.batchSize = value; },
        accepted: [1, 512],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 513, code: 'out-of-range' },
            { value: 1.5, code: 'invalid-field' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'SGD momentum',
        path: 'recipe.training.optimizer.momentum',
        make: momentumBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.optimizer.momentum = value; },
        accepted: [0, 1 - Number.EPSILON],
        rejected: [
            { value: -Number.EPSILON, code: 'out-of-range' },
            { value: 1, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    ...(['beta1', 'beta2'] as const).map((beta): NumericBoundaryCase => ({
        name: `Adam ${beta}`,
        path: `recipe.training.optimizer.${beta}`,
        make: adamBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.optimizer[beta] = value; },
        accepted: [0, 1 - Number.EPSILON],
        rejected: [
            { value: -Number.EPSILON, code: 'out-of-range' },
            { value: 1, code: 'out-of-range' },
            { value: Number.POSITIVE_INFINITY, code: 'invalid-field' },
        ],
    })),
    {
        name: 'Adam epsilon',
        path: 'recipe.training.optimizer.epsilon',
        make: adamBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.optimizer.epsilon = value; },
        accepted: [Number.MIN_VALUE, 1],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1.000_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'penalty coefficient',
        path: 'recipe.objective.penalty.coefficient',
        make: penaltyBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.objective.penalty.coefficient = value; },
        accepted: [Number.MIN_VALUE, 1],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1.000_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'global clipping maximum',
        path: 'recipe.training.gradientClipping.maximumNorm',
        make: clippingBoundaryDocument,
        set: (candidate, value) => {
            candidate.recipe.training.gradientClipping.maximumNorm = value;
        },
        accepted: [Number.MIN_VALUE, 1_000_000],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1_000_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'Huber delta',
        path: 'recipe.objective.dataLoss.delta',
        make: () => cloneDocument(VALID_HUBER_DOCUMENT),
        set: (candidate, value) => { candidate.recipe.objective.dataLoss.delta = value; },
        accepted: [Number.MIN_VALUE, 1_000_000],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1_000_001, code: 'out-of-range' },
            { value: Number.NEGATIVE_INFINITY, code: 'invalid-field' },
        ],
    },
    {
        name: 'step interval',
        path: 'recipe.training.schedule.interval',
        make: stepBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.schedule.interval = value; },
        accepted: [1, 1_000_000_000],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1_000_000_001, code: 'out-of-range' },
            { value: 1.5, code: 'invalid-field' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'step gamma',
        path: 'recipe.training.schedule.gamma',
        make: stepBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.schedule.gamma = value; },
        accepted: [Number.MIN_VALUE, 1],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1.000_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'cosine total steps',
        path: 'recipe.training.schedule.totalSteps',
        make: cosineBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.schedule.totalSteps = value; },
        accepted: [1, 1_000_000_000],
        rejected: [
            { value: 0, code: 'out-of-range' },
            { value: 1_000_000_001, code: 'out-of-range' },
            { value: 1.5, code: 'invalid-field' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
    {
        name: 'cosine minimum rate',
        path: 'recipe.training.schedule.minimumRate',
        make: cosineBoundaryDocument,
        set: (candidate, value) => { candidate.recipe.training.schedule.minimumRate = value; },
        accepted: [0, 0.03],
        rejected: [
            { value: -Number.EPSILON, code: 'out-of-range' },
            { value: 0.030_001, code: 'out-of-range' },
            { value: Number.NaN, code: 'invalid-field' },
        ],
    },
];

describe('numeric and deterministic split boundaries', () => {
    it.each(NUMERIC_BOUNDARIES)('accepts both $name limits', (boundary) => {
        for (const value of boundary.accepted) {
            const candidate = boundary.make?.() ?? cloneDocument();
            boundary.set(candidate, value);
            expect(validateExperimentDocument(candidate), `${boundary.name}=${value}`).toMatchObject({
                ok: true,
            });
        }
    });

    it.each(NUMERIC_BOUNDARIES)('rejects values just outside or malformed for $name', (boundary) => {
        for (const rejection of boundary.rejected) {
            const candidate = boundary.make?.() ?? cloneDocument();
            boundary.set(candidate, rejection.value);
            expectIssue(candidate, rejection.code, boundary.path);
        }
    });

    it('rejects an individually bounded split that produces an empty train population', () => {
        const candidate = cloneDocument();
        candidate.recipe.data.sampleCount = 2;
        candidate.recipe.data.trainFraction = 0.1;
        candidate.recipe.training.batchSize = 1;

        expectIssue(candidate, 'resource-limit', 'recipe.data.trainFraction');
    });

    it('accepts the adjacent two-sample split with positive train and test populations', () => {
        const candidate = cloneDocument();
        candidate.recipe.data.sampleCount = 2;
        candidate.recipe.data.trainFraction = 0.9;
        candidate.recipe.training.batchSize = 1;

        expect(validateExperimentDocument(candidate)).toMatchObject({ ok: true });
    });

    it('rejects a batch larger than the derived training population', () => {
        const candidate = cloneDocument();
        candidate.recipe.training.batchSize = 151;

        expectIssue(candidate, 'resource-limit', 'recipe.training.batchSize');
    });
});

const FEATURE_REGISTRY = [
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

describe('feature and architecture resource limits', () => {
    it('accepts the one-feature and complete ordered-registry boundaries', () => {
        for (const featureIds of [['x'], FEATURE_REGISTRY]) {
            const candidate = cloneDocument();
            candidate.recipe.inputs.featureIds = [...featureIds];

            expect(validateExperimentDocument(candidate)).toMatchObject({ ok: true });
        }
    });

    it('rejects empty and over-limit feature lists', () => {
        const empty = cloneDocument();
        empty.recipe.inputs.featureIds = [];
        expectIssue(empty, 'resource-limit', 'recipe.inputs.featureIds');

        const overLimit = cloneDocument();
        overLimit.recipe.inputs.featureIds = [...FEATURE_REGISTRY, 'x'];
        expectIssue(overLimit, 'resource-limit', 'recipe.inputs.featureIds');
    });

    it('rejects unknown, duplicate, and non-canonical feature ordering independently', () => {
        const unknown = cloneDocument();
        unknown.recipe.inputs.featureIds = ['x', 'bogus'];
        expectIssue(unknown, 'invalid-field', 'recipe.inputs.featureIds[1]');

        const duplicate = cloneDocument();
        duplicate.recipe.inputs.featureIds = ['x', 'x'];
        expectIssue(duplicate, 'duplicate-feature', 'recipe.inputs.featureIds[1]');

        const outOfOrder = cloneDocument();
        outOfOrder.recipe.inputs.featureIds = ['y', 'x'];
        expectIssue(outOfOrder, 'invalid-field', 'recipe.inputs.featureIds[1]');
    });

    it('rejects a non-array feature list', () => {
        const candidate = cloneDocument();
        candidate.recipe.inputs.featureIds = { x: true, y: true };

        expectIssue(candidate, 'invalid-field', 'recipe.inputs.featureIds');
    });

    it('accepts zero and six hidden layers, including the largest valid architecture', () => {
        const noHidden = cloneDocument();
        noHidden.recipe.model.hiddenLayers = [];
        expect(validateExperimentDocument(noHidden)).toMatchObject({ ok: true });

        const largest = cloneDocument();
        largest.recipe.inputs.featureIds = [...FEATURE_REGISTRY];
        largest.recipe.model.hiddenLayers = [16, 16, 16, 16, 16, 16];
        expect(validateExperimentDocument(largest)).toMatchObject({ ok: true });
    });

    it('rejects seven hidden layers and a non-array layer list', () => {
        const tooDeep = cloneDocument();
        tooDeep.recipe.model.hiddenLayers = [1, 1, 1, 1, 1, 1, 1];
        expectIssue(tooDeep, 'resource-limit', 'recipe.model.hiddenLayers');

        const notAnArray = cloneDocument();
        notAnArray.recipe.model.hiddenLayers = 4;
        expectIssue(notAnArray, 'invalid-field', 'recipe.model.hiddenLayers');
    });

    it.each([
        { width: 1, ok: true },
        { width: 16, ok: true },
        { width: 0, ok: false, code: 'out-of-range' },
        { width: 17, ok: false, code: 'out-of-range' },
        { width: 1.5, ok: false, code: 'invalid-field' },
        { width: Number.NaN, ok: false, code: 'invalid-field' },
    ])('validates hidden width $width at its exact boundary', ({ width, ok, code }) => {
        const candidate = cloneDocument();
        candidate.recipe.model.hiddenLayers = [width];

        if (ok) {
            expect(validateExperimentDocument(candidate)).toMatchObject({ ok: true });
        } else {
            expectIssue(candidate, code!, 'recipe.model.hiddenLayers[0]');
        }
    });

    it('derives the bias-inclusive trainable parameter count before allocation', () => {
        const candidate = cloneDocument();
        candidate.recipe.inputs.featureIds = [...FEATURE_REGISTRY];
        candidate.recipe.model.hiddenLayers = [20, 20, 20, 20, 20, 20];

        expectIssue(candidate, 'resource-limit', 'recipe.model.hiddenLayers');
    });

    it('rejects unsupported hidden activation and initialization values', () => {
        const activation = cloneDocument();
        activation.recipe.model.hiddenActivation = 'softmax';
        expectIssue(activation, 'invalid-field', 'recipe.model.hiddenActivation');

        const initialization = cloneDocument();
        initialization.recipe.model.initialization = 'random';
        expectIssue(initialization, 'invalid-field', 'recipe.model.initialization');
    });
});

describe('task, optimizer, schedule, clipping, and objective discriminants', () => {
    it.each([
        {
            name: 'binary task with regression dataset',
            source: VALID_BINARY_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.task.dataset = 'reg-plane';
            },
            path: 'recipe.task.dataset',
        },
        {
            name: 'multiclass task with binary dataset',
            source: VALID_MULTICLASS_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.task.dataset = 'circle';
            },
            path: 'recipe.task.dataset',
        },
        {
            name: 'regression task with multiclass dataset',
            source: VALID_REGRESSION_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.task.dataset = 'three-class-clusters';
            },
            path: 'recipe.task.dataset',
        },
        {
            name: 'binary task with regression loss',
            source: VALID_BINARY_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.dataLoss = { kind: 'mean-squared-error' };
            },
            path: 'recipe.objective.dataLoss.kind',
        },
        {
            name: 'multiclass task with binary loss',
            source: VALID_MULTICLASS_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.dataLoss = {
                    kind: 'binary-cross-entropy-with-logits',
                };
            },
            path: 'recipe.objective.dataLoss.kind',
        },
        {
            name: 'regression task with multiclass loss',
            source: VALID_REGRESSION_DOCUMENT,
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.dataLoss = {
                    kind: 'categorical-cross-entropy-with-logits',
                };
            },
            path: 'recipe.objective.dataLoss.kind',
        },
    ])('rejects $name', ({ source, mutate, path }) => {
        const candidate = cloneDocument(source);
        mutate(candidate);

        expectIssue(candidate, 'incompatible-task', path);
    });

    it.each([
        {
            name: 'task',
            path: 'recipe.task.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.task.kind = 'classification';
            },
        },
        {
            name: 'dataset',
            path: 'recipe.task.dataset',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.task.dataset = 'not-a-dataset';
            },
        },
        {
            name: 'data loss',
            path: 'recipe.objective.dataLoss.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.dataLoss.kind = 'cross-entropy';
            },
        },
        {
            name: 'penalty',
            path: 'recipe.objective.penalty.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.penalty.kind = 'elastic-net';
            },
        },
        {
            name: 'schedule',
            path: 'recipe.training.schedule.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.schedule.kind = 'exponential';
            },
        },
        {
            name: 'optimizer',
            path: 'recipe.training.optimizer.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.optimizer.kind = 'rmsprop';
            },
        },
        {
            name: 'gradient clipping',
            path: 'recipe.training.gradientClipping.kind',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.gradientClipping.kind = 'per-layer';
            },
        },
    ])('rejects an unknown $name discriminant', ({ mutate, path }) => {
        const candidate = cloneDocument();
        mutate(candidate);

        expectIssue(candidate, 'invalid-field', path);
    });

    it('requires the exact reduction, penalty target, and clipping scope literals', () => {
        const reduction = cloneDocument();
        reduction.recipe.objective.reduction = 'sum';
        expectIssue(reduction, 'invalid-field', 'recipe.objective.reduction');

        const penalty = penaltyBoundaryDocument();
        penalty.recipe.objective.penalty.applyTo = 'biases';
        expectIssue(penalty, 'invalid-field', 'recipe.objective.penalty.applyTo');

        const clipping = clippingBoundaryDocument();
        clipping.recipe.training.gradientClipping.scope = 'data-gradient';
        expectIssue(
            clipping,
            'invalid-field',
            'recipe.training.gradientClipping.scope',
        );
    });

    it.each([
        {
            name: 'constant schedule',
            path: 'recipe.training.schedule.interval',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.schedule.interval = 10;
            },
        },
        {
            name: 'plain SGD',
            path: 'recipe.training.optimizer.momentum',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.optimizer.momentum = 0.9;
            },
        },
        {
            name: 'no clipping',
            path: 'recipe.training.gradientClipping.maximumNorm',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.training.gradientClipping.maximumNorm = 1;
            },
        },
        {
            name: 'no penalty',
            path: 'recipe.objective.penalty.coefficient',
            mutate: (candidate: MutableDocument) => {
                candidate.recipe.objective.penalty.coefficient = 0;
            },
        },
    ])('rejects fields inactive for $name', ({ mutate, path }) => {
        const candidate = cloneDocument();
        mutate(candidate);

        expectIssue(candidate, 'unknown-field', path);
    });

    it.each([
        {
            name: 'step interval',
            make: stepBoundaryDocument,
            path: 'recipe.training.schedule.interval',
        },
        {
            name: 'cosine minimum rate',
            make: cosineBoundaryDocument,
            path: 'recipe.training.schedule.minimumRate',
        },
        {
            name: 'momentum coefficient',
            make: momentumBoundaryDocument,
            path: 'recipe.training.optimizer.momentum',
        },
        {
            name: 'Adam epsilon',
            make: adamBoundaryDocument,
            path: 'recipe.training.optimizer.epsilon',
        },
        {
            name: 'global clipping scope',
            make: clippingBoundaryDocument,
            path: 'recipe.training.gradientClipping.scope',
        },
        {
            name: 'L1 coefficient',
            make: penaltyBoundaryDocument,
            path: 'recipe.objective.penalty.coefficient',
        },
        {
            name: 'Huber delta',
            make: () => cloneDocument(VALID_HUBER_DOCUMENT),
            path: 'recipe.objective.dataLoss.delta',
        },
    ])('rejects a missing active $name', ({ make, path }) => {
        const candidate = make();
        const segments = path.split('.');
        const field = segments.pop()!;
        let parent = candidate;
        for (const segment of segments) parent = parent[segment];
        delete parent[field];

        expectIssue(candidate, 'missing-field', path);
    });
});

describe('strict records and bounded issue collection', () => {
    const recordLayers = [
        { name: 'envelope', path: '', target: (candidate: MutableDocument) => candidate },
        { name: 'recipe', path: 'recipe.', target: (candidate: MutableDocument) => candidate.recipe },
        { name: 'data', path: 'recipe.data.', target: (candidate: MutableDocument) => candidate.recipe.data },
        { name: 'inputs', path: 'recipe.inputs.', target: (candidate: MutableDocument) => candidate.recipe.inputs },
        { name: 'model', path: 'recipe.model.', target: (candidate: MutableDocument) => candidate.recipe.model },
        { name: 'training', path: 'recipe.training.', target: (candidate: MutableDocument) => candidate.recipe.training },
        { name: 'task', path: 'recipe.task.', target: (candidate: MutableDocument) => candidate.recipe.task },
        { name: 'objective', path: 'recipe.objective.', target: (candidate: MutableDocument) => candidate.recipe.objective },
        { name: 'loss', path: 'recipe.objective.dataLoss.', target: (candidate: MutableDocument) => candidate.recipe.objective.dataLoss },
        { name: 'penalty', path: 'recipe.objective.penalty.', target: (candidate: MutableDocument) => candidate.recipe.objective.penalty },
        { name: 'schedule', path: 'recipe.training.schedule.', target: (candidate: MutableDocument) => candidate.recipe.training.schedule },
        { name: 'optimizer', path: 'recipe.training.optimizer.', target: (candidate: MutableDocument) => candidate.recipe.training.optimizer },
        { name: 'clipping', path: 'recipe.training.gradientClipping.', target: (candidate: MutableDocument) => candidate.recipe.training.gradientClipping },
        { name: 'view', path: 'view.', target: (candidate: MutableDocument) => candidate.view },
    ] as const;

    it.each(recordLayers)('rejects unknown fields at the $name layer', ({ path, target }) => {
        const candidate = cloneDocument();
        target(candidate).unexpected = true;

        expectIssue(candidate, 'unknown-field', `${path}unexpected`);
    });

    it.each([
        { name: 'envelope', path: 'recipe', mutate: (candidate: MutableDocument) => { delete candidate.recipe; } },
        { name: 'recipe', path: 'recipe.data', mutate: (candidate: MutableDocument) => { delete candidate.recipe.data; } },
        { name: 'data', path: 'recipe.data.seed', mutate: (candidate: MutableDocument) => { delete candidate.recipe.data.seed; } },
        { name: 'inputs', path: 'recipe.inputs.featureIds', mutate: (candidate: MutableDocument) => { delete candidate.recipe.inputs.featureIds; } },
        { name: 'model', path: 'recipe.model.seed', mutate: (candidate: MutableDocument) => { delete candidate.recipe.model.seed; } },
        { name: 'training', path: 'recipe.training.optimizer', mutate: (candidate: MutableDocument) => { delete candidate.recipe.training.optimizer; } },
        { name: 'task', path: 'recipe.task.dataset', mutate: (candidate: MutableDocument) => { delete candidate.recipe.task.dataset; } },
        { name: 'objective', path: 'recipe.objective.reduction', mutate: (candidate: MutableDocument) => { delete candidate.recipe.objective.reduction; } },
        { name: 'loss', path: 'recipe.objective.dataLoss.kind', mutate: (candidate: MutableDocument) => { delete candidate.recipe.objective.dataLoss.kind; } },
        { name: 'penalty', path: 'recipe.objective.penalty.kind', mutate: (candidate: MutableDocument) => { delete candidate.recipe.objective.penalty.kind; } },
        { name: 'schedule', path: 'recipe.training.schedule.kind', mutate: (candidate: MutableDocument) => { delete candidate.recipe.training.schedule.kind; } },
        { name: 'optimizer', path: 'recipe.training.optimizer.kind', mutate: (candidate: MutableDocument) => { delete candidate.recipe.training.optimizer.kind; } },
        { name: 'clipping', path: 'recipe.training.gradientClipping.kind', mutate: (candidate: MutableDocument) => { delete candidate.recipe.training.gradientClipping.kind; } },
        { name: 'view', path: 'view.showTestData', mutate: (candidate: MutableDocument) => { delete candidate.view.showTestData; } },
    ])('rejects missing fields at the $name layer', ({ path, mutate }) => {
        const candidate = cloneDocument();
        mutate(candidate);

        expectIssue(candidate, 'missing-field', path);
    });

    it.each([
        { name: 'root', path: '$', mutate: (_candidate: MutableDocument) => [] },
        { name: 'recipe', path: 'recipe', mutate: (candidate: MutableDocument) => { candidate.recipe = []; return candidate; } },
        { name: 'data', path: 'recipe.data', mutate: (candidate: MutableDocument) => { candidate.recipe.data = []; return candidate; } },
        { name: 'inputs', path: 'recipe.inputs', mutate: (candidate: MutableDocument) => { candidate.recipe.inputs = []; return candidate; } },
        { name: 'model', path: 'recipe.model', mutate: (candidate: MutableDocument) => { candidate.recipe.model = []; return candidate; } },
        { name: 'training', path: 'recipe.training', mutate: (candidate: MutableDocument) => { candidate.recipe.training = []; return candidate; } },
        { name: 'task', path: 'recipe.task', mutate: (candidate: MutableDocument) => { candidate.recipe.task = []; return candidate; } },
        { name: 'objective', path: 'recipe.objective', mutate: (candidate: MutableDocument) => { candidate.recipe.objective = []; return candidate; } },
        { name: 'loss', path: 'recipe.objective.dataLoss', mutate: (candidate: MutableDocument) => { candidate.recipe.objective.dataLoss = []; return candidate; } },
        { name: 'penalty', path: 'recipe.objective.penalty', mutate: (candidate: MutableDocument) => { candidate.recipe.objective.penalty = []; return candidate; } },
        { name: 'schedule', path: 'recipe.training.schedule', mutate: (candidate: MutableDocument) => { candidate.recipe.training.schedule = []; return candidate; } },
        { name: 'optimizer', path: 'recipe.training.optimizer', mutate: (candidate: MutableDocument) => { candidate.recipe.training.optimizer = []; return candidate; } },
        { name: 'clipping', path: 'recipe.training.gradientClipping', mutate: (candidate: MutableDocument) => { candidate.recipe.training.gradientClipping = []; return candidate; } },
        { name: 'view', path: 'view', mutate: (candidate: MutableDocument) => { candidate.view = []; return candidate; } },
    ])('rejects an array where the $name record is required', ({ path, mutate }) => {
        const candidate = mutate(cloneDocument());

        expectIssue(candidate, 'invalid-field', path);
    });

    it('reports legacy state and unsupported versions distinctly', () => {
        expectIssue(
            { network: {}, training: {}, data: {}, features: {}, ui: {} },
            'legacy-state',
            '$',
        );

        const previous = cloneDocument();
        previous.schemaVersion = 1;
        expectIssue(previous, 'unsupported-version', 'schemaVersion');

        const future = cloneDocument();
        future.schemaVersion = 3;
        expectIssue(future, 'unsupported-version', 'schemaVersion');
    });

    it('requires the envelope discriminant and boolean view flags', () => {
        const kind = cloneDocument();
        kind.kind = 'experiment';
        expectIssue(kind, 'invalid-field', 'kind');

        const showTestData = cloneDocument();
        showTestData.view.showTestData = 0;
        expectIssue(showTestData, 'invalid-field', 'view.showTestData');

        const discretizeOutput = cloneDocument();
        discretizeOutput.view.discretizeOutput = 'false';
        expectIssue(discretizeOutput, 'invalid-field', 'view.discretizeOutput');
    });

    it('caps adversarial issue collection at exactly 100 actionable entries', () => {
        const candidate = cloneDocument();
        for (let index = 0; index < 150; index++) {
            candidate[`unknown${index}`] = index;
        }

        const result = validateExperimentDocument(candidate);

        expect(result.ok).toBe(false);
        if (result.ok) throw new Error('Expected bounded validation failure');
        expect(result.issues).toHaveLength(100);
        for (const issue of result.issues) {
            expect(issue.code).toBe('unknown-field');
            expect(issue.path).toMatch(/^unknown\d+$/u);
            expect(issue.message.length).toBeGreaterThan(0);
        }
    });
});

describe('validated compiler transaction and built-in default', () => {
    it.each(VALID_TASK_DOCUMENTS)(
        'validates and derives the exact $name output contract',
        ({ document, expectedTask }) => {
            const candidate = cloneDocument(document);
            const validation = validateExperimentDocument(candidate);
            expect(validation).toMatchObject({ ok: true });
            if (!validation.ok) throw new Error('Expected valid task fixture');

            const compiled = compileValidatedExperiment(validation.value.recipe);

            expect(compiled.network).toEqual({
                inputSize: 2,
                hiddenLayers: [4, 4],
                outputSize: expectedTask.outputSize,
                activation: 'tanh',
                outputActivation: expectedTask.outputActivation,
                weightInit: 'xavier',
                seed: 42,
            });
            expect(compiled.task).toEqual(expectedTask);
            expect(compiled.data).toEqual({
                dataset: document.recipe.task.dataset,
                sampleCount: 300,
                trainFraction: 0.5,
                noise: 0,
                seed: 42,
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
            expect(compiled.objective.spec).toEqual(document.recipe.objective);
            expect(compiled.training.objective).toBe(compiled.objective);
        },
    );

    it('accepts the second regression loss variant', () => {
        expect(validateExperimentDocument(cloneDocument(VALID_HUBER_DOCUMENT))).toMatchObject({
            ok: true,
        });
    });

    it('exports the complete validated default document beside legacy constants', () => {
        expect(EXPERIMENT_SCHEMA_VERSION).toBe(2);
        expect(DEFAULT_EXPERIMENT_DOCUMENT).toEqual(VALID_BINARY_DOCUMENT);

        const validation = validateExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(validation).toEqual({ ok: true, value: DEFAULT_EXPERIMENT_DOCUMENT });
        expect(validateFromPublicBarrel).toBe(validateExperimentDocument);
    });

    it('keeps the compiler helper out of the public barrel and allocation-free', () => {
        expect(publicBarrelSource).not.toMatch(/compileValidatedExperiment/u);
        expect(experimentSchemaSource).toMatch(/compileExperimentRecipe/u);
        expect(experimentSchemaSource).not.toMatch(/new (?:Network|PRNG)\b/u);
        expect(experimentSchemaSource).not.toMatch(/generateDataset/u);
    });
});
