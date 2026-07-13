export const VALID_BINARY_DOCUMENT = {
    kind: 'nn-playground-experiment',
    schemaVersion: 2,
    recipe: {
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
            seed: 42,
        },
        training: {
            batchSize: 10,
            learningRate: 0.03,
            schedule: { kind: 'constant' },
            optimizer: { kind: 'sgd' },
            gradientClipping: { kind: 'none' },
        },
        task: {
            kind: 'binary-classification',
            dataset: 'circle',
        },
        objective: {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        },
    },
    view: {
        showTestData: false,
        discretizeOutput: false,
    },
} as const;

export const VALID_MULTICLASS_DOCUMENT = {
    ...VALID_BINARY_DOCUMENT,
    recipe: {
        ...VALID_BINARY_DOCUMENT.recipe,
        task: {
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
        },
        objective: {
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
            penalty: { kind: 'l2', coefficient: 0.1, applyTo: 'weights' },
            reduction: 'mean-per-sample',
        },
    },
} as const;

export const VALID_REGRESSION_DOCUMENT = {
    ...VALID_BINARY_DOCUMENT,
    recipe: {
        ...VALID_BINARY_DOCUMENT.recipe,
        task: {
            kind: 'regression',
            dataset: 'reg-plane',
        },
        objective: {
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        },
    },
} as const;

export const VALID_HUBER_DOCUMENT = {
    ...VALID_REGRESSION_DOCUMENT,
    recipe: {
        ...VALID_REGRESSION_DOCUMENT.recipe,
        objective: {
            ...VALID_REGRESSION_DOCUMENT.recipe.objective,
            dataLoss: { kind: 'huber', delta: 0.75 },
        },
    },
} as const;

export const VALID_TASK_DOCUMENTS = [
    {
        name: 'binary classification',
        document: VALID_BINARY_DOCUMENT,
        expectedTask: {
            kind: 'binary-classification',
            dataset: 'circle',
            outputSize: 1,
            outputActivation: 'sigmoid',
            target: { kind: 'scalar', values: [0, 1] },
        },
    },
    {
        name: 'multiclass classification',
        document: VALID_MULTICLASS_DOCUMENT,
        expectedTask: {
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
            outputSize: 3,
            outputActivation: 'softmax',
            target: { kind: 'one-hot', length: 3 },
        },
    },
    {
        name: 'regression',
        document: VALID_REGRESSION_DOCUMENT,
        expectedTask: {
            kind: 'regression',
            dataset: 'reg-plane',
            outputSize: 1,
            outputActivation: 'linear',
            target: { kind: 'scalar', finite: true },
        },
    },
] as const;
