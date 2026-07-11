import type {
    BinaryDatasetId,
    DatasetContract,
    DatasetId,
    MulticlassDatasetId,
    RegressionDatasetId,
} from './types.js';

export const BINARY_DATASET_IDS = Object.freeze([
    'circle',
    'xor',
    'gauss',
    'spiral',
    'moons',
    'checkerboard',
    'rings',
    'heart',
] as const satisfies readonly BinaryDatasetId[]);

export const MULTICLASS_DATASET_IDS = Object.freeze([
    'three-class-clusters',
] as const satisfies readonly MulticlassDatasetId[]);

export const REGRESSION_DATASET_IDS = Object.freeze([
    'reg-plane',
    'reg-gauss',
] as const satisfies readonly RegressionDatasetId[]);

export const DATASET_IDS = Object.freeze([
    ...BINARY_DATASET_IDS,
    ...MULTICLASS_DATASET_IDS,
    ...REGRESSION_DATASET_IDS,
] as const satisfies readonly DatasetId[]);

const INPUT_DOMAIN = Object.freeze({
    x: Object.freeze([-1, 1] as const),
    y: Object.freeze([-1, 1] as const),
});
const BINARY_TARGET = Object.freeze({
    kind: 'binary' as const,
    values: Object.freeze([0, 1] as const),
});
const COORDINATE_NOISE = Object.freeze({
    minimum: 0,
    maximum: 100,
    meaning: 'coordinate-perturbation' as const,
});
const TARGET_NOISE = Object.freeze({
    minimum: 0,
    maximum: 100,
    meaning: 'target-perturbation' as const,
});

function binaryContract(id: BinaryDatasetId): DatasetContract {
    return Object.freeze({
        id,
        taskKind: 'binary-classification',
        inputDomain: INPUT_DOMAIN,
        targetDomain: BINARY_TARGET,
        noise: COORDINATE_NOISE,
        generatorVersion: 2,
    });
}

const DATASET_CONTRACTS: Readonly<Record<DatasetId, DatasetContract>> = Object.freeze({
    circle: binaryContract('circle'),
    xor: binaryContract('xor'),
    gauss: binaryContract('gauss'),
    spiral: binaryContract('spiral'),
    moons: binaryContract('moons'),
    checkerboard: binaryContract('checkerboard'),
    rings: binaryContract('rings'),
    heart: binaryContract('heart'),
    'three-class-clusters': Object.freeze({
        id: 'three-class-clusters',
        taskKind: 'multiclass-classification',
        inputDomain: INPUT_DOMAIN,
        targetDomain: Object.freeze({ kind: 'classes', classCount: 3 }),
        noise: COORDINATE_NOISE,
        generatorVersion: 2,
    }),
    'reg-plane': Object.freeze({
        id: 'reg-plane',
        taskKind: 'regression',
        inputDomain: INPUT_DOMAIN,
        targetDomain: Object.freeze({
            kind: 'continuous',
            boundsForNoise: (noise: number) => [-2 - 0.06 * noise, 2 + 0.06 * noise] as const,
        }),
        noise: TARGET_NOISE,
        generatorVersion: 2,
    }),
    'reg-gauss': Object.freeze({
        id: 'reg-gauss',
        taskKind: 'regression',
        inputDomain: INPUT_DOMAIN,
        targetDomain: Object.freeze({
            kind: 'continuous',
            boundsForNoise: (noise: number) => [-0.06 * noise, 2 + 0.06 * noise] as const,
        }),
        noise: TARGET_NOISE,
        generatorVersion: 2,
    }),
});

export function getDatasetContract(id: DatasetId): DatasetContract {
    const contract = (DATASET_CONTRACTS as Readonly<Record<string, DatasetContract>>)[id];
    if (!contract) {
        throw new RangeError(`Unknown dataset ID: ${String(id)}`);
    }
    return contract;
}
