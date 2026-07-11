// ── Dataset generators ──
// All generators produce finite points inside their versioned contracts.

import { getDatasetContract } from './datasetContracts.js';
import type {
    DataPoint,
    DataSplit,
    DatasetContract,
    DatasetGenerationRequest,
    DatasetId,
    DatasetType,
} from './types.js';
import { normalizeUint32Seed, PRNG } from './prng.js';

const DEFAULT_NUM_SAMPLES = 300;
const DEFAULT_TRAIN_FRACTION = 0.5;
const MINIMUM_SAMPLE_COUNT = 2;
const MAXIMUM_SAMPLE_COUNT = 1_000;
const MINIMUM_TRAIN_FRACTION = 0.1;
const MAXIMUM_TRAIN_FRACTION = 0.9;
const MAXIMUM_COORDINATE_ATTEMPTS = 10_000;
const THREE_CLASS_DEFAULT_NOISE = 8;
const THREE_CLASS_DEFAULT_TRAIN_RATIO = 0.5;
const THREE_CLASS_LABELS = Object.freeze([0, 1, 2] as const);
const THREE_CLASS_CENTERS = [
    { x: -0.55, y: -0.35, label: 0 },
    { x: 0.55, y: -0.35, label: 1 },
    { x: 0, y: 0.55, label: 2 },
] as const;

export interface ThreeClassClusterDatasetContract {
    id: 'three-class-clusters';
    problemType: 'classification';
    classCount: 3;
    classLabels: readonly [0, 1, 2];
    outputSize: 3;
    outputActivation: 'softmax';
    lossType: 'categoricalCrossEntropy';
    defaultNumSamples: number;
    defaultNoise: number;
    defaultTrainRatio: number;
    generate: typeof generateThreeClassClusters;
}

export type DatasetGenerationErrorCode =
    | 'invalid-settings'
    | 'coordinate-attempt-limit'
    | 'invalid-generated-target';

/** Structured failure emitted instead of returning an invalid version-2 sample. */
export class DatasetGenerationError extends Error {
    readonly name = 'DatasetGenerationError';

    constructor(
        readonly code: DatasetGenerationErrorCode,
        readonly dataset: string,
        message: string,
        readonly attempts?: number,
    ) {
        super(message);
    }
}

interface NormalizedDatasetGenerationRequest {
    dataset: DatasetId;
    sampleCount: number;
    noise: number;
    seed: number;
    trainFraction: number;
}

/** Strict version-2 generator. It validates every setting before sample allocation. */
export function generateDatasetV2(request: DatasetGenerationRequest): DataSplit {
    const normalized = validateDatasetGenerationRequest(request);
    return generateValidatedDataset(normalized);
}

/** Compatibility adapter for the pre-version-2 positional generator API. */
export function generateDataset(
    type: DatasetType,
    numSamples: number = DEFAULT_NUM_SAMPLES,
    noise: number = 0,
    trainRatio: number = 0.5,
    seed: number = 42,
): DataSplit {
    const requestedSamples = normalizeSampleCount(numSamples);
    return generateValidatedDataset({
        dataset: type,
        sampleCount: requestedSamples,
        noise: normalizeLegacyNoise(noise),
        trainFraction: normalizeLegacyTrainFraction(requestedSamples, trainRatio),
        seed: normalizeUint32Seed(seed),
    });
}

/** Generate a bounded three-class cluster split for future multiclass wiring. */
export function generateThreeClassClusters(
    numSamples: number = DEFAULT_NUM_SAMPLES,
    noise: number = 0,
    trainRatio: number = 0.5,
    seed: number = 42,
): DataSplit {
    const requestedSamples = normalizeSampleCount(numSamples);
    return generateValidatedDataset({
        dataset: 'three-class-clusters',
        sampleCount: requestedSamples,
        noise: normalizeLegacyNoise(noise),
        trainFraction: normalizeLegacyTrainFraction(requestedSamples, trainRatio),
        seed: normalizeUint32Seed(seed),
    });
}

export const THREE_CLASS_CLUSTER_DATASET_CONTRACT: ThreeClassClusterDatasetContract = Object.freeze({
    id: 'three-class-clusters',
    problemType: 'classification',
    classCount: 3,
    classLabels: THREE_CLASS_LABELS,
    outputSize: 3,
    outputActivation: 'softmax',
    lossType: 'categoricalCrossEntropy',
    defaultNumSamples: DEFAULT_NUM_SAMPLES,
    defaultNoise: THREE_CLASS_DEFAULT_NOISE,
    defaultTrainRatio: THREE_CLASS_DEFAULT_TRAIN_RATIO,
    generate: (
        numSamples = DEFAULT_NUM_SAMPLES,
        noise = THREE_CLASS_DEFAULT_NOISE,
        trainRatio = THREE_CLASS_DEFAULT_TRAIN_RATIO,
        seed = 42,
    ) => generateThreeClassClusters(numSamples, noise, trainRatio, seed),
});

function normalizeSampleCount(numSamples: number): number {
    return Number.isFinite(numSamples) ? Math.max(0, Math.floor(numSamples)) : DEFAULT_NUM_SAMPLES;
}

function normalizeLegacyNoise(noise: number): number {
    return Number.isFinite(noise) ? Math.max(0, noise) : 0;
}

function normalizeLegacyTrainFraction(sampleCount: number, trainRatio: number): number {
    if (sampleCount <= 1) return DEFAULT_TRAIN_FRACTION;
    return Number.isFinite(trainRatio) && trainRatio > 0 && trainRatio < 1
        ? trainRatio
        : DEFAULT_TRAIN_FRACTION;
}

function validateDatasetGenerationRequest(
    request: DatasetGenerationRequest,
): NormalizedDatasetGenerationRequest {
    const candidate = request as DatasetGenerationRequest | null | undefined;
    if (!candidate || typeof candidate !== 'object') {
        throw new DatasetGenerationError(
            'invalid-settings',
            'unknown',
            'Dataset generation requires a request object',
        );
    }

    const dataset = typeof candidate.dataset === 'string' ? candidate.dataset : 'unknown';
    let contract: DatasetContract;

    try {
        contract = getDatasetContract(candidate.dataset as DatasetId);
    } catch {
        throw new DatasetGenerationError(
            'invalid-settings',
            dataset,
            `Dataset generation requires one of the eleven registered dataset IDs; received ${dataset}`,
        );
    }

    if (!Number.isInteger(candidate?.sampleCount)
        || candidate.sampleCount < MINIMUM_SAMPLE_COUNT
        || candidate.sampleCount > MAXIMUM_SAMPLE_COUNT) {
        throw new DatasetGenerationError(
            'invalid-settings',
            dataset,
            `Dataset sampleCount must be an integer from ${MINIMUM_SAMPLE_COUNT} through ${MAXIMUM_SAMPLE_COUNT}`,
        );
    }
    if (!Number.isFinite(candidate.noise)
        || candidate.noise < contract.noise.minimum
        || candidate.noise > contract.noise.maximum) {
        throw new DatasetGenerationError(
            'invalid-settings',
            dataset,
            `Dataset noise must be finite and from ${contract.noise.minimum} through ${contract.noise.maximum}`,
        );
    }

    const trainFraction = candidate.trainFraction ?? DEFAULT_TRAIN_FRACTION;
    if (!Number.isFinite(trainFraction)
        || trainFraction < MINIMUM_TRAIN_FRACTION
        || trainFraction > MAXIMUM_TRAIN_FRACTION) {
        throw new DatasetGenerationError(
            'invalid-settings',
            dataset,
            `Dataset trainFraction must be finite and from ${MINIMUM_TRAIN_FRACTION} through ${MAXIMUM_TRAIN_FRACTION}`,
        );
    }

    let seed: number;
    try {
        seed = normalizeUint32Seed(candidate.seed);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new DatasetGenerationError('invalid-settings', dataset, message);
    }

    return {
        dataset: candidate.dataset,
        sampleCount: candidate.sampleCount,
        noise: candidate.noise,
        seed,
        trainFraction,
    };
}

function generateValidatedDataset(request: NormalizedDatasetGenerationRequest): DataSplit {
    const rng = new PRNG(request.seed);
    const contract = getDatasetContract(request.dataset);
    const pairedClassCount = Math.ceil(request.sampleCount / 2);
    let points: DataPoint[];

    switch (request.dataset) {
        case 'circle': points = genCircle(pairedClassCount, request.noise, rng, contract); break;
        case 'xor': points = genXor(request.sampleCount, request.noise, rng, contract); break;
        case 'gauss': points = genGauss(pairedClassCount, request.noise, rng, contract); break;
        case 'spiral': points = genSpiral(pairedClassCount, request.noise, rng, contract); break;
        case 'moons': points = genMoons(pairedClassCount, request.noise, rng, contract); break;
        case 'checkerboard': points = genCheckerboard(pairedClassCount, request.noise, rng, contract); break;
        case 'rings': points = genRings(request.sampleCount, request.noise, rng, contract); break;
        case 'heart': points = genHeart(pairedClassCount, request.noise, rng, contract); break;
        case 'three-class-clusters':
            points = genThreeClassClusters(request.sampleCount, request.noise, rng, contract);
            break;
        case 'reg-plane': points = genRegPlane(request.sampleCount, request.noise, rng, contract); break;
        case 'reg-gauss': points = genRegGauss(request.sampleCount, request.noise, rng, contract); break;
    }

    rng.shuffle(points);
    points = points.slice(0, request.sampleCount);
    const splitIdx = getSplitIndex(points.length, request.trainFraction);
    return {
        train: points.slice(0, splitIdx),
        test: points.slice(splitIdx),
    };
}

function getSplitIndex(total: number, trainRatio: number): number {
    if (total <= 0) return 0;
    if (total === 1) return 1;

    const ratio = Number.isFinite(trainRatio) && trainRatio > 0 && trainRatio < 1
        ? trainRatio
        : 0.5;
    const splitIdx = Math.floor(total * ratio);
    return Math.min(total - 1, Math.max(1, splitIdx));
}

function generateBoundedPoint(
    dataset: DatasetId,
    contract: DatasetContract,
    noise: number,
    createCandidate: () => DataPoint,
): DataPoint {
    for (let attempt = 1; attempt <= MAXIMUM_COORDINATE_ATTEMPTS; attempt++) {
        const candidate = createCandidate();
        if (!isCoordinateInsideContract(candidate, contract)) continue;
        if (!isTargetInsideContract(candidate.label, contract, noise)) {
            throw new DatasetGenerationError(
                'invalid-generated-target',
                dataset,
                `Dataset ${dataset} produced a target outside its declared contract`,
                attempt,
            );
        }
        return candidate;
    }

    throw new DatasetGenerationError(
        'coordinate-attempt-limit',
        dataset,
        `Dataset ${dataset} could not produce an in-domain coordinate in ${MAXIMUM_COORDINATE_ATTEMPTS} attempts`,
        MAXIMUM_COORDINATE_ATTEMPTS,
    );
}

function isCoordinateInsideContract(point: DataPoint, contract: DatasetContract): boolean {
    return Number.isFinite(point.x)
        && Number.isFinite(point.y)
        && point.x >= contract.inputDomain.x[0]
        && point.x <= contract.inputDomain.x[1]
        && point.y >= contract.inputDomain.y[0]
        && point.y <= contract.inputDomain.y[1];
}

function isTargetInsideContract(label: number, contract: DatasetContract, noise: number): boolean {
    if (!Number.isFinite(label)) return false;
    switch (contract.targetDomain.kind) {
        case 'binary': return label === 0 || label === 1;
        case 'classes':
            return Number.isInteger(label) && label >= 0 && label < contract.targetDomain.classCount;
        case 'continuous': {
            const [minimum, maximum] = contract.targetDomain.boundsForNoise(noise);
            return label >= minimum && label <= maximum;
        }
    }
}

// ── Classification datasets ──

function genThreeClassClusters(
    n: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    const scale = 0.08 + noise * 0.004;

    for (let i = 0; i < n; i++) {
        const center = THREE_CLASS_CENTERS[i % THREE_CLASS_CENTERS.length];
        points.push(generateBoundedPoint('three-class-clusters', contract, noise, () => ({
            x: center.x + rng.gaussian(0, scale),
            y: center.y + rng.gaussian(0, scale),
            label: center.label,
        })));
    }

    return points;
}

function genCircle(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    const radius = 0.5;
    for (let i = 0; i < pointsPerClass; i++) {
        // Inner circle (label 0)
        points.push(generateBoundedPoint('circle', contract, noise, () => {
            const r0 = rng.range(0, radius * 0.5);
            const a0 = rng.range(0, 2 * Math.PI);
            return {
                x: r0 * Math.cos(a0) + rng.gaussian(0, noise * 0.01),
                y: r0 * Math.sin(a0) + rng.gaussian(0, noise * 0.01),
                label: 0,
            };
        }));
        // Outer ring (label 1)
        points.push(generateBoundedPoint('circle', contract, noise, () => {
            const r1 = rng.range(radius * 0.7, radius);
            const a1 = rng.range(0, 2 * Math.PI);
            return {
                x: r1 * Math.cos(a1) + rng.gaussian(0, noise * 0.01),
                y: r1 * Math.sin(a1) + rng.gaussian(0, noise * 0.01),
                label: 1,
            };
        }));
    }
    return points;
}

function genXor(n: number, noise: number, rng: PRNG, contract: DatasetContract): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        const quadrant = i % 4;
        const xMin = quadrant === 0 || quadrant === 3 ? -1 : 0;
        const yMin = quadrant === 2 || quadrant === 3 ? -1 : 0;
        const label = quadrant === 0 || quadrant === 2 ? 0 : 1;

        points.push(generateBoundedPoint('xor', contract, noise, () => ({
            x: rng.range(xMin, xMin + 1) + rng.gaussian(0, noise * 0.01),
            y: rng.range(yMin, yMin + 1) + rng.gaussian(0, noise * 0.01),
            label,
        })));
    }
    return points;
}

function genGauss(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    const scale = 0.3 + noise * 0.005;
    for (let i = 0; i < pointsPerClass; i++) {
        points.push(generateBoundedPoint('gauss', contract, noise, () => ({
            x: rng.gaussian(-0.3, scale),
            y: rng.gaussian(-0.3, scale),
            label: 0,
        })));
        points.push(generateBoundedPoint('gauss', contract, noise, () => ({
            x: rng.gaussian(0.3, scale),
            y: rng.gaussian(0.3, scale),
            label: 1,
        })));
    }
    return points;
}

function genSpiral(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < pointsPerClass; i++) {
        for (let cls = 0; cls < 2; cls++) {
            points.push(generateBoundedPoint('spiral', contract, noise, () => {
                const r = (i / pointsPerClass) * 0.8;
                const t = (cls * Math.PI)
                    + (i / pointsPerClass) * 3 * Math.PI
                    + rng.gaussian(0, noise * 0.04);
                return {
                    x: r * Math.sin(t),
                    y: r * Math.cos(t),
                    label: cls,
                };
            }));
        }
    }
    return points;
}

function genMoons(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < pointsPerClass; i++) {
        // Upper moon (label 0)
        points.push(generateBoundedPoint('moons', contract, noise, () => {
            const angle0 = Math.PI * (i / pointsPerClass);
            return {
                x: Math.cos(angle0) * 0.5 + rng.gaussian(0, noise * 0.01),
                y: Math.sin(angle0) * 0.5 + rng.gaussian(0, noise * 0.01),
                label: 0,
            };
        }));
        // Lower moon (label 1)
        points.push(generateBoundedPoint('moons', contract, noise, () => {
            const angle1 = Math.PI * (i / pointsPerClass);
            return {
                x: 0.5 - Math.cos(angle1) * 0.5 + rng.gaussian(0, noise * 0.01),
                y: -Math.sin(angle1) * 0.5 + 0.3 + rng.gaussian(0, noise * 0.01),
                label: 1,
            };
        }));
    }
    return points;
}

function genCheckerboard(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    const n = pointsPerClass * 2;
    for (let i = 0; i < n; i++) {
        points.push(generateBoundedPoint('checkerboard', contract, noise, () => {
            const cleanX = rng.range(-1, 1);
            const cleanY = rng.range(-1, 1);
            const cx = cleanX >= 0 ? 1 : 0;
            const cy = cleanY >= 0 ? 1 : 0;
            return {
                x: cleanX + rng.gaussian(0, noise * 0.005),
                y: cleanY + rng.gaussian(0, noise * 0.005),
                label: (cx + cy) % 2,
            };
        }));
    }
    return points;
}

function genRings(n: number, noise: number, rng: PRNG, contract: DatasetContract): DataPoint[] {
    const points: DataPoint[] = [];
    const rings = 3;
    for (let i = 0; i < n; i++) {
        const ring = i % rings;
        const rMin = (ring * 0.9) / rings;
        const rMax = ((ring + 1) * 0.9) / rings;
        const label = ring % 2;
        points.push(generateBoundedPoint('rings', contract, noise, () => {
            const r = rng.range(rMin, rMax);
            const angle = rng.range(0, 2 * Math.PI);
            return {
                x: r * Math.cos(angle) + rng.gaussian(0, noise * 0.008),
                y: r * Math.sin(angle) + rng.gaussian(0, noise * 0.008),
                label,
            };
        }));
    }
    return points;
}

function genHeart(
    pointsPerClass: number,
    noise: number,
    rng: PRNG,
    contract: DatasetContract,
): DataPoint[] {
    const points: DataPoint[] = [];
    const n = pointsPerClass * 2;
    for (let i = 0; i < n; i++) {
        points.push(generateBoundedPoint('heart', contract, noise, () => {
            const cleanX = rng.range(-1, 1);
            const cleanY = rng.range(-1, 1);
            const x2 = cleanX * cleanX;
            const y2 = cleanY * cleanY;
            const inner = x2 + y2 - 0.6;
            const val = inner * inner * inner - x2 * y2 * cleanY;
            return {
                x: cleanX + rng.gaussian(0, noise * 0.005),
                y: cleanY + rng.gaussian(0, noise * 0.005),
                label: val < 0 ? 1 : 0,
            };
        }));
    }
    return points;
}

// ── Regression datasets ──

function genRegPlane(n: number, noise: number, rng: PRNG, contract: DatasetContract): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        points.push(generateBoundedPoint('reg-plane', contract, noise, () => {
            const x = rng.range(-1, 1);
            const y = rng.range(-1, 1);
            return {
                x,
                y,
                label: x + y + rng.gaussian(0, noise * 0.02),
            };
        }));
    }
    return points;
}

function genRegGauss(n: number, noise: number, rng: PRNG, contract: DatasetContract): DataPoint[] {
    const points: DataPoint[] = [];
    for (let i = 0; i < n; i++) {
        points.push(generateBoundedPoint('reg-gauss', contract, noise, () => {
            const x = rng.range(-1, 1);
            const y = rng.range(-1, 1);
            const v =
                Math.exp(-((x - 0.3) ** 2 + (y - 0.3) ** 2) / 0.2)
                + Math.exp(-((x + 0.3) ** 2 + (y + 0.3) ** 2) / 0.2);
            return {
                x,
                y,
                label: v + rng.gaussian(0, noise * 0.02),
            };
        }));
    }
    return points;
}

/** Get default problem type for a dataset. */
export function getDefaultProblemType(dataset: DatasetType): 'classification' | 'regression' {
    return dataset.startsWith('reg-') ? 'regression' : 'classification';
}
