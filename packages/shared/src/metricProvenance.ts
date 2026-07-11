import type {
    ConfusionMatrixData,
    MulticlassConfusionMatrixData,
} from '@nn-playground/engine';
import { createStableJsonSnapshot } from './canonicalJson.js';

export interface ModelRevision {
    readonly generationId: number;
    readonly revision: number;
    readonly step: number;
    readonly epoch: number;
}

export interface DatasetRevision {
    readonly generatorVersion: number;
    readonly datasetKey: string;
    readonly trainCount: number;
    readonly testCount: number;
}

export interface FullSplitBasis {
    readonly kind: 'full-split';
    readonly split: 'train' | 'test';
    readonly sampleCount: number;
    readonly populationCount: number;
}

export interface EvaluationValues {
    readonly dataLoss: number;
    readonly accuracy?: number;
    readonly confusionMatrix?: ConfusionMatrixData | MulticlassConfusionMatrixData;
}

export type EvaluationTrigger =
    | 'initial'
    | 'cadence'
    | 'manual-step'
    | 'pause'
    | 'checkpoint'
    | 'save'
    | 'stop-condition'
    | 'restore';

export interface PairedEvaluation {
    readonly evaluationId: number;
    readonly trigger: EvaluationTrigger;
    readonly model: ModelRevision;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly train: {
        readonly basis: FullSplitBasis & { readonly split: 'train' };
        readonly values: EvaluationValues;
    };
    readonly test: {
        readonly basis: FullSplitBasis & { readonly split: 'test' };
        readonly values: EvaluationValues;
    };
    readonly objective: {
        readonly regularizationPenalty: number;
        readonly trainTotalObjective: number;
    };
}

export interface LiveTrainingSignal {
    readonly model: ModelRevision;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly basis: {
        readonly kind: 'mini-batch-ema';
        readonly alpha: number;
        readonly latestBatchSize: number;
        readonly throughStep: number;
    };
    readonly dataLoss: number;
}

export type ArtifactBasis =
    | FullSplitBasis
    | {
        readonly kind: 'bounded-sample';
        readonly split: 'train' | 'test';
        readonly sampleCount: number;
        readonly populationCount: number;
    }
    | {
        readonly kind: 'prediction-grid';
        readonly pointCount: number;
        readonly domain: readonly [number, number, number, number];
    }
    | {
        readonly kind: 'parameter-grid';
        readonly sampleCount: number;
        readonly parameterPositions: number;
    };

export interface ArtifactProvenance {
    readonly model: ModelRevision;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly basis: ArtifactBasis;
}

export type TrainingTrendPoint = LiveTrainingSignal;
export type EvaluationPoint = PairedEvaluation;

export interface EvaluationPolicy {
    readonly everySteps: number;
    readonly forceOnPause: true;
    readonly forceOnManualStep: true;
    readonly forceOnCheckpoint: true;
    readonly forceOnSave: true;
}

export const DEFAULT_EVALUATION_POLICY: EvaluationPolicy = Object.freeze({
    everySteps: 50,
    forceOnPause: true,
    forceOnManualStep: true,
    forceOnCheckpoint: true,
    forceOnSave: true,
});

const EVALUATION_TRIGGERS = new Set<EvaluationTrigger>([
    'initial',
    'cadence',
    'manual-step',
    'pause',
    'checkpoint',
    'save',
    'stop-condition',
    'restore',
]);

function fail(path: string, message: string): never {
    throw new TypeError(`${path}: ${message}`);
}

function record(value: unknown, path: string): Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return fail(path, 'must be a record');
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
        return fail(path, 'must have a plain record prototype');
    }
    for (const key of Reflect.ownKeys(value)) {
        if (typeof key !== 'string') fail(`${path}.${String(key)}`, 'is not allowed');
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor === undefined || !('value' in descriptor)) {
            fail(`${path}.${key}`, 'must be a data property');
        }
        if (!descriptor.enumerable) {
            fail(`${path}.${key}`, 'must be enumerable');
        }
    }
    return value as Record<string, unknown>;
}

function densePlainArray(value: unknown, path: string, length: number): unknown[] {
    if (!Array.isArray(value) || value.length !== length) {
        return fail(path, `must be a dense array of length ${length}`);
    }
    if (Object.getPrototypeOf(value) !== Array.prototype) {
        return fail(path, 'must have the standard array prototype');
    }
    const expected = new Set(Array.from({ length }, (_, index) => String(index)));
    for (const key of Reflect.ownKeys(value)) {
        if (key === 'length') continue;
        if (typeof key !== 'string' || !expected.has(key)) {
            fail(`${path}.${String(key)}`, 'is not allowed');
        }
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
            fail(`${path}[${key}]`, 'must be an enumerable data property');
        }
    }
    for (let index = 0; index < length; index++) {
        if (!Object.prototype.hasOwnProperty.call(value, index)) {
            fail(`${path}[${index}]`, 'is required');
        }
    }
    return value;
}

function exactRecord(
    value: unknown,
    path: string,
    keys: readonly string[],
): Record<string, unknown> {
    const result = record(value, path);
    const allowed = new Set(keys);
    for (const key of Reflect.ownKeys(result)) {
        if (typeof key !== 'string' || !allowed.has(key)) {
            fail(`${path}.${String(key)}`, 'is not allowed');
        }
    }
    for (const key of keys) {
        if (!Object.prototype.hasOwnProperty.call(result, key)) {
            fail(`${path}.${key}`, 'is required');
        }
    }
    return result;
}

function exactRecordWithOptional(
    value: unknown,
    path: string,
    required: readonly string[],
    optional: readonly string[],
): Record<string, unknown> {
    const result = record(value, path);
    const allowed = new Set([...required, ...optional]);
    for (const key of Reflect.ownKeys(result)) {
        if (typeof key !== 'string' || !allowed.has(key)) {
            fail(`${path}.${String(key)}`, 'is not allowed');
        }
    }
    for (const key of required) {
        if (!Object.prototype.hasOwnProperty.call(result, key)) {
            fail(`${path}.${key}`, 'is required');
        }
    }
    return result;
}

function safeInteger(value: unknown, path: string, minimum: number, maximum: number): number {
    if (typeof value !== 'number'
        || !Number.isSafeInteger(value)
        || value < minimum
        || value > maximum) {
        return fail(path, `must be an integer from ${minimum} to ${maximum}`);
    }
    return value;
}

function finiteNumber(value: unknown, path: string, minimum = 0): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum) {
        return fail(path, `must be a finite number at least ${minimum}`);
    }
    return value;
}

function identity(value: unknown, path: string): string {
    if (typeof value !== 'string' || value.length === 0 || value.length > 1_024) {
        return fail(path, 'must be a non-empty bounded identity string');
    }
    return value;
}

function assertModel(value: unknown, path: string): asserts value is ModelRevision {
    const model = exactRecord(value, path, ['generationId', 'revision', 'step', 'epoch']);
    safeInteger(model['generationId'], `${path}.generationId`, 1, Number.MAX_SAFE_INTEGER);
    safeInteger(model['revision'], `${path}.revision`, 0, Number.MAX_SAFE_INTEGER);
    safeInteger(model['step'], `${path}.step`, 0, Number.MAX_SAFE_INTEGER);
    safeInteger(model['epoch'], `${path}.epoch`, 0, Number.MAX_SAFE_INTEGER);
}

function assertDataset(value: unknown, path: string): asserts value is DatasetRevision {
    const dataset = exactRecord(
        value,
        path,
        ['generatorVersion', 'datasetKey', 'trainCount', 'testCount'],
    );
    safeInteger(dataset['generatorVersion'], `${path}.generatorVersion`, 1, 4_294_967_295);
    identity(dataset['datasetKey'], `${path}.datasetKey`);
    const trainCount = safeInteger(dataset['trainCount'], `${path}.trainCount`, 1, 1_000);
    const testCount = safeInteger(dataset['testCount'], `${path}.testCount`, 1, 1_000);
    if (trainCount + testCount > 1_000) {
        fail(path, 'train and test populations must total at most 1000');
    }
}

function splitPopulation(dataset: DatasetRevision, split: 'train' | 'test'): number {
    return split === 'train' ? dataset.trainCount : dataset.testCount;
}

function assertFullSplitBasis(
    value: unknown,
    path: string,
    dataset: DatasetRevision,
    expectedSplit?: 'train' | 'test',
): asserts value is FullSplitBasis {
    const basis = exactRecord(
        value,
        path,
        ['kind', 'split', 'sampleCount', 'populationCount'],
    );
    if (basis['kind'] !== 'full-split') fail(`${path}.kind`, 'must be full-split');
    if (basis['split'] !== 'train' && basis['split'] !== 'test') {
        fail(`${path}.split`, 'must be train or test');
    }
    if (expectedSplit !== undefined && basis['split'] !== expectedSplit) {
        fail(`${path}.split`, `must be ${expectedSplit}`);
    }
    const population = splitPopulation(dataset, basis['split']);
    const sampleCount = safeInteger(
        basis['sampleCount'],
        `${path}.sampleCount`,
        1,
        population,
    );
    const populationCount = safeInteger(
        basis['populationCount'],
        `${path}.populationCount`,
        1,
        population,
    );
    if (sampleCount !== population) {
        fail(`${path}.sampleCount`, `must equal the complete population ${population}`);
    }
    if (populationCount !== population) {
        fail(`${path}.populationCount`, `must equal the complete population ${population}`);
    }
}

function confusionCount(value: unknown, path: string): number {
    return safeInteger(value, path, 0, 1_000);
}

function assertConfusionMatrix(value: unknown, path: string, sampleCount: number): number {
    const matrix = record(value, path);
    if ('classCount' in matrix || 'counts' in matrix || 'classLabels' in matrix) {
        const multiclass = exactRecord(
            matrix,
            path,
            ['classCount', 'classLabels', 'counts'],
        );
        if (multiclass['classCount'] !== 3) fail(`${path}.classCount`, 'must be 3');
        const labels = densePlainArray(multiclass['classLabels'], `${path}.classLabels`, 3);
        if (labels[0] !== 0
            || labels[1] !== 1
            || labels[2] !== 2) {
            fail(`${path}.classLabels`, 'must be [0, 1, 2]');
        }
        const counts = densePlainArray(multiclass['counts'], `${path}.counts`, 9);
        let total = 0;
        for (let index = 0; index < 9; index++) {
            if (!Object.prototype.hasOwnProperty.call(counts, index)) {
                fail(`${path}.counts[${index}]`, 'is required');
            }
            total += confusionCount(counts[index], `${path}.counts[${index}]`);
        }
        if (total !== sampleCount) fail(path, `counts must total ${sampleCount}`);
        return (
            confusionCount(counts[0], `${path}.counts[0]`)
            + confusionCount(counts[4], `${path}.counts[4]`)
            + confusionCount(counts[8], `${path}.counts[8]`)
        ) / sampleCount;
    }

    const binary = exactRecord(matrix, path, ['tp', 'tn', 'fp', 'fn']);
    const total = confusionCount(binary['tp'], `${path}.tp`)
        + confusionCount(binary['tn'], `${path}.tn`)
        + confusionCount(binary['fp'], `${path}.fp`)
        + confusionCount(binary['fn'], `${path}.fn`);
    if (total !== sampleCount) fail(path, `counts must total ${sampleCount}`);
    return (
        confusionCount(binary['tp'], `${path}.tp`)
        + confusionCount(binary['tn'], `${path}.tn`)
    ) / sampleCount;
}

function assertEvaluationValues(
    value: unknown,
    path: string,
    sampleCount: number,
): asserts value is EvaluationValues {
    const values = exactRecordWithOptional(
        value,
        path,
        ['dataLoss'],
        ['accuracy', 'confusionMatrix'],
    );
    finiteNumber(values['dataLoss'], `${path}.dataLoss`);
    if (Object.prototype.hasOwnProperty.call(values, 'accuracy')) {
        const accuracy = finiteNumber(values['accuracy'], `${path}.accuracy`);
        if (accuracy > 1) fail(`${path}.accuracy`, 'must be at most 1');
    }
    if (Object.prototype.hasOwnProperty.call(values, 'confusionMatrix')) {
        const matrixAccuracy = assertConfusionMatrix(
            values['confusionMatrix'],
            `${path}.confusionMatrix`,
            sampleCount,
        );
        if (Object.prototype.hasOwnProperty.call(values, 'accuracy')
            && Math.abs((values['accuracy'] as number) - matrixAccuracy) > 1e-12) {
            fail(`${path}.accuracy`, 'must equal the accuracy derived from confusion counts');
        }
    }
}

function assertPairedEvaluationSnapshot(
    value: unknown,
): asserts value is PairedEvaluation {
    const pair = exactRecord(
        value,
        'evaluation',
        ['evaluationId', 'trigger', 'model', 'dataset', 'objectiveKey', 'train', 'test', 'objective'],
    );
    safeInteger(pair['evaluationId'], 'evaluationId', 1, Number.MAX_SAFE_INTEGER);
    if (typeof pair['trigger'] !== 'string'
        || !EVALUATION_TRIGGERS.has(pair['trigger'] as EvaluationTrigger)) {
        fail('trigger', 'is not supported');
    }
    assertModel(pair['model'], 'model');
    assertDataset(pair['dataset'], 'dataset');
    identity(pair['objectiveKey'], 'objectiveKey');

    const train = exactRecord(pair['train'], 'train', ['basis', 'values']);
    assertFullSplitBasis(train['basis'], 'train.basis', pair['dataset'], 'train');
    assertEvaluationValues(train['values'], 'train.values', pair['dataset'].trainCount);
    const test = exactRecord(pair['test'], 'test', ['basis', 'values']);
    assertFullSplitBasis(test['basis'], 'test.basis', pair['dataset'], 'test');
    assertEvaluationValues(test['values'], 'test.values', pair['dataset'].testCount);

    const objective = exactRecord(
        pair['objective'],
        'objective',
        ['regularizationPenalty', 'trainTotalObjective'],
    );
    const penalty = finiteNumber(
        objective['regularizationPenalty'],
        'objective.regularizationPenalty',
    );
    const total = finiteNumber(
        objective['trainTotalObjective'],
        'objective.trainTotalObjective',
    );
    const expected = train['values'].dataLoss + penalty;
    if (!Number.isFinite(expected)) {
        fail('objective.trainTotalObjective', 'component sum must remain finite');
    }
    const tolerance = 1e-12 * Math.max(1, Math.abs(expected));
    if (Math.abs(total - expected) > tolerance) {
        fail(
            'objective.trainTotalObjective',
            'must equal train data loss plus regularization penalty',
        );
    }
}

export function parsePairedEvaluation(value: unknown): PairedEvaluation {
    const snapshot = createStableJsonSnapshot(value);
    assertPairedEvaluationSnapshot(snapshot);
    return snapshot;
}

export function assertPairedEvaluation(value: unknown): void {
    void parsePairedEvaluation(value);
}

function assertLiveTrainingSignalSnapshot(
    value: unknown,
): asserts value is LiveTrainingSignal {
    const signal = exactRecord(
        value,
        'signal',
        ['model', 'dataset', 'objectiveKey', 'basis', 'dataLoss'],
    );
    assertModel(signal['model'], 'model');
    assertDataset(signal['dataset'], 'dataset');
    identity(signal['objectiveKey'], 'objectiveKey');
    const basis = exactRecord(
        signal['basis'],
        'basis',
        ['kind', 'alpha', 'latestBatchSize', 'throughStep'],
    );
    if (basis['kind'] !== 'mini-batch-ema') fail('basis.kind', 'must be mini-batch-ema');
    const alpha = finiteNumber(basis['alpha'], 'basis.alpha');
    if (alpha <= 0 || alpha > 1) fail('basis.alpha', 'must be greater than 0 and at most 1');
    safeInteger(
        basis['latestBatchSize'],
        'basis.latestBatchSize',
        1,
        signal['dataset'].trainCount,
    );
    const throughStep = safeInteger(
        basis['throughStep'],
        'basis.throughStep',
        0,
        Number.MAX_SAFE_INTEGER,
    );
    if (throughStep !== signal['model'].step) {
        fail('basis.throughStep', 'must equal model.step');
    }
    finiteNumber(signal['dataLoss'], 'dataLoss');
}

export function parseLiveTrainingSignal(value: unknown): LiveTrainingSignal {
    const snapshot = createStableJsonSnapshot(value);
    assertLiveTrainingSignalSnapshot(snapshot);
    return snapshot;
}

export function assertLiveTrainingSignal(value: unknown): void {
    void parseLiveTrainingSignal(value);
}

function assertArtifactProvenanceSnapshot(
    value: unknown,
): asserts value is ArtifactProvenance {
    const artifact = exactRecord(
        value,
        'artifact',
        ['model', 'dataset', 'objectiveKey', 'basis'],
    );
    assertModel(artifact['model'], 'model');
    assertDataset(artifact['dataset'], 'dataset');
    identity(artifact['objectiveKey'], 'objectiveKey');
    const basis = record(artifact['basis'], 'basis');
    switch (basis['kind']) {
        case 'full-split':
            assertFullSplitBasis(basis, 'basis', artifact['dataset']);
            return;
        case 'bounded-sample': {
            const bounded = exactRecord(
                basis,
                'basis',
                ['kind', 'split', 'sampleCount', 'populationCount'],
            );
            if (bounded['split'] !== 'train' && bounded['split'] !== 'test') {
                fail('basis.split', 'must be train or test');
            }
            const population = splitPopulation(artifact['dataset'], bounded['split']);
            const populationCount = safeInteger(
                bounded['populationCount'],
                'basis.populationCount',
                1,
                population,
            );
            if (populationCount !== population) {
                fail('basis.populationCount', `must equal ${population}`);
            }
            safeInteger(bounded['sampleCount'], 'basis.sampleCount', 1, populationCount);
            return;
        }
        case 'prediction-grid': {
            const grid = exactRecord(basis, 'basis', ['kind', 'pointCount', 'domain']);
            safeInteger(grid['pointCount'], 'basis.pointCount', 1, 1_000_000);
            const domain = densePlainArray(grid['domain'], 'basis.domain', 4);
            const numericDomain = domain.map((entry, index) => {
                if (typeof entry !== 'number' || !Number.isFinite(entry)) {
                    fail(`basis.domain[${index}]`, 'must be finite');
                }
                return entry;
            });
            if (numericDomain[0] >= numericDomain[1] || numericDomain[2] >= numericDomain[3]) {
                fail('basis.domain', 'must have increasing x and y bounds');
            }
            return;
        }
        case 'parameter-grid': {
            const grid = exactRecord(
                basis,
                'basis',
                ['kind', 'sampleCount', 'parameterPositions'],
            );
            safeInteger(
                grid['sampleCount'],
                'basis.sampleCount',
                1,
                artifact['dataset'].trainCount + artifact['dataset'].testCount,
            );
            safeInteger(
                grid['parameterPositions'],
                'basis.parameterPositions',
                1,
                1_000_000,
            );
            return;
        }
        default:
            fail('basis.kind', 'is not supported');
    }
}

export function parseArtifactProvenance(value: unknown): ArtifactProvenance {
    const snapshot = createStableJsonSnapshot(value);
    assertArtifactProvenanceSnapshot(snapshot);
    return snapshot;
}

export function assertArtifactProvenance(value: unknown): void {
    void parseArtifactProvenance(value);
}
