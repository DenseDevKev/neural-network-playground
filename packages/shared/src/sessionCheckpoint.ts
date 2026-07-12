import {
    validateNetworkSessionStateV2,
    type NetworkSessionStateV2,
    type OptimizerSpecV2,
} from '@nn-playground/engine';
import {
    parsePairedEvaluation,
    type DatasetRevision,
    type ModelRevision,
    type PairedEvaluation,
} from './metricProvenance.js';

export const SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES = 262_144 as const;

export type SessionOptimizerStateV2 = NetworkSessionStateV2['optimizer'];

export interface SessionCheckpointV2 {
    readonly kind: 'nn-playground-session-checkpoint';
    readonly schemaVersion: 2;
    readonly recipeFingerprint: string;
    readonly objectiveKey: string;
    readonly datasetKey: string;
    readonly model: ModelRevision;
    readonly evaluation: PairedEvaluation;
    readonly network: NetworkSessionStateV2['network'];
    readonly optimizer: SessionOptimizerStateV2;
    readonly cursor: {
        readonly epoch: number;
        readonly batchStart: number;
        readonly shuffledIndices: Uint32Array;
    };
    readonly trajectoryGuarantee: 'parameters-and-optimizer-only';
}

export interface SessionCheckpointValidationContext {
    readonly recipeFingerprint: string;
    readonly objectiveKey: string;
    readonly dataset: DatasetRevision;
    readonly layerSizes: readonly number[];
    readonly optimizer: OptimizerSpecV2;
}

type PlainRecord = Record<string, unknown>;

function fail(path: string, message: string): never {
    throw new TypeError(`${path}: ${message}`);
}

function exactRecord(
    value: unknown,
    path: string,
    keys: readonly string[],
): PlainRecord {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return fail(path, 'must be a plain record');
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
        return fail(path, 'must have a plain record prototype');
    }
    const expected = new Set(keys);
    const ownKeys = Reflect.ownKeys(value);
    if (ownKeys.length !== keys.length
        || ownKeys.some((key) => typeof key !== 'string' || !expected.has(key))) {
        return fail(path, `must contain exactly ${[...keys].sort().join(', ')}`);
    }
    for (const key of keys) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
            fail(`${path}.${key}`, 'must be an enumerable data property');
        }
    }
    return value as PlainRecord;
}

function denseArray(value: unknown, path: string, length: number): unknown[] {
    if (!Array.isArray(value) || value.length !== length) {
        return fail(path, `must be a dense array of length ${length}`);
    }
    if (Object.getPrototypeOf(value) !== Array.prototype) {
        return fail(path, 'must have the standard Array prototype');
    }
    const expectedKeys = Array.from({ length }, (_, index) => String(index));
    const ownKeys = Reflect.ownKeys(value).filter((key) => key !== 'length');
    if (ownKeys.length !== expectedKeys.length
        || ownKeys.some((key, index) => key !== expectedKeys[index])) {
        return fail(path, `must be a dense array of length ${length}`);
    }
    for (const key of expectedKeys) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
            return fail(`${path}[${key}]`, 'must be an enumerable data property');
        }
    }
    return value;
}

function safeInteger(value: unknown, path: string, minimum: number): number {
    if (!Number.isSafeInteger(value) || (value as number) < minimum) {
        return fail(path, `must be a safe integer at least ${minimum}`);
    }
    return value as number;
}

function boundedIdentity(value: unknown, path: string): string {
    if (typeof value !== 'string' || value.length === 0 || value.length > 1_024) {
        return fail(path, 'must be a non-empty bounded identity string');
    }
    return value;
}

function parseModel(value: unknown, path: string): ModelRevision {
    const model = exactRecord(value, path, ['generationId', 'revision', 'step', 'epoch']);
    return {
        generationId: safeInteger(model['generationId'], `${path}.generationId`, 1),
        revision: safeInteger(model['revision'], `${path}.revision`, 0),
        step: safeInteger(model['step'], `${path}.step`, 0),
        epoch: safeInteger(model['epoch'], `${path}.epoch`, 0),
    };
}

function sameModel(left: ModelRevision, right: ModelRevision): boolean {
    return left.generationId === right.generationId
        && left.revision === right.revision
        && left.step === right.step
        && left.epoch === right.epoch;
}

function sameDataset(left: DatasetRevision, right: DatasetRevision): boolean {
    return left.generatorVersion === right.generatorVersion
        && left.datasetKey === right.datasetKey
        && left.trainCount === right.trainCount
        && left.testCount === right.testCount;
}

interface TypedArrayBudget {
    readonly seen: Set<ArrayBufferLike>;
    used: number;
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Float64Array.prototype) as object;
const GET_TYPED_ARRAY_TAG = Object.getOwnPropertyDescriptor(
    TYPED_ARRAY_PROTOTYPE,
    Symbol.toStringTag,
)?.get;

function typedArrayTag(value: unknown): string | undefined {
    if (typeof GET_TYPED_ARRAY_TAG !== 'function') {
        throw new Error('missing intrinsic TypedArray brand getter');
    }
    try {
        return Reflect.apply(GET_TYPED_ARRAY_TAG, value, []) as string | undefined;
    } catch {
        return undefined;
    }
}

function inspectView(
    value: unknown,
    constructor: typeof Float64Array | typeof Uint32Array,
    path: string,
    budget: TypedArrayBudget,
): Float64Array | Uint32Array {
    if (!ArrayBuffer.isView(value) || typedArrayTag(value) !== constructor.name) {
        return fail(path, `must be a ${constructor.name}`);
    }
    const view = value as Float64Array | Uint32Array;
    if (Object.prototype.toString.call(view.buffer) === '[object SharedArrayBuffer]') {
        return fail(path, 'must not use SharedArrayBuffer backing');
    }
    if (view.byteOffset !== 0 || view.byteLength !== view.buffer.byteLength) {
        return fail(path, 'must use a tight full-backing view, not a subview');
    }
    if (budget.seen.has(view.buffer)) {
        return fail(path, 'must not repeat or alias a typed-array backing buffer');
    }
    budget.seen.add(view.buffer);
    budget.used += view.byteLength;
    if (budget.used > SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES) {
        return fail(
            path,
            `typed-array payload must not exceed ${SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES} bytes`,
        );
    }
    return view;
}

function inspectFloat64List(
    value: unknown,
    path: string,
    lengths: readonly number[],
    budget: TypedArrayBudget,
): Float64Array[] {
    const list = denseArray(value, path, lengths.length);
    const clones: Float64Array[] = [];
    for (let index = 0; index < lengths.length; index++) {
        const view = inspectView(list[index], Float64Array, `${path}[${index}]`, budget);
        if (view.length !== lengths[index]) {
            fail(`${path}[${index}]`, `must contain ${lengths[index]} elements`);
        }
        clones.push(Float64Array.from(view));
    }
    return clones;
}

function inspectNetworkAndOptimizer(
    checkpoint: PlainRecord,
    context: SessionCheckpointValidationContext,
    budget: TypedArrayBudget,
): NetworkSessionStateV2 {
    if (!Array.isArray(context.layerSizes) || context.layerSizes.length < 2) {
        fail('context.layerSizes', 'must contain at least input and output sizes');
    }
    const layerSizes = context.layerSizes.map((size, index) => (
        safeInteger(size, `context.layerSizes[${index}]`, 1)
    ));
    const network = exactRecord(checkpoint['network'], 'checkpoint.network', ['layers']);
    const layerValues = denseArray(
        network['layers'],
        'checkpoint.network.layers',
        layerSizes.length - 1,
    );
    const weightLengths: number[] = [];
    const biasLengths: number[] = [];
    const clonedLayers: NetworkSessionStateV2['network']['layers'] = [];
    for (let index = 0; index < layerValues.length; index++) {
        const path = `checkpoint.network.layers[${index}]`;
        const layer = exactRecord(
            layerValues[index],
            path,
            ['inputSize', 'outputSize', 'weights', 'biases'],
        );
        const inputSize = layerSizes[index];
        const outputSize = layerSizes[index + 1];
        if (layer['inputSize'] !== inputSize) fail(`${path}.inputSize`, `must equal ${inputSize}`);
        if (layer['outputSize'] !== outputSize) fail(`${path}.outputSize`, `must equal ${outputSize}`);
        const weightLength = inputSize * outputSize;
        if (!Number.isSafeInteger(weightLength)) fail(path, 'parameter count exceeds safe range');
        const weights = inspectView(layer['weights'], Float64Array, `${path}.weights`, budget);
        const biases = inspectView(layer['biases'], Float64Array, `${path}.biases`, budget);
        if (weights.length !== weightLength) {
            fail(`${path}.weights`, `must contain ${weightLength} elements`);
        }
        if (biases.length !== outputSize) {
            fail(`${path}.biases`, `must contain ${outputSize} elements`);
        }
        weightLengths.push(weightLength);
        biasLengths.push(outputSize);
        clonedLayers.push({
            inputSize,
            outputSize,
            weights: Float64Array.from(weights),
            biases: Float64Array.from(biases),
        });
    }

    const optimizer = checkpoint['optimizer'];
    const optimizerRecord = exactRecord(
        optimizer,
        'checkpoint.optimizer',
        context.optimizer.kind === 'sgd'
            ? ['kind', 'optimizerStep']
            : context.optimizer.kind === 'sgd-momentum'
                ? ['kind', 'optimizerStep', 'weightVelocity', 'biasVelocity']
                : [
                    'kind',
                    'optimizerStep',
                    'firstWeightMoment',
                    'firstBiasMoment',
                    'secondWeightMoment',
                    'secondBiasMoment',
                ],
    );
    if (optimizerRecord['kind'] !== context.optimizer.kind) {
        fail('checkpoint.optimizer.kind', `must equal ${context.optimizer.kind}`);
    }
    safeInteger(optimizerRecord['optimizerStep'], 'checkpoint.optimizer.optimizerStep', 0);
    let clonedOptimizer: NetworkSessionStateV2['optimizer'];
    if (context.optimizer.kind === 'sgd') {
        clonedOptimizer = {
            kind: 'sgd',
            optimizerStep: optimizerRecord['optimizerStep'] as number,
        };
    } else if (context.optimizer.kind === 'sgd-momentum') {
        const weightVelocity = inspectFloat64List(
            optimizerRecord['weightVelocity'],
            'checkpoint.optimizer.weightVelocity',
            weightLengths,
            budget,
        );
        const biasVelocity = inspectFloat64List(
            optimizerRecord['biasVelocity'],
            'checkpoint.optimizer.biasVelocity',
            biasLengths,
            budget,
        );
        clonedOptimizer = {
            kind: 'sgd-momentum',
            optimizerStep: optimizerRecord['optimizerStep'] as number,
            weightVelocity,
            biasVelocity,
        };
    } else if (context.optimizer.kind === 'adam') {
        const firstWeightMoment = inspectFloat64List(
            optimizerRecord['firstWeightMoment'],
            'checkpoint.optimizer.firstWeightMoment',
            weightLengths,
            budget,
        );
        const firstBiasMoment = inspectFloat64List(
            optimizerRecord['firstBiasMoment'],
            'checkpoint.optimizer.firstBiasMoment',
            biasLengths,
            budget,
        );
        const secondWeightMoment = inspectFloat64List(
            optimizerRecord['secondWeightMoment'],
            'checkpoint.optimizer.secondWeightMoment',
            weightLengths,
            budget,
        );
        const secondBiasMoment = inspectFloat64List(
            optimizerRecord['secondBiasMoment'],
            'checkpoint.optimizer.secondBiasMoment',
            biasLengths,
            budget,
        );
        clonedOptimizer = {
            kind: 'adam',
            optimizerStep: optimizerRecord['optimizerStep'] as number,
            firstWeightMoment,
            firstBiasMoment,
            secondWeightMoment,
            secondBiasMoment,
        };
    } else {
        return fail('context.optimizer.kind', 'must be sgd, sgd-momentum, or adam');
    }

    return validateNetworkSessionStateV2(
        { network: { layers: clonedLayers }, optimizer: clonedOptimizer },
        { layerSizes, maximumBytes: SESSION_CHECKPOINT_MAX_TYPED_ARRAY_BYTES },
        context.optimizer,
    );
}

function parseCursor(
    value: unknown,
    model: ModelRevision,
    trainCount: number,
    budget: TypedArrayBudget,
): SessionCheckpointV2['cursor'] {
    const cursor = exactRecord(
        value,
        'checkpoint.cursor',
        ['epoch', 'batchStart', 'shuffledIndices'],
    );
    const epoch = safeInteger(cursor['epoch'], 'checkpoint.cursor.epoch', 0);
    const batchStart = safeInteger(cursor['batchStart'], 'checkpoint.cursor.batchStart', 0);
    if (epoch !== model.epoch) fail('checkpoint.cursor.epoch', 'must equal checkpoint model epoch');
    if (batchStart > trainCount) {
        fail('checkpoint.cursor.batchStart', `must be at most trainCount ${trainCount}`);
    }
    const indices = inspectView(
        cursor['shuffledIndices'],
        Uint32Array,
        'checkpoint.cursor.shuffledIndices',
        budget,
    );
    if (indices.length !== trainCount) {
        fail('checkpoint.cursor.shuffledIndices', `must contain ${trainCount} entries`);
    }
    const seen = new Uint8Array(trainCount);
    for (let index = 0; index < indices.length; index++) {
        const sampleIndex = indices[index];
        if (sampleIndex >= trainCount) {
            fail(`checkpoint.cursor.shuffledIndices[${index}]`, `must be below ${trainCount}`);
        }
        if (seen[sampleIndex] !== 0) {
            fail(`checkpoint.cursor.shuffledIndices[${index}]`, 'must not repeat an entry');
        }
        seen[sampleIndex] = 1;
    }
    return { epoch, batchStart, shuffledIndices: new Uint32Array(indices) };
}

/** Validate a complete V2 checkpoint and return a fully detached clone. */
export function validateSessionCheckpointV2(
    value: unknown,
    context: SessionCheckpointValidationContext,
): SessionCheckpointV2 {
    const checkpoint = exactRecord(value, 'checkpoint', [
        'kind',
        'schemaVersion',
        'recipeFingerprint',
        'objectiveKey',
        'datasetKey',
        'model',
        'evaluation',
        'network',
        'optimizer',
        'cursor',
        'trajectoryGuarantee',
    ]);
    if (checkpoint['kind'] !== 'nn-playground-session-checkpoint') {
        fail('checkpoint.kind', 'must be nn-playground-session-checkpoint');
    }
    if (checkpoint['schemaVersion'] !== 2) fail('checkpoint.schemaVersion', 'must equal 2');
    const recipeFingerprint = boundedIdentity(
        checkpoint['recipeFingerprint'],
        'checkpoint.recipeFingerprint',
    );
    const objectiveKey = boundedIdentity(checkpoint['objectiveKey'], 'checkpoint.objectiveKey');
    const datasetKey = boundedIdentity(checkpoint['datasetKey'], 'checkpoint.datasetKey');
    if (recipeFingerprint !== context.recipeFingerprint) {
        fail('checkpoint.recipeFingerprint', 'does not match the active recipe');
    }
    if (objectiveKey !== context.objectiveKey) {
        fail('checkpoint.objectiveKey', 'does not match the active objective');
    }
    if (datasetKey !== context.dataset.datasetKey) {
        fail('checkpoint.datasetKey', 'does not match the active dataset');
    }
    if (checkpoint['trajectoryGuarantee'] !== 'parameters-and-optimizer-only') {
        fail(
            'checkpoint.trajectoryGuarantee',
            'must be parameters-and-optimizer-only',
        );
    }

    const model = parseModel(checkpoint['model'], 'checkpoint.model');
    const evaluation = parsePairedEvaluation(checkpoint['evaluation']);
    if (evaluation.trigger !== 'initial' && evaluation.trigger !== 'checkpoint') {
        fail('checkpoint.evaluation.trigger', 'must be initial or checkpoint');
    }
    const isInitialSeed = evaluation.trigger === 'initial';
    if (isInitialSeed && (
        evaluation.evaluationId !== 1
        || model.revision !== 0
        || model.step !== 0
        || model.epoch !== 0
    )) {
        fail(
            'checkpoint.evaluation.trigger',
            'initial is reserved for the revision-zero step-zero seed pair',
        );
    }
    if (!sameModel(model, evaluation.model)) {
        fail('checkpoint.evaluation.model', 'must equal checkpoint.model');
    }
    if (evaluation.objectiveKey !== objectiveKey) {
        fail('checkpoint.evaluation.objectiveKey', 'must equal checkpoint.objectiveKey');
    }
    if (!sameDataset(evaluation.dataset, context.dataset)) {
        fail('checkpoint.evaluation.dataset', 'must equal the active dataset revision');
    }

    const budget: TypedArrayBudget = { seen: new Set(), used: 0 };
    const session = inspectNetworkAndOptimizer(checkpoint, context, budget);
    if (session.optimizer.optimizerStep !== model.step) {
        fail(
            'checkpoint.optimizer.optimizerStep',
            'must equal checkpoint.model.step',
        );
    }
    const cursor = parseCursor(
        checkpoint['cursor'],
        model,
        context.dataset.trainCount,
        budget,
    );
    if (isInitialSeed && cursor.batchStart !== 0) {
        fail('checkpoint.cursor.batchStart', 'initial checkpoint cursor must begin at zero');
    }

    return {
        kind: 'nn-playground-session-checkpoint',
        schemaVersion: 2,
        recipeFingerprint,
        objectiveKey,
        datasetKey,
        model,
        evaluation,
        network: session.network,
        optimizer: session.optimizer,
        cursor,
        trajectoryGuarantee: 'parameters-and-optimizer-only',
    };
}
