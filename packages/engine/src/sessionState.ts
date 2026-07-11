import type {
    ExpectedNetworkSessionShape,
    NetworkSessionStateV2,
    OptimizerSpecV2,
} from './types.js';

const NETWORK_SESSION_MAXIMUM_BYTES = 256 * 1024;
const FLOAT64_BYTES_PER_ELEMENT = 8;

type TypedArrayIntrinsicGetter = (this: unknown) => number;

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Float64Array.prototype) as object;

function getTypedArrayIntrinsicGetter(
    propertyName: 'length' | 'byteLength',
): TypedArrayIntrinsicGetter {
    const getter = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, propertyName)?.get;
    if (typeof getter !== 'function') {
        throw new Error(`missing intrinsic TypedArray ${propertyName} getter`);
    }
    return getter;
}

const GET_TYPED_ARRAY_LENGTH = getTypedArrayIntrinsicGetter('length');
const GET_TYPED_ARRAY_BYTE_LENGTH = getTypedArrayIntrinsicGetter('byteLength');

type RecordValue = Record<string, unknown>;

function assertRecord(value: unknown, name: string): asserts value is RecordValue {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) {
        throw new RangeError(`${name} must be an object`);
    }
}

function assertExactKeys(value: RecordValue, expectedKeys: readonly string[], name: string): void {
    const actualKeys = Object.keys(value).sort();
    const sortedExpectedKeys = [...expectedKeys].sort();
    if (
        actualKeys.length !== sortedExpectedKeys.length ||
        actualKeys.some((key, index) => key !== sortedExpectedKeys[index])
    ) {
        throw new RangeError(`${name} must contain exactly ${sortedExpectedKeys.join(', ')}`);
    }
}

function assertNonNegativeInteger(value: unknown, name: string): asserts value is number {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value as number) < 0) {
        throw new RangeError(`${name} must be a non-negative integer`);
    }
}

function validateExpectedShape(expectedShape: ExpectedNetworkSessionShape): readonly number[] {
    assertRecord(expectedShape, 'expectedShape');
    assertExactKeys(expectedShape, ['layerSizes', 'maximumBytes'], 'expectedShape');
    if (expectedShape.maximumBytes !== NETWORK_SESSION_MAXIMUM_BYTES) {
        throw new RangeError(`expectedShape.maximumBytes must be ${NETWORK_SESSION_MAXIMUM_BYTES}`);
    }
    if (!Array.isArray(expectedShape.layerSizes) || expectedShape.layerSizes.length < 2) {
        throw new RangeError('expectedShape.layerSizes must contain at least input and output sizes');
    }
    for (let index = 0; index < expectedShape.layerSizes.length; index++) {
        const size = expectedShape.layerSizes[index];
        if (!Number.isSafeInteger(size) || size <= 0) {
            throw new RangeError(`expectedShape.layerSizes[${index}] must be a positive safe integer`);
        }
    }
    return expectedShape.layerSizes;
}

function expectedOptimizerKind(expectedOptimizer: OptimizerSpecV2): OptimizerSpecV2['kind'] {
    assertRecord(expectedOptimizer, 'expectedOptimizer');
    switch (expectedOptimizer.kind) {
        case 'sgd':
        case 'sgd-momentum':
        case 'adam':
            return expectedOptimizer.kind;
        default:
            throw new RangeError('expectedOptimizer.kind must be sgd, sgd-momentum, or adam');
    }
}

function inspectFloat64Buffer(
    value: unknown,
    name: string,
): { buffer: Float64Array; length: number; byteLength: number } {
    if (!(value instanceof Float64Array)) {
        throw new RangeError(`${name} must be a Float64Array`);
    }

    let length: number;
    let byteLength: number;
    try {
        length = Reflect.apply(GET_TYPED_ARRAY_LENGTH, value, []);
        byteLength = Reflect.apply(GET_TYPED_ARRAY_BYTE_LENGTH, value, []);
    } catch {
        throw new RangeError(`${name} must be a compatible Float64Array`);
    }
    if (
        !Number.isSafeInteger(length) ||
        length < 0 ||
        !Number.isSafeInteger(byteLength) ||
        byteLength !== length * FLOAT64_BYTES_PER_ELEMENT
    ) {
        throw new RangeError(`${name} must use a valid Float64 representation`);
    }
    return { buffer: value, length, byteLength };
}

function cloneFiniteBuffer(
    value: unknown,
    expectedLength: number,
    name: string,
    byteBudget: { used: number; maximum: number },
): Float64Array {
    const inspected = inspectFloat64Buffer(value, name);
    if (inspected.length !== expectedLength) {
        throw new RangeError(`${name} must be a Float64Array of length ${expectedLength}`);
    }
    byteBudget.used += inspected.byteLength;
    if (byteBudget.used > byteBudget.maximum) {
        throw new RangeError(`network session typed arrays must not exceed ${byteBudget.maximum} bytes`);
    }
    const clone = new Float64Array(inspected.length);
    for (let index = 0; index < inspected.length; index++) {
        const element = inspected.buffer[index];
        if (!Number.isFinite(element)) {
            throw new RangeError(`${name}[${index}] must be finite`);
        }
        clone[index] = element;
    }
    return clone;
}

function cloneFiniteBufferList(
    value: unknown,
    expectedLengths: readonly number[],
    name: string,
    byteBudget: { used: number; maximum: number },
): Float64Array[] {
    if (!Array.isArray(value) || value.length !== expectedLengths.length) {
        throw new RangeError(`${name} must have ${expectedLengths.length} layers`);
    }
    return expectedLengths.map((expectedLength, layerIndex) => cloneFiniteBuffer(
        value[layerIndex],
        expectedLength,
        `${name}[${layerIndex}]`,
        byteBudget,
    ));
}

/** Validate a V2 engine session payload and return a fully detached clone. */
export function validateNetworkSessionStateV2(
    value: unknown,
    expectedShape: ExpectedNetworkSessionShape,
    expectedOptimizer: OptimizerSpecV2,
): NetworkSessionStateV2 {
    const layerSizes = validateExpectedShape(expectedShape);
    const optimizerKind = expectedOptimizerKind(expectedOptimizer);
    const byteBudget = { used: 0, maximum: NETWORK_SESSION_MAXIMUM_BYTES };

    assertRecord(value, 'network session state');
    assertExactKeys(value, ['network', 'optimizer'], 'network session state');
    assertRecord(value.network, 'network session state.network');
    assertExactKeys(value.network, ['layers'], 'network session state.network');
    if (!Array.isArray(value.network.layers) || value.network.layers.length !== layerSizes.length - 1) {
        throw new RangeError(`network session state.network.layers must have ${layerSizes.length - 1} layers`);
    }

    const weightLengths: number[] = [];
    const biasLengths: number[] = [];
    const layers = value.network.layers.map((layerValue, layerIndex) => {
        const name = `network session state.network.layers[${layerIndex}]`;
        assertRecord(layerValue, name);
        assertExactKeys(layerValue, ['inputSize', 'outputSize', 'weights', 'biases'], name);
        const inputSize = layerSizes[layerIndex];
        const outputSize = layerSizes[layerIndex + 1];
        if (layerValue.inputSize !== inputSize) {
            throw new RangeError(`${name}.inputSize must be ${inputSize}`);
        }
        if (layerValue.outputSize !== outputSize) {
            throw new RangeError(`${name}.outputSize must be ${outputSize}`);
        }
        const weightLength = inputSize * outputSize;
        if (!Number.isSafeInteger(weightLength)) {
            throw new RangeError(`${name} parameter count exceeds the safe integer range`);
        }
        weightLengths.push(weightLength);
        biasLengths.push(outputSize);
        return {
            inputSize,
            outputSize,
            weights: cloneFiniteBuffer(layerValue.weights, weightLength, `${name}.weights`, byteBudget),
            biases: cloneFiniteBuffer(layerValue.biases, outputSize, `${name}.biases`, byteBudget),
        };
    });

    assertRecord(value.optimizer, 'network session state.optimizer');
    if (value.optimizer.kind !== optimizerKind) {
        throw new RangeError(`network session optimizer kind must be ${optimizerKind}`);
    }
    assertNonNegativeInteger(value.optimizer.optimizerStep, 'network session optimizer step');
    const optimizerStep = value.optimizer.optimizerStep;

    switch (optimizerKind) {
        case 'sgd':
            assertExactKeys(value.optimizer, ['kind', 'optimizerStep'], 'network session state.optimizer');
            return {
                network: { layers },
                optimizer: { kind: 'sgd', optimizerStep },
            };
        case 'sgd-momentum':
            assertExactKeys(
                value.optimizer,
                ['kind', 'optimizerStep', 'weightVelocity', 'biasVelocity'],
                'network session state.optimizer',
            );
            return {
                network: { layers },
                optimizer: {
                    kind: 'sgd-momentum',
                    optimizerStep,
                    weightVelocity: cloneFiniteBufferList(
                        value.optimizer.weightVelocity,
                        weightLengths,
                        'network session state.optimizer.weightVelocity',
                        byteBudget,
                    ),
                    biasVelocity: cloneFiniteBufferList(
                        value.optimizer.biasVelocity,
                        biasLengths,
                        'network session state.optimizer.biasVelocity',
                        byteBudget,
                    ),
                },
            };
        case 'adam':
            assertExactKeys(
                value.optimizer,
                [
                    'kind',
                    'optimizerStep',
                    'firstWeightMoment',
                    'firstBiasMoment',
                    'secondWeightMoment',
                    'secondBiasMoment',
                ],
                'network session state.optimizer',
            );
            return {
                network: { layers },
                optimizer: {
                    kind: 'adam',
                    optimizerStep,
                    firstWeightMoment: cloneFiniteBufferList(
                        value.optimizer.firstWeightMoment,
                        weightLengths,
                        'network session state.optimizer.firstWeightMoment',
                        byteBudget,
                    ),
                    firstBiasMoment: cloneFiniteBufferList(
                        value.optimizer.firstBiasMoment,
                        biasLengths,
                        'network session state.optimizer.firstBiasMoment',
                        byteBudget,
                    ),
                    secondWeightMoment: cloneFiniteBufferList(
                        value.optimizer.secondWeightMoment,
                        weightLengths,
                        'network session state.optimizer.secondWeightMoment',
                        byteBudget,
                    ),
                    secondBiasMoment: cloneFiniteBufferList(
                        value.optimizer.secondBiasMoment,
                        biasLengths,
                        'network session state.optimizer.secondBiasMoment',
                        byteBudget,
                    ),
                },
            };
    }
}
