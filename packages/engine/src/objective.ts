import type {
    ClipResult,
    CompiledObjective,
    GradientClipSpecV2,
    GradientDiagnostics,
    MutableNumericArray,
    ObjectiveBreakdown,
    ObjectiveNetworkConfig,
    ObjectiveSpecV2,
    PenaltySpecV2,
} from './types.js';

const DISTRIBUTION_SUM_TOLERANCE = 1e-5;
const MAX_OBJECTIVE_PARAMETER = 1_000_000;

function assertFinite(value: number, name: string): void {
    if (!Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
    }
}

function assertFiniteNonNegative(value: number, name: string): void {
    if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`${name} must be finite and non-negative`);
    }
}

function assertFinitePositiveAtMost(value: number, maximum: number, name: string): void {
    if (!Number.isFinite(value) || value <= 0 || value > maximum) {
        throw new RangeError(`${name} must be finite and in range (0, ${maximum}]`);
    }
}

function assertFiniteBuffer(values: ArrayLike<number>, name: string): void {
    if (values == null || !Number.isInteger(values.length) || values.length < 0) {
        throw new RangeError(`${name} must be an array-like numeric buffer`);
    }
    for (let i = 0; i < values.length; i++) {
        assertFinite(values[i], `${name}[${i}]`);
    }
}

function assertVectorPair(
    left: ArrayLike<number>,
    right: ArrayLike<number>,
    leftName: string,
    rightName: string,
): number {
    if (left == null || right == null || left.length === 0) {
        throw new RangeError(`${leftName} and ${rightName} must not be empty`);
    }
    if (left.length !== right.length) {
        throw new RangeError(`${leftName} and ${rightName} must have the same length`);
    }
    assertFiniteBuffer(left, leftName);
    assertFiniteBuffer(right, rightName);
    return left.length;
}

function assertVectorLength(values: ArrayLike<number>, expectedLength: number, name: string): void {
    if (values == null || values.length !== expectedLength) {
        throw new RangeError(`${name} must have length ${expectedLength}`);
    }
    assertFiniteBuffer(values, name);
}

function assertDestinationLength(
    destination: MutableNumericArray,
    expectedLength: number,
): void {
    if (destination == null || destination.length !== expectedLength) {
        throw new RangeError(`destination must have length ${expectedLength}`);
    }
}

function assertBinaryTarget(target: number): void {
    if (target !== 0 && target !== 1) {
        throw new RangeError('binary cross-entropy target must be exactly 0 or 1');
    }
}

function assertCategoricalLogitsAndTarget(
    logits: ArrayLike<number>,
    target: ArrayLike<number>,
): number {
    const outputCount = assertVectorPair(logits, target, 'logits', 'target');
    let targetSum = 0;
    for (let i = 0; i < outputCount; i++) {
        if (target[i] < 0) {
            throw new RangeError('target values must be finite and non-negative');
        }
        targetSum += target[i];
    }
    if (Math.abs(targetSum - 1) > DISTRIBUTION_SUM_TOLERANCE) {
        throw new RangeError('target values must sum to 1');
    }
    return outputCount;
}

function stableSigmoid(logit: number): number {
    if (logit >= 0) {
        return 1 / (1 + Math.exp(-logit));
    }
    const exponential = Math.exp(logit);
    return exponential / (1 + exponential);
}

function binaryCrossEntropyWithLogitsUnchecked(logit: number, target: number): number {
    return Math.max(logit, 0) - logit * target + Math.log1p(Math.exp(-Math.abs(logit)));
}

/** Stable binary cross-entropy for one logit and a target in {0, 1}. */
export function binaryCrossEntropyWithLogits(logit: number, target: number): number {
    assertFinite(logit, 'logit');
    assertBinaryTarget(target);
    return binaryCrossEntropyWithLogitsUnchecked(logit, target);
}

/** Derivative of binary cross-entropy with respect to its input logit. */
export function binaryCrossEntropyLogitDelta(logit: number, target: number): number {
    assertFinite(logit, 'logit');
    assertBinaryTarget(target);
    return stableSigmoid(logit) - target;
}

function categoricalMaximum(logits: ArrayLike<number>): number {
    let maximum = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maximum) maximum = logits[i];
    }
    return maximum;
}

function categoricalShiftedExponentialSum(
    logits: ArrayLike<number>,
    maximum: number,
): number {
    let exponentialSum = 0;
    for (let i = 0; i < logits.length; i++) {
        exponentialSum += Math.exp(logits[i] - maximum);
    }
    return exponentialSum;
}

/** Stable categorical cross-entropy for normalized targets and raw logits. */
export function categoricalCrossEntropyWithLogits(
    logits: ArrayLike<number>,
    target: ArrayLike<number>,
): number {
    const outputCount = assertCategoricalLogitsAndTarget(logits, target);
    const maximum = categoricalMaximum(logits);
    const exponentialSum = categoricalShiftedExponentialSum(logits, maximum);
    const logExponentialSum = Math.log(exponentialSum);
    let loss = 0;
    for (let i = 0; i < outputCount; i++) {
        if (target[i] === 0) continue;
        loss += target[i] * ((maximum - logits[i]) + logExponentialSum);
    }
    return loss;
}

/** Softmax(logits) minus target, the categorical output-logit derivative. */
export function categoricalCrossEntropyLogitDelta(
    logits: ArrayLike<number>,
    target: ArrayLike<number>,
): number[] {
    const outputCount = assertCategoricalLogitsAndTarget(logits, target);
    const maximum = categoricalMaximum(logits);
    const inverseExponentialSum = 1 / categoricalShiftedExponentialSum(logits, maximum);
    const delta = new Array<number>(outputCount);
    for (let i = 0; i < outputCount; i++) {
        delta[i] = Math.exp(logits[i] - maximum) * inverseExponentialSum - target[i];
    }
    return delta;
}

/** Scalar true-MSE kernel used by the compatibility loss adapter. */
export function meanSquaredErrorScalar(prediction: number, target: number): number {
    assertFinite(prediction, 'prediction');
    assertFinite(target, 'target');
    return (prediction - target) ** 2;
}

/** Scalar true-MSE derivative used by the compatibility loss adapter. */
export function meanSquaredErrorScalarDelta(prediction: number, target: number): number {
    assertFinite(prediction, 'prediction');
    assertFinite(target, 'target');
    return 2 * (prediction - target);
}

/** Mean squared error across output coordinates. */
export function meanSquaredError(
    prediction: ArrayLike<number>,
    target: ArrayLike<number>,
): number {
    const outputCount = assertVectorPair(prediction, target, 'prediction', 'target');
    let sum = 0;
    for (let i = 0; i < outputCount; i++) {
        sum += (prediction[i] - target[i]) ** 2;
    }
    return sum / outputCount;
}

/** Prediction derivative of mean squared error across output coordinates. */
export function meanSquaredErrorDelta(
    prediction: ArrayLike<number>,
    target: ArrayLike<number>,
): number[] {
    const outputCount = assertVectorPair(prediction, target, 'prediction', 'target');
    const inverseOutputCount = 1 / outputCount;
    const delta = new Array<number>(outputCount);
    for (let i = 0; i < outputCount; i++) {
        delta[i] = 2 * (prediction[i] - target[i]) * inverseOutputCount;
    }
    return delta;
}

function assertHuberDelta(delta: number): void {
    assertFinitePositiveAtMost(delta, MAX_OBJECTIVE_PARAMETER, 'Huber delta');
}

function huberElement(error: number, delta: number): number {
    const absoluteError = Math.abs(error);
    return absoluteError <= delta
        ? 0.5 * error * error
        : delta * (absoluteError - 0.5 * delta);
}

function huberElementDelta(error: number, delta: number): number {
    return Math.abs(error) <= delta ? error : delta * Math.sign(error);
}

/** Conventional Huber loss, averaged across output coordinates. */
export function huberLoss(
    prediction: ArrayLike<number>,
    target: ArrayLike<number>,
    delta: number,
): number {
    assertHuberDelta(delta);
    const outputCount = assertVectorPair(prediction, target, 'prediction', 'target');
    let sum = 0;
    for (let i = 0; i < outputCount; i++) {
        sum += huberElement(prediction[i] - target[i], delta);
    }
    return sum / outputCount;
}

/** Prediction derivative of Huber loss, averaged across output coordinates. */
export function huberLossDelta(
    prediction: ArrayLike<number>,
    target: ArrayLike<number>,
    delta: number,
): number[] {
    assertHuberDelta(delta);
    const outputCount = assertVectorPair(prediction, target, 'prediction', 'target');
    const inverseOutputCount = 1 / outputCount;
    const result = new Array<number>(outputCount);
    for (let i = 0; i < outputCount; i++) {
        result[i] = huberElementDelta(prediction[i] - target[i], delta) * inverseOutputCount;
    }
    return result;
}

function assertPenaltyCoefficient(coefficient: number): void {
    assertFinitePositiveAtMost(coefficient, 1, 'penalty coefficient');
}

/** L1 penalty for one flat weight buffer. */
export function l1Penalty(weights: ArrayLike<number>, coefficient: number): number {
    assertPenaltyCoefficient(coefficient);
    assertFiniteBuffer(weights, 'weights');
    let sum = 0;
    for (let i = 0; i < weights.length; i++) sum += Math.abs(weights[i]);
    const penalty = coefficient * sum;
    assertFinite(penalty, 'L1 penalty');
    return penalty;
}

/** L1 penalty derivative for one flat weight buffer. */
export function l1PenaltyGradient(weights: ArrayLike<number>, coefficient: number): number[] {
    assertPenaltyCoefficient(coefficient);
    assertFiniteBuffer(weights, 'weights');
    const gradient = new Array<number>(weights.length);
    for (let i = 0; i < weights.length; i++) {
        gradient[i] = weights[i] === 0 ? 0 : coefficient * Math.sign(weights[i]);
    }
    return gradient;
}

/** L2 penalty for one flat weight buffer. */
export function l2Penalty(weights: ArrayLike<number>, coefficient: number): number {
    assertPenaltyCoefficient(coefficient);
    assertFiniteBuffer(weights, 'weights');
    let sumSquares = 0;
    for (let i = 0; i < weights.length; i++) sumSquares += weights[i] * weights[i];
    const penalty = 0.5 * coefficient * sumSquares;
    assertFinite(penalty, 'L2 penalty');
    return penalty;
}

/** L2 penalty derivative for one flat weight buffer. */
export function l2PenaltyGradient(weights: ArrayLike<number>, coefficient: number): number[] {
    assertPenaltyCoefficient(coefficient);
    assertFiniteBuffer(weights, 'weights');
    const gradient = new Array<number>(weights.length);
    for (let i = 0; i < weights.length; i++) gradient[i] = coefficient * weights[i];
    return gradient;
}

interface NormAccumulator {
    scale: number;
    scaledSumSquares: number;
}

function createNormAccumulator(): NormAccumulator {
    return { scale: 0, scaledSumSquares: 1 };
}

function addToNorm(accumulator: NormAccumulator, value: number, name: string): void {
    assertFinite(value, name);
    const absoluteValue = Math.abs(value);
    if (absoluteValue === 0) return;
    if (accumulator.scale < absoluteValue) {
        const ratio = accumulator.scale / absoluteValue;
        accumulator.scaledSumSquares = 1 + accumulator.scaledSumSquares * ratio * ratio;
        accumulator.scale = absoluteValue;
        return;
    }
    const ratio = absoluteValue / accumulator.scale;
    accumulator.scaledSumSquares += ratio * ratio;
}

function finishNorm(accumulator: NormAccumulator): number {
    if (accumulator.scale === 0) return 0;
    const norm = accumulator.scale * Math.sqrt(accumulator.scaledSumSquares);
    assertFinite(norm, 'gradient norm');
    return norm;
}

function addBuffersToNorm(
    accumulator: NormAccumulator,
    buffers: readonly ArrayLike<number>[],
    name: string,
): void {
    for (let bufferIndex = 0; bufferIndex < buffers.length; bufferIndex++) {
        const buffer = buffers[bufferIndex];
        assertFiniteBuffer(buffer, `${name}[${bufferIndex}]`);
        for (let i = 0; i < buffer.length; i++) {
            addToNorm(accumulator, buffer[i], `${name}[${bufferIndex}][${i}]`);
        }
    }
}

/** Global Euclidean norm over weight and bias buffers. */
export function gradientNorm(
    weightGradients: readonly ArrayLike<number>[],
    biasGradients: readonly ArrayLike<number>[] = [],
): number {
    const accumulator = createNormAccumulator();
    addBuffersToNorm(accumulator, weightGradients, 'weightGradients');
    addBuffersToNorm(accumulator, biasGradients, 'biasGradients');
    return finishNorm(accumulator);
}

function assertGradientClipSpec(spec: GradientClipSpecV2): void {
    if (spec == null || typeof spec !== 'object') {
        throw new RangeError('gradient clipping specification must be an object');
    }
    if (spec.kind === 'none') return;
    if (spec.kind !== 'global-norm') {
        throw new RangeError('unsupported gradient clipping kind');
    }
    if (spec.scope !== 'total-objective-gradient') {
        throw new RangeError('global gradient clipping must cover the total objective gradient');
    }
    assertFinitePositiveAtMost(spec.maximumNorm, MAX_OBJECTIVE_PARAMETER, 'maximum gradient norm');
}

/** Resolve a pure global clipping decision without applying optimizer semantics. */
export function computeGradientTransform(
    totalGradientNorm: number,
    clipSpec: GradientClipSpecV2,
): ClipResult {
    assertFiniteNonNegative(totalGradientNorm, 'totalGradientNorm');
    assertGradientClipSpec(clipSpec);
    if (clipSpec.kind === 'none' || totalGradientNorm <= clipSpec.maximumNorm) {
        return { totalGradientNorm, clippedGradientNorm: totalGradientNorm, clipScale: 1 };
    }
    return {
        totalGradientNorm,
        clippedGradientNorm: clipSpec.maximumNorm,
        clipScale: clipSpec.maximumNorm / totalGradientNorm,
    };
}

/** Apply one clip scale to the complete weight-and-bias objective gradient. */
export function applyGradientTransformInto(
    weightGradients: readonly MutableNumericArray[],
    biasGradients: readonly MutableNumericArray[],
    dataGradientNorm: number,
    penaltyGradientNorm: number,
    clipSpec: GradientClipSpecV2,
): GradientDiagnostics {
    assertFiniteNonNegative(dataGradientNorm, 'dataGradientNorm');
    assertFiniteNonNegative(penaltyGradientNorm, 'penaltyGradientNorm');
    const totalGradientNorm = gradientNorm(weightGradients, biasGradients);
    const transform = computeGradientTransform(totalGradientNorm, clipSpec);

    if (transform.clipScale !== 1) {
        for (const buffers of [weightGradients, biasGradients]) {
            for (const buffer of buffers) {
                for (let i = 0; i < buffer.length; i++) buffer[i] *= transform.clipScale;
            }
        }
    }

    return {
        dataGradientNorm,
        penaltyGradientNorm,
        ...transform,
    };
}

/** Build the explicit data-loss + model-penalty objective decomposition. */
export function buildObjectiveBreakdown(
    dataLoss: number,
    regularizationPenalty: number,
): ObjectiveBreakdown {
    assertFiniteNonNegative(dataLoss, 'dataLoss');
    assertFiniteNonNegative(regularizationPenalty, 'regularizationPenalty');
    const totalObjective = dataLoss + regularizationPenalty;
    assertFinite(totalObjective, 'totalObjective');
    return { dataLoss, regularizationPenalty, totalObjective };
}

function validateObjectiveSpec(spec: ObjectiveSpecV2): ObjectiveSpecV2 {
    if (spec == null || typeof spec !== 'object') {
        throw new RangeError('objective specification must be an object');
    }
    if (spec.reduction !== 'mean-per-sample') {
        throw new RangeError('objective reduction must be mean-per-sample');
    }

    let dataLoss: ObjectiveSpecV2['dataLoss'];
    switch (spec.dataLoss?.kind) {
        case 'binary-cross-entropy-with-logits':
        case 'categorical-cross-entropy-with-logits':
        case 'mean-squared-error':
            dataLoss = { kind: spec.dataLoss.kind };
            break;
        case 'huber':
            assertHuberDelta(spec.dataLoss.delta);
            dataLoss = { kind: 'huber', delta: spec.dataLoss.delta };
            break;
        default:
            throw new RangeError('unsupported objective data-loss kind');
    }

    const penaltySpec = spec.penalty as PenaltySpecV2 | undefined;
    let penalty: PenaltySpecV2;
    if (penaltySpec?.kind === 'none') {
        penalty = { kind: 'none' };
    } else if (penaltySpec?.kind === 'l1' || penaltySpec?.kind === 'l2') {
        assertPenaltyCoefficient(penaltySpec.coefficient);
        if (penaltySpec.applyTo !== 'weights') {
            throw new RangeError('regularization penalties apply to weights only');
        }
        penalty = {
            kind: penaltySpec.kind,
            coefficient: penaltySpec.coefficient,
            applyTo: 'weights',
        };
    } else {
        throw new RangeError('unsupported objective penalty kind');
    }

    return { dataLoss, penalty, reduction: 'mean-per-sample' };
}

function validateObjectiveNetworkConfig(config: ObjectiveNetworkConfig): void {
    if (config == null || typeof config !== 'object') {
        throw new RangeError('objective network configuration must be an object');
    }
    if (!Number.isInteger(config.outputSize) || config.outputSize <= 0) {
        throw new RangeError('objective network outputSize must be a positive integer');
    }
}

function assertObjectiveCompatibility(
    spec: ObjectiveSpecV2,
    config: ObjectiveNetworkConfig,
): void {
    switch (spec.dataLoss.kind) {
        case 'binary-cross-entropy-with-logits':
            if (config.outputSize !== 1 || config.outputActivation !== 'sigmoid') {
                throw new RangeError('binary cross-entropy requires one sigmoid output');
            }
            return;
        case 'categorical-cross-entropy-with-logits':
            if (config.outputSize !== 3 || config.outputActivation !== 'softmax') {
                throw new RangeError('categorical cross-entropy requires three softmax outputs');
            }
            return;
        case 'mean-squared-error':
        case 'huber':
            if (config.outputSize !== 1 || config.outputActivation !== 'linear') {
                throw new RangeError('regression objectives require one linear output');
            }
    }
}

function assertCompiledSampleVectors(
    logits: ArrayLike<number>,
    outputs: ArrayLike<number>,
    target: ArrayLike<number>,
    outputSize: number,
): void {
    assertVectorLength(logits, outputSize, 'logits');
    assertVectorLength(outputs, outputSize, 'outputs');
    assertVectorLength(target, outputSize, 'target');
}

function assertWeightGradientShapes(
    weights: readonly ArrayLike<number>[],
    gradients: readonly MutableNumericArray[],
): void {
    if (weights.length !== gradients.length) {
        throw new RangeError('weights and weightGradients must have the same layer count');
    }
    for (let layer = 0; layer < weights.length; layer++) {
        if (weights[layer].length !== gradients[layer].length) {
            throw new RangeError(`weights[${layer}] and weightGradients[${layer}] must have the same length`);
        }
        assertFiniteBuffer(weights[layer], `weights[${layer}]`);
        assertFiniteBuffer(gradients[layer], `weightGradients[${layer}]`);
    }
}

function penaltyForBuffers(
    weights: readonly ArrayLike<number>[],
    penaltySpec: PenaltySpecV2,
): number {
    let penalty = 0;
    for (let layer = 0; layer < weights.length; layer++) {
        const weightsForLayer = weights[layer];
        assertFiniteBuffer(weightsForLayer, `weights[${layer}]`);
        if (penaltySpec.kind === 'l1') {
            for (let i = 0; i < weightsForLayer.length; i++) {
                penalty += penaltySpec.coefficient * Math.abs(weightsForLayer[i]);
            }
        } else if (penaltySpec.kind === 'l2') {
            for (let i = 0; i < weightsForLayer.length; i++) {
                penalty += 0.5 * penaltySpec.coefficient * weightsForLayer[i] * weightsForLayer[i];
            }
        }
    }
    assertFinite(penalty, 'regularization penalty');
    return penalty;
}

function addPenaltyGradientToBuffers(
    weights: readonly ArrayLike<number>[],
    gradients: readonly MutableNumericArray[],
    penaltySpec: PenaltySpecV2,
): number {
    assertWeightGradientShapes(weights, gradients);
    if (penaltySpec.kind === 'none') return 0;

    const normAccumulator = createNormAccumulator();
    for (let layer = 0; layer < weights.length; layer++) {
        for (let i = 0; i < weights[layer].length; i++) {
            const penaltyGradient = penaltySpec.kind === 'l1'
                ? (weights[layer][i] === 0 ? 0 : penaltySpec.coefficient * Math.sign(weights[layer][i]))
                : penaltySpec.coefficient * weights[layer][i];
            addToNorm(normAccumulator, penaltyGradient, `penaltyGradient[${layer}][${i}]`);
            assertFinite(gradients[layer][i] + penaltyGradient, `combinedGradient[${layer}][${i}]`);
        }
    }

    for (let layer = 0; layer < weights.length; layer++) {
        for (let i = 0; i < weights[layer].length; i++) {
            gradients[layer][i] += penaltySpec.kind === 'l1'
                ? (weights[layer][i] === 0 ? 0 : penaltySpec.coefficient * Math.sign(weights[layer][i]))
                : penaltySpec.coefficient * weights[layer][i];
        }
    }
    return finishNorm(normAccumulator);
}

/** Compile a validated objective into allocation-free sample and model operations. */
export function compileObjective(
    spec: ObjectiveSpecV2,
    networkConfig: ObjectiveNetworkConfig,
): CompiledObjective {
    const compiledSpec = validateObjectiveSpec(spec);
    validateObjectiveNetworkConfig(networkConfig);
    assertObjectiveCompatibility(compiledSpec, networkConfig);
    const outputSize = networkConfig.outputSize;

    return {
        spec: compiledSpec,
        evaluateDataSample(logits, outputs, target): number {
            assertCompiledSampleVectors(logits, outputs, target, outputSize);
            switch (compiledSpec.dataLoss.kind) {
                case 'binary-cross-entropy-with-logits':
                    return binaryCrossEntropyWithLogits(logits[0], target[0]);
                case 'categorical-cross-entropy-with-logits':
                    return categoricalCrossEntropyWithLogits(logits, target);
                case 'mean-squared-error':
                    return meanSquaredError(outputs, target);
                case 'huber':
                    return huberLoss(outputs, target, compiledSpec.dataLoss.delta);
            }
        },
        seedOutputDeltaInto(logits, outputs, target, destination): void {
            assertCompiledSampleVectors(logits, outputs, target, outputSize);
            assertDestinationLength(destination, outputSize);
            switch (compiledSpec.dataLoss.kind) {
                case 'binary-cross-entropy-with-logits':
                    destination[0] = binaryCrossEntropyLogitDelta(logits[0], target[0]);
                    return;
                case 'categorical-cross-entropy-with-logits': {
                    assertCategoricalLogitsAndTarget(logits, target);
                    const maximum = categoricalMaximum(logits);
                    const inverseExponentialSum = 1 / categoricalShiftedExponentialSum(logits, maximum);
                    for (let i = 0; i < outputSize; i++) {
                        destination[i] = Math.exp(logits[i] - maximum) * inverseExponentialSum - target[i];
                    }
                    return;
                }
                case 'mean-squared-error': {
                    const inverseOutputCount = 1 / outputSize;
                    for (let i = 0; i < outputSize; i++) {
                        destination[i] = 2 * (outputs[i] - target[i]) * inverseOutputCount;
                    }
                    return;
                }
                case 'huber': {
                    const inverseOutputCount = 1 / outputSize;
                    for (let i = 0; i < outputSize; i++) {
                        destination[i] = huberElementDelta(
                            outputs[i] - target[i],
                            compiledSpec.dataLoss.delta,
                        ) * inverseOutputCount;
                    }
                }
            }
        },
        regularizationPenalty(weights): number {
            return penaltyForBuffers(weights, compiledSpec.penalty);
        },
        addPenaltyGradientInto(weights, weightGradients): number {
            return addPenaltyGradientToBuffers(weights, weightGradients, compiledSpec.penalty);
        },
    };
}
