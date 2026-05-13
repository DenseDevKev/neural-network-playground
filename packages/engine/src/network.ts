// ── Core dense feed-forward network ──
// Packed Float32Array storage for every hot-path buffer (weights, biases,
// gradients, optimizer moments, activation scratch). All inner loops are
// plain arithmetic over typed arrays — no nested number[][][] walking, no
// per-step allocations, no polymorphic function-pointer dispatch over
// activation/loss/optimizer objects. Behaviour and public API are
// preserved: the nested accessors (`getWeights`, `getBiases`, `serialize`,
// `getSnapshot` with `includeParams`) still return `number[][][]` /
// `number[][]` for callers that expect the legacy shape.

import type {
    NetworkConfig,
    TrainingConfig,
    SerializedNetwork,
    NetworkSnapshot,
    HistoryPoint,
    Metrics,
    LossType,
    LayerStats,
    ActivationType,
    PredictionTrace,
    ActivationHistogramOptions,
    ActivationHistogramResult,
    ActivationHistogramLayer,
    NetworkCheckpoint,
    BackpropExplanation,
    BackpropExplanationLayer,
    BackpropExplanationStatus,
    LossLandscapeProbe,
    LossLandscapeProbeOptions,
    LossLandscapeParameter,
} from './types.js';
import { getLoss } from './losses.js';
import { computeLearningRate, validateLRSchedule } from './schedules.js';
import { initWeightsInto, initBiasesInto } from './initialization.js';
import { transformPoint } from './features.js';
import type { FeatureSpec } from './features.js';
import { PRNG } from './prng.js';

// ── Activation kernels ──────────────────────────────────────────────────────
// Each kernel is monomorphic — V8 sees exactly one shape at each call site
// once the Network constructor has captured the appropriate reference.
//
// We store every parameter / gradient / scratch buffer as Float64Array so
// forward/backward arithmetic matches the legacy float64 semantics. The
// big wins (packed layout, zero per-step allocations, monomorphic dispatch,
// fused gradient clip) don't depend on 32-bit storage — they come from
// getting rid of nested number[] indirection.

type Buf = Float64Array;
type LayerActFn = (pre: Buf, out: Buf, n: number) => void;
type LayerDActFn = (delta: Buf, pre: Buf, out: Buf, n: number) => void;

function actRelu(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const z = pre[i];
        out[i] = z > 0 ? z : 0;
    }
}
function actTanh(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) out[i] = Math.tanh(pre[i]);
}
function actSigmoid(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-pre[i]));
}
function actLinear(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) out[i] = pre[i];
}
function actLeakyRelu(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const z = pre[i];
        out[i] = z > 0 ? z : 0.01 * z;
    }
}
function actElu(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const z = pre[i];
        out[i] = z >= 0 ? z : Math.exp(z) - 1;
    }
}
function actSwish(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const z = pre[i];
        out[i] = z / (1 + Math.exp(-z));
    }
}
function stableSoftplusScalar(x: number): number {
    return x > 0
        ? x + Math.log1p(Math.exp(-x))
        : Math.log1p(Math.exp(x));
}
function stableSigmoidScalar(x: number): number {
    if (x >= 0) {
        return 1 / (1 + Math.exp(-x));
    }
    const ex = Math.exp(x);
    return ex / (1 + ex);
}
function actSoftplus(pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) out[i] = stableSoftplusScalar(pre[i]);
}

function pickAct(kind: ActivationType): LayerActFn {
    switch (kind) {
        case 'relu': return actRelu;
        case 'tanh': return actTanh;
        case 'sigmoid': return actSigmoid;
        case 'linear': return actLinear;
        case 'leakyRelu': return actLeakyRelu;
        case 'elu': return actElu;
        case 'swish': return actSwish;
        case 'softplus': return actSoftplus;
    }
}

// In-place element-wise multiplication of `delta` by the activation
// derivative. This mirrors the semantics of activations.ts:
//   relu/leakyRelu/elu use the output value (and, for elu, the pre-act);
//   tanh/sigmoid use the output;
//   swish/softplus need the pre-act to recover the sigmoid factor.

function dActRelu(delta: Buf, _pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) if (out[i] <= 0) delta[i] = 0;
}
function dActTanh(delta: Buf, _pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const o = out[i];
        delta[i] *= 1 - o * o;
    }
}
function dActSigmoid(delta: Buf, _pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const o = out[i];
        delta[i] *= o * (1 - o);
    }
}
function dActLinear(_d: Buf, _p: Buf, _o: Buf, _n: number): void { /* identity */ }

function dActLeakyRelu(delta: Buf, _pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) if (out[i] <= 0) delta[i] *= 0.01;
}
function dActElu(delta: Buf, pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        if (pre[i] < 0) delta[i] *= out[i] + 1;
    }
}
function dActSwish(delta: Buf, pre: Buf, out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        const sig = 1 / (1 + Math.exp(-pre[i]));
        delta[i] *= out[i] + sig * (1 - out[i]);
    }
}
function dActSoftplus(delta: Buf, pre: Buf, _out: Buf, n: number): void {
    for (let i = 0; i < n; i++) {
        delta[i] *= stableSigmoidScalar(pre[i]);
    }
}

function pickDAct(kind: ActivationType): LayerDActFn {
    switch (kind) {
        case 'relu': return dActRelu;
        case 'tanh': return dActTanh;
        case 'sigmoid': return dActSigmoid;
        case 'linear': return dActLinear;
        case 'leakyRelu': return dActLeakyRelu;
        case 'elu': return dActElu;
        case 'swish': return dActSwish;
        case 'softplus': return dActSoftplus;
    }
}

function assertLayerSize(value: number, name: string): void {
    if (!Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
        throw new RangeError(`${name} must be a finite positive integer`);
    }
}

function validatedLayerSizes(config: NetworkConfig): number[] {
    assertLayerSize(config.inputSize, 'inputSize');
    if (!Array.isArray(config.hiddenLayers)) {
        throw new RangeError('hiddenLayers must be an array');
    }
    for (let i = 0; i < config.hiddenLayers.length; i++) {
        assertLayerSize(config.hiddenLayers[i], `hiddenLayers[${i}]`);
    }
    assertLayerSize(config.outputSize, 'outputSize');
    return [config.inputSize, ...config.hiddenLayers, config.outputSize];
}

function assertFiniteValue(value: number, name: string): void {
    if (!Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
    }
}

function assertFiniteInRange(
    value: number,
    name: string,
    min: number,
    max: number,
    options: { minInclusive?: boolean; maxInclusive?: boolean } = {},
): void {
    const minOk = options.minInclusive === true ? value >= min : value > min;
    const maxOk = options.maxInclusive === true ? value <= max : value < max;
    if (!Number.isFinite(value) || !minOk || !maxOk) {
        const minLabel = options.minInclusive === true ? `[${min}` : `(${min}`;
        const maxLabel = options.maxInclusive === true ? `${max}]` : `${max})`;
        throw new RangeError(`${name} must be finite and in range ${minLabel}, ${maxLabel}`);
    }
}

function assertVectorShape(
    vector: ArrayLike<number> | undefined,
    expectedLen: number,
    name: string,
): asserts vector is ArrayLike<number> {
    if (vector == null || vector.length !== expectedLen) {
        throw new RangeError(`${name} must have length ${expectedLen}`);
    }
    for (let i = 0; i < expectedLen; i++) {
        assertFiniteValue(vector[i], `${name}[${i}]`);
    }
}

function assertBatchShapes(inputs: number[][], targets: number[][], inputSize: number, outputSize: number): void {
    if (!Array.isArray(inputs) || !Array.isArray(targets)) {
        throw new RangeError('inputs and targets must be arrays');
    }
    if (inputs.length !== targets.length) {
        throw new RangeError('inputs and targets must have the same batch length');
    }
    for (let i = 0; i < inputs.length; i++) {
        assertVectorShape(inputs[i], inputSize, `inputs[${i}]`);
        assertVectorShape(targets[i], outputSize, `targets[${i}]`);
    }
}

function assertIndexedBatchSelection(
    inputs: number[][],
    targets: number[][],
    indices: ArrayLike<number> | undefined,
    start: number,
    end: number,
    inputSize: number,
    outputSize: number,
): void {
    if (!Array.isArray(inputs) || !Array.isArray(targets)) {
        throw new RangeError('inputs and targets must be arrays');
    }
    if (inputs.length !== targets.length) {
        throw new RangeError('inputs and targets must have the same batch length');
    }
    if (indices == null || !Number.isInteger(indices.length)) {
        throw new RangeError('indices must be an array-like object');
    }
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > indices.length) {
        throw new RangeError('start and end must describe a valid indices range');
    }

    for (let i = start; i < end; i++) {
        const sampleIdx = indices[i];
        if (!Number.isInteger(sampleIdx) || sampleIdx < 0 || sampleIdx >= inputs.length) {
            throw new RangeError(`indices[${i}] must reference a valid sample`);
        }
        assertVectorShape(inputs[sampleIdx], inputSize, `inputs[${sampleIdx}]`);
        assertVectorShape(targets[sampleIdx], outputSize, `targets[${sampleIdx}]`);
    }
}

function assertSerializedParams(
    weights: number[][][],
    biases: number[][],
    layerSizes: number[],
): void {
    const layerCount = layerSizes.length - 1;
    if (!Array.isArray(weights) || weights.length !== layerCount) {
        throw new RangeError(`serialized weights must have ${layerCount} layers`);
    }
    if (!Array.isArray(biases) || biases.length !== layerCount) {
        throw new RangeError(`serialized biases must have ${layerCount} layers`);
    }

    for (let l = 0; l < layerCount; l++) {
        const fanIn = layerSizes[l];
        const fanOut = layerSizes[l + 1];
        const layer = weights[l];
        if (!Array.isArray(layer) || layer.length !== fanOut) {
            throw new RangeError(`serialized weights[${l}] must have ${fanOut} rows`);
        }
        for (let n = 0; n < fanOut; n++) {
            const row = layer[n];
            if (!Array.isArray(row) || row.length !== fanIn) {
                throw new RangeError(`serialized weights[${l}][${n}] must have ${fanIn} values`);
            }
            for (let k = 0; k < fanIn; k++) {
                assertFiniteValue(row[k], `serialized weights[${l}][${n}][${k}]`);
            }
        }

        const biasLayer = biases[l];
        if (!Array.isArray(biasLayer) || biasLayer.length !== fanOut) {
            throw new RangeError(`serialized biases[${l}] must have ${fanOut} values`);
        }
        for (let n = 0; n < fanOut; n++) {
            assertFiniteValue(biasLayer[n], `serialized biases[${l}][${n}]`);
        }
    }
}

function clonePackedBuffers(buffers: readonly Float64Array[]): Float64Array[] {
    return buffers.map((buffer) => new Float64Array(buffer));
}

function assertPackedBufferList(
    buffers: readonly Float64Array[] | undefined,
    expectedLengths: readonly number[],
    name: string,
): asserts buffers is readonly Float64Array[] {
    if (!Array.isArray(buffers) || buffers.length !== expectedLengths.length) {
        throw new RangeError(`${name} must have ${expectedLengths.length} layers`);
    }
    for (let l = 0; l < expectedLengths.length; l++) {
        const buffer = buffers[l];
        if (!(buffer instanceof Float64Array) || buffer.length !== expectedLengths[l]) {
            throw new RangeError(`${name}[${l}] must be a Float64Array of length ${expectedLengths[l]}`);
        }
        for (let i = 0; i < buffer.length; i++) {
            assertFiniteValue(buffer[i], `${name}[${l}][${i}]`);
        }
    }
}

function assertOptimizerStateShape(
    checkpoint: NetworkCheckpoint,
    weightLengths: readonly number[],
    biasLengths: readonly number[],
): void {
    if (checkpoint.hasAdamState && !checkpoint.hasMomentumState) {
        throw new RangeError('checkpoint Adam state requires momentum state');
    }

    if (checkpoint.hasMomentumState) {
        assertPackedBufferList(checkpoint.mWeights, weightLengths, 'checkpoint.mWeights');
        assertPackedBufferList(checkpoint.mBiases, biasLengths, 'checkpoint.mBiases');
    } else if (checkpoint.mWeights.length !== 0 || checkpoint.mBiases.length !== 0) {
        throw new RangeError('checkpoint momentum buffers must be empty when momentum state is absent');
    }

    if (checkpoint.hasAdamState) {
        assertPackedBufferList(checkpoint.vWeights, weightLengths, 'checkpoint.vWeights');
        assertPackedBufferList(checkpoint.vBiases, biasLengths, 'checkpoint.vBiases');
    } else if (checkpoint.vWeights.length !== 0 || checkpoint.vBiases.length !== 0) {
        throw new RangeError('checkpoint Adam buffers must be empty when Adam state is absent');
    }
}

function assertTrainingHyperparams(training: TrainingConfig): void {
    assertFiniteInRange(training.learningRate, 'learningRate', 0, Number.POSITIVE_INFINITY);
    assertFiniteInRange(training.batchSize, 'batchSize', 0, Number.POSITIVE_INFINITY);
    assertFiniteInRange(training.regularizationRate, 'regularizationRate', 0, Number.POSITIVE_INFINITY, {
        minInclusive: true,
    });
    assertFiniteInRange(training.momentum, 'momentum', 0, 1, {
        minInclusive: true,
        maxInclusive: true,
    });
    assertFiniteInRange(training.adamBeta1 ?? 0.9, 'adamBeta1', 0, 1, { minInclusive: true });
    assertFiniteInRange(training.adamBeta2 ?? 0.999, 'adamBeta2', 0, 1, { minInclusive: true });
    assertFiniteInRange(training.adamEps ?? 1e-8, 'adamEps', 0, Number.POSITIVE_INFINITY);

    const clip = training.gradientClip;
    if (clip != null) {
        assertFiniteInRange(clip, 'gradientClip', 0, Number.POSITIVE_INFINITY);
    }
    validateLRSchedule(training.lrSchedule);

    switch (training.optimizer) {
        case 'sgd':
            break;
        case 'sgdMomentum':
            break;
        case 'adam':
            break;
        default:
            throw new RangeError('optimizer must be one of sgd, sgdMomentum, or adam');
    }
}

function normalizePositiveIntegerOption(
    value: number | undefined,
    name: string,
    fallback: number,
    max: number,
): number {
    const next = value ?? fallback;
    if (!Number.isInteger(next) || next <= 0 || next > max) {
        throw new RangeError(`${name} must be a positive integer no greater than ${max}`);
    }
    return next;
}

function activationKindForLayer(config: NetworkConfig, layerIndex: number): ActivationType {
    return layerIndex === config.hiddenLayers.length ? config.outputActivation : config.activation;
}

function isSaturatedActivation(value: number, kind: ActivationType, threshold: number): boolean {
    switch (kind) {
        case 'sigmoid':
            return value <= 1 - threshold || value >= threshold;
        case 'tanh':
            return Math.abs(value) >= threshold;
        default:
            return false;
    }
}

interface AbsAggregate {
    sum: number;
    max: number;
    count: number;
}

interface MomentAggregate {
    sum: number;
    sumSquares: number;
    count: number;
}

function createAbsAggregate(): AbsAggregate {
    return { sum: 0, max: 0, count: 0 };
}

function createMomentAggregate(): MomentAggregate {
    return { sum: 0, sumSquares: 0, count: 0 };
}

function addAbsValues(aggregate: AbsAggregate, values: Float64Array): void {
    for (let i = 0; i < values.length; i++) {
        const abs = Math.abs(values[i]);
        aggregate.sum += abs;
        if (abs > aggregate.max) aggregate.max = abs;
    }
    aggregate.count += values.length;
}

function addMomentValues(aggregate: MomentAggregate, values: Float64Array): void {
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        aggregate.sum += value;
        aggregate.sumSquares += value * value;
    }
    aggregate.count += values.length;
}

function finishAbsAggregate(aggregate: AbsAggregate): { mean: number; max: number } {
    return {
        mean: aggregate.count > 0 ? aggregate.sum / aggregate.count : 0,
        max: aggregate.max,
    };
}

function finishMomentAggregate(aggregate: MomentAggregate): { mean: number; std: number } {
    if (aggregate.count === 0) return { mean: 0, std: 0 };
    const mean = aggregate.sum / aggregate.count;
    const variance = Math.max(0, aggregate.sumSquares / aggregate.count - mean * mean);
    return { mean, std: Math.sqrt(variance) };
}

function summarizeScaledAbsBuffers(
    first: Float64Array,
    second: Float64Array,
    scale: number,
): { mean: number; max: number } {
    const aggregate = createAbsAggregate();
    for (let i = 0; i < first.length; i++) {
        const abs = Math.abs(first[i] * scale);
        aggregate.sum += abs;
        if (abs > aggregate.max) aggregate.max = abs;
    }
    for (let i = 0; i < second.length; i++) {
        const abs = Math.abs(second[i] * scale);
        aggregate.sum += abs;
        if (abs > aggregate.max) aggregate.max = abs;
    }
    aggregate.count = first.length + second.length;
    return finishAbsAggregate(aggregate);
}

function summarizeAbsDiffBuffers(
    beforeFirst: Float64Array,
    afterFirst: Float64Array,
    beforeSecond: Float64Array,
    afterSecond: Float64Array,
): { mean: number; max: number } {
    const aggregate = createAbsAggregate();
    for (let i = 0; i < beforeFirst.length; i++) {
        const abs = Math.abs(afterFirst[i] - beforeFirst[i]);
        aggregate.sum += abs;
        if (abs > aggregate.max) aggregate.max = abs;
    }
    for (let i = 0; i < beforeSecond.length; i++) {
        const abs = Math.abs(afterSecond[i] - beforeSecond[i]);
        aggregate.sum += abs;
        if (abs > aggregate.max) aggregate.max = abs;
    }
    aggregate.count = beforeFirst.length + beforeSecond.length;
    return finishAbsAggregate(aggregate);
}

function getBackpropStatus(meanUpdate: number, clipScale: number): BackpropExplanationStatus {
    if (clipScale < 1) return 'clipped';
    if (meanUpdate < 1e-6) return 'tiny';
    if (meanUpdate > 0.1) return 'large';
    return 'healthy';
}

function describeBackpropLayer(status: BackpropExplanationStatus): string {
    switch (status) {
        case 'clipped':
            return 'Global gradient clipping scaled this layer update.';
        case 'tiny':
            return 'The previewed update is tiny; learning may move slowly here.';
        case 'large':
            return 'The previewed update is large; watch for unstable loss movement.';
        case 'healthy':
            return 'The previewed update is in a moderate range.';
    }
}

interface LossLandscapeInternalParameter {
    layerIndex: number;
    neuronIndex: number;
    inputIndex: number;
    flatIndex: number;
}

function normalizeLossLandscapeGridSize(value: number | undefined): number {
    const gridSize = value ?? 7;
    if (!Number.isInteger(gridSize) || gridSize < 3 || gridSize > 7 || gridSize % 2 === 0) {
        throw new RangeError('loss landscape gridSize must be an odd integer between 3 and 7');
    }
    return gridSize;
}

function normalizeLossLandscapeMaxSamples(value: number | undefined): number {
    return normalizePositiveIntegerOption(value, 'loss landscape maxSamples', 64, 64);
}

function normalizeLossLandscapeRadius(value: number | undefined): number {
    const radius = value ?? 0.1;
    assertFiniteInRange(radius, 'loss landscape radius', 0, 1, { maxInclusive: true });
    return radius;
}

function buildLossLandscapeOffsets(gridSize: number, radius: number): number[] {
    const center = Math.floor(gridSize / 2);
    const step = radius / center;
    const offsets = new Array<number>(gridSize);
    for (let i = 0; i < gridSize; i++) {
        offsets[i] = (i - center) * step;
    }
    offsets[center] = 0;
    return offsets;
}

function toLossLandscapeParameter(parameter: LossLandscapeInternalParameter): LossLandscapeParameter {
    return {
        kind: 'weight',
        layerIndex: parameter.layerIndex,
        neuronIndex: parameter.neuronIndex,
        inputIndex: parameter.inputIndex,
        label: `w${parameter.layerIndex}[${parameter.neuronIndex},${parameter.inputIndex}]`,
    };
}

// ── Network ─────────────────────────────────────────────────────────────────

export class Network {
    readonly config: NetworkConfig;

    // Layer dimensions: [inputSize, ...hiddenLayers, outputSize]
    private layerSizes: number[];

    // Packed weight / bias storage. One Float64Array per layer. Weights are
    // stored row-major as [fanOut × fanIn]: `weights[l][n * fanIn + w]` is
    // the weight from previous-layer neuron w into this-layer neuron n.
    private weights: Float64Array[] = [];
    private biases: Float64Array[] = [];

    // Gradient accumulators — shaped identically to weights / biases.
    private weightGrads: Float64Array[] = [];
    private biasGrads: Float64Array[] = [];
    private recentWeightGrads: Float64Array[] = [];

    // Optimizer state. Allocated lazily on the first applyGradients call
    // that actually needs it (SGD stays empty, momentum allocates `m*`,
    // Adam additionally allocates `v*`).
    private mWeights: Float64Array[] = [];
    private mBiases: Float64Array[] = [];
    private vWeights: Float64Array[] = [];
    private vBiases: Float64Array[] = [];
    private hasMomentumState = false;
    private hasAdamState = false;
    private activeOptimizer: TrainingConfig['optimizer'] | null = null;
    private optimizerStep = 0;

    // Forward-pass scratch — preallocated once, reused every call.
    private preActs: Float64Array[] = [];
    private outputs: Float64Array[] = [];
    private deltas: Float64Array[] = [];
    private inputScratch: Float64Array;

    // Monomorphic activation references — resolved once at construction.
    private actFwdHidden: LayerActFn;
    private actFwdOutput: LayerActFn;
    private actDHidden: LayerDActFn;
    private actDOutput: LayerDActFn;

    private currentStep = 0;

    constructor(config: NetworkConfig, seed?: number) {
        const layerSizes = validatedLayerSizes(config);
        this.config = { ...config, hiddenLayers: [...config.hiddenLayers] };
        this.layerSizes = layerSizes;
        const rng = new PRNG(seed ?? this.config.seed);

        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            const w = new Float64Array(fanOut * fanIn);
            initWeightsInto(w, fanIn, fanOut, this.config.weightInit, rng);
            this.weights.push(w);

            const b = new Float64Array(fanOut);
            initBiasesInto(b);
            this.biases.push(b);

            this.weightGrads.push(new Float64Array(fanOut * fanIn));
            this.biasGrads.push(new Float64Array(fanOut));
            this.recentWeightGrads.push(new Float64Array(fanOut * fanIn));

            this.preActs.push(new Float64Array(fanOut));
            this.outputs.push(new Float64Array(fanOut));
            this.deltas.push(new Float64Array(fanOut));
        }

        this.inputScratch = new Float64Array(this.config.inputSize);

        this.actFwdHidden = pickAct(this.config.activation);
        this.actFwdOutput = pickAct(this.config.outputActivation);
        this.actDHidden = pickDAct(this.config.activation);
        this.actDOutput = pickDAct(this.config.outputActivation);
    }

    // ── Forward ──────────────────────────────────────────────────────────────

    /**
     * Zero-copy forward pass. Writes pre-activations / post-activations
     * into the network's scratch buffers and returns a view of the last
     * layer's outputs. DO NOT mutate the returned array — it aliases
     * internal state.
     */
    private forwardInto(input: ArrayLike<number>): Float64Array {
        const numLayers = this.weights.length;
        const scratch = this.inputScratch;
        const inLen = this.layerSizes[0];
        assertVectorShape(input, inLen, 'input');
        for (let i = 0; i < inLen; i++) scratch[i] = input[i];

        let prev: Float64Array = scratch;
        let prevLen = inLen;

        for (let l = 0; l < numLayers; l++) {
            const fanOut = this.layerSizes[l + 1];
            const w = this.weights[l];
            const b = this.biases[l];
            const pre = this.preActs[l];
            const out = this.outputs[l];

            for (let n = 0; n < fanOut; n++) {
                let sum = b[n];
                const rowStart = n * prevLen;
                for (let k = 0; k < prevLen; k++) sum += w[rowStart + k] * prev[k];
                pre[n] = sum;
            }

            if (l === numLayers - 1) this.actFwdOutput(pre, out, fanOut);
            else this.actFwdHidden(pre, out, fanOut);

            prev = out;
            prevLen = fanOut;
        }

        return prev;
    }

    /** Forward pass — returns a fresh number[] for the public API. */
    forward(input: number[]): number[] {
        const out = this.forwardInto(input);
        const copy = new Array<number>(out.length);
        for (let i = 0, n = out.length; i < n; i++) copy[i] = out[i];
        return copy;
    }

    /** Capture a pure-copy trace for one prediction without mutating training state. */
    tracePrediction(
        input: number[],
        target?: number[],
        lossType?: LossType,
        huberDelta?: number,
    ): PredictionTrace {
        if (target != null) {
            assertVectorShape(target, this.config.outputSize, 'target');
        }
        const out = this.forwardInto(input);

        const output = new Array<number>(out.length);
        for (let i = 0, n = out.length; i < n; i++) output[i] = out[i];

        let lossContribution: number | undefined;
        if (target != null && lossType != null) {
            const lossFn = getLoss(lossType, { huberDelta });
            let sum = 0;
            for (let i = 0; i < output.length; i++) {
                sum += lossFn.loss(output[i], target[i]);
            }
            lossContribution = output.length === 0 ? 0 : sum / output.length;
        }

        return {
            input: [...input],
            target: target == null ? undefined : [...target],
            output,
            prediction: output.length === 1 ? output[0] : [...output],
            lossContribution,
            layers: this.outputs.map((layerOutput, layerIndex) => ({
                layerIndex,
                preActivations: Array.from(this.preActs[layerIndex]),
                activations: Array.from(layerOutput),
            })),
        };
    }

    // ── Backward ─────────────────────────────────────────────────────────────

    backward(target: number[], lossType: LossType, huberDelta?: number): void {
        assertVectorShape(target, this.config.outputSize, 'target');
        const lossFn = getLoss(lossType, { huberDelta });
        const dloss = lossFn.dloss;
        const numLayers = this.weights.length;
        const outIdx = numLayers - 1;
        const useSigmoidCrossEntropyDelta = (
            lossType === 'crossEntropy' && this.config.outputActivation === 'sigmoid'
        );

        // Seed output-layer delta = dloss(output, target).
        const outputs = this.outputs[outIdx];
        const outDelta = this.deltas[outIdx];
        const outLen = outputs.length;
        for (let i = 0; i < outLen; i++) {
            outDelta[i] = useSigmoidCrossEntropyDelta
                ? outputs[i] - target[i]
                : dloss(outputs[i], target[i]);
        }
        if (!useSigmoidCrossEntropyDelta) {
            // Multiply by output activation derivative.
            this.actDOutput(outDelta, this.preActs[outIdx], outputs, outLen);
        }

        // Walk backwards accumulating weight/bias grads and propagating delta.
        for (let l = outIdx; l >= 0; l--) {
            const fanOut = this.layerSizes[l + 1];
            const fanIn = this.layerSizes[l];
            const prevOut: Float64Array = l > 0 ? this.outputs[l - 1] : this.inputScratch;
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            const w = this.weights[l];
            const dl = this.deltas[l];

            // ∂L/∂W[l][n, k] += δ[l][n] · prevOut[k]
            // ∂L/∂b[l][n]    += δ[l][n]
            for (let n = 0; n < fanOut; n++) {
                const d = dl[n];
                bg[n] += d;
                const rowStart = n * fanIn;
                for (let k = 0; k < fanIn; k++) wg[rowStart + k] += d * prevOut[k];
            }

            if (l > 0) {
                // Propagate: δ[l-1][k] = Σ_n δ[l][n] · w[l][n, k].
                const prevDelta = this.deltas[l - 1];
                for (let k = 0; k < fanIn; k++) prevDelta[k] = 0;
                for (let n = 0; n < fanOut; n++) {
                    const d = dl[n];
                    const rowStart = n * fanIn;
                    for (let k = 0; k < fanIn; k++) prevDelta[k] += d * w[rowStart + k];
                }
                // Then multiply by the hidden activation derivative in place.
                this.actDHidden(prevDelta, this.preActs[l - 1], this.outputs[l - 1], fanIn);
            }
        }
    }

    // ── Apply gradients ──────────────────────────────────────────────────────

    /**
     * One-pass average → (optional clip) → optimizer update → zero grads.
     * Global-norm gradient clipping fuses into the same sweep: the first
     * pass averages in place and accumulates Σg² simultaneously; if the
     * norm exceeds the clip, we apply a scale multiplier inside the
     * optimizer kernel (still one extra pass over params, same as before
     * but without the extra buffer copies).
     */
    applyGradients(training: TrainingConfig, batchSize: number): void {
        if (!Number.isFinite(batchSize) || batchSize <= 0) {
            throw new RangeError('gradient normalization count must be finite and greater than 0');
        }
        assertTrainingHyperparams(training);
        this.prepareOptimizer(training.optimizer);
        const lr = computeLearningRate(training.learningRate, this.currentStep, training.lrSchedule);
        assertFiniteInRange(lr, 'effective learningRate', 0, Number.POSITIVE_INFINITY);
        const invB = 1 / batchSize;

        // Pass 1: average grads, accumulate squared norm.
        let sqSum = 0;
        for (let l = 0; l < this.weightGrads.length; l++) {
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            for (let i = 0, n = wg.length; i < n; i++) {
                const g = wg[i] * invB;
                wg[i] = g;
                sqSum += g * g;
            }
            for (let i = 0, n = bg.length; i < n; i++) {
                const g = bg[i] * invB;
                bg[i] = g;
                sqSum += g * g;
            }
        }

        // Resolve global-norm clip scale. `scale === 1` is the common case
        // and the optimizer kernels strip the multiply when that's true.
        let scale = 1;
        const clip = training.gradientClip;
        if (clip != null && clip > 0) {
            const norm = Math.sqrt(sqSum);
            if (norm > clip) scale = clip / norm;
        }

        // Preserve the gradients that were actually applied so inspection
        // stats remain useful after the accumulators are zeroed below.
        for (let l = 0; l < this.weightGrads.length; l++) {
            const wg = this.weightGrads[l];
            const recent = this.recentWeightGrads[l];
            if (scale === 1) {
                recent.set(wg);
            } else {
                for (let i = 0, n = wg.length; i < n; i++) recent[i] = wg[i] * scale;
            }
        }

        // Pass 2: optimizer update.
        switch (training.optimizer) {
            case 'sgd':
                this.stepSGD(lr, scale, training.regularization, training.regularizationRate);
                break;
            case 'sgdMomentum':
                this.ensureMomentumState();
                this.stepMomentum(
                    lr, scale,
                    training.regularization, training.regularizationRate,
                    training.momentum ?? 0.9,
                );
                break;
            case 'adam':
                this.ensureAdamState();
                this.stepAdam(
                    lr, scale,
                    training.regularization, training.regularizationRate,
                    training.adamBeta1 ?? 0.9,
                    training.adamBeta2 ?? 0.999,
                    training.adamEps ?? 1e-8,
                );
                break;
        }

        // Zero-fill for the next batch.
        for (let l = 0; l < this.weightGrads.length; l++) {
            this.weightGrads[l].fill(0);
            this.biasGrads[l].fill(0);
        }
        this.currentStep++;
        this.optimizerStep++;
    }

    private prepareOptimizer(optimizer: TrainingConfig['optimizer']): void {
        if (this.activeOptimizer === optimizer) return;
        this.clearOptimizerState();
        this.optimizerStep = 0;
        this.activeOptimizer = optimizer;
    }

    private clearOptimizerState(): void {
        this.hasMomentumState = false;
        this.hasAdamState = false;
        this.mWeights = [];
        this.mBiases = [];
        this.vWeights = [];
        this.vBiases = [];
    }

    private ensureMomentumState(): void {
        if (this.hasMomentumState) return;
        this.mWeights = this.weights.map((w) => new Float64Array(w.length));
        this.mBiases = this.biases.map((b) => new Float64Array(b.length));
        this.hasMomentumState = true;
    }

    private ensureAdamState(): void {
        if (this.hasAdamState) return;
        this.ensureMomentumState();
        this.vWeights = this.weights.map((w) => new Float64Array(w.length));
        this.vBiases = this.biases.map((b) => new Float64Array(b.length));
        this.hasAdamState = true;
    }

    private stepSGD(
        lr: number,
        scale: number,
        reg: 'none' | 'l1' | 'l2',
        regRate: number,
    ): void {
        const scaleUnity = scale === 1;
        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            const bn = b.length;

            if (scaleUnity) {
                for (let i = 0; i < bn; i++) b[i] -= lr * bg[i];
            } else {
                for (let i = 0; i < bn; i++) b[i] -= lr * bg[i] * scale;
            }

            const wn = w.length;
            if (reg === 'l2') {
                if (scaleUnity) {
                    for (let i = 0; i < wn; i++) w[i] -= lr * (wg[i] + regRate * w[i]);
                } else {
                    for (let i = 0; i < wn; i++) w[i] -= lr * (wg[i] * scale + regRate * w[i]);
                }
            } else if (reg === 'l1') {
                if (scaleUnity) {
                    for (let i = 0; i < wn; i++) w[i] -= lr * (wg[i] + regRate * Math.sign(w[i]));
                } else {
                    for (let i = 0; i < wn; i++) w[i] -= lr * (wg[i] * scale + regRate * Math.sign(w[i]));
                }
            } else {
                if (scaleUnity) {
                    for (let i = 0; i < wn; i++) w[i] -= lr * wg[i];
                } else {
                    for (let i = 0; i < wn; i++) w[i] -= lr * wg[i] * scale;
                }
            }
        }
    }

    private stepMomentum(
        lr: number,
        scale: number,
        reg: 'none' | 'l1' | 'l2',
        regRate: number,
        mom: number,
    ): void {
        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            const mw = this.mWeights[l];
            const mb = this.mBiases[l];

            const bn = b.length;
            for (let i = 0; i < bn; i++) {
                const g = bg[i] * scale;
                const m = mom * mb[i] + g;
                mb[i] = m;
                b[i] -= lr * m;
            }

            const wn = w.length;
            for (let i = 0; i < wn; i++) {
                let g = wg[i] * scale;
                if (reg === 'l2') g += regRate * w[i];
                else if (reg === 'l1') g += regRate * Math.sign(w[i]);
                const m = mom * mw[i] + g;
                mw[i] = m;
                w[i] -= lr * m;
            }
        }
    }

    private stepAdam(
        lr: number,
        scale: number,
        reg: 'none' | 'l1' | 'l2',
        regRate: number,
        beta1: number,
        beta2: number,
        eps: number,
    ): void {
        // Bias-corrected step counter starts at 1 on the first call
        // (matches legacy behaviour: `step + 1` inside the old optimizer).
        const step = this.optimizerStep + 1;
        const b1c = 1 - Math.pow(beta1, step);
        const b2c = 1 - Math.pow(beta2, step);
        const oneMinusB1 = 1 - beta1;
        const oneMinusB2 = 1 - beta2;

        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            const mw = this.mWeights[l];
            const mb = this.mBiases[l];
            const vw = this.vWeights[l];
            const vb = this.vBiases[l];

            const bn = b.length;
            for (let i = 0; i < bn; i++) {
                const g = bg[i] * scale;
                const m = beta1 * mb[i] + oneMinusB1 * g;
                mb[i] = m;
                const v = beta2 * vb[i] + oneMinusB2 * g * g;
                vb[i] = v;
                const mHat = m / b1c;
                const vHat = v / b2c;
                b[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }

            const wn = w.length;
            for (let i = 0; i < wn; i++) {
                let g = wg[i] * scale;
                if (reg === 'l2') g += regRate * w[i];
                else if (reg === 'l1') g += regRate * Math.sign(w[i]);
                const m = beta1 * mw[i] + oneMinusB1 * g;
                mw[i] = m;
                const v = beta2 * vw[i] + oneMinusB2 * g * g;
                vw[i] = v;
                const mHat = m / b1c;
                const vHat = v / b2c;
                w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
        }
    }

    // ── Mini-batch train ─────────────────────────────────────────────────────

    trainBatch(inputs: number[][], targets: number[][], training: TrainingConfig): number {
        assertBatchShapes(inputs, targets, this.config.inputSize, this.config.outputSize);
        if (inputs.length === 0) return 0;

        return this.trainValidatedBatch(inputs, targets, 0, inputs.length, training);
    }

    trainBatchIndexed(
        inputs: number[][],
        targets: number[][],
        indices: ArrayLike<number>,
        start: number,
        end: number,
        training: TrainingConfig,
    ): number {
        assertIndexedBatchSelection(
            inputs,
            targets,
            indices,
            start,
            end,
            this.config.inputSize,
            this.config.outputSize,
        );
        if (start === end) return 0;

        return this.trainValidatedBatch(inputs, targets, start, end, training, indices);
    }

    private trainValidatedBatch(
        inputs: number[][],
        targets: number[][],
        start: number,
        end: number,
        training: TrainingConfig,
        indices?: ArrayLike<number>,
    ): number {
        const lossFn = getLoss(training.lossType, { huberDelta: training.huberDelta });
        const lossScalar = lossFn.loss;
        let totalLoss = 0;
        let count = 0;

        for (let s = start; s < end; s++) {
            const sampleIdx = indices == null ? s : indices[s];
            const out = this.forwardInto(inputs[sampleIdx]);
            const tgt = targets[sampleIdx];
            for (let o = 0, n = out.length; o < n; o++) {
                totalLoss += lossScalar(out[o], tgt[o]);
                count++;
            }
            this.backward(tgt, training.lossType, training.huberDelta);
        }

        this.applyGradients(training, (end - start) * this.config.outputSize);
        return count > 0 ? totalLoss / count : 0;
    }

    explainBackpropStep(
        inputs: number[][],
        targets: number[][],
        training: TrainingConfig,
    ): BackpropExplanation {
        assertBatchShapes(inputs, targets, this.config.inputSize, this.config.outputSize);
        if (inputs.length === 0) {
            throw new RangeError('backprop explanation batch must contain at least one sample');
        }

        const dryRun = new Network(this.config);
        dryRun.restoreCheckpoint(this.createCheckpoint());
        return dryRun.explainBackpropStepInPlace(inputs, targets, training);
    }

    private explainBackpropStepInPlace(
        inputs: number[][],
        targets: number[][],
        training: TrainingConfig,
    ): BackpropExplanation {
        assertTrainingHyperparams(training);
        const beforeWeights = clonePackedBuffers(this.weights);
        const beforeBiases = clonePackedBuffers(this.biases);
        const signalAggregates = this.deltas.map(() => createAbsAggregate());
        const activationAggregates = this.outputs.map(() => createMomentAggregate());
        const lossFn = getLoss(training.lossType, { huberDelta: training.huberDelta });
        let totalLoss = 0;
        let count = 0;

        for (let s = 0; s < inputs.length; s++) {
            const out = this.forwardInto(inputs[s]);
            for (let l = 0; l < this.outputs.length; l++) {
                addMomentValues(activationAggregates[l], this.outputs[l]);
            }
            const target = targets[s];
            for (let o = 0; o < out.length; o++) {
                totalLoss += lossFn.loss(out[o], target[o]);
                count++;
            }
            this.backward(target, training.lossType, training.huberDelta);
            for (let l = 0; l < this.deltas.length; l++) {
                addAbsValues(signalAggregates[l], this.deltas[l]);
            }
        }

        const invB = 1 / (inputs.length * this.config.outputSize);
        let sqSum = 0;
        for (let l = 0; l < this.weightGrads.length; l++) {
            const wg = this.weightGrads[l];
            const bg = this.biasGrads[l];
            for (let i = 0; i < wg.length; i++) {
                const g = wg[i] * invB;
                sqSum += g * g;
            }
            for (let i = 0; i < bg.length; i++) {
                const g = bg[i] * invB;
                sqSum += g * g;
            }
        }

        const globalGradientNorm = Math.sqrt(sqSum);
        const clip = training.gradientClip;
        const globalClipScale = clip != null && clip > 0 && globalGradientNorm > clip
            ? clip / globalGradientNorm
            : 1;
        const gradientScale = invB * globalClipScale;
        const gradientSummaries = this.weightGrads.map((weightGrad, layerIndex) => (
            summarizeScaledAbsBuffers(weightGrad, this.biasGrads[layerIndex], gradientScale)
        ));

        this.applyGradients(training, inputs.length * this.config.outputSize);

        const layers: BackpropExplanationLayer[] = [];
        for (let l = 0; l < this.weights.length; l++) {
            const signal = finishAbsAggregate(signalAggregates[l]);
            const gradient = gradientSummaries[l];
            const update = summarizeAbsDiffBuffers(
                beforeWeights[l],
                this.weights[l],
                beforeBiases[l],
                this.biases[l],
            );
            const activations = finishMomentAggregate(activationAggregates[l]);
            const status = getBackpropStatus(update.mean, globalClipScale);
            layers.push({
                layerIndex: l,
                meanAbsErrorSignal: signal.mean,
                maxAbsErrorSignal: signal.max,
                meanAbsGradient: gradient.mean,
                maxAbsGradient: gradient.max,
                meanAbsUpdate: update.mean,
                maxAbsUpdate: update.max,
                meanActivation: activations.mean,
                activationStd: activations.std,
                status,
                note: describeBackpropLayer(status),
            });
        }

        const clipped = globalClipScale < 1;
        const healthyCount = layers.filter((layer) => layer.status === 'healthy').length;
        const summary = clipped
            ? `Backprop preview clipped the global gradient by ${globalClipScale.toFixed(3)}.`
            : `Backprop preview found ${healthyCount} healthy layer update${healthyCount === 1 ? '' : 's'}.`;

        return {
            batchSize: inputs.length,
            loss: count > 0 ? totalLoss / count : 0,
            learningRate: computeLearningRate(training.learningRate, this.currentStep - 1, training.lrSchedule),
            globalGradientNorm,
            globalClipScale,
            clipped,
            layers,
            summary,
        };
    }

    probeLossLandscape(
        inputs: number[][],
        targets: number[][],
        training: TrainingConfig,
        options: LossLandscapeProbeOptions = {},
    ): LossLandscapeProbe {
        assertBatchShapes(inputs, targets, this.config.inputSize, this.config.outputSize);
        assertTrainingHyperparams(training);
        if (inputs.length === 0) {
            throw new RangeError('loss landscape probe batch must contain at least one sample');
        }

        const gridSize = normalizeLossLandscapeGridSize(options.gridSize);
        const maxSamples = normalizeLossLandscapeMaxSamples(options.maxSamples);
        const radius = normalizeLossLandscapeRadius(options.radius);
        const sampleCount = Math.min(inputs.length, maxSamples);
        const [axisA, axisB] = this.getDefaultLossLandscapeAxes();
        const offsets = buildLossLandscapeOffsets(gridSize, radius);
        const checkpoint = this.createCheckpoint();
        const dryRun = new Network(this.config);
        const losses = new Float32Array(gridSize * gridSize);
        let minLoss = Number.POSITIVE_INFINITY;
        let maxLoss = Number.NEGATIVE_INFINITY;
        let bestLoss = Number.POSITIVE_INFINITY;
        let bestRow = 0;
        let bestCol = 0;

        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                dryRun.restoreCheckpoint(checkpoint);
                dryRun.applyLossLandscapeOffset(axisA, offsets[col]);
                dryRun.applyLossLandscapeOffset(axisB, offsets[row]);
                const loss = Math.fround(dryRun.evaluateLossOnly(
                    inputs,
                    targets,
                    sampleCount,
                    training.lossType,
                    training.huberDelta,
                ));
                const index = row * gridSize + col;
                losses[index] = loss;
                if (loss < minLoss) minLoss = loss;
                if (loss > maxLoss) maxLoss = loss;
                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestRow = row;
                    bestCol = col;
                }
            }
        }

        const center = Math.floor(gridSize / 2);
        const centerLoss = losses[center * gridSize + center];
        return {
            gridSize,
            sampleCount,
            radius,
            axisA: {
                parameter: toLossLandscapeParameter(axisA),
                offsets: [...offsets],
            },
            axisB: {
                parameter: toLossLandscapeParameter(axisB),
                offsets: [...offsets],
            },
            losses,
            centerLoss,
            minLoss,
            maxLoss,
            best: {
                row: bestRow,
                col: bestCol,
                loss: bestLoss,
                offsetA: offsets[bestCol],
                offsetB: offsets[bestRow],
            },
            summary: `Loss surface probe found best loss ${bestLoss.toFixed(4)} near ${toLossLandscapeParameter(axisA).label} ${offsets[bestCol].toFixed(3)} and ${toLossLandscapeParameter(axisB).label} ${offsets[bestRow].toFixed(3)}.`,
        };
    }

    private getDefaultLossLandscapeAxes(): [LossLandscapeInternalParameter, LossLandscapeInternalParameter] {
        const axes: LossLandscapeInternalParameter[] = [];
        for (let layerIndex = 0; layerIndex < this.weights.length; layerIndex++) {
            const fanIn = this.layerSizes[layerIndex];
            const layerWeights = this.weights[layerIndex];
            for (let flatIndex = 0; flatIndex < layerWeights.length; flatIndex++) {
                axes.push({
                    layerIndex,
                    neuronIndex: Math.floor(flatIndex / fanIn),
                    inputIndex: flatIndex % fanIn,
                    flatIndex,
                });
                if (axes.length === 2) {
                    return [axes[0], axes[1]];
                }
            }
        }
        throw new RangeError('loss landscape probe requires at least two trainable weights');
    }

    private applyLossLandscapeOffset(parameter: LossLandscapeInternalParameter, offset: number): void {
        this.weights[parameter.layerIndex][parameter.flatIndex] += offset;
    }

    private evaluateLossOnly(
        inputs: number[][],
        targets: number[][],
        sampleCount: number,
        lossType: LossType,
        huberDelta?: number,
    ): number {
        const lossFn = getLoss(lossType, { huberDelta });
        let lossSum = 0;
        let lossCount = 0;
        for (let i = 0; i < sampleCount; i++) {
            const pred = this.forwardInto(inputs[i]);
            const target = targets[i];
            for (let o = 0; o < pred.length; o++) {
                lossSum += lossFn.loss(pred[o], target[o]);
                lossCount++;
            }
        }
        return lossCount > 0 ? lossSum / lossCount : 0;
    }

    // ── Predict helpers ──────────────────────────────────────────────────────

    predict(input: number[]): number[] {
        return this.forward(input);
    }

    predictBatch(inputs: number[][]): number[][] {
        return inputs.map((inp) => this.forward(inp));
    }

    predictGrid(gridInputs: number[][]): Float32Array {
        const len = gridInputs.length;
        const res = new Float32Array(len);
        for (let i = 0; i < len; i++) {
            res[i] = this.forwardInto(gridInputs[i])[0];
        }
        return res;
    }

    predictGridWithNeurons(
        gridInputs: number[][],
    ): { outputGrid: Float32Array; neuronGrids: Float32Array[] } {
        const numLayers = this.weights.length;
        const gridLen = gridInputs.length;

        const neuronGrids: Float32Array[] = [];
        for (let l = 0; l < numLayers; l++) {
            for (let n = 0; n < this.layerSizes[l + 1]; n++) {
                neuronGrids.push(new Float32Array(gridLen));
            }
        }

        const outputGrid = new Float32Array(gridLen);

        for (let i = 0; i < gridLen; i++) {
            const out = this.forwardInto(gridInputs[i]);
            outputGrid[i] = out[0];

            let idx = 0;
            for (let l = 0; l < numLayers; l++) {
                const layerOut = this.outputs[l];
                for (let n = 0, len = layerOut.length; n < len; n++) {
                    neuronGrids[idx][i] = layerOut[n];
                    idx++;
                }
            }
        }

        return { outputGrid, neuronGrids };
    }

    predictGridInto(gridInputs: number[][], target: Float32Array | Float64Array): void {
        for (let i = 0, len = gridInputs.length; i < len; i++) {
            target[i] = this.forwardInto(gridInputs[i])[0];
        }
    }

    predictGridWithNeuronsInto(
        gridInputs: number[][],
        outputTarget: Float32Array | Float64Array,
        neuronTarget: Float32Array | Float64Array,
    ): void {
        const numLayers = this.weights.length;
        const gridLen = gridInputs.length;

        for (let i = 0; i < gridLen; i++) {
            const out = this.forwardInto(gridInputs[i]);
            outputTarget[i] = out[0];

            let neuronIdx = 0;
            for (let l = 0; l < numLayers; l++) {
                const layerOut = this.outputs[l];
                for (let n = 0, len = layerOut.length; n < len; n++) {
                    neuronTarget[neuronIdx * gridLen + i] = layerOut[n];
                    neuronIdx++;
                }
            }
        }
    }

    // ── Flat accessors (preferred; zero-copy where possible) ────────────────

    // The wire-format to the main thread stays Float32Array: the UI only
    // renders 3 decimals of precision, and halving the transferable size
    // matters more than extra mantissa bits for display-only payloads.
    getWeightsFlat(): { buffer: Float32Array; layerSizes: number[] } {
        let total = 0;
        for (const w of this.weights) total += w.length;
        const buffer = new Float32Array(total);
        let off = 0;
        for (const w of this.weights) {
            buffer.set(w, off);
            off += w.length;
        }
        return { buffer, layerSizes: [...this.layerSizes] };
    }

    getBiasesFlat(): Float32Array {
        let total = 0;
        for (const b of this.biases) total += b.length;
        const buffer = new Float32Array(total);
        let off = 0;
        for (const b of this.biases) {
            buffer.set(b, off);
            off += b.length;
        }
        return buffer;
    }

    getTotalNeuronCount(): number {
        let count = 0;
        for (let l = 0; l < this.weights.length; l++) count += this.layerSizes[l + 1];
        return count;
    }

    // ── Evaluate ─────────────────────────────────────────────────────────────

    evaluate(
        inputs: number[][],
        targets: number[][],
        lossType: LossType,
        problemType: 'classification' | 'regression',
        huberDelta?: number,
    ): Metrics {
        const lossFn = getLoss(lossType, { huberDelta });
        const lossScalar = lossFn.loss;
        assertBatchShapes(inputs, targets, this.config.inputSize, this.config.outputSize);
        const N = inputs.length;
        const usePublicForward = this.forward !== Network.prototype.forward;

        let lossSum = 0;
        let lossCount = 0;

        if (problemType === 'classification' && this.config.outputSize === 1) {
            let correct = 0;
            let tp = 0, tn = 0, fp = 0, fn = 0;
            for (let i = 0; i < N; i++) {
                const pred = usePublicForward ? this.forward(inputs[i]) : this.forwardInto(inputs[i]);
                const tgt = targets[i];
                for (let o = 0, outLen = pred.length; o < outLen; o++) {
                    lossSum += lossScalar(pred[o], tgt[o]);
                    lossCount++;
                }
                const predClass = pred[0] >= 0.5 ? 1 : 0;
                const tgtClass = tgt[0];
                if (predClass === tgtClass) correct++;
                if (predClass === 1 && tgtClass === 1) tp++;
                else if (predClass === 0 && tgtClass === 0) tn++;
                else if (predClass === 1 && tgtClass === 0) fp++;
                else if (predClass === 0 && tgtClass === 1) fn++;
            }
            return {
                loss: lossCount > 0 ? lossSum / lossCount : 0,
                accuracy: N > 0 ? correct / N : 0,
                confusionMatrix: { tp, tn, fp, fn },
            };
        }

        if (problemType === 'classification') {
            let correct = 0;
            for (let i = 0; i < N; i++) {
                const pred = usePublicForward ? this.forward(inputs[i]) : this.forwardInto(inputs[i]);
                const tgt = targets[i];
                for (let o = 0, outLen = pred.length; o < outLen; o++) {
                    lossSum += lossScalar(pred[o], tgt[o]);
                    lossCount++;
                }
                let maxIdx = 0;
                for (let o = 1; o < pred.length; o++) {
                    if (pred[o] > pred[maxIdx]) maxIdx = o;
                }
                let tgtIdx = 0;
                for (let o = 1; o < tgt.length; o++) {
                    if (tgt[o] > tgt[tgtIdx]) tgtIdx = o;
                }
                if (maxIdx === tgtIdx) correct++;
            }
            return {
                loss: lossCount > 0 ? lossSum / lossCount : 0,
                accuracy: N > 0 ? correct / N : 0,
            };
        }

        // Regression.
        for (let i = 0; i < N; i++) {
            const pred = usePublicForward ? this.forward(inputs[i]) : this.forwardInto(inputs[i]);
            const tgt = targets[i];
            for (let o = 0, outLen = pred.length; o < outLen; o++) {
                lossSum += lossScalar(pred[o], tgt[o]);
                lossCount++;
            }
        }
        return { loss: lossCount > 0 ? lossSum / lossCount : 0 };
    }

    // ── Snapshot / inspection / (de)serialization ───────────────────────────

    getSnapshot(
        step: number,
        epoch: number,
        trainMetrics: Metrics,
        testMetrics: Metrics,
        outputGrid: ArrayLike<number>,
        gridSize: number,
        options?: { includeParams?: boolean },
    ): NetworkSnapshot {
        const includeParams = options?.includeParams ?? true;
        const historyPoint: HistoryPoint = {
            step,
            trainLoss: trainMetrics.loss,
            testLoss: testMetrics.loss,
            trainAccuracy: trainMetrics.accuracy,
            testAccuracy: testMetrics.accuracy,
        };

        return {
            step,
            epoch,
            weights: includeParams ? this.getWeights() : [],
            biases: includeParams ? this.getBiases() : [],
            trainLoss: trainMetrics.loss,
            testLoss: testMetrics.loss,
            trainMetrics,
            testMetrics,
            outputGrid,
            gridSize,
            historyPoint,
        };
    }

    /** Nested view of the weights — reconstructed from the packed buffers. */
    getWeights(): number[][][] {
        const out: number[][][] = [];
        for (let l = 0; l < this.weights.length; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            const w = this.weights[l];
            const layer: number[][] = [];
            for (let n = 0; n < fanOut; n++) {
                const row = new Array<number>(fanIn);
                const rowStart = n * fanIn;
                for (let k = 0; k < fanIn; k++) row[k] = w[rowStart + k];
                layer.push(row);
            }
            out.push(layer);
        }
        return out;
    }

    /** Nested view of the biases — copied out of the packed buffer. */
    getBiases(): number[][] {
        const out: number[][] = [];
        for (const b of this.biases) {
            const row = new Array<number>(b.length);
            for (let i = 0; i < b.length; i++) row[i] = b[i];
            out.push(row);
        }
        return out;
    }

    getStep(): number {
        return this.currentStep;
    }

    createCheckpoint(): NetworkCheckpoint {
        return {
            config: { ...this.config, hiddenLayers: [...this.config.hiddenLayers] },
            currentStep: this.currentStep,
            optimizerStep: this.optimizerStep,
            activeOptimizer: this.activeOptimizer,
            hasMomentumState: this.hasMomentumState,
            hasAdamState: this.hasAdamState,
            weights: clonePackedBuffers(this.weights),
            biases: clonePackedBuffers(this.biases),
            mWeights: this.hasMomentumState ? clonePackedBuffers(this.mWeights) : [],
            mBiases: this.hasMomentumState ? clonePackedBuffers(this.mBiases) : [],
            vWeights: this.hasAdamState ? clonePackedBuffers(this.vWeights) : [],
            vBiases: this.hasAdamState ? clonePackedBuffers(this.vBiases) : [],
        };
    }

    restoreCheckpoint(checkpoint: NetworkCheckpoint): void {
        if (checkpoint == null || typeof checkpoint !== 'object') {
            throw new RangeError('checkpoint must be an object');
        }
        if (
            !Number.isFinite(checkpoint.currentStep) ||
            !Number.isInteger(checkpoint.currentStep) ||
            checkpoint.currentStep < 0
        ) {
            throw new RangeError('checkpoint currentStep must be a non-negative integer');
        }
        if (
            !Number.isFinite(checkpoint.optimizerStep) ||
            !Number.isInteger(checkpoint.optimizerStep) ||
            checkpoint.optimizerStep < 0
        ) {
            throw new RangeError('checkpoint optimizerStep must be a non-negative integer');
        }
        if (
            checkpoint.activeOptimizer !== null &&
            checkpoint.activeOptimizer !== 'sgd' &&
            checkpoint.activeOptimizer !== 'sgdMomentum' &&
            checkpoint.activeOptimizer !== 'adam'
        ) {
            throw new RangeError('checkpoint activeOptimizer must be null, sgd, sgdMomentum, or adam');
        }

        const checkpointLayerSizes = validatedLayerSizes(checkpoint.config);
        if (
            checkpointLayerSizes.length !== this.layerSizes.length ||
            checkpointLayerSizes.some((size, index) => size !== this.layerSizes[index])
        ) {
            throw new RangeError('checkpoint network shape does not match this network');
        }
        if (
            checkpoint.config.activation !== this.config.activation ||
            checkpoint.config.outputActivation !== this.config.outputActivation
        ) {
            throw new RangeError('checkpoint activation config does not match this network');
        }

        const weightLengths = this.weights.map((buffer) => buffer.length);
        const biasLengths = this.biases.map((buffer) => buffer.length);
        assertPackedBufferList(checkpoint.weights, weightLengths, 'checkpoint.weights');
        assertPackedBufferList(checkpoint.biases, biasLengths, 'checkpoint.biases');
        assertOptimizerStateShape(checkpoint, weightLengths, biasLengths);

        for (let l = 0; l < this.weights.length; l++) {
            this.weights[l].set(checkpoint.weights[l]);
            this.biases[l].set(checkpoint.biases[l]);
            this.weightGrads[l].fill(0);
            this.biasGrads[l].fill(0);
            this.recentWeightGrads[l].fill(0);
        }

        this.hasMomentumState = checkpoint.hasMomentumState;
        this.hasAdamState = checkpoint.hasAdamState;
        this.mWeights = checkpoint.hasMomentumState ? clonePackedBuffers(checkpoint.mWeights) : [];
        this.mBiases = checkpoint.hasMomentumState ? clonePackedBuffers(checkpoint.mBiases) : [];
        this.vWeights = checkpoint.hasAdamState ? clonePackedBuffers(checkpoint.vWeights) : [];
        this.vBiases = checkpoint.hasAdamState ? clonePackedBuffers(checkpoint.vBiases) : [];
        this.activeOptimizer = checkpoint.activeOptimizer;
        this.optimizerStep = checkpoint.optimizerStep;
        this.currentStep = checkpoint.currentStep;
    }

    /** Read a single weight. Useful for tests and gradient checks that need
     *  to inspect individual parameters without reifying the full nested
     *  weight array. `layerIdx` indexes the layer-to-layer matrix (0 for
     *  input→first-hidden, etc.). */
    getWeight(layerIdx: number, neuronIdx: number, prevIdx: number): number {
        const fanIn = this.layerSizes[layerIdx];
        return this.weights[layerIdx][neuronIdx * fanIn + prevIdx];
    }

    /** Write a single weight in place. The change is visible to the next
     *  `forward` / `forwardInto` call. Used by finite-difference gradient
     *  checks that need to perturb one parameter at a time. */
    setWeight(layerIdx: number, neuronIdx: number, prevIdx: number, value: number): void {
        assertFiniteValue(value, 'weight');
        const fanIn = this.layerSizes[layerIdx];
        this.weights[layerIdx][neuronIdx * fanIn + prevIdx] = value;
    }

    getBias(layerIdx: number, neuronIdx: number): number {
        return this.biases[layerIdx][neuronIdx];
    }

    setBias(layerIdx: number, neuronIdx: number, value: number): void {
        assertFiniteValue(value, 'bias');
        this.biases[layerIdx][neuronIdx] = value;
    }

    /** Nested view of the current weight gradients. Used by gradient_check
     *  tests and any external diagnostic that wants to inspect the most
     *  recent backward pass without touching the packed internals. */
    getWeightGrads(): number[][][] {
        const out: number[][][] = [];
        for (let l = 0; l < this.weightGrads.length; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            const wg = this.weightGrads[l];
            const layer: number[][] = [];
            for (let n = 0; n < fanOut; n++) {
                const row = new Array<number>(fanIn);
                const rowStart = n * fanIn;
                for (let k = 0; k < fanIn; k++) row[k] = wg[rowStart + k];
                layer.push(row);
            }
            out.push(layer);
        }
        return out;
    }

    getBiasGrads(): number[][] {
        const out: number[][] = [];
        for (const bg of this.biasGrads) {
            const row = new Array<number>(bg.length);
            for (let i = 0; i < bg.length; i++) row[i] = bg[i];
            out.push(row);
        }
        return out;
    }

    getLayerStats(): LayerStats[] {
        const stats: LayerStats[] = [];
        for (let l = 0; l < this.weights.length; l++) {
            // Mean |w|
            let sumAbsW = 0;
            const w = this.weights[l];
            for (let i = 0; i < w.length; i++) sumAbsW += Math.abs(w[i]);
            const meanAbsWeight = w.length > 0 ? sumAbsW / w.length : 0;

            // Mean |g|
            let sumAbsG = 0;
            let wg = this.weightGrads[l];
            let hasCurrentGradient = false;
            for (let i = 0; i < wg.length; i++) {
                if (wg[i] !== 0) {
                    hasCurrentGradient = true;
                    break;
                }
            }
            if (!hasCurrentGradient) wg = this.recentWeightGrads[l];
            for (let i = 0; i < wg.length; i++) sumAbsG += Math.abs(wg[i]);
            const meanAbsGradient = wg.length > 0 ? sumAbsG / wg.length : 0;

            // Activation stats over the most recent forward pass.
            const acts = this.outputs[l];
            const len = acts.length;
            let sum = 0;
            for (let i = 0; i < len; i++) sum += acts[i];
            const meanActivation = len > 0 ? sum / len : 0;
            let sumSq = 0;
            for (let i = 0; i < len; i++) {
                const d = acts[i] - meanActivation;
                sumSq += d * d;
            }
            const activationStd = len > 0 ? Math.sqrt(sumSq / len) : 0;

            stats.push({ meanActivation, activationStd, meanAbsWeight, meanAbsGradient });
        }
        return stats;
    }

    computeActivationHistograms(
        inputs: readonly ArrayLike<number>[],
        options: ActivationHistogramOptions = {},
    ): ActivationHistogramResult {
        if (!Array.isArray(inputs)) {
            throw new RangeError('inputs must be an array');
        }

        const binCount = normalizePositiveIntegerOption(options.binCount, 'binCount', 12, 64);
        const maxSamples = normalizePositiveIntegerOption(options.maxSamples, 'maxSamples', 128, 4096);
        const zeroThreshold = options.zeroThreshold ?? 0.05;
        const saturationThreshold = options.saturationThreshold ?? 0.95;
        assertFiniteInRange(zeroThreshold, 'zeroThreshold', 0, Number.POSITIVE_INFINITY, {
            minInclusive: true,
        });
        assertFiniteInRange(saturationThreshold, 'saturationThreshold', 0, 1, {
            minInclusive: true,
            maxInclusive: true,
        });

        const layerCount = this.outputs.length;
        const sampleCount = Math.min(inputs.length, maxSamples);
        const mins = new Array<number>(layerCount).fill(Number.POSITIVE_INFINITY);
        const maxes = new Array<number>(layerCount).fill(Number.NEGATIVE_INFINITY);
        const totals = new Array<number>(layerCount).fill(0);
        const nearZeroCounts = new Array<number>(layerCount).fill(0);
        const saturatedCounts = new Array<number>(layerCount).fill(0);

        const sampleIndexFor = (ordinal: number): number => (
            sampleCount === inputs.length
                ? ordinal
                : Math.floor((ordinal * inputs.length) / sampleCount)
        );

        for (let sampleOrdinal = 0; sampleOrdinal < sampleCount; sampleOrdinal++) {
            const sampleIndex = sampleIndexFor(sampleOrdinal);
            const input = inputs[sampleIndex];
            assertVectorShape(input, this.config.inputSize, `inputs[${sampleIndex}]`);
            this.forwardInto(input);

            for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
                const out = this.outputs[layerIndex];
                const activationKind = activationKindForLayer(this.config, layerIndex);
                for (let i = 0; i < out.length; i++) {
                    const value = out[i];
                    if (value < mins[layerIndex]) mins[layerIndex] = value;
                    if (value > maxes[layerIndex]) maxes[layerIndex] = value;
                    if (Math.abs(value) <= zeroThreshold) nearZeroCounts[layerIndex]++;
                    if (isSaturatedActivation(value, activationKind, saturationThreshold)) {
                        saturatedCounts[layerIndex]++;
                    }
                    totals[layerIndex]++;
                }
            }
        }

        const layers: ActivationHistogramLayer[] = [];
        const bins = new Float32Array(layerCount * binCount);
        for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
            const hasValues = totals[layerIndex] > 0;
            const minActivation = hasValues ? mins[layerIndex] : 0;
            const maxActivation = hasValues ? maxes[layerIndex] : 0;
            const range = maxActivation - minActivation;
            const binStart = range === 0 ? minActivation - 0.5 : minActivation;
            const binWidth = (range === 0 ? 1 : range) / binCount;
            layers.push({
                layerIndex,
                binCount,
                binStart,
                binWidth,
                minActivation,
                maxActivation,
                totalCount: totals[layerIndex],
                nearZeroCount: nearZeroCounts[layerIndex],
                saturatedCount: saturatedCounts[layerIndex],
            });
        }

        for (let sampleOrdinal = 0; sampleOrdinal < sampleCount; sampleOrdinal++) {
            const sampleIndex = sampleIndexFor(sampleOrdinal);
            this.forwardInto(inputs[sampleIndex]);

            for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
                const layer = layers[layerIndex];
                const out = this.outputs[layerIndex];
                const offset = layerIndex * binCount;
                for (let i = 0; i < out.length; i++) {
                    const rawBin = Math.floor((out[i] - layer.binStart) / layer.binWidth);
                    const binIndex = Math.min(binCount - 1, Math.max(0, rawBin));
                    bins[offset + binIndex]++;
                }
            }
        }

        return { layers, bins };
    }

    /** Re-initialize weights/biases from the configured seed (or a new one)
     *  and clear all optimizer state + step counter. */
    reset(seed?: number): void {
        const rng = new PRNG(seed ?? this.config.seed);
        for (let l = 0; l < this.weights.length; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            initWeightsInto(this.weights[l], fanIn, fanOut, this.config.weightInit, rng);
            initBiasesInto(this.biases[l]);
            this.weightGrads[l].fill(0);
            this.biasGrads[l].fill(0);
            this.recentWeightGrads[l].fill(0);
        }
        this.clearOptimizerState();
        this.activeOptimizer = null;
        this.optimizerStep = 0;
        this.currentStep = 0;
    }

    serialize(): SerializedNetwork {
        assertSerializedParams(this.getWeights(), this.getBiases(), this.layerSizes);
        return {
            config: { ...this.config },
            weights: this.getWeights(),
            biases: this.getBiases(),
        };
    }

    static deserialize(data: SerializedNetwork): Network {
        if (data == null || typeof data !== 'object') {
            throw new RangeError('serialized network must be an object');
        }
        const net = new Network(data.config);
        assertSerializedParams(data.weights, data.biases, net.layerSizes);
        for (let l = 0; l < net.weights.length; l++) {
            const fanIn = net.layerSizes[l];
            const fanOut = net.layerSizes[l + 1];
            const packed = net.weights[l];
            const layer = data.weights[l];
            for (let n = 0; n < fanOut; n++) {
                const row = layer[n];
                const rowStart = n * fanIn;
                for (let k = 0; k < fanIn; k++) packed[rowStart + k] = row[k];
            }
            const bSrc = data.biases[l];
            const bDst = net.biases[l];
            for (let i = 0; i < bDst.length; i++) bDst[i] = bSrc[i];
        }
        return net;
    }
}

// ── Utility: build the prediction grid inputs ──────────────────────────────
// Unchanged from the legacy implementation — still returns number[][] so
// consumers (including the worker) see the same interface.

export function buildGridInputs(
    gridSize: number,
    activeFeatures: FeatureSpec[],
): number[][] {
    if (!Number.isInteger(gridSize) || gridSize < 2) {
        throw new RangeError('gridSize must be an integer greater than or equal to 2');
    }

    const inputs: number[][] = [];
    for (let gy = 0; gy < gridSize; gy++) {
        for (let gx = 0; gx < gridSize; gx++) {
            const x = -1 + (2 * gx) / (gridSize - 1);
            const y = 1 - (2 * gy) / (gridSize - 1);
            inputs.push(transformPoint(x, y, activeFeatures));
        }
    }
    return inputs;
}
