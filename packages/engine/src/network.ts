// ── Core dense feed-forward network ──
// Fully transparent: every weight, bias, activation, and gradient is accessible.

import type {
    NetworkConfig,
    TrainingConfig,
    SerializedNetwork,
    NetworkSnapshot,
    HistoryPoint,
    Metrics,
    LossType,
    LayerStats,
} from './types.js';
import { getActivation } from './activations.js';
import { getLoss } from './losses.js';
import { getOptimizer, createOptimizerState } from './optimizers.js';
import type { OptimizerState, OptimizerHyperparams } from './optimizers.js';
import { computeLearningRate } from './schedules.js';
import { initWeights, initBiases } from './initialization.js';
import { transformPoint } from './features.js';
import type { FeatureSpec } from './features.js';
import { PRNG } from './prng.js';

export class Network {
    readonly config: NetworkConfig;

    // Layer dimensions: [inputSize, ...hiddenLayers, outputSize]
    private layerSizes: number[];

    // weights[l][n][w] — layer l, neuron n, weight w (from prev layer neuron w)
    private weights: number[][][];
    // biases[l][n]
    private biases: number[][];

    // Forward pass cache
    private layerOutputs: number[][] = [];  // post-activation
    private layerPreActs: number[][] = [];  // pre-activation
    private input: number[] = [];

    // Gradient accumulators (accumulated over batch, then averaged)
    private weightGrads: number[][][] = [];
    private biasGrads: number[][] = [];

    // Optimizer state
    private optState: OptimizerState = [];
    private currentStep = 0;

    constructor(config: NetworkConfig, seed?: number) {
        this.config = { ...config };
        const rng = new PRNG(seed ?? config.seed);

        this.layerSizes = [
            config.inputSize,
            ...config.hiddenLayers,
            config.outputSize,
        ];

        // Initialize weights and biases
        this.weights = [];
        this.biases = [];
        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            this.weights.push(initWeights(fanIn, fanOut, config.weightInit, rng));
            this.biases.push(initBiases(fanOut));
        }

        this.initGradients();
    }

    private initGradients(): void {
        this.weightGrads = [];
        this.biasGrads = [];
        for (let l = 0; l < this.weights.length; l++) {
            this.weightGrads.push(
                this.weights[l].map((row) => new Array(row.length).fill(0)),
            );
            this.biasGrads.push(new Array(this.biases[l].length).fill(0));
        }
    }

    /** Forward pass — returns output layer values. */
    forward(input: number[]): number[] {
        this.input = input;
        this.layerOutputs = [];
        this.layerPreActs = [];

        let current = input;
        const numLayers = this.weights.length;

        for (let l = 0; l < numLayers; l++) {
            const isOutput = l === numLayers - 1;
            const activation = getActivation(
                isOutput ? this.config.outputActivation : this.config.activation,
            );

            const preActs: number[] = [];
            const outputs: number[] = [];

            const layerWeights = this.weights[l];
            const layerBiases = this.biases[l];
            for (let n = 0; n < layerWeights.length; n++) {
                let sum = layerBiases[n];
                const neuronWeights = layerWeights[n];
                for (let w = 0, len = neuronWeights.length; w < len; w++) {
                    sum += neuronWeights[w] * current[w];
                }
                preActs.push(sum);
                outputs.push(activation.f(sum));
            }

            this.layerPreActs.push(preActs);
            this.layerOutputs.push(outputs);
            current = outputs;
        }

        return current;
    }

    /** Backward pass — accumulates gradients. */
    backward(target: number[], lossType: LossType, huberDelta?: number): void {
        const lossFn = getLoss(lossType, { huberDelta });
        const numLayers = this.weights.length;

        // Output layer deltas
        const outputAct = getActivation(this.config.outputActivation);
        const outputLayerOutputs = this.layerOutputs[numLayers - 1];
        const outputLayerPreActs = this.layerPreActs[numLayers - 1];
        let deltas: number[] = new Array(outputLayerOutputs.length);
        for (let i = 0, len = outputLayerOutputs.length; i < len; i++) {
            const o = outputLayerOutputs[i];
            const dLoss = lossFn.dloss(o, target[i]);
            const dAct = outputAct.df(outputLayerPreActs[i], o);
            deltas[i] = dLoss * dAct;
        }

        // Backpropagate through layers
        for (let l = numLayers - 1; l >= 0; l--) {
            const prevOutput = l > 0 ? this.layerOutputs[l - 1] : this.input;
            const layerWeights = this.weights[l];
            const layerWeightGrads = this.weightGrads[l];
            const layerBiasGrads = this.biasGrads[l];

            for (let n = 0; n < deltas.length; n++) {
                const delta = deltas[n];
                layerBiasGrads[n] += delta;

                const neuronWeights = layerWeights[n];
                const neuronWeightGrads = layerWeightGrads[n];
                for (let w = 0; w < neuronWeights.length; w++) {
                    neuronWeightGrads[w] += delta * prevOutput[w];
                }
            }

            if (l > 0) {
                const act = getActivation(this.config.activation);
                const newDeltas: number[] = new Array(this.layerSizes[l]).fill(0);
                for (let n = 0; n < deltas.length; n++) {
                    const delta = deltas[n];
                    const neuronWeights = layerWeights[n];
                    for (let w = 0; w < neuronWeights.length; w++) {
                        newDeltas[w] += delta * neuronWeights[w];
                    }
                }
                deltas = newDeltas.map((d, i) =>
                    d * act.df(this.layerPreActs[l - 1][i], this.layerOutputs[l - 1][i]),
                );
            }
        }
    }

    /**
     * Apply accumulated gradients using the optimizer.
     *
     * Pipeline:
     *   1. Average gradients in-place by batch size.
     *   2. Apply global-norm gradient clipping (scales the whole gradient
     *      vector by `clip / ‖g‖₂` when the norm exceeds `clip`). This
     *      preserves gradient direction, unlike element-wise clipping.
     *   3. Apply regularization + optimizer update per parameter.
     */
    applyGradients(training: TrainingConfig, batchSize: number): void {
        const opt = getOptimizer(training.optimizer);

        // Lazily create optimizer state (includes one bias slot per neuron).
        if (this.optState.length === 0 && opt.stateSize > 0) {
            const sizes = this.layerSizes.slice(1);
            const prevSizes = this.layerSizes.slice(0, -1);
            this.optState = createOptimizerState(sizes, prevSizes, opt.stateSize);
        }

        const lr = computeLearningRate(training.learningRate, this.currentStep, training.lrSchedule);
        const { gradientClip, regularization, regularizationRate } = training;
        const optStateSize = opt.stateSize;
        const hyper: OptimizerHyperparams = {
            momentum: training.momentum,
            beta1: training.adamBeta1,
            beta2: training.adamBeta2,
            eps: training.adamEps,
        };
        const scratch: number[] = optStateSize > 0 ? new Array(optStateSize).fill(0) : [];

        // ── Pass 1: average gradients in place.
        const invBatch = 1 / batchSize;
        for (let l = 0; l < this.weightGrads.length; l++) {
            const lBiasGrads = this.biasGrads[l];
            const lWeightGrads = this.weightGrads[l];
            for (let n = 0; n < lWeightGrads.length; n++) {
                lBiasGrads[n] *= invBatch;
                const nGrads = lWeightGrads[n];
                for (let w = 0; w < nGrads.length; w++) {
                    nGrads[w] *= invBatch;
                }
            }
        }

        // ── Pass 2: global-norm clipping.
        if (gradientClip != null && gradientClip > 0) {
            let sqSum = 0;
            for (let l = 0; l < this.weightGrads.length; l++) {
                const lBiasGrads = this.biasGrads[l];
                const lWeightGrads = this.weightGrads[l];
                for (let n = 0; n < lWeightGrads.length; n++) {
                    const bg = lBiasGrads[n];
                    sqSum += bg * bg;
                    const nGrads = lWeightGrads[n];
                    for (let w = 0; w < nGrads.length; w++) {
                        const g = nGrads[w];
                        sqSum += g * g;
                    }
                }
            }
            const norm = Math.sqrt(sqSum);
            if (norm > gradientClip) {
                const scale = gradientClip / norm;
                for (let l = 0; l < this.weightGrads.length; l++) {
                    const lBiasGrads = this.biasGrads[l];
                    const lWeightGrads = this.weightGrads[l];
                    for (let n = 0; n < lWeightGrads.length; n++) {
                        lBiasGrads[n] *= scale;
                        const nGrads = lWeightGrads[n];
                        for (let w = 0; w < nGrads.length; w++) {
                            nGrads[w] *= scale;
                        }
                    }
                }
            }
        }

        // ── Pass 3: regularization + optimizer update.
        for (let l = 0; l < this.weights.length; l++) {
            const lWeights = this.weights[l];
            const lBiases = this.biases[l];
            const lBiasGrads = this.biasGrads[l];
            const lWeightGrads = this.weightGrads[l];
            const lOptState = this.optState[l];
            const fanIn = this.layerSizes[l];
            const biasStateIdx = fanIn; // bias stored at the "+1" slot

            for (let n = 0; n < lWeights.length; n++) {
                // Bias — routed through the optimizer (no regularization on biases).
                const bg = lBiasGrads[n];
                if (optStateSize > 0) {
                    for (let s = 0; s < optStateSize; s++) {
                        scratch[s] = lOptState[s][n][biasStateIdx];
                    }
                    lBiases[n] = opt.update(lBiases[n], bg, lr, scratch, this.currentStep, hyper);
                    for (let s = 0; s < optStateSize; s++) {
                        lOptState[s][n][biasStateIdx] = scratch[s];
                    }
                } else {
                    lBiases[n] = opt.update(lBiases[n], bg, lr, scratch, this.currentStep, hyper);
                }

                const nWeights = lWeights[n];
                const nWeightGrads = lWeightGrads[n];

                for (let w = 0; w < nWeights.length; w++) {
                    let g = nWeightGrads[w];

                    // Weight regularization (biases exempted).
                    if (regularization === 'l1') {
                        g += regularizationRate * Math.sign(nWeights[w]);
                    } else if (regularization === 'l2') {
                        g += regularizationRate * nWeights[w];
                    }

                    if (optStateSize > 0) {
                        for (let s = 0; s < optStateSize; s++) {
                            scratch[s] = lOptState[s][n][w];
                        }
                        nWeights[w] = opt.update(nWeights[w], g, lr, scratch, this.currentStep, hyper);
                        for (let s = 0; s < optStateSize; s++) {
                            lOptState[s][n][w] = scratch[s];
                        }
                    } else {
                        nWeights[w] = opt.update(nWeights[w], g, lr, scratch, this.currentStep, hyper);
                    }
                }
            }
        }

        // Reset gradient accumulators.
        this.zeroGradients();
        this.currentStep++;
    }

    private zeroGradients(): void {
        for (let l = 0; l < this.weightGrads.length; l++) {
            for (let n = 0; n < this.weightGrads[l].length; n++) {
                this.biasGrads[l][n] = 0;
                for (let w = 0; w < this.weightGrads[l][n].length; w++) {
                    this.weightGrads[l][n][w] = 0;
                }
            }
        }
    }

    /**
     * Train on one mini-batch. Returns the mean loss, averaged across all
     * samples and output units.
     */
    trainBatch(
        inputs: number[][],
        targets: number[][],
        training: TrainingConfig,
    ): number {
        const lossFn = getLoss(training.lossType, { huberDelta: training.huberDelta });
        let totalLoss = 0;
        let count = 0;

        for (let i = 0; i < inputs.length; i++) {
            const output = this.forward(inputs[i]);
            this.backward(targets[i], training.lossType, training.huberDelta);
            const tgt = targets[i];
            for (let o = 0, outLen = output.length; o < outLen; o++) {
                totalLoss += lossFn.loss(output[o], tgt[o]);
                count++;
            }
        }

        this.applyGradients(training, inputs.length);
        return count > 0 ? totalLoss / count : 0;
    }

    /** Predict a single input (no gradient tracking). */
    predict(input: number[]): number[] {
        return this.forward(input);
    }

    /** Predict a batch. */
    predictBatch(inputs: number[][]): number[][] {
        return inputs.map((inp) => this.forward(inp));
    }

    /**
     * Compute predictions over a regular grid for heatmap rendering.
     * Returns predictions as a flat array.
     */
    predictGrid(
        gridInputs: number[][],
    ): Float32Array {
        const len = gridInputs.length;
        const res = new Float32Array(len);
        for (let i = 0; i < len; i++) {
            res[i] = this.forward(gridInputs[i])[0];
        }
        return res;
    }

    /**
     * Predict grid and collect per-neuron activations for mini heatmaps.
     * Returns { outputGrid, neuronGrids } where neuronGrids is a flat array
     * indexed as [layerIndex][neuronIndex] = Float32Array(gridSize * gridSize).
     */
    predictGridWithNeurons(
        gridInputs: number[][],
    ): { outputGrid: Float32Array; neuronGrids: Float32Array[] } {
        const numLayers = this.weights.length;
        const gridLen = gridInputs.length;

        // Initialize neuronGrids: one Float32Array per neuron
        const neuronGrids: Float32Array[] = [];
        for (let l = 0; l < numLayers; l++) {
            for (let n = 0; n < this.weights[l].length; n++) {
                neuronGrids.push(new Float32Array(gridLen));
            }
        }

        const outputGrid = new Float32Array(gridLen);

        for (let i = 0; i < gridLen; i++) {
            const output = this.forward(gridInputs[i]);
            outputGrid[i] = output[0];

            // Collect per-neuron activations
            let idx = 0;
            for (let l = 0; l < numLayers; l++) {
                for (let n = 0; n < this.layerOutputs[l].length; n++) {
                    neuronGrids[idx][i] = this.layerOutputs[l][n];
                    idx++;
                }
            }
        }

        return { outputGrid, neuronGrids };
    }

    /**
     * Predict grid and write results directly into a pre-allocated Float32Array.
     * This avoids creating intermediate number[] arrays.
     */
    predictGridInto(
        gridInputs: number[][],
        target: Float32Array,
    ): void {
        for (let i = 0; i < gridInputs.length; i++) {
            target[i] = this.forward(gridInputs[i])[0];
        }
    }

    /**
     * Predict grid + per-neuron activations, writing directly into Float32Arrays.
     * `outputTarget` receives the output predictions.
     * `neuronTarget` receives all neuron activations, flattened layer-by-layer.
     */
    predictGridWithNeuronsInto(
        gridInputs: number[][],
        outputTarget: Float32Array,
        neuronTarget: Float32Array,
    ): void {
        const numLayers = this.weights.length;
        // Total neurons across all layers
        let totalNeurons = 0;
        for (let l = 0; l < numLayers; l++) {
            totalNeurons += this.weights[l].length;
        }

        for (let i = 0; i < gridInputs.length; i++) {
            const output = this.forward(gridInputs[i]);
            outputTarget[i] = output[0];

            // Write per-neuron activations at offset: neuronIndex * gridLength + gridPosition
            let neuronIdx = 0;
            for (let l = 0; l < numLayers; l++) {
                for (let n = 0; n < this.layerOutputs[l].length; n++) {
                    neuronTarget[neuronIdx * gridInputs.length + i] = this.layerOutputs[l][n];
                    neuronIdx++;
                }
            }
        }
    }

    /**
     * Pack all weights into a flat Float32Array (row-major, layer by layer).
     * Returns { buffer, layerSizes } where layerSizes = [inputSize, h1, h2, ..., outputSize].
     */
    getWeightsFlat(): { buffer: Float32Array; layerSizes: number[] } {
        let totalWeights = 0;
        for (const layer of this.weights) {
            for (const neuron of layer) {
                totalWeights += neuron.length;
            }
        }
        const buffer = new Float32Array(totalWeights);
        let offset = 0;
        for (const layer of this.weights) {
            for (const neuron of layer) {
                buffer.set(neuron, offset);
                offset += neuron.length;
            }
        }
        return { buffer, layerSizes: [...this.layerSizes] };
    }

    /**
     * Pack all biases into a flat Float32Array (layer by layer).
     */
    getBiasesFlat(): Float32Array {
        let totalBiases = 0;
        for (const layer of this.biases) {
            totalBiases += layer.length;
        }
        const buffer = new Float32Array(totalBiases);
        let offset = 0;
        for (const layer of this.biases) {
            buffer.set(layer, offset);
            offset += layer.length;
        }
        return buffer;
    }

    /**
     * Count total neurons across all hidden + output layers.
     */
    getTotalNeuronCount(): number {
        let count = 0;
        for (const layer of this.weights) {
            count += layer.length;
        }
        return count;
    }


    /** Evaluate loss+accuracy on a dataset. */
    evaluate(
        inputs: number[][],
        targets: number[][],
        lossType: LossType,
        problemType: 'classification' | 'regression',
        huberDelta?: number,
    ): Metrics {
        const lossFn = getLoss(lossType, { huberDelta });
        const predsAll: number[][] = inputs.map((inp) => this.forward(inp).slice());

        // Loss is averaged across (samples × output units).
        let lossSum = 0;
        let lossCount = 0;
        for (let i = 0; i < predsAll.length; i++) {
            const pred = predsAll[i];
            const tgt = targets[i];
            for (let o = 0, outLen = pred.length; o < outLen; o++) {
                lossSum += lossFn.loss(pred[o], tgt[o]);
                lossCount++;
            }
        }
        const loss = lossCount > 0 ? lossSum / lossCount : 0;

        // Single-output classification: accuracy + confusion matrix.
        if (problemType === 'classification' && this.config.outputSize === 1) {
            let correct = 0;
            let tp = 0;
            let tn = 0;
            let fp = 0;
            let fn = 0;

            for (let i = 0; i < predsAll.length; i++) {
                const predClass = predsAll[i][0] >= 0.5 ? 1 : 0;
                const tgtClass = targets[i][0];

                if (predClass === tgtClass) correct++;

                if (predClass === 1 && tgtClass === 1) tp++;
                else if (predClass === 0 && tgtClass === 0) tn++;
                else if (predClass === 1 && tgtClass === 0) fp++;
                else if (predClass === 0 && tgtClass === 1) fn++;
            }
            return {
                loss,
                accuracy: predsAll.length > 0 ? correct / predsAll.length : 0,
                confusionMatrix: { tp, tn, fp, fn },
            };
        }

        // Multi-output classification: argmax accuracy (if targets are class indices).
        if (problemType === 'classification') {
            let correct = 0;
            for (let i = 0; i < predsAll.length; i++) {
                const pred = predsAll[i];
                let maxIdx = 0;
                for (let o = 1; o < pred.length; o++) {
                    if (pred[o] > pred[maxIdx]) maxIdx = o;
                }
                const tgt = targets[i];
                let tgtIdx = 0;
                for (let o = 1; o < tgt.length; o++) {
                    if (tgt[o] > tgt[tgtIdx]) tgtIdx = o;
                }
                if (maxIdx === tgtIdx) correct++;
            }
            return {
                loss,
                accuracy: predsAll.length > 0 ? correct / predsAll.length : 0,
            };
        }

        return { loss };
    }

    /**
     * Generate a snapshot for the UI to render.
     *
     * By default, weights and biases are deep-copied into the snapshot so that
     * callers (code export, serialisation, unit tests) can mutate freely.
     * Pass `includeParams: false` to skip the copies on hot paths; the snapshot
     * will have empty `weights`/`biases` arrays and the caller is expected to
     * obtain flat buffers through `getWeightsFlat()` / `getBiasesFlat()`.
     */
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
            weights: includeParams
                ? this.weights.map((l) => l.map((n) => [...n]))
                : [],
            biases: includeParams ? this.biases.map((l) => [...l]) : [],
            trainLoss: trainMetrics.loss,
            testLoss: testMetrics.loss,
            trainMetrics,
            testMetrics,
            outputGrid,
            gridSize,
            historyPoint,
        };
    }

    /** Get raw weights (for inspection). */
    getWeights(): number[][][] {
        return this.weights;
    }

    /** Get raw biases. */
    getBiases(): number[][] {
        return this.biases;
    }

    /** Get the current step count. */
    getStep(): number {
        return this.currentStep;
    }

    /** Compute per-layer statistics for the inspection panel. */
    getLayerStats(): LayerStats[] {
        const stats: LayerStats[] = [];
        for (let l = 0; l < this.weights.length; l++) {
            // Mean absolute weight
            let sumAbsW = 0, countW = 0;
            for (const neuron of this.weights[l]) {
                for (const w of neuron) {
                    sumAbsW += Math.abs(w);
                    countW++;
                }
            }
            const meanAbsWeight = countW > 0 ? sumAbsW / countW : 0;

            // Mean absolute gradient
            let sumAbsG = 0, countG = 0;
            if (this.weightGrads[l]) {
                for (const neuron of this.weightGrads[l]) {
                    for (const g of neuron) {
                        sumAbsG += Math.abs(g);
                        countG++;
                    }
                }
            }
            const meanAbsGradient = countG > 0 ? sumAbsG / countG : 0;

            // Activation stats (from most recent forward pass)
            let meanActivation = 0, activationStd = 0;
            if (this.layerOutputs[l]) {
                const acts = this.layerOutputs[l];
                const n = acts.length;
                meanActivation = acts.reduce((a, b) => a + b, 0) / n;
                activationStd = Math.sqrt(
                    acts.reduce((a, b) => a + (b - meanActivation) ** 2, 0) / n,
                );
            }

            stats.push({ meanActivation, activationStd, meanAbsWeight, meanAbsGradient });
        }
        return stats;
    }

    /** Reset the network with a new seed, clearing optimizer state. */
    reset(seed?: number): void {
        const rng = new PRNG(seed ?? this.config.seed);
        for (let l = 0; l < this.weights.length; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            this.weights[l] = initWeights(fanIn, fanOut, this.config.weightInit, rng);
            this.biases[l] = initBiases(fanOut);
        }
        this.initGradients();
        this.optState = [];
        this.currentStep = 0;
    }

    /** Serialize the network for save/restore. */
    serialize(): SerializedNetwork {
        return {
            config: { ...this.config },
            weights: this.weights.map((l) => l.map((n) => [...n])),
            biases: this.biases.map((l) => [...l]),
        };
    }

    /** Restore a network from serialized data. */
    static deserialize(data: SerializedNetwork): Network {
        const net = new Network(data.config);
        net.weights = data.weights.map((l) => l.map((n) => [...n]));
        net.biases = data.biases.map((l) => [...l]);
        return net;
    }
}

// ── Utility: build the prediction grid inputs ──

/**
 * Generate a regular grid of (x, y) coords in [-1, 1]² and transform
 * them through the active features. Returns feature-transformed inputs.
 */
export function buildGridInputs(
    gridSize: number,
    activeFeatures: FeatureSpec[],
): number[][] {
    const inputs: number[][] = [];
    for (let gy = 0; gy < gridSize; gy++) {
        for (let gx = 0; gx < gridSize; gx++) {
            const x = -1 + (2 * gx) / (gridSize - 1);
            const y = 1 - (2 * gy) / (gridSize - 1); // flip y for canvas coords
            inputs.push(transformPoint(x, y, activeFeatures));
        }
    }
    return inputs;
}
