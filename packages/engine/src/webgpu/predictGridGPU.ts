// ── WebGPU decision-boundary grid predictor (AS-4) ──────────────────────────
// Computes the same forward pass as Network.predictGridInto /
// Network.predictGridWithNeuronsInto, but on the GPU. Used by the worker
// for the heatmap visualization grid only — training stays on CPU because
// Adam parity across f32/f64 is a research project, and the per-step launch
// overhead of GPU dispatch overwhelms the cost of a small-batch step
// anyway.
//
// Shape constraints:
//   - Per-layer width capped at MAX_GPU_WIDTH (compile-time constant in the
//     WGSL shader). Any layer exceeding this falls back to CPU at the
//     `WebGPUGridPredictor` factory call.
//   - Up to MAX_GPU_LAYERS layers (input + hidden + output). Same fallback.
//
// Memory model:
//   - weights / biases / inputs / outputs / neuron activations are all f32
//     storage buffers. Updated in place per frame via queue.writeBuffer.
//   - Layer offsets and per-layer sizes live in a small uniform buffer,
//     re-uploaded only on shape change.
//
// Synchronization:
//   - predict*() returns a Promise that resolves after the readback copy
//     completes. Callers (the worker) await it before publishing the
//     snapshot via the seqlock.

import type { ActivationType } from '../types.js';
import {
    GPUBufferUsage,
    GPUMapMode,
    GPUShaderStage,
    type GPUBindGroup,
    type GPUBindGroupLayout,
    type GPUBuffer,
    type GPUComputePipeline,
    type GPUDevice,
} from './types.js';

/** Per-layer width cap (must match the const in the shader). */
export const MAX_GPU_WIDTH = 64;
/** Max layers including input + output (must match the const in the shader). */
export const MAX_GPU_LAYERS = 8;

const ACTIVATION_ID: Record<ActivationType, number> = {
    relu: 0,
    tanh: 1,
    sigmoid: 2,
    linear: 3,
    leakyRelu: 4,
    elu: 5,
    swish: 6,
    softplus: 7,
};

// ── Shader source ──────────────────────────────────────────────────────────
// Embedded as a string template so we don't depend on any bundler magic
// (?raw imports). Shader body deliberately mirrors the CPU implementation
// in network.ts so parity tests have a chance.
//
// Layout, in order of binding:
//   0 (storage, read)        weights         — packed Float32Array, all layers
//   1 (storage, read)        biases          — packed Float32Array, all layers
//   2 (storage, read)        gridInputs      — flattened: gridLen * inputSize
//   3 (storage, read_write)  outputGrid      — gridLen scalars (last-layer neuron 0)
//   4 (storage, read_write)  neuronGrids     — totalNeurons * gridLen scalars
//                                              (column-major: stride=gridLen)
//   5 (uniform)              params          — scalars (gridLen, inputSize, …)
//   6 (storage, read)        layerSizes      — u32[MAX_GPU_LAYERS]
//   7 (storage, read)        weightOffsets   — u32[MAX_GPU_LAYERS-1]
//   8 (storage, read)        biasOffsets     — u32[MAX_GPU_LAYERS-1]
//   9 (storage, read)        neuronOffsets   — u32[MAX_GPU_LAYERS-1]
//                                              cumulative sum of fanOuts

const SHADER_SOURCE = /* wgsl */`
const MAX_WIDTH: u32 = ${MAX_GPU_WIDTH}u;
const MAX_LAYERS: u32 = ${MAX_GPU_LAYERS}u;

@group(0) @binding(0) var<storage, read>       weights      : array<f32>;
@group(0) @binding(1) var<storage, read>       biases       : array<f32>;
@group(0) @binding(2) var<storage, read>       gridInputs   : array<f32>;
@group(0) @binding(3) var<storage, read_write> outputGrid   : array<f32>;
@group(0) @binding(4) var<storage, read_write> neuronGrids  : array<f32>;

struct Meta {
    gridLen        : u32,
    inputSize      : u32,
    numLayers      : u32,    // count of weight matrices = layers - 1
    actHidden      : u32,
    actOutput      : u32,
    writeNeurons   : u32,    // 0 = skip neuronGrids, 1 = write
    _pad0          : u32,
    _pad1          : u32,
};
@group(0) @binding(5) var<uniform>            params       : Meta;
@group(0) @binding(6) var<storage, read>       layerSizes    : array<u32>;
@group(0) @binding(7) var<storage, read>       weightOffsets : array<u32>;
@group(0) @binding(8) var<storage, read>       biasOffsets   : array<u32>;
@group(0) @binding(9) var<storage, read>       neuronOffsets : array<u32>;

fn apply_activation(x: f32, id: u32) -> f32 {
    // Branches in switch statements compile to a select on most backends,
    // so this stays branch-free at runtime for any single thread.
    switch (id) {
        case 0u: { return max(0.0, x); }                          // relu
        case 1u: { return tanh(x); }                              // tanh
        case 2u: { return 1.0 / (1.0 + exp(-x)); }                // sigmoid
        case 3u: { return x; }                                    // linear
        case 4u: { return select(0.01 * x, x, x > 0.0); }         // leakyRelu
        case 5u: { return select(exp(x) - 1.0, x, x >= 0.0); }    // elu
        case 6u: { return x / (1.0 + exp(-x)); }                  // swish
        case 7u: {                                                // softplus
            if (x > 0.0) {
                return x + log(1.0 + exp(-x));
            }
            return log(1.0 + exp(x));
        }
        default: { return x; }
    }
}

@compute @workgroup_size(64)
fn forward_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel : u32 = gid.x;
    if (pixel >= params.gridLen) {
        return;
    }

    // Two ping-pong scratch arrays. WGSL forbids dynamic-size private
    // arrays, but fixed-size arrays of MAX_WIDTH cover every supported
    // network. The runtime cap matches the playground's per-layer maximum.
    var bufA : array<f32, MAX_WIDTH>;
    var bufB : array<f32, MAX_WIDTH>;

    // Load the input vector for this pixel.
    let inputBase : u32 = pixel * params.inputSize;
    let inputSize = params.inputSize;
    for (var i : u32 = 0u; i < inputSize; i = i + 1u) {
        bufA[i] = gridInputs[inputBase + i];
    }
    var prevLen : u32 = inputSize;
    var srcIsA : bool = true;

    let numLayers = params.numLayers;
    for (var l : u32 = 0u; l < numLayers; l = l + 1u) {
        let fanIn  : u32 = layerSizes[l];
        let fanOut : u32 = layerSizes[l + 1u];
        let wBase  : u32 = weightOffsets[l];
        let bBase  : u32 = biasOffsets[l];
        let isLast : bool = (l + 1u == numLayers);
        let actId  : u32 = select(params.actHidden, params.actOutput, isLast);

        for (var n : u32 = 0u; n < fanOut; n = n + 1u) {
            var sum : f32 = biases[bBase + n];
            let rowStart : u32 = wBase + n * fanIn;
            for (var k : u32 = 0u; k < fanIn; k = k + 1u) {
                let prevVal : f32 = select(bufB[k], bufA[k], srcIsA);
                sum = sum + weights[rowStart + k] * prevVal;
            }
            let out : f32 = apply_activation(sum, actId);
            if (srcIsA) { bufB[n] = out; } else { bufA[n] = out; }
        }

        // Optionally write this layer's activations to the per-neuron
        // grid so the visualization can show the inner heatmaps.
        if (params.writeNeurons == 1u) {
            let nBase : u32 = neuronOffsets[l];
            for (var n : u32 = 0u; n < fanOut; n = n + 1u) {
                let val : f32 = select(bufA[n], bufB[n], srcIsA);
                neuronGrids[(nBase + n) * params.gridLen + pixel] = val;
            }
        }

        srcIsA = !srcIsA;
        prevLen = fanOut;
    }

    // Last layer's output[0] is the decision-boundary value. After the
    // final swap, the produced layer lives in the buffer we just wrote to:
    //   if srcIsA is now true, that means we just wrote into bufA on the
    //   last iteration → output is in bufA. (And vice versa.)
    let result : f32 = select(bufB[0], bufA[0], srcIsA);
    outputGrid[pixel] = result;
}
`;

// Width cap helper used by the factory and parity tests.
export function exceedsGpuShape(layerSizes: number[]): boolean {
    if (layerSizes.length > MAX_GPU_LAYERS) return true;
    for (const w of layerSizes) {
        if (w > MAX_GPU_WIDTH) return true;
    }
    return false;
}

// Cumulative-sum helpers for the offset arrays. Keep these near the shader
// so the indexing scheme is obvious from one read.
function cumulativeWeightOffsets(layerSizes: number[]): Uint32Array {
    const arr = new Uint32Array(MAX_GPU_LAYERS);
    let off = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        arr[l] = off;
        off += layerSizes[l] * layerSizes[l + 1];
    }
    return arr;
}

function cumulativeBiasOffsets(layerSizes: number[]): Uint32Array {
    const arr = new Uint32Array(MAX_GPU_LAYERS);
    let off = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        arr[l] = off;
        off += layerSizes[l + 1];
    }
    return arr;
}

function cumulativeNeuronOffsets(layerSizes: number[]): Uint32Array {
    const arr = new Uint32Array(MAX_GPU_LAYERS);
    let off = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        arr[l] = off;
        off += layerSizes[l + 1];
    }
    return arr;
}

function paddedLayerSizes(layerSizes: number[]): Uint32Array {
    const arr = new Uint32Array(MAX_GPU_LAYERS);
    for (let i = 0; i < layerSizes.length; i++) arr[i] = layerSizes[i];
    return arr;
}

function totalWeights(layerSizes: number[]): number {
    let total = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        total += layerSizes[l] * layerSizes[l + 1];
    }
    return Math.max(1, total);
}

function totalBiases(layerSizes: number[]): number {
    let total = 0;
    for (let l = 0; l < layerSizes.length - 1; l++) {
        total += layerSizes[l + 1];
    }
    return Math.max(1, total);
}

function totalHiddenAndOutput(layerSizes: number[]): number {
    let total = 0;
    for (let l = 1; l < layerSizes.length; l++) total += layerSizes[l];
    return Math.max(1, total);
}

/**
 * Owns one WebGPU compute pipeline + its persistent storage buffers for a
 * specific network shape. Reuse across frames; rebuild (dispose + new
 * instance) on shape change.
 */
export class WebGPUGridPredictor {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private bindGroup: GPUBindGroup;

    private weightsBuf: GPUBuffer;
    private biasesBuf: GPUBuffer;
    private inputsBuf: GPUBuffer;
    private outputBuf: GPUBuffer;
    private outputReadBuf: GPUBuffer;
    private neuronBuf: GPUBuffer;
    private neuronReadBuf: GPUBuffer;
    private metaBuf: GPUBuffer;
    private layerSizesBuf: GPUBuffer;
    private weightOffsetsBuf: GPUBuffer;
    private biasOffsetsBuf: GPUBuffer;
    private neuronOffsetsBuf: GPUBuffer;

    /** Guards to prevent double-mapping readback buffers. */
    private outputMapped = false;
    private neuronMapped = false;

    private layerSizes: number[];
    private actHiddenId: number;
    private actOutputId: number;
    private gridLen: number;
    private inputSize: number;
    private neuronTotal: number;
    private weightTotal: number;
    private biasTotal: number;

    constructor(args: {
        device: GPUDevice;
        layerSizes: number[];
        gridLen: number;
        hiddenActivation: ActivationType;
        outputActivation: ActivationType;
    }) {
        if (exceedsGpuShape(args.layerSizes)) {
            throw new Error(
                `WebGPUGridPredictor: network shape ${JSON.stringify(args.layerSizes)} ` +
                `exceeds compile-time caps (max ${MAX_GPU_WIDTH} per layer, ${MAX_GPU_LAYERS} layers).`,
            );
        }

        this.device = args.device;
        this.layerSizes = [...args.layerSizes];
        this.actHiddenId = ACTIVATION_ID[args.hiddenActivation];
        this.actOutputId = ACTIVATION_ID[args.outputActivation];
        this.gridLen = args.gridLen;
        this.inputSize = args.layerSizes[0];
        this.weightTotal = totalWeights(args.layerSizes);
        this.biasTotal = totalBiases(args.layerSizes);
        this.neuronTotal = totalHiddenAndOutput(args.layerSizes);

        // ── Buffers ──────────────────────────────────────────────────────
        const dev = this.device;
        const F32 = 4;
        const U32 = 4;

        this.weightsBuf = dev.createBuffer({
            size: this.weightTotal * F32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.biasesBuf = dev.createBuffer({
            size: this.biasTotal * F32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.inputsBuf = dev.createBuffer({
            size: this.gridLen * this.inputSize * F32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.outputBuf = dev.createBuffer({
            size: this.gridLen * F32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.outputReadBuf = dev.createBuffer({
            size: this.gridLen * F32,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.neuronBuf = dev.createBuffer({
            size: this.neuronTotal * this.gridLen * F32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.neuronReadBuf = dev.createBuffer({
            size: this.neuronTotal * this.gridLen * F32,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.metaBuf = dev.createBuffer({
            // Meta struct: 8 u32 fields (with padding) = 32 bytes.
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.layerSizesBuf = dev.createBuffer({
            size: MAX_GPU_LAYERS * U32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.weightOffsetsBuf = dev.createBuffer({
            size: MAX_GPU_LAYERS * U32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.biasOffsetsBuf = dev.createBuffer({
            size: MAX_GPU_LAYERS * U32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.neuronOffsetsBuf = dev.createBuffer({
            size: MAX_GPU_LAYERS * U32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Upload offset arrays (constant for this shape).
        dev.queue.writeBuffer(this.layerSizesBuf, 0, paddedLayerSizes(this.layerSizes));
        dev.queue.writeBuffer(this.weightOffsetsBuf, 0, cumulativeWeightOffsets(this.layerSizes));
        dev.queue.writeBuffer(this.biasOffsetsBuf, 0, cumulativeBiasOffsets(this.layerSizes));
        dev.queue.writeBuffer(this.neuronOffsetsBuf, 0, cumulativeNeuronOffsets(this.layerSizes));

        // ── Pipeline ─────────────────────────────────────────────────────
        const layout = this.buildBindGroupLayout();
        const pipelineLayout = dev.createPipelineLayout({ bindGroupLayouts: [layout] });
        const module = dev.createShaderModule({ code: SHADER_SOURCE });
        this.pipeline = dev.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: 'forward_grid' },
        });
        this.bindGroup = this.buildBindGroup(layout);
    }

    private buildBindGroupLayout(): GPUBindGroupLayout {
        const COMPUTE = GPUShaderStage.COMPUTE;
        const ROS = { type: 'read-only-storage' as const };
        const RWS = { type: 'storage' as const };
        const UNI = { type: 'uniform' as const };
        return this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: COMPUTE, buffer: ROS },
                { binding: 1, visibility: COMPUTE, buffer: ROS },
                { binding: 2, visibility: COMPUTE, buffer: ROS },
                { binding: 3, visibility: COMPUTE, buffer: RWS },
                { binding: 4, visibility: COMPUTE, buffer: RWS },
                { binding: 5, visibility: COMPUTE, buffer: UNI },
                { binding: 6, visibility: COMPUTE, buffer: ROS },
                { binding: 7, visibility: COMPUTE, buffer: ROS },
                { binding: 8, visibility: COMPUTE, buffer: ROS },
                { binding: 9, visibility: COMPUTE, buffer: ROS },
            ],
        });
    }

    private buildBindGroup(layout: GPUBindGroupLayout): GPUBindGroup {
        return this.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.weightsBuf } },
                { binding: 1, resource: { buffer: this.biasesBuf } },
                { binding: 2, resource: { buffer: this.inputsBuf } },
                { binding: 3, resource: { buffer: this.outputBuf } },
                { binding: 4, resource: { buffer: this.neuronBuf } },
                { binding: 5, resource: { buffer: this.metaBuf } },
                { binding: 6, resource: { buffer: this.layerSizesBuf } },
                { binding: 7, resource: { buffer: this.weightOffsetsBuf } },
                { binding: 8, resource: { buffer: this.biasOffsetsBuf } },
                { binding: 9, resource: { buffer: this.neuronOffsetsBuf } },
            ],
        });
    }

    /** Push freshly-trained weights/biases to the GPU. Cheap (no realloc). */
    updateWeights(flatWeights: Float32Array, flatBiases: Float32Array): void {
        if (flatWeights.length < this.weightTotal) {
            throw new Error('updateWeights: flatWeights too short for this shape');
        }
        if (flatBiases.length < this.biasTotal) {
            throw new Error('updateWeights: flatBiases too short for this shape');
        }
        // Copy at most weightTotal/biasTotal elements — the engine's flat
        // accessors emit exactly the right size, but be defensive.
        this.device.queue.writeBuffer(
            this.weightsBuf, 0,
            flatWeights, 0, this.weightTotal,
        );
        this.device.queue.writeBuffer(
            this.biasesBuf, 0,
            flatBiases, 0, this.biasTotal,
        );
    }

    /** Push the (constant per network shape) flattened grid inputs. */
    setGridInputs(flatGridInputs: Float32Array): void {
        if (flatGridInputs.length !== this.gridLen * this.inputSize) {
            throw new Error('setGridInputs: length mismatch');
        }
        this.device.queue.writeBuffer(this.inputsBuf, 0, flatGridInputs);
    }

    /** Run the compute pass; copy output to a mappable buffer; return on read. */
    async predictGridInto(outputDst: Float32Array): Promise<void> {
        await this.dispatch(false);
        await this.readbackOutput(outputDst);
    }

    /** Same as above but also fills the per-neuron grids. */
    async predictGridWithNeuronsInto(
        outputDst: Float32Array,
        neuronDst: Float32Array,
    ): Promise<void> {
        await this.dispatch(true);
        await Promise.all([
            this.readbackOutput(outputDst),
            this.readbackNeurons(neuronDst),
        ]);
    }

    private writeMeta(writeNeurons: boolean): void {
        const meta = new Uint32Array([
            this.gridLen,
            this.inputSize,
            this.layerSizes.length - 1, // numLayers (weight matrices)
            this.actHiddenId,
            this.actOutputId,
            writeNeurons ? 1 : 0,
            0, 0, // padding
        ]);
        this.device.queue.writeBuffer(this.metaBuf, 0, meta);
    }

    private async dispatch(writeNeurons: boolean): Promise<void> {
        this.writeMeta(writeNeurons);
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        // workgroup_size = 64 in the shader; one thread per pixel.
        const groupCount = Math.ceil(this.gridLen / 64);
        pass.dispatchWorkgroups(groupCount);
        pass.end();
        // Copy output(s) into mappable buffers as part of the same submission
        // so we only synchronise once with the device.
        enc.copyBufferToBuffer(this.outputBuf, 0, this.outputReadBuf, 0, this.gridLen * 4);
        if (writeNeurons) {
            enc.copyBufferToBuffer(
                this.neuronBuf, 0,
                this.neuronReadBuf, 0,
                this.neuronTotal * this.gridLen * 4,
            );
        }
        const cb = enc.finish();
        this.device.queue.submit([cb]);
        await this.device.queue.onSubmittedWorkDone();
    }

    private async readbackOutput(dst: Float32Array): Promise<void> {
        // Guard: skip if already mapped (concurrent prediction).
        if (this.outputMapped) return;
        this.outputMapped = true;
        try {
            await this.outputReadBuf.mapAsync(GPUMapMode.READ);
            const range = this.outputReadBuf.getMappedRange(0, this.gridLen * 4);
            const view = new Float32Array(range);
            // Copy out before unmap; the underlying ArrayBuffer is invalidated
            // when the buffer is unmapped.
            if (dst.length >= this.gridLen) {
                dst.set(view.subarray(0, this.gridLen));
            } else {
                dst.set(view.subarray(0, dst.length));
            }
            this.outputReadBuf.unmap();
        } finally {
            this.outputMapped = false;
        }
    }

    private async readbackNeurons(dst: Float32Array): Promise<void> {
        // Guard: skip if already mapped (concurrent prediction).
        if (this.neuronMapped) return;
        this.neuronMapped = true;
        try {
            const total = this.neuronTotal * this.gridLen;
            await this.neuronReadBuf.mapAsync(GPUMapMode.READ);
            const range = this.neuronReadBuf.getMappedRange(0, total * 4);
            const view = new Float32Array(range);
            if (dst.length >= total) {
                dst.set(view.subarray(0, total));
            } else {
                dst.set(view.subarray(0, dst.length));
            }
            this.neuronReadBuf.unmap();
        } finally {
            this.neuronMapped = false;
        }
    }

    /** Free GPU resources. After dispose() the predictor must not be used. */
    dispose(): void {
        this.weightsBuf.destroy();
        this.biasesBuf.destroy();
        this.inputsBuf.destroy();
        this.outputBuf.destroy();
        this.outputReadBuf.destroy();
        this.neuronBuf.destroy();
        this.neuronReadBuf.destroy();
        this.metaBuf.destroy();
        this.layerSizesBuf.destroy();
        this.weightOffsetsBuf.destroy();
        this.biasOffsetsBuf.destroy();
        this.neuronOffsetsBuf.destroy();
    }
}

/** Convenience: flatten gridInputs (number[][]) into a Float32Array. */
export function flattenGridInputs(gridInputs: number[][]): Float32Array {
    if (gridInputs.length === 0) return new Float32Array(0);
    const inputSize = gridInputs[0].length;
    const flat = new Float32Array(gridInputs.length * inputSize);
    for (let i = 0; i < gridInputs.length; i++) {
        flat.set(gridInputs[i], i * inputSize);
    }
    return flat;
}
