// ── Minimal WebGPU type stubs (AS-4) ────────────────────────────────────────
// The engine package targets `ES2022` only (no DOM lib, no @webgpu/types) so
// it stays runnable in Node test environments without polyfills. These
// hand-rolled aliases cover the surface our predictor actually uses; they
// intentionally lean on `unknown` for opaque types so we never accidentally
// assume more than what the spec guarantees.
//
// At call sites we cast `(navigator as any).gpu` once to obtain the
// adapter — that's the only place a structural assumption escapes this file.

export type GPUBufferUsageFlags = number;
export type GPUMapModeFlags = number;
export type GPUShaderStageFlags = number;

export interface GPUAdapter {
    requestDevice(descriptor?: unknown): Promise<GPUDevice>;
}

export interface GPUDevice {
    readonly queue: GPUQueue;
    createBuffer(descriptor: {
        size: number;
        usage: GPUBufferUsageFlags;
        mappedAtCreation?: boolean;
    }): GPUBuffer;
    createShaderModule(descriptor: { code: string }): GPUShaderModule;
    createBindGroupLayout(descriptor: {
        entries: Array<{
            binding: number;
            visibility: GPUShaderStageFlags;
            buffer?: { type?: 'uniform' | 'storage' | 'read-only-storage' };
        }>;
    }): GPUBindGroupLayout;
    createPipelineLayout(descriptor: {
        bindGroupLayouts: GPUBindGroupLayout[];
    }): GPUPipelineLayout;
    createComputePipeline(descriptor: {
        layout: GPUPipelineLayout;
        compute: { module: GPUShaderModule; entryPoint: string };
    }): GPUComputePipeline;
    createBindGroup(descriptor: {
        layout: GPUBindGroupLayout;
        entries: Array<{
            binding: number;
            resource: { buffer: GPUBuffer; offset?: number; size?: number };
        }>;
    }): GPUBindGroup;
    createCommandEncoder(): GPUCommandEncoder;
    destroy?(): void;
}

export interface GPUBuffer {
    readonly size: number;
    mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

export interface GPUQueue {
    writeBuffer(
        buffer: GPUBuffer,
        bufferOffset: number,
        data: ArrayBufferView | ArrayBuffer,
        dataOffset?: number,
        size?: number,
    ): void;
    submit(commandBuffers: GPUCommandBuffer[]): void;
    onSubmittedWorkDone(): Promise<void>;
}

export interface GPUShaderModule { readonly _brand?: 'shader' }
export interface GPUBindGroupLayout { readonly _brand?: 'bgl' }
export interface GPUBindGroup { readonly _brand?: 'bg' }
export interface GPUPipelineLayout { readonly _brand?: 'pl' }
export interface GPUComputePipeline { readonly _brand?: 'cp' }
export interface GPUCommandBuffer { readonly _brand?: 'cb' }

export interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
}

export interface GPUCommandEncoder {
    beginComputePass(): GPUComputePassEncoder;
    copyBufferToBuffer(
        src: GPUBuffer, srcOffset: number,
        dst: GPUBuffer, dstOffset: number,
        size: number,
    ): void;
    finish(): GPUCommandBuffer;
}

// Spec constants (subset). Mirrors the values in WebGPU's published spec —
// using literals here means we don't need to import the spec's enum object
// at runtime, which would require @webgpu/types.
export const GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
} as const;

export const GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
} as const;

export const GPUShaderStage = {
    COMPUTE: 0x0004,
} as const;

/** Shape of `navigator.gpu` we use. Cast to this on the call site. */
export interface GPUNavigator {
    requestAdapter(options?: { powerPreference?: 'low-power' | 'high-performance' }): Promise<GPUAdapter | null>;
}
