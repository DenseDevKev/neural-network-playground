import { describe, it, expect } from 'vitest';
import { Network, buildGridInputs } from '../../network.js';
import { detectWebGPU, resetWebGPUDetectionCache } from '../detect.js';
import {
    WebGPUGridPredictor,
    exceedsGpuShape,
    flattenGridInputs,
} from '../predictGridGPU.js';
import { defaultFeatureFlags, getActiveFeatures } from '../../features.js';
import type { NetworkConfig } from '../../types.js';

describe('WebGPUGridPredictor parity', () => {
    it('returns null when the WebGPU API is unavailable', async () => {
        const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

        try {
            resetWebGPUDetectionCache();
            Object.defineProperty(globalThis, 'navigator', {
                configurable: true,
                value: {},
            });

            await expect(detectWebGPU()).resolves.toBeNull();
        } finally {
            if (originalNavigator) {
                Object.defineProperty(globalThis, 'navigator', originalNavigator);
            } else {
                delete (globalThis as { navigator?: unknown }).navigator;
            }
            resetWebGPUDetectionCache();
        }
    });

    it('matches Network.predictGridInto element-wise to within 1e-4 when GPU is available', async () => {
        const device = await detectWebGPU();
        const config: NetworkConfig = {
            inputSize: 2,
            hiddenLayers: [8, 8],
            outputSize: 1,
            activation: 'tanh',
            outputActivation: 'sigmoid',
            weightInit: 'xavier',
            seed: 42,
        };
        if (exceedsGpuShape([config.inputSize, ...config.hiddenLayers, config.outputSize])) {
            throw new Error('Test config exceeds GPU shape caps; bug in test or constants');
        }

        const cpu = new Network(config, config.seed);
        const features = getActiveFeatures(defaultFeatureFlags());
        // Use only the first two features so the input vector matches inputSize.
        const limited = features.slice(0, 2);
        const gridSize = 16;
        const gridInputs = buildGridInputs(gridSize, limited);
        const gridLen = gridInputs.length;

        const cpuOut = new Float32Array(gridLen);
        cpu.predictGridInto(gridInputs, cpuOut);
        expect(cpuOut).toHaveLength(gridSize * gridSize);
        for (let i = 0; i < cpuOut.length; i++) {
            expect(Number.isFinite(cpuOut[i])).toBe(true);
        }

        if (!device) {
            expect(device).toBeNull();
            return;
        }

        const layerSizes = [config.inputSize, ...config.hiddenLayers, config.outputSize];
        const gpu = new WebGPUGridPredictor({
            device,
            layerSizes,
            gridLen,
            hiddenActivation: config.activation,
            outputActivation: config.outputActivation,
        });
        const flat = cpu.getWeightsFlat();
        gpu.updateWeights(flat.buffer, cpu.getBiasesFlat());
        gpu.setGridInputs(flattenGridInputs(gridInputs));

        const gpuOut = new Float32Array(gridLen);
        await gpu.predictGridInto(gpuOut);
        gpu.dispose();

        for (let i = 0; i < gridLen; i++) {
            expect(gpuOut[i]).toBeCloseTo(cpuOut[i], 4);
        }
    });

    it('keeps softplus finite and compares with GPU when available', async () => {
        const device = await detectWebGPU();
        const config: NetworkConfig = {
            inputSize: 1,
            hiddenLayers: [],
            outputSize: 1,
            activation: 'tanh',
            outputActivation: 'softplus',
            weightInit: 'zeros',
            seed: 42,
        };
        const cpu = new Network(config, config.seed);
        cpu.setWeight(0, 0, 0, 1000);
        const gridInputs = [[1], [-1], [0]];
        const gridLen = gridInputs.length;
        const cpuOut = new Float32Array(gridLen);
        cpu.predictGridInto(gridInputs, cpuOut);

        expect(cpuOut[0]).toBeCloseTo(1000, 4);
        expect(cpuOut[1]).toBeCloseTo(0, 4);
        expect(cpuOut[2]).toBeCloseTo(Math.log(2), 4);
        for (let i = 0; i < cpuOut.length; i++) {
            expect(Number.isFinite(cpuOut[i])).toBe(true);
        }

        if (!device) {
            expect(device).toBeNull();
            return;
        }

        const layerSizes = [config.inputSize, ...config.hiddenLayers, config.outputSize];
        const gpu = new WebGPUGridPredictor({
            device,
            layerSizes,
            gridLen,
            hiddenActivation: config.activation,
            outputActivation: config.outputActivation,
        });
        const flat = cpu.getWeightsFlat();
        gpu.updateWeights(flat.buffer, cpu.getBiasesFlat());
        gpu.setGridInputs(flattenGridInputs(gridInputs));

        const gpuOut = new Float32Array(gridLen);
        await gpu.predictGridInto(gpuOut);
        gpu.dispose();

        for (let i = 0; i < gridLen; i++) {
            expect(gpuOut[i]).toBeCloseTo(cpuOut[i], 4);
        }
    });

    it('rejects shapes that exceed the compile-time width cap', () => {
        // 65 > MAX_GPU_WIDTH (64) — must reject before we ever touch a GPU.
        expect(exceedsGpuShape([2, 65, 1])).toBe(true);
        // Reasonable shape — must accept.
        expect(exceedsGpuShape([2, 16, 16, 1])).toBe(false);
    });
});
