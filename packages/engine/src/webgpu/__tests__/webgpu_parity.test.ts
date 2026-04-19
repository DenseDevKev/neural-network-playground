import { describe, it, expect } from 'vitest';
import { Network, buildGridInputs } from '../../network.js';
import { detectWebGPU } from '../detect.js';
import {
    WebGPUGridPredictor,
    exceedsGpuShape,
    flattenGridInputs,
} from '../predictGridGPU.js';
import { defaultFeatureFlags, getActiveFeatures } from '../../features.js';
import type { NetworkConfig } from '../../types.js';

describe('WebGPUGridPredictor parity', () => {
    it('matches Network.predictGridInto element-wise to within 1e-4 (skips without GPU)', async () => {
        const device = await detectWebGPU();
        if (!device) {
            // Expected on CI / Node — skip without failing.
            return;
        }

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
