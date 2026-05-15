import { describe, expect, it } from 'vitest';
import { DEFAULT_TRAINING, PRESETS } from '../index.js';
import type { DatasetType } from '@nn-playground/engine';

const PUBLIC_DATASETS = new Set<DatasetType>([
    'circle',
    'xor',
    'gauss',
    'spiral',
    'moons',
    'checkerboard',
    'rings',
    'heart',
    'reg-plane',
    'reg-gauss',
]);

describe('preset registry multiclass guardrails', () => {
    it('keeps built-in presets on public scalar dataset contracts', () => {
        for (const preset of PRESETS) {
            const dataset = preset.config.data?.dataset;
            const network = preset.config.network;
            const training = preset.config.training ?? DEFAULT_TRAINING;

            expect(dataset, preset.id).toBeDefined();
            expect(PUBLIC_DATASETS.has(dataset!), preset.id).toBe(true);
            expect(network?.outputSize, preset.id).toBe(1);
            expect(network?.outputActivation, preset.id).not.toBe('softmax');
            expect(training.lossType, preset.id).not.toBe('categoricalCrossEntropy');
        }
    });
});
