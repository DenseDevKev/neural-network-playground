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
    'three-class-clusters',
]);

describe('preset registry multiclass guardrails', () => {
    it('exposes exactly one approved multiclass preset', () => {
        const multiclassPresets = PRESETS.filter((preset) => preset.config.data?.dataset === 'three-class-clusters');

        expect(multiclassPresets.map((preset) => preset.id)).toEqual(['three-class-clusters']);
        expect(multiclassPresets[0].config).toMatchObject({
            data: { dataset: 'three-class-clusters', problemType: 'classification' },
            network: { outputSize: 3, outputActivation: 'softmax' },
            training: { lossType: 'categoricalCrossEntropy' },
        });
    });

    it('keeps built-in presets on scalar contracts unless they are the approved multiclass tuple', () => {
        for (const preset of PRESETS) {
            const dataset = preset.config.data?.dataset;
            const network = preset.config.network;
            const training = preset.config.training ?? DEFAULT_TRAINING;

            expect(dataset, preset.id).toBeDefined();
            expect(PUBLIC_DATASETS.has(dataset!), preset.id).toBe(true);
            if (preset.id === 'three-class-clusters') {
                expect(network?.outputSize, preset.id).toBe(3);
                expect(network?.outputActivation, preset.id).toBe('softmax');
                expect(training.lossType, preset.id).toBe('categoricalCrossEntropy');
                continue;
            }
            expect(network?.outputSize, preset.id).toBe(1);
            expect(network?.outputActivation, preset.id).not.toBe('softmax');
            expect(training.lossType, preset.id).not.toBe('categoricalCrossEntropy');
        }
    });
});
