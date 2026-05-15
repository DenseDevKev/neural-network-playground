import { beforeEach, describe, expect, it } from 'vitest';
import { usePlaygroundStore } from './usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    PRESETS,
    decodeUrlState,
} from '@nn-playground/shared';
import type { DatasetType } from '@nn-playground/engine';
import type { Preset } from '@nn-playground/shared';

const PUBLIC_DATASETS: readonly DatasetType[] = [
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
];

function expectScalarRuntimeConfig() {
    const { network, training } = usePlaygroundStore.getState();
    expect(network.outputSize).toBe(1);
    expect(network.outputActivation).not.toBe('softmax');
    expect(training.lossType).not.toBe('categoricalCrossEntropy');
}

function seedHiddenMulticlassState() {
    usePlaygroundStore.setState((state) => ({
        network: {
            ...state.network,
            outputSize: 3,
            outputActivation: 'softmax',
        },
        training: {
            ...state.training,
            lossType: 'categoricalCrossEntropy',
        },
    }));
}

describe('usePlaygroundStore compatibility guards', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            training: { ...DEFAULT_TRAINING },
            data: { ...DEFAULT_DATA },
            features: { ...DEFAULT_FEATURES },
            ui: { showTestData: false, discretizeOutput: false },
        });
        window.location.hash = '';
    });

    it('keeps loss/output activation compatible when loss changes', () => {
        usePlaygroundStore.getState().setLossType('mse');
        expect(usePlaygroundStore.getState().training.lossType).toBe('mse');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('linear');

        usePlaygroundStore.getState().setLossType('crossEntropy');
        expect(usePlaygroundStore.getState().training.lossType).toBe('crossEntropy');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('sigmoid');
    });

    it('keeps public dataset actions on single-output scalar contracts', () => {
        for (const dataset of PUBLIC_DATASETS) {
            seedHiddenMulticlassState();
            usePlaygroundStore.getState().setDataset(dataset);
            expectScalarRuntimeConfig();
        }
    });

    it('keeps built-in presets on single-output scalar contracts', () => {
        for (const preset of PRESETS) {
            seedHiddenMulticlassState();
            usePlaygroundStore.getState().applyPreset(preset);
            expectScalarRuntimeConfig();
        }
    });

    it('clamps future single-output preset contracts before syncing them to the URL', () => {
        const malformedPreset: Preset = {
            id: 'future-preset',
            title: 'Future Preset',
            description: 'Synthetic guard fixture',
            learningGoal: 'Keep public presets scalar until multiclass UI is ready.',
            difficulty: 'advanced',
            config: {
                data: { ...DEFAULT_DATA, dataset: 'xor', problemType: 'classification' },
                network: {
                    ...DEFAULT_NETWORK,
                    inputSize: 2,
                    outputSize: 3,
                    outputActivation: 'sigmoid',
                    seed: DEFAULT_DATA.seed,
                },
                features: { ...DEFAULT_FEATURES },
                training: { ...DEFAULT_TRAINING, lossType: 'crossEntropy' },
            },
        };

        usePlaygroundStore.getState().applyPreset(malformedPreset);

        expectScalarRuntimeConfig();
        expect(window.location.hash).not.toContain('os=3');
    });

    it('does not publish hidden multiclass contracts when syncing the URL', () => {
        seedHiddenMulticlassState();

        usePlaygroundStore.getState().syncToUrl();

        expect(window.location.hash).not.toContain('categoricalCrossEntropy');
        expect(window.location.hash).not.toContain('softmax');
        const decoded = decodeUrlState(window.location.hash.slice(1));
        expect(decoded.network.outputSize).toBe(1);
        expect(decoded.network.outputActivation).not.toBe('softmax');
        expect(decoded.training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it('updates advanced hyperparameters without disturbing unrelated config', () => {
        const store = usePlaygroundStore.getState();

        store.setMomentum(0.65);
        store.setGradientClip(0.5);
        store.setAdamBetas(0.8, 0.98);
        store.setHuberDelta(0.75);
        store.setLRSchedule({ type: 'step', stepSize: 20, gamma: 0.5 });
        store.setWeightInit('he');
        store.setLossType('mse');
        store.setOutputActivation('linear');

        expect(usePlaygroundStore.getState().training).toMatchObject({
            momentum: 0.65,
            gradientClip: 0.5,
            adamBeta1: 0.8,
            adamBeta2: 0.98,
            huberDelta: 0.75,
            lrSchedule: { type: 'step', stepSize: 20, gamma: 0.5 },
        });
        expect(usePlaygroundStore.getState().network.weightInit).toBe('he');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('linear');
        expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
    });

    it('sanitizes learning-rate schedules before they enter store state', () => {
        const store = usePlaygroundStore.getState();

        store.setLRSchedule({ type: 'step', stepSize: 0, gamma: 2 } as any);
        expect(usePlaygroundStore.getState().training.lrSchedule).toEqual({
            type: 'step',
            stepSize: 1,
            gamma: 0.5,
        });

        store.setLRSchedule({ type: 'cosine', totalSteps: 0, minLr: 0.1 } as any);
        expect(usePlaygroundStore.getState().training.lrSchedule).toEqual({
            type: 'cosine',
            totalSteps: 1,
            minLr: DEFAULT_TRAINING.learningRate,
        });

        store.setLRSchedule({ type: 'constant' } as any);
        expect(usePlaygroundStore.getState().training.lrSchedule).toBeUndefined();
    });

    it('reshuffles data by changing only the existing data seed', () => {
        const before = usePlaygroundStore.getState();

        before.reshuffleDataSeed();

        const after = usePlaygroundStore.getState();
        expect(after.data.seed).toBe(before.data.seed + 1);
        expect(after.data.dataset).toBe(before.data.dataset);
        expect(after.data.noise).toBe(before.data.noise);
        expect(after.data.numSamples).toBe(before.data.numSamples);
        expect(after.data.trainTestRatio).toBe(before.data.trainTestRatio);
        expect(after.network.seed).toBe(before.network.seed);
    });
});
