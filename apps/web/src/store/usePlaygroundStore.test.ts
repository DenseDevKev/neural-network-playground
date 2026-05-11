import { beforeEach, describe, expect, it } from 'vitest';
import { usePlaygroundStore } from './usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('usePlaygroundStore compatibility guards', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            training: { ...DEFAULT_TRAINING },
            data: { ...DEFAULT_DATA },
            features: { ...DEFAULT_FEATURES },
            ui: { showTestData: false, discretizeOutput: false },
        });
    });

    it('keeps loss/output activation compatible when loss changes', () => {
        usePlaygroundStore.getState().setLossType('mse');
        expect(usePlaygroundStore.getState().training.lossType).toBe('mse');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('linear');

        usePlaygroundStore.getState().setLossType('crossEntropy');
        expect(usePlaygroundStore.getState().training.lossType).toBe('crossEntropy');
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('sigmoid');
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
});
