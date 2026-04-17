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
            ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
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
});
