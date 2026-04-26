import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { NetworkGraphSVG } from './NetworkGraphSVG.tsx';
import {
    getFrameVersions,
    resetFrameBuffer,
    updateFrameBuffer,
} from '../../worker/frameBuffer.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

describe('NetworkGraphSVG', () => {
    const originalResizeObserver = window.ResizeObserver;

    beforeEach(() => {
        resetFrameBuffer();
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            outputGridVersion: 0,
            neuronGridsVersion: 0,
            paramsVersion: 0,
            layerStatsVersion: 0,
            confusionMatrixVersion: 0,
            trainPoints: [],
            testPoints: [],
        });

        class ResizeObserverMock {
            observe() {}
            disconnect() {}
        }
        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: ResizeObserverMock,
        });

        act(() => {
            updateFrameBuffer({
                weights: new Float32Array([0.3, -0.5, 0.7, -0.2, 0.9, -0.4]),
                biases: new Float32Array([0.1, -0.1, 0.05]),
                weightLayout: { layerSizes: [2, 2, 1] },
            });
            useTrainingStore.getState().setFrameVersions(getFrameVersions());
        });
        usePlaygroundStore.setState({
            network: {
                ...usePlaygroundStore.getState().network,
                hiddenLayers: [2],
                activation: 'tanh',
            },
        });
    });

    afterEach(() => {
        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: originalResizeObserver,
        });
    });

    it('shows node details from keyboard focus and clears them on blur', () => {
        render(<NetworkGraphSVG />);

        const nodeTarget = screen.getByRole('button', {
            name: /Hidden 1, Neuron 1.*Bias: 0\.1000.*Activation: tanh/,
        });
        fireEvent.focus(nodeTarget);

        expect(screen.getByText('Hidden 1, Neuron 1')).toBeInTheDocument();
        expect(screen.getByText('Bias: 0.1000')).toBeInTheDocument();
        expect(screen.getByText('Activation: tanh')).toBeInTheDocument();

        fireEvent.blur(nodeTarget);
        expect(screen.queryByText('Hidden 1, Neuron 1')).not.toBeInTheDocument();
    });

    it('shows edge details from keyboard focus and clears them on blur', () => {
        render(<NetworkGraphSVG />);

        const edgeTarget = screen.getByRole('button', {
            name: /Weight: 0\.3000.*Connection: Input 1 to Hidden 1, Neuron 1/,
        });
        fireEvent.focus(edgeTarget);

        expect(screen.getByText('Weight: 0.3000')).toBeInTheDocument();
        expect(screen.getByText('Layer 1, [0→0]')).toBeInTheDocument();

        fireEvent.blur(edgeTarget);
        expect(screen.queryByText('Weight: 0.3000')).not.toBeInTheDocument();
    });
});
