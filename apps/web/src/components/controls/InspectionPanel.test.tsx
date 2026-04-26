import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { InspectionPanel } from './InspectionPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

const workerApi = vi.hoisted(() => ({
    getPredictionTrace: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: () => workerApi,
}));

describe('InspectionPanel demand', () => {
    beforeEach(() => {
        workerApi.getPredictionTrace.mockReset();
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
            demand: { ...DEFAULT_DEMAND, needLayerStats: false },
        });
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
        });
    });

    afterEach(() => {
        usePlaygroundStore.setState({
            demand: { ...DEFAULT_DEMAND, needLayerStats: false },
        });
    });

    it('requests layer stats only while inspection is mounted', () => {
        const { unmount } = render(<InspectionPanel />);

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(true);

        unmount();

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(false);
    });

    it('requests an on-demand prediction trace for the selected training sample', async () => {
        workerApi.getPredictionTrace.mockResolvedValue({
            runId: 1,
            step: 12,
            sample: { source: 'train', index: 0, x: 0.25, y: -0.5, label: 1 },
            trace: {
                input: [0.25, -0.5],
                target: [1],
                output: [0.82],
                prediction: 0.82,
                lossContribution: 0.19,
                layers: [
                    { layerIndex: 0, preActivations: [0.1, -0.2], activations: [0.1, -0.197] },
                    { layerIndex: 1, preActivations: [1.5], activations: [0.82] },
                ],
            },
        });
        useTrainingStore.setState({
            trainPoints: [{ x: 0.25, y: -0.5, label: 1 }],
            testPoints: [],
        });

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /trace prediction/i }));

        await waitFor(() => {
            expect(workerApi.getPredictionTrace).toHaveBeenCalledWith({ source: 'train', index: 0 });
        });
        expect(await screen.findByText('0.8200')).toBeInTheDocument();
        expect(screen.getByText('Output')).toBeInTheDocument();
        expect(screen.getByText(/Layer 1/i)).toBeInTheDocument();
        expect(screen.getByText('loss')).toBeInTheDocument();
        expect(screen.getByText('0.1900')).toBeInTheDocument();
    });

    it('shows a deterministic empty state when no selected sample exists', () => {
        render(<InspectionPanel />);

        expect(screen.getByText(/No training samples are available yet/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /trace prediction/i })).toBeDisabled();
    });
});
