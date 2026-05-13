import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { InspectionPanel } from './InspectionPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { updateFrameBuffer, resetFrameBuffer } from '../../worker/frameBuffer.ts';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

const workerApi = vi.hoisted(() => ({
    getPredictionTrace: vi.fn(),
    getBackpropExplanation: vi.fn(),
    getLossLandscapeProbe: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: () => workerApi,
}));

describe('InspectionPanel demand', () => {
    beforeEach(() => {
        workerApi.getPredictionTrace.mockReset();
        workerApi.getBackpropExplanation.mockReset();
        workerApi.getLossLandscapeProbe.mockReset();
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
            demand: { ...DEFAULT_DEMAND, needLayerStats: false, needActivationHistograms: false },
        });
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            activationHistogramsVersion: 0,
            trainPoints: [],
            testPoints: [],
        });
        resetFrameBuffer();
    });

    afterEach(() => {
        usePlaygroundStore.setState({
            demand: { ...DEFAULT_DEMAND, needLayerStats: false, needActivationHistograms: false },
        });
        resetFrameBuffer();
    });

    it('requests layer stats and activation histograms only while inspection is mounted', () => {
        const { unmount } = render(<InspectionPanel />);

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(true);
        expect(usePlaygroundStore.getState().demand.needActivationHistograms).toBe(true);

        unmount();

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(false);
        expect(usePlaygroundStore.getState().demand.needActivationHistograms).toBe(false);
    });

    it('renders an accessible activation histogram summary from the frame buffer', () => {
        usePlaygroundStore.setState((state) => ({
            network: { ...state.network, hiddenLayers: [4] },
        }));
        updateFrameBuffer({
            layerStats: [
                { meanActivation: 0.25, activationStd: 0.15, meanAbsWeight: 0.2, meanAbsGradient: 0.01 },
                { meanActivation: 0.75, activationStd: 0.05, meanAbsWeight: 0.3, meanAbsGradient: 0.02 },
            ],
            activationHistogramBins: new Float32Array([1, 3, 0, 2, 0, 1, 4, 0]),
            activationHistogramLayout: {
                binCount: 4,
                layers: [
                    {
                        layerIndex: 0,
                        binCount: 4,
                        binStart: -1,
                        binWidth: 0.5,
                        minActivation: -1,
                        maxActivation: 1,
                        totalCount: 6,
                        nearZeroCount: 2,
                        saturatedCount: 1,
                    },
                    {
                        layerIndex: 1,
                        binCount: 4,
                        binStart: 0,
                        binWidth: 0.25,
                        minActivation: 0,
                        maxActivation: 1,
                        totalCount: 5,
                        nearZeroCount: 1,
                        saturatedCount: 4,
                    },
                ],
            },
        });
        useTrainingStore.setState({
            activationHistogramsVersion: 1,
            frameVersion: 1,
        });

        render(<InspectionPanel />);

        expect(screen.getByRole('region', { name: /activation histogram explorer/i })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: /histogram layer/i })).toHaveValue('0');
        expect(screen.getAllByText(/Hidden 1 activations/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/33\.3% near zero/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/16\.7% near activation limits/i).length).toBeGreaterThan(0);

        fireEvent.change(screen.getByRole('combobox', { name: /histogram layer/i }), {
            target: { value: '1' },
        });

        expect(screen.getAllByText(/Output activations/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/80\.0% near activation limits/i).length).toBeGreaterThan(0);
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

    it('requests and renders a one-shot backprop preview', async () => {
        workerApi.getBackpropExplanation.mockResolvedValue({
            runId: 1,
            step: 7,
            epoch: 2,
            explanation: {
                batchSize: 5,
                loss: 0.1234,
                learningRate: 0.03,
                globalGradientNorm: 0.0042,
                globalClipScale: 1,
                clipped: false,
                summary: 'Backprop preview found 2 healthy layer updates.',
                layers: [
                    {
                        layerIndex: 0,
                        meanAbsErrorSignal: 0.012,
                        maxAbsErrorSignal: 0.02,
                        meanAbsGradient: 0.003,
                        maxAbsGradient: 0.01,
                        meanAbsUpdate: 0.0009,
                        maxAbsUpdate: 0.002,
                        meanActivation: 0.4,
                        activationStd: 0.1,
                        status: 'healthy',
                        note: 'The previewed update is in a moderate range.',
                    },
                    {
                        layerIndex: 1,
                        meanAbsErrorSignal: 0.02,
                        maxAbsErrorSignal: 0.05,
                        meanAbsGradient: 0.004,
                        maxAbsGradient: 0.015,
                        meanAbsUpdate: 0.001,
                        maxAbsUpdate: 0.004,
                        meanActivation: 0.6,
                        activationStd: 0.2,
                        status: 'healthy',
                        note: 'The previewed update is in a moderate range.',
                    },
                ],
            },
        });

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /preview backprop/i }));

        await waitFor(() => {
            expect(workerApi.getBackpropExplanation).toHaveBeenCalledTimes(1);
        });
        expect(await screen.findByText(/Preview from step 7 \/ epoch 2/i)).toBeInTheDocument();
        expect(screen.getByRole('region', { name: /slow-motion backprop preview/i })).toBeInTheDocument();
        expect(screen.getByRole('status', { name: /backprop preview status/i })).toHaveTextContent(
            /Backprop preview found 2 healthy layer updates/i,
        );
        expect(screen.getByRole('list', { name: /backprop layer summaries/i })).toBeInTheDocument();
        expect(screen.getByText(/batch 5/i)).toBeInTheDocument();
        expect(screen.getByText(/gradient norm 0\.004/i)).toBeInTheDocument();
        expect(screen.getByText(/Hidden 1/i)).toBeInTheDocument();
        expect(screen.getAllByText(/healthy/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/mean update 0\.001/i).length).toBeGreaterThan(0);
    });

    it('activates backprop preview from the keyboard', async () => {
        workerApi.getBackpropExplanation.mockResolvedValue({
            runId: 1,
            step: 0,
            epoch: 0,
            explanation: {
                batchSize: 1,
                loss: 0,
                learningRate: 0.03,
                globalGradientNorm: 0,
                globalClipScale: 1,
                clipped: false,
                summary: 'Backprop preview found 0 healthy layer updates.',
                layers: [],
            },
        });
        render(<InspectionPanel />);

        const button = screen.getByRole('button', { name: /preview backprop/i });
        button.focus();
        expect(button).toHaveFocus();
        await userEvent.keyboard('{Enter}');

        await waitFor(() => {
            expect(workerApi.getBackpropExplanation).toHaveBeenCalledTimes(1);
        });
    });

    it('guards duplicate backprop requests while loading', async () => {
        let resolvePreview: (value: unknown) => void = () => {};
        workerApi.getBackpropExplanation.mockReturnValue(new Promise((resolve) => {
            resolvePreview = resolve;
        }));

        render(<InspectionPanel />);
        const button = screen.getByRole('button', { name: /preview backprop/i });

        fireEvent.click(button);
        fireEvent.click(button);

        expect(workerApi.getBackpropExplanation).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: /previewing backprop/i })).toBeDisabled();

        resolvePreview({
            runId: 1,
            step: 0,
            epoch: 0,
            explanation: {
                batchSize: 1,
                loss: 0,
                learningRate: 0.03,
                globalGradientNorm: 0,
                globalClipScale: 1,
                clipped: false,
                summary: 'Backprop preview found 0 healthy layer updates.',
                layers: [],
            },
        });

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /preview backprop/i })).not.toBeDisabled();
        });
    });

    it('renders backprop worker errors accessibly', async () => {
        workerApi.getBackpropExplanation.mockRejectedValue(
            new Error('Backprop preview is unavailable at the epoch shuffle boundary; step once before previewing.'),
        );

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /preview backprop/i }));

        expect(await screen.findByRole('status', { name: /backprop preview status/i })).toHaveTextContent(
            /Backprop preview failed: Backprop preview is unavailable at the epoch shuffle boundary/i,
        );
    });

    it('requests and renders a one-shot loss landscape probe', async () => {
        workerApi.getLossLandscapeProbe.mockResolvedValue({
            runId: 2,
            step: 9,
            epoch: 1,
            probe: {
                gridSize: 3,
                sampleCount: 12,
                radius: 0.1,
                axisA: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 0, inputIndex: 0, label: 'W1[0,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                axisB: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 1, inputIndex: 0, label: 'W1[1,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                losses: [0.62, 0.58, 0.5, 0.55, 0.49, 0.45, 0.53, 0.44, 0.4],
                centerLoss: 0.49,
                minLoss: 0.4,
                maxLoss: 0.62,
                best: { row: 2, col: 2, loss: 0.4, offsetA: 0.1, offsetB: 0.1 },
                summary: 'Loss surface probe found best loss 0.4000 near W1[0,0] 0.100 and W1[1,0] 0.100.',
            },
        });

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /probe loss surface/i }));

        await waitFor(() => {
            expect(workerApi.getLossLandscapeProbe).toHaveBeenCalledTimes(1);
        });
        expect(screen.getByRole('region', { name: /loss landscape probe/i })).toBeInTheDocument();
        expect(screen.getByRole('img', { name: /local 2d loss slice/i })).toBeInTheDocument();
        expect(screen.getByRole('status', { name: /loss landscape probe status/i })).toHaveTextContent(
            /best loss 0\.4000/i,
        );
        expect(screen.getByText(/Probe from step 9 \/ epoch 1/i)).toBeInTheDocument();
        expect(screen.getByText(/center 0\.490/i)).toBeInTheDocument();
        expect(screen.getByText(/min 0\.400/i)).toBeInTheDocument();
        expect(screen.getByText(/max 0\.620/i)).toBeInTheDocument();
        expect(screen.getByText(/sampled 12 points on a 3 by 3 grid/i)).toBeInTheDocument();
        expect(screen.getByText(/best direction W1\[0,0\] \+0\.100, W1\[1,0\] \+0\.100/i)).toBeInTheDocument();
    });

    it('activates loss landscape probing from the keyboard', async () => {
        workerApi.getLossLandscapeProbe.mockResolvedValue({
            runId: 1,
            step: 0,
            epoch: 0,
            probe: {
                gridSize: 3,
                sampleCount: 1,
                radius: 0.1,
                axisA: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 0, inputIndex: 0, label: 'W1[0,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                axisB: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 1, inputIndex: 0, label: 'W1[1,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                losses: [1, 1, 1, 1, 0.9, 1, 1, 1, 1],
                centerLoss: 0.9,
                minLoss: 0.9,
                maxLoss: 1,
                best: { row: 1, col: 1, loss: 0.9, offsetA: 0, offsetB: 0 },
                summary: 'Loss surface probe found best loss 0.9000 at the current weights.',
            },
        });
        render(<InspectionPanel />);

        const button = screen.getByRole('button', { name: /probe loss surface/i });
        button.focus();
        expect(button).toHaveFocus();
        await userEvent.keyboard('{Enter}');

        await waitFor(() => {
            expect(workerApi.getLossLandscapeProbe).toHaveBeenCalledTimes(1);
        });
    });

    it('guards duplicate loss landscape requests while loading', async () => {
        let resolveProbe: (value: unknown) => void = () => {};
        workerApi.getLossLandscapeProbe.mockReturnValue(new Promise((resolve) => {
            resolveProbe = resolve;
        }));

        render(<InspectionPanel />);
        const button = screen.getByRole('button', { name: /probe loss surface/i });

        fireEvent.click(button);
        fireEvent.click(button);

        expect(workerApi.getLossLandscapeProbe).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: /probing loss surface/i })).toBeDisabled();

        resolveProbe({
            runId: 1,
            step: 0,
            epoch: 0,
            probe: {
                gridSize: 3,
                sampleCount: 1,
                radius: 0.1,
                axisA: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 0, inputIndex: 0, label: 'W1[0,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                axisB: {
                    parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 1, inputIndex: 0, label: 'W1[1,0]' },
                    offsets: [-0.1, 0, 0.1],
                },
                losses: [1, 1, 1, 1, 0.9, 1, 1, 1, 1],
                centerLoss: 0.9,
                minLoss: 0.9,
                maxLoss: 1,
                best: { row: 1, col: 1, loss: 0.9, offsetA: 0, offsetB: 0 },
                summary: 'Loss surface probe found best loss 0.9000 at the current weights.',
            },
        });

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /probe loss surface/i })).not.toBeDisabled();
        });
    });

    it('renders loss landscape worker errors accessibly', async () => {
        workerApi.getLossLandscapeProbe.mockRejectedValue(new Error('Loss landscape probe needs at least two trainable weights.'));

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /probe loss surface/i }));

        expect(await screen.findByRole('status', { name: /loss landscape probe status/i })).toHaveTextContent(
            /Loss landscape probe failed: Loss landscape probe needs at least two trainable weights/i,
        );
    });

    it('shows a deterministic empty state when no selected sample exists', () => {
        render(<InspectionPanel />);

        expect(screen.getByText(/No training samples are available yet/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /trace prediction/i })).toBeDisabled();
    });
});
