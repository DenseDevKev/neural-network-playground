import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { PREPARED_PRESETS, DEFAULT_DEMAND } from '@nn-playground/shared';
import { InspectionPanel } from './InspectionPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameVersions, resetFrameBuffer, updateFrameBuffer } from '../../worker/frameBuffer.ts';

const workerApi = vi.hoisted(() => ({
    getPredictionTraceV2: vi.fn(),
    getBackpropExplanationV2: vi.fn(),
    getObjectiveLandscapeV2: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: () => workerApi,
}));

const MODEL = { generationId: 1, revision: 12, step: 12, epoch: 1 } as const;
const DATASET = {
    generatorVersion: 1,
    datasetKey: 'dataset-v2',
    trainCount: 210,
    testCount: 90,
} as const;

function oneLayerPrepared() {
    const result = PREPARED_PRESETS.find((entry) => entry.id === 'circle-one-layer')?.prepared;
    if (!result) throw new Error('missing circle-one-layer');
    return result;
}

function installCurrentEvidence() {
    useTrainingStore.setState({
        latestLiveSignal: {
            model: MODEL,
            dataset: DATASET,
            objectiveKey: 'objective-v2',
            basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 12 },
            dataLoss: 0.4,
        },
        latestEvaluation: null,
    });
}

function backpropResponse() {
    return {
        runId: 1,
        model: MODEL,
        dataset: DATASET,
        objectiveKey: 'objective-v2',
        basis: { kind: 'next-mini-batch', sampleCount: 5, populationCount: 210 },
        explanation: {
            batchSize: 5,
            learningRate: 0.03,
            objective: { dataLoss: 0.12, regularizationPenalty: 0.01, totalObjective: 0.13 },
            gradients: {
                dataGradientNorm: 0.004,
                penaltyGradientNorm: 0.002,
                totalGradientNorm: 0.006,
                clippedGradientNorm: 0.006,
                clipScale: 1,
            },
            summary: 'Backprop preview found 1 healthy layer update.',
            layers: [{
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
            }],
        },
    } as const;
}

function landscapeResponse() {
    return {
        runId: 1,
        model: MODEL,
        provenance: {
            model: MODEL,
            dataset: DATASET,
            objectiveKey: 'objective-v2',
            basis: { kind: 'parameter-grid', sampleCount: 12, parameterPositions: 9 },
        },
        probe: {
            basis: 'training-objective',
            gridSize: 3,
            sampleCount: 12,
            parameterPositionCount: 9,
            radius: 0.1,
            axisA: { parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 0, inputIndex: 0, label: 'W1[0,0]' }, offsets: [-0.1, 0, 0.1] },
            axisB: { parameter: { kind: 'weight', layerIndex: 0, neuronIndex: 1, inputIndex: 0, label: 'W1[1,0]' }, offsets: [-0.1, 0, 0.1] },
            objectives: [0.62, 0.58, 0.5, 0.55, 0.49, 0.45, 0.53, 0.44, 0.4],
            centerObjective: 0.49,
            minObjective: 0.4,
            maxObjective: 0.62,
            best: { row: 2, col: 2, objective: 0.4, offsetA: 0.1, offsetB: 0.1 },
            summary: 'Training-objective surface found best objective 0.4000.',
        },
    } as const;
}

describe('InspectionPanel V2 evidence', () => {
    beforeEach(() => {
        Object.values(workerApi).forEach((mock) => mock.mockReset());
        const prepared = oneLayerPrepared();
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared },
            prepared,
            demand: { ...DEFAULT_DEMAND, needLayerStats: false, needActivationHistograms: false },
        });
        useTrainingStore.setState({
            snapshot: null,
            trainPoints: [],
            testPoints: [],
            latestLiveSignal: null,
            latestEvaluation: null,
        });
        resetFrameBuffer();
        useTrainingStore.setState(getFrameVersions());
    });

    afterEach(() => resetFrameBuffer());

    it('requests layer statistics only while mounted', () => {
        const { unmount } = render(<InspectionPanel />);
        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(true);
        expect(usePlaygroundStore.getState().demand.needActivationHistograms).toBe(true);
        unmount();
        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(false);
    });

    it('labels activation statistics as N of M from artifact provenance', () => {
        updateFrameBuffer({
            layerStats: [
                { meanActivation: 0.25, activationStd: 0.15, meanAbsWeight: 0.2, meanAbsGradient: 0.01 },
                { meanActivation: 0.75, activationStd: 0.05, meanAbsWeight: 0.3, meanAbsGradient: 0.02 },
            ],
            layerStatsProvenance: {
                model: MODEL,
                dataset: DATASET,
                objectiveKey: 'objective-v2',
                basis: { kind: 'bounded-sample', split: 'train', sampleCount: 128, populationCount: 210 },
            },
            layerStatsGradientRevision: 12,
        });
        useTrainingStore.setState(getFrameVersions());

        render(<InspectionPanel />);
        expect(screen.getByText('Activation statistics across 128 of 210 training examples'))
            .toBeInTheDocument();
    });

    it('renders sample data loss and model penalty from a current V2 prediction trace', async () => {
        installCurrentEvidence();
        useTrainingStore.setState({ trainPoints: [{ x: 0.25, y: -0.5, label: 1 }] });
        workerApi.getPredictionTraceV2.mockResolvedValue({
            runId: 1,
            model: MODEL,
            dataset: DATASET,
            objectiveKey: 'objective-v2',
            sample: { source: 'train', index: 0, x: 0.25, y: -0.5, label: 1 },
            trace: {
                input: [0.25, -0.5],
                target: [1],
                output: [0.82],
                prediction: 0.82,
                sampleDataLoss: 0.19,
                regularizationPenalty: 0.03,
                layers: [{ layerIndex: 0, preActivations: [1.5], activations: [0.82] }],
            },
        });

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /trace prediction/i }));

        expect(await screen.findByText('sample data loss')).toBeInTheDocument();
        expect(screen.getByText('0.1900')).toBeInTheDocument();
        expect(screen.getByText('model penalty')).toBeInTheDocument();
        expect(screen.getByText('0.0300')).toBeInTheDocument();
    });

    it('shows the complete objective and gradient breakdown for backprop', async () => {
        installCurrentEvidence();
        workerApi.getBackpropExplanationV2.mockResolvedValue(backpropResponse());
        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /preview backprop/i }));

        expect(await screen.findByText(/Preview from step 12 \/ epoch 1/i)).toBeInTheDocument();
        expect(screen.getByText(/data loss 0\.120/i)).toBeInTheDocument();
        expect(screen.getByText(/penalty 0\.010/i)).toBeInTheDocument();
        expect(screen.getByText(/training objective 0\.130/i)).toBeInTheDocument();
        expect(screen.getByText(/complete objective gradient 0\.006/i)).toBeInTheDocument();
    });

    it('states the training-objective sample and parameter-grid basis', async () => {
        installCurrentEvidence();
        workerApi.getObjectiveLandscapeV2.mockResolvedValue(landscapeResponse());
        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /probe loss surface/i }));

        expect(await screen.findByText('Training objective on a parameter grid')).toBeInTheDocument();
        expect(screen.getByText(/sampled 12 training examples across 9 parameter positions/i))
            .toBeInTheDocument();
        expect(screen.getByRole('img', { name: /Training-objective parameter grid/i }))
            .toBeInTheDocument();
    });

    it('drops a diagnostic response when the model revision changed while it was pending', async () => {
        installCurrentEvidence();
        let resolve!: (value: ReturnType<typeof backpropResponse>) => void;
        workerApi.getBackpropExplanationV2.mockReturnValue(new Promise((next) => { resolve = next; }));
        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /preview backprop/i }));
        await act(async () => {
            useTrainingStore.setState((state) => ({
                latestLiveSignal: state.latestLiveSignal
                    ? { ...state.latestLiveSignal, model: { ...MODEL, revision: 13, step: 13 } }
                    : null,
            }));
            resolve(backpropResponse());
        });

        await waitFor(() => expect(workerApi.getBackpropExplanationV2).toHaveBeenCalledTimes(1));
        expect(screen.queryByText(/Preview from step 12/i)).not.toBeInTheDocument();
    });

    it('keeps trace unavailable without a target-bearing sample', () => {
        installCurrentEvidence();
        render(<InspectionPanel />);
        expect(screen.getByRole('button', { name: /trace prediction/i })).toBeDisabled();
    });
});
