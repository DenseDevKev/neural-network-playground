import { fireEvent, render, screen } from '@testing-library/react';
import { DEFAULT_DEMAND, PREPARED_PRESETS } from '@nn-playground/shared';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameVersions, resetFrameBuffer } from '../../worker/frameBuffer.ts';
import { InspectionPanel } from './InspectionPanel.tsx';

const workerApi = vi.hoisted(() => ({
    getPredictionTraceV2: vi.fn(),
    getBackpropExplanationV2: vi.fn(),
    getObjectiveLandscapeV2: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: async () => workerApi,
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

describe('InspectionPanel integration', () => {
    beforeEach(() => {
        Object.values(workerApi).forEach((mock) => mock.mockReset());
        const prepared = oneLayerPrepared();
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared },
            demand: {
                ...DEFAULT_DEMAND,
                needLayerStats: false,
                needActivationHistograms: false,
            },
        });
        useTrainingStore.setState({
            trainPoints: [],
            testPoints: [],
            latestLiveSignal: null,
            latestEvaluation: null,
        });
        resetFrameBuffer();
        useTrainingStore.setState(getFrameVersions());
    });

    afterEach(() => resetFrameBuffer());

    it('keeps the public wrapper free of visualization-demand ownership', () => {
        const initialDemand = usePlaygroundStore.getState().demand;
        const { unmount } = render(<InspectionPanel />);

        expect(usePlaygroundStore.getState().demand).toBe(initialDemand);

        unmount();
        expect(usePlaygroundStore.getState().demand).toBe(initialDemand);
    });

    it('renders a worker prediction trace through the controller and view', async () => {
        useTrainingStore.setState({
            trainPoints: [{ x: 0.25, y: -0.5, label: 1 }],
            latestLiveSignal: {
                model: MODEL,
                dataset: DATASET,
                objectiveKey: 'objective-v2',
                basis: {
                    kind: 'mini-batch-ema',
                    alpha: 0.1,
                    latestBatchSize: 10,
                    throughStep: 12,
                },
                dataLoss: 0.4,
            },
        });
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
                layers: [{
                    layerIndex: 0,
                    preActivations: [1.5],
                    activations: [0.82],
                }],
            },
        });

        render(<InspectionPanel />);
        fireEvent.click(screen.getByRole('button', { name: /trace prediction/i }));

        expect(await screen.findByText('sample data loss')).toBeInTheDocument();
        expect(screen.getByText('0.1900')).toBeInTheDocument();
        expect(screen.getByText('model penalty')).toBeInTheDocument();
        expect(screen.getByText('0.0300')).toBeInTheDocument();
        expect(screen.getByText('Trace from training sample 0 · model step 12 · revision 12'))
            .toBeInTheDocument();
    });
});
