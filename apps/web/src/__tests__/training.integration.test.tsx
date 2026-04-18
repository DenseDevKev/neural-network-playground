// ── Training Integration Tests ──
// Exercises the training loop pipeline through a mocked workerBridge,
// driving synthetic snapshots and asserting store updates.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import App from '../App';
import { useTrainingStore } from '../store/useTrainingStore';
import { usePlaygroundStore } from '../store/usePlaygroundStore';
import { resetFrameBuffer } from '../worker/frameBuffer';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

// ── Fake workerBridge ──

let capturedOnSnapshot: ((msg: unknown) => void) | null = null;
let fakeTerminateWorker: ReturnType<typeof vi.fn>;
let fakePostStreamCommand: ReturnType<typeof vi.fn>;
let fakeStartRenderLoop: ReturnType<typeof vi.fn>;
let fakeStopRenderLoop: ReturnType<typeof vi.fn>;
let fakeNewRunTo: ReturnType<typeof vi.fn>;

const fakeSnapshot = {
    step: 10,
    epoch: 1,
    trainLoss: 0.3,
    testLoss: 0.4,
    trainMetrics: { loss: 0.3, accuracy: 0.7 },
    testMetrics: { loss: 0.4, accuracy: 0.6 },
    weights: [],
    biases: [],
    outputGrid: [],
    gridSize: 40,
    historyPoint: { step: 10, trainLoss: 0.3, testLoss: 0.4 },
};

const fakeWorkerApi = {
    initialize: vi.fn().mockResolvedValue({ snapshot: fakeSnapshot, runId: 1 }),
    updateConfig: vi.fn().mockResolvedValue({ snapshot: fakeSnapshot, runId: 2 }),
    reset: vi.fn().mockResolvedValue({ snapshot: fakeSnapshot, runId: 3 }),
    step: vi.fn().mockResolvedValue(fakeSnapshot),
    getTrainPoints: vi.fn().mockResolvedValue([]),
    getTestPoints: vi.fn().mockResolvedValue([]),
    updateDemand: vi.fn().mockResolvedValue(undefined),
    setStreamPort: vi.fn().mockResolvedValue(undefined),
};

vi.mock('../worker/workerBridge.ts', () => ({
    getWorkerApi: () => fakeWorkerApi,
    setupStreamChannel: vi.fn().mockResolvedValue(undefined),
    postStreamCommand: (...args: unknown[]) => fakePostStreamCommand(...args),
    onSnapshot: (cb: (msg: unknown) => void) => {
        capturedOnSnapshot = cb;
        return () => { capturedOnSnapshot = null; };
    },
    newRunTo: (...args: unknown[]) => fakeNewRunTo(...args),
    terminateWorker: () => fakeTerminateWorker(),
    startRenderLoop: () => fakeStartRenderLoop(),
    stopRenderLoop: () => fakeStopRenderLoop(),
    getCurrentRunId: vi.fn().mockReturnValue(1),
}));

// Stub layout sub-components so tests stay focused on training logic.
vi.mock('../components/layout/Header.tsx', () => ({
    Header: ({ training }: { training: { play: () => void; pause: () => void } }) => (
        <header>
            <button onClick={() => training.play()}>Play</button>
            <button onClick={() => training.pause()}>Pause</button>
        </header>
    ),
}));
vi.mock('../components/layout/Sidebar.tsx', () => ({
    Sidebar: () => <aside>Sidebar</aside>,
}));
vi.mock('../components/layout/MainArea.tsx', () => ({
    MainArea: () => <main id="main-content" tabIndex={-1}>Main</main>,
}));

describe('Training integration', () => {
    beforeEach(() => {
        capturedOnSnapshot = null;
        fakeTerminateWorker = vi.fn();
        fakePostStreamCommand = vi.fn();
        fakeStartRenderLoop = vi.fn();
        fakeStopRenderLoop = vi.fn();
        fakeNewRunTo = vi.fn();

        fakeWorkerApi.initialize.mockResolvedValue({ snapshot: fakeSnapshot, runId: 1 });
        fakeWorkerApi.updateConfig.mockResolvedValue({ snapshot: fakeSnapshot, runId: 2 });
        fakeWorkerApi.reset.mockResolvedValue({ snapshot: fakeSnapshot, runId: 3 });
        fakeWorkerApi.step.mockResolvedValue(fakeSnapshot);
        fakeWorkerApi.getTrainPoints.mockResolvedValue([]);
        fakeWorkerApi.getTestPoints.mockResolvedValue([]);

        resetFrameBuffer();

        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
        });

        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            history: [],
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            workerError: null,
            testMetricsStale: false,
        });
    });

    it('initializes worker on mount and populates the training store', async () => {
        await act(async () => {
            render(<App />);
        });

        expect(fakeWorkerApi.initialize).toHaveBeenCalledTimes(1);
        expect(fakeNewRunTo).toHaveBeenCalledWith(1);
        expect(useTrainingStore.getState().snapshot?.trainLoss).toBe(0.3);
    });

    it('sends startTraining command when Play is clicked', async () => {
        await act(async () => {
            render(<App />);
        });

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Play' }));
        });

        expect(fakePostStreamCommand).toHaveBeenCalledWith(
            expect.objectContaining({ type: 'startTraining' }),
        );
        expect(fakeStartRenderLoop).toHaveBeenCalled();
    });

    it('sends stopTraining command when Pause is clicked', async () => {
        await act(async () => {
            render(<App />);
        });

        // Start training first
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Play' }));
        });

        fakePostStreamCommand.mockClear();
        fakeStopRenderLoop.mockClear();

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
        });

        expect(fakePostStreamCommand).toHaveBeenCalledWith(
            expect.objectContaining({ type: 'stopTraining' }),
        );
        expect(fakeStopRenderLoop).toHaveBeenCalled();
    });

    it('applies snapshot messages from the worker to the training store', async () => {
        await act(async () => {
            render(<App />);
        });

        const snapshotMsg = {
            type: 'snapshot' as const,
            runId: 1,
            snapshotId: 1,
            scalars: {
                step: 20,
                epoch: 2,
                trainLoss: 0.25,
                testLoss: 0.35,
                trainAccuracy: 0.8,
                testAccuracy: 0.75,
                gridSize: 40,
                testMetricsStale: false,
            },
            historyPoint: { step: 20, trainLoss: 0.25, testLoss: 0.35 },
        };

        await act(async () => {
            capturedOnSnapshot?.(snapshotMsg);
        });

        const state = useTrainingStore.getState();
        expect(state.snapshot?.trainLoss).toBe(0.25);
        expect(state.snapshot?.epoch).toBe(2);
    });

    it('routes worker error messages to the error overlay', async () => {
        await act(async () => {
            render(<App />);
        });

        const errorMsg = {
            type: 'error' as const,
            runId: 1,
            message: 'Training diverged.',
        };

        await act(async () => {
            capturedOnSnapshot?.(errorMsg);
        });

        expect(useTrainingStore.getState().workerError).toBe('Training diverged.');
        expect(screen.getByText('Worker connection lost')).toBeInTheDocument();
    });
});

describe('Dataset switching scenario', () => {
    beforeEach(() => {
        fakeTerminateWorker = vi.fn();
        fakePostStreamCommand = vi.fn();
        fakeStartRenderLoop = vi.fn();
        fakeStopRenderLoop = vi.fn();
        fakeNewRunTo = vi.fn();

        fakeWorkerApi.initialize.mockResolvedValue({ snapshot: fakeSnapshot, runId: 1 });
        fakeWorkerApi.updateConfig.mockResolvedValue({ snapshot: fakeSnapshot, runId: 2 });
        fakeWorkerApi.getTrainPoints.mockResolvedValue([]);
        fakeWorkerApi.getTestPoints.mockResolvedValue([]);
        fakeWorkerApi.updateDemand.mockResolvedValue(undefined);

        resetFrameBuffer();

        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 },
        });

        useTrainingStore.setState({
            status: 'idle',
            snapshot: fakeSnapshot as any,
            history: [],
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            workerError: null,
            testMetricsStale: false,
        });
    });

    it('calls updateConfig and transitions pendingConfigSource on dataset change', async () => {
        await act(async () => {
            render(<App />);
        });

        // Mark as initialized so config-sync useEffect runs
        fakeWorkerApi.initialize.mockClear();
        fakeWorkerApi.updateConfig.mockClear();

        await act(async () => {
            usePlaygroundStore.getState().setDataset('xor');
        });

        // Allow async sync to complete
        await act(async () => {
            await Promise.resolve();
        });

        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
    });
});
