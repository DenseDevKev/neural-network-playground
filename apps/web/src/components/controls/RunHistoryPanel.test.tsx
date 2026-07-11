import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RunHistoryPanel } from './RunHistoryPanel.tsx';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    useExperimentMemoryStore,
} from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    resetFrameBuffer,
    updateFrameBuffer,
} from '../../worker/frameBuffer.ts';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

function makeRecord(overrides: Partial<ExperimentRunRecordV1> = {}): ExperimentRunRecordV1 {
    return {
        schemaVersion: 1,
        id: 'run-1',
        createdAt: '2026-04-26T00:00:00.000Z',
        updatedAt: '2026-04-26T00:00:00.000Z',
        title: 'Saved XOR',
        config: {
            data: { ...DEFAULT_DATA, dataset: 'xor' },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, hiddenLayers: [4, 4], seed: DEFAULT_DATA.seed },
            training: { ...DEFAULT_TRAINING },
            features: { ...DEFAULT_FEATURES },
            ui: { showTestData: true, discretizeOutput: false },
        },
        summary: {
            status: 'paused',
            pauseReason: 'manual',
            step: 120,
            epoch: 3,
            trainLoss: 0.22,
            testLoss: 0.31,
            trainMetrics: { loss: 0.22, accuracy: 0.9 },
            testMetrics: { loss: 0.31, accuracy: 0.8 },
        },
        network: null,
        history: [{ step: 120, trainLoss: 0.22, testLoss: 0.31 }],
        ...overrides,
    };
}

function makeApprovedMulticlassConfig() {
    return {
        data: {
            ...DEFAULT_DATA,
            dataset: 'three-class-clusters' as const,
            problemType: 'classification' as const,
        },
        network: {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            hiddenLayers: [],
            outputSize: 3,
            outputActivation: 'softmax' as const,
            seed: DEFAULT_DATA.seed,
        },
        training: {
            ...DEFAULT_TRAINING,
            lossType: 'categoricalCrossEntropy' as const,
        },
        features: { ...DEFAULT_FEATURES },
        ui: { showTestData: true, discretizeOutput: false },
    };
}

function makeApprovedMulticlassRecord(overrides: Partial<ExperimentRunRecordV1> = {}): ExperimentRunRecordV1 {
    return makeRecord({
        id: 'multiclass',
        title: 'Three class clusters',
        config: makeApprovedMulticlassConfig(),
        network: null,
        ...overrides,
    });
}

const workerAuthoredMulticlassConfusion = {
    classCount: 3,
    classLabels: [0, 1, 2],
    counts: [2, 0, 1, 0, 3, 0, 1, 0, 4],
} as const;

describe('RunHistoryPanel', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useExperimentMemoryStore.getState().clearRecords();
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });
        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'paused',
            snapshot: null,
            pauseReason: null,
            arenaSummaries: null,
            arenaSummariesVersion: 0,
        });
        resetFrameBuffer();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders an empty state when no runs are saved', () => {
        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.getByText('History is the saved-run record surface.')).toBeInTheDocument();
        expect(screen.getByText('No saved runs')).toBeInTheDocument();
    });

    it('does not create a new legacy V1 record from the current V2 runtime', async () => {
        useTrainingStore.setState({
            snapshot: {
                step: 5,
                epoch: 1,
                trainLoss: 0.4,
                testLoss: 0.5,
                trainMetrics: { loss: 0.4, accuracy: 0.8 },
                testMetrics: { loss: 0.5, accuracy: 0.7 },
                weights: [[[0.1, 0.2]]],
                biases: [[0.3]],
                outputGrid: [],
                gridSize: 50,
                historyPoint: { step: 5, trainLoss: 0.4, testLoss: 0.5 },
            } as any,
        });
        const storageBefore = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        const saveButton = screen.getByRole('button', { name: 'Save current run' });
        expect(saveButton).toBeDisabled();
        await userEvent.click(saveButton);

        expect(screen.getByText(/v2 run saving is unavailable until provenance-aware records/i))
            .toBeInTheDocument();
        expect(useExperimentMemoryStore.getState().records).toHaveLength(0);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(storageBefore);
    });

    it('keeps approved multiclass V2 state out of legacy V1 storage', async () => {
        const multiclassConfig = makeApprovedMulticlassConfig();
        usePlaygroundStore.setState(multiclassConfig);
        useTrainingStore.setState({
            snapshot: {
                step: 7,
                epoch: 1,
                trainLoss: 0.36,
                testLoss: 0.44,
                trainMetrics: { loss: 0.36, accuracy: 0.76 },
                testMetrics: { loss: 0.44, accuracy: 0.7 },
                weights: [[[0.1, 0.2], [0.3, -0.2], [-0.1, 0.4]]],
                biases: [[0.01, -0.02, 0.03]],
                outputGrid: [],
                gridSize: 50,
                historyPoint: { step: 7, trainLoss: 0.36, testLoss: 0.44 },
            } as any,
        });
        const storageBefore = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        const saveButton = screen.getByRole('button', { name: 'Save current run' });
        expect(saveButton).toBeDisabled();
        await userEvent.click(saveButton);

        expect(useExperimentMemoryStore.getState().records).toHaveLength(0);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(storageBefore);
    });

    it('does not create legacy bytes from worker-authored multiclass confusion data', async () => {
        const multiclassConfig = makeApprovedMulticlassConfig();
        usePlaygroundStore.setState(multiclassConfig);
        updateFrameBuffer({ multiclassConfusionMatrix: workerAuthoredMulticlassConfusion });
        useTrainingStore.setState({
            snapshot: {
                step: 7,
                epoch: 1,
                trainLoss: 0.36,
                testLoss: 0.44,
                trainMetrics: { loss: 0.36, accuracy: 0.76 },
                testMetrics: {
                    loss: 0.44,
                    accuracy: 0.7,
                    multiclassConfusionMatrix: workerAuthoredMulticlassConfusion,
                },
                weights: [[[0.1, 0.2], [0.3, -0.2], [-0.1, 0.4]]],
                biases: [[0.01, -0.02, 0.03]],
                outputGrid: [],
                gridSize: 50,
                historyPoint: { step: 7, trainLoss: 0.36, testLoss: 0.44 },
            } as any,
        });
        const storageBefore = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        const saveButton = screen.getByRole('button', { name: 'Save current run' });
        expect(saveButton).toBeDisabled();
        await userEvent.click(saveButton);

        expect(useExperimentMemoryStore.getState().records).toHaveLength(0);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(storageBefore);
    });

    it('keeps V1 saved runs read-only without entering the V2 runtime', () => {
        const onRestore = vi.fn();
        const onInitializeArena = vi.fn();
        const onStepArena = vi.fn();
        const store = usePlaygroundStore.getState();
        const applyRecipe = vi.spyOn(store, 'applyRecipe');
        const replaceDocument = vi.spyOn(store, 'replaceDocument');
        const priorPrepared = store.prepared;
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord());
        });
        const priorBytes = JSON.stringify(useExperimentMemoryStore.getState().records[0]);

        render(
            <RunHistoryPanel
                onRestore={onRestore}
                onInitializeArena={onInitializeArena}
                onStepArena={onStepArena}
            />,
        );

        expect(screen.getByText('Legacy V1 record')).toBeInTheDocument();
        expect(screen.getByText(/read-only and incompatible with the v2 experiment runtime/i))
            .toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /restore/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /live arena/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
        expect(usePlaygroundStore.getState().prepared).toBe(priorPrepared);
        expect(JSON.stringify(useExperimentMemoryStore.getState().records[0])).toBe(priorBytes);
        expect(applyRecipe).not.toHaveBeenCalled();
        expect(replaceDocument).not.toHaveBeenCalled();
        expect(onRestore).not.toHaveBeenCalled();
        expect(onInitializeArena).not.toHaveBeenCalled();
        expect(onStepArena).not.toHaveBeenCalled();
    });

    it('shows existing-data comparison summaries between saved runs', () => {
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'baseline',
                title: 'Baseline',
                updatedAt: '2026-04-26T00:00:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 100,
                    trainLoss: 0.4,
                    testLoss: 0.6,
                },
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'tuned',
                title: 'Tuned model',
                updatedAt: '2026-04-26T00:01:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 180,
                    trainLoss: 0.25,
                    testLoss: 0.42,
                },
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.getByRole('group', { name: 'Comparison for Tuned model against Baseline' })).toBeInTheDocument();
        expect(screen.getByText('Compared with Baseline')).toBeInTheDocument();
        expect(screen.getAllByText('Legacy V1 record')).toHaveLength(2);
        expect(screen.getAllByText(/read-only and incompatible with the V2 experiment runtime/i)).toHaveLength(2);
        expect(screen.getByRole('group', { name: 'Comparison for Tuned model against Baseline' }))
            .toHaveTextContent(/not directly comparable/i);
        expect(screen.queryByText('Train loss -0.1500')).not.toBeInTheDocument();
        expect(screen.queryByText(/next adjustment:/i)).not.toBeInTheDocument();
    });

    it('does not fabricate a current V2 versus legacy V1 winner', () => {
        useTrainingStore.setState({
            snapshot: {
                step: 220,
                epoch: 4,
                trainLoss: 0.2,
                testLoss: 0.36,
                trainMetrics: { loss: 0.2, accuracy: 0.92 },
                testMetrics: { loss: 0.36, accuracy: 0.84 },
                weights: [[[0.1, 0.2]]],
                biases: [[0.3]],
                outputGrid: [],
                gridSize: 50,
            } as any,
        });
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'previous',
                title: 'Previous run',
                updatedAt: '2026-04-26T00:00:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 180,
                    trainLoss: 0.25,
                    testLoss: 0.42,
                },
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'best',
                title: 'Best saved',
                updatedAt: '2026-04-26T00:01:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 160,
                    trainLoss: 0.21,
                    testLoss: 0.33,
                },
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.queryByRole('region', { name: 'Comparison loop' })).not.toBeInTheDocument();
        expect(screen.queryByText('Best saved vs current')).not.toBeInTheDocument();
        expect(screen.queryByText(/which performed better/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/lower test loss/i)).not.toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Legacy saved-run comparison' }))
            .toHaveTextContent(/not directly comparable/i);
    });

    it('does not rewrite legacy bytes through title editing', () => {
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord());
        });
        const recordBefore = JSON.stringify(useExperimentMemoryStore.getState().records[0]);
        const storageBefore = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY);

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.queryByLabelText('Title for Saved XOR')).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Update title for Saved XOR' }))
            .not.toBeInTheDocument();
        expect(JSON.stringify(useExperimentMemoryStore.getState().records[0])).toBe(recordBefore);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(storageBefore);
    });

    it('renders a read-only side-by-side legacy comparison with accessible model regions', async () => {
        const user = userEvent.setup();
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'baseline',
                title: 'Baseline',
                updatedAt: '2026-04-26T00:00:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 100,
                    trainLoss: 0.4,
                    testLoss: 0.6,
                },
                history: [
                    { step: 0, trainLoss: 0.7, testLoss: 0.8 },
                    { step: 100, trainLoss: 0.4, testLoss: 0.6 },
                ],
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'tuned',
                title: 'Tuned model',
                updatedAt: '2026-04-26T00:01:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 180,
                    trainLoss: 0.25,
                    testLoss: 0.42,
                },
                history: [
                    { step: 0, trainLoss: 0.7, testLoss: 0.8 },
                    { step: 180, trainLoss: 0.25, testLoss: 0.42 },
                ],
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.getByRole('region', { name: 'Legacy saved-run comparison' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Model A: Tuned model' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Model B: Baseline' })).toBeInTheDocument();
        expect(screen.getByRole('group', { name: 'Legacy comparison summary' }))
            .toHaveTextContent(/not directly comparable/i);
        expect(screen.getByRole('group', { name: 'Legacy comparison summary' }))
            .not.toHaveTextContent('lower test loss');

        await user.selectOptions(screen.getByLabelText('Model A run'), 'baseline');

        expect(screen.getByRole('region', { name: 'Model A: Baseline' })).toBeInTheDocument();
        expect(screen.getByRole('group', { name: 'Legacy comparison summary' }))
            .toHaveTextContent(/not directly comparable/i);
    });

    it('renders architecture comparison rows for selected saved runs', async () => {
        const user = userEvent.setup();
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'baseline',
                title: 'Baseline',
                updatedAt: '2026-04-26T00:00:00.000Z',
                config: {
                    ...makeRecord().config,
                    data: { ...DEFAULT_DATA, dataset: 'circle', numSamples: 300, noise: 0.05 },
                    network: {
                        ...DEFAULT_NETWORK,
                        inputSize: 2,
                        hiddenLayers: [4],
                        activation: 'tanh',
                        outputActivation: 'sigmoid',
                    },
                    training: {
                        ...DEFAULT_TRAINING,
                        optimizer: 'sgd',
                        learningRate: 0.03,
                        batchSize: 10,
                        lossType: 'crossEntropy',
                        regularization: 'none',
                        regularizationRate: 0,
                    },
                    features: { ...DEFAULT_FEATURES },
                },
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'tuned',
                title: 'Tuned model',
                updatedAt: '2026-04-26T00:01:00.000Z',
                config: {
                    ...makeRecord().config,
                    data: { ...DEFAULT_DATA, dataset: 'xor', numSamples: 500, noise: 0.1 },
                    network: {
                        ...DEFAULT_NETWORK,
                        inputSize: 4,
                        hiddenLayers: [8, 4],
                        activation: 'relu',
                        outputActivation: 'sigmoid',
                    },
                    training: {
                        ...DEFAULT_TRAINING,
                        optimizer: 'adam',
                        learningRate: 0.01,
                        batchSize: 16,
                        lossType: 'crossEntropy',
                        regularization: 'l2',
                        regularizationRate: 0.003,
                    },
                    features: { ...DEFAULT_FEATURES, xSquared: true, xy: true },
                },
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        const architecture = screen.getByRole('group', { name: 'Architecture comparison' });
        expect(architecture).toHaveTextContent('Hidden layers A [8, 4] / B [4]');
        expect(architecture).toHaveTextContent('Total hidden units A 12 / B 4 (+8)');
        expect(architecture).toHaveTextContent('Activation A relu / B tanh');
        expect(architecture).toHaveTextContent('Output/loss A sigmoid + crossEntropy / B sigmoid + crossEntropy');
        expect(architecture).toHaveTextContent('Optimizer/lr A adam @ 0.01 / B sgd @ 0.03');
        expect(architecture).toHaveTextContent('Batch size A 16 / B 10 (+6)');
        expect(architecture).toHaveTextContent('Regularization A l2 0.003 / B none 0');
        expect(architecture).toHaveTextContent('Data A xor, 500 samples, noise 0.1 / B circle, 300 samples, noise 0.05');
        expect(architecture).toHaveTextContent('Features A x, y, xSquared, xy / B x, y');

        await user.selectOptions(screen.getByLabelText('Model A run'), 'baseline');

        expect(screen.getByRole('group', { name: 'Architecture comparison' })).toHaveTextContent(
            'Total hidden units A 4 / B 4 (same)',
        );
    });

    it('does not expose live execution for scalar legacy records', () => {
        const onInitializeArena = vi.fn();
        const onStepArena = vi.fn();
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'baseline',
                title: 'Baseline',
                updatedAt: '2026-04-26T00:00:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 100,
                    trainLoss: 0.4,
                    testLoss: 0.6,
                },
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'tuned',
                title: 'Tuned model',
                updatedAt: '2026-04-26T00:01:00.000Z',
                summary: {
                    ...makeRecord().summary,
                    step: 180,
                    trainLoss: 0.25,
                    testLoss: 0.42,
                },
            }));
        });

        render(
            <RunHistoryPanel
                onRestore={vi.fn()}
                onInitializeArena={onInitializeArena}
                onStepArena={onStepArena}
            />,
        );

        expect(screen.getByRole('region', { name: 'Legacy saved-run comparison' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /live arena/i })).not.toBeInTheDocument();
        expect(screen.getByText(/cannot be restored or executed in the live V2 arena/i)).toBeInTheDocument();
        expect(onInitializeArena).not.toHaveBeenCalled();
        expect(onStepArena).not.toHaveBeenCalled();
    });

    it('keeps multiclass V1 records static and read-only', () => {
        const onInitializeArena = vi.fn();
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'scalar',
                title: 'Scalar baseline',
                updatedAt: '2026-04-26T00:00:00.000Z',
            }));
            useExperimentMemoryStore.getState().saveRecord(makeApprovedMulticlassRecord({
                id: 'multiclass',
                title: 'Three class clusters',
                updatedAt: '2026-04-26T00:01:00.000Z',
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} onInitializeArena={onInitializeArena} />);

        expect(screen.getByRole('region', { name: 'Model A: Three class clusters' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /live arena/i })).not.toBeInTheDocument();
        expect(screen.getByText(/cannot be restored or executed in the live V2 arena/i)).toBeInTheDocument();
        expect(onInitializeArena).not.toHaveBeenCalled();
    });

    it('renders accessible loss-history thumbnails from saved history points', () => {
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                history: [
                    { step: 1, trainLoss: 0.9, testLoss: 1.1 },
                    { step: 2, trainLoss: 0.6, testLoss: 0.8 },
                    { step: 3, trainLoss: 0.3, testLoss: 0.5 },
                ],
            }));
            useExperimentMemoryStore.getState().saveRecord(makeRecord({
                id: 'no-history',
                title: 'No history',
                updatedAt: '2026-04-26T00:01:00.000Z',
                history: [],
            }));
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.getByRole('img', {
            name: 'Loss thumbnail for Saved XOR: 3 points, train loss 0.9000 to 0.3000, test loss 1.1000 to 0.5000.',
        })).toBeInTheDocument();
        expect(screen.getByText('No loss history thumbnail')).toBeInTheDocument();
    });

    it('exports a markdown report for a saved run', async () => {
        const createObjectURL = vi.fn(() => 'blob:report');
        const revokeObjectURL = vi.fn();
        Object.defineProperty(URL, 'createObjectURL', { value: createObjectURL, configurable: true });
        Object.defineProperty(URL, 'revokeObjectURL', { value: revokeObjectURL, configurable: true });
        const click = vi.fn();
        vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
            const element = document.createElementNS('http://www.w3.org/1999/xhtml', tagName);
            if (tagName === 'a') Object.defineProperty(element, 'click', { value: click });
            return element as HTMLElement;
        });
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord());
        });

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        await userEvent.click(screen.getByRole('button', { name: /export report for saved xor/i }));

        expect(createObjectURL).toHaveBeenCalledTimes(1);
        const report = await (createObjectURL.mock.calls[0][0] as Blob).text();
        expect(report).toContain('- Optimizer: sgd');
        expect(report).toContain('- Active features: x, y');
        expect(report).toContain('- Generalization gap: 0.0900');
        expect(report).toContain('- Legacy parameter snapshot present: no (not executable in V2)');
        expect(report).toContain('Legacy V1 record');
        expect(report).toContain('read-only and incompatible with the V2 experiment runtime');
        expect(click).toHaveBeenCalledTimes(1);
        expect(revokeObjectURL).toHaveBeenCalledWith('blob:report');
    });
});
