import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RunHistoryPanel } from './RunHistoryPanel.tsx';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import {
    type ArenaModelSummary,
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

function makeArenaSummaries(): ArenaModelSummary[] {
    return [
        {
            side: 'A',
            label: 'Tuned model',
            status: 'paused',
            pauseReason: 'manual',
            step: 181,
            epoch: 3,
            trainLoss: 0.24,
            testLoss: 0.4,
            trainAccuracy: 0.91,
            testAccuracy: 0.84,
        },
        {
            side: 'B',
            label: 'Baseline',
            status: 'paused',
            pauseReason: 'manual',
            step: 101,
            epoch: 2,
            trainLoss: 0.38,
            testLoss: 0.58,
            trainAccuracy: 0.78,
            testAccuracy: 0.7,
        },
    ];
}

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
    });

    it('renders an empty state when no runs are saved', () => {
        render(<RunHistoryPanel onRestore={vi.fn()} />);

        expect(screen.getByText('No saved runs')).toBeInTheDocument();
    });

    it('saves the current run when a snapshot exists', async () => {
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

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        expect(screen.getByText(/circle at step 5/i)).toBeInTheDocument();
        expect(useExperimentMemoryStore.getState().records).toHaveLength(1);
    });

    it('saves an approved multiclass current run without silently dropping it', async () => {
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

        render(<RunHistoryPanel onRestore={vi.fn()} />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        expect(screen.getByText(/three-class-clusters at step 7/i)).toBeInTheDocument();
        expect(useExperimentMemoryStore.getState().records[0].config.network.outputSize).toBe(3);
        expect(useExperimentMemoryStore.getState().records[0].config.training.lossType).toBe('categoricalCrossEntropy');
    });

    it('restores a saved run config and calls reset', async () => {
        const onRestore = vi.fn();
        act(() => {
            useExperimentMemoryStore.getState().saveRecord(makeRecord());
        });

        render(<RunHistoryPanel onRestore={onRestore} />);
        await userEvent.click(screen.getByRole('button', { name: /restore config for saved xor/i }));

        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
        expect(usePlaygroundStore.getState().network.hiddenLayers).toEqual([4, 4]);
        expect(onRestore).toHaveBeenCalledTimes(1);
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
        expect(screen.getByText('Train loss -0.1500')).toBeInTheDocument();
        expect(screen.getByText('Test loss -0.1800')).toBeInTheDocument();
        expect(screen.getByText('Gap -0.0300')).toBeInTheDocument();
        expect(screen.getByText('Steps +80')).toBeInTheDocument();
    });

    it('renders a side-by-side arena from two saved runs with accessible model regions', async () => {
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

        expect(screen.getByRole('region', { name: 'Side-by-side model arena' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Model A: Tuned model' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Model B: Baseline' })).toBeInTheDocument();
        expect(screen.getByRole('group', { name: 'Arena comparison summary' })).toHaveTextContent(
            'Model A lower test loss by 0.1800',
        );
        expect(screen.getByRole('group', { name: 'Arena comparison summary' })).toHaveTextContent(
            'Model A trained 80 more steps',
        );

        await user.selectOptions(screen.getByLabelText('Model A run'), 'baseline');

        expect(screen.getByRole('region', { name: 'Model A: Baseline' })).toBeInTheDocument();
        expect(screen.getByRole('group', { name: 'Arena comparison summary' })).toHaveTextContent(
            'Both models have the same test loss.',
        );
    });

    it('starts and steps the live scalar arena with accessible summaries', async () => {
        const user = userEvent.setup();
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

        await user.click(screen.getByRole('button', { name: 'Start live arena with selected saved runs' }));

        expect(onInitializeArena).toHaveBeenCalledWith(
            expect.objectContaining({ id: 'tuned' }),
            expect.objectContaining({ id: 'baseline' }),
        );

        act(() => {
            useTrainingStore.setState({
                arenaSummaries: makeArenaSummaries(),
                arenaSummariesVersion: 1,
            });
        });

        const summaries = screen.getByRole('group', { name: 'Live arena scalar summaries' });
        expect(summaries).toHaveTextContent('Model A live: Tuned model');
        expect(summaries).toHaveTextContent('test 0.4000');
        expect(summaries).toHaveTextContent('Model B live: Baseline');
        expect(summaries).toHaveTextContent('accuracy 84.0% / 70.0%');

        await user.keyboard('[Tab]');
        await user.click(screen.getByRole('button', { name: 'Step live arena once' }));

        expect(onStepArena).toHaveBeenCalledTimes(1);
    });

    it('keeps the live arena scalar-only when saved multiclass records are visible', async () => {
        const user = userEvent.setup();
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

        const startButton = screen.getByRole('button', { name: 'Start live arena with selected saved runs' });
        expect(screen.getByRole('region', { name: 'Model A: Three class clusters' })).toBeInTheDocument();
        expect(startButton).toBeDisabled();
        expect(screen.getByText(/live arena supports saved scalar runs/i)).toBeInTheDocument();

        await user.click(startButton);

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
        expect(report).toContain('- Saved parameters: no');
        expect(click).toHaveBeenCalledTimes(1);
        expect(revokeObjectURL).toHaveBeenCalledWith('blob:report');
    });
});
