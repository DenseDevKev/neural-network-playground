import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RunHistoryPanel } from './RunHistoryPanel.tsx';
import { useExperimentMemoryStore } from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
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
