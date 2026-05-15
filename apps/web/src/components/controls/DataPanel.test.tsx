import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DataPanel } from './DataPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { DEFAULT_DATA } from '@nn-playground/shared';

describe('DataPanel loading feedback', () => {
    beforeEach(() => {
        usePlaygroundStore.setState((state) => ({
            data: {
                ...state.data,
                dataset: 'circle',
                problemType: 'classification',
                noise: 0,
                trainTestRatio: 0.5,
                numSamples: DEFAULT_DATA.numSamples,
                seed: 42,
            },
        }));

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [{ x: 0, y: 0, label: 0 }, { x: 1, y: 1, label: 1 }],
            testPoints: [{ x: -1, y: -1, label: 0 }],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            featuresConfigLoading: false,
            trainingConfigLoading: false,
            presetConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
        });
    });

    it('shows the inline loading state when the dataset changes', async () => {
        const user = userEvent.setup();

        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'XOR' }));

        expect(screen.getByRole('status')).toHaveTextContent('Generating data...');
        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
    });

    it('shows data-specific config errors and allows retrying', async () => {
        const user = userEvent.setup();

        useTrainingStore.setState({
            configError: 'Failed to generate data',
            configErrorSource: 'data',
        });

        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByText('Failed to generate data')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });

    it('explains cause and effect in data tooltips', () => {
        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByText('Cause: XOR alternates labels by quadrant. Effect: a straight boundary fails, so hidden layers have something meaningful to learn.')).toBeInTheDocument();
        expect(screen.getByText('Cause: more noise blurs class edges. Effect: training loss may flatten and test accuracy becomes harder to improve.')).toBeInTheDocument();
    });

    it('does not start a data transaction for the already-active dataset', async () => {
        const user = userEvent.setup();

        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Circle' }));

        expect(screen.queryByRole('status')).not.toBeInTheDocument();
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
    });

    it('shows accessible train/test split counts from runtime points', () => {
        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByLabelText('Train/test split: 2 train, 1 test')).toBeInTheDocument();
        expect(screen.getByText('Train 2')).toBeInTheDocument();
        expect(screen.getByText('Test 1')).toBeInTheDocument();
    });

    it('summarizes bounded dataset parameters for comparison', () => {
        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByLabelText('Dataset settings: 300 samples, 0 noise, 50% train')).toBeInTheDocument();
    });

    it('changes sample count through bounded preset controls', async () => {
        const user = userEvent.setup();

        render(<DataPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: '600 samples' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
        expect(usePlaygroundStore.getState().data.numSamples).toBe(600);
        expect(screen.getByRole('button', { name: '600 samples' })).toHaveAttribute('aria-pressed', 'true');
    });

    it('exposes the approved three-class dataset as a complete multiclass tuple', async () => {
        const user = userEvent.setup();

        render(<DataPanel onReset={vi.fn()} />);

        const button = screen.getByRole('button', { name: 'Three-Class' });
        await user.click(button);

        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
        expect(usePlaygroundStore.getState().data.dataset).toBe('three-class-clusters');
        expect(usePlaygroundStore.getState().data.problemType).toBe('classification');
        expect(usePlaygroundStore.getState().network.outputSize).toBe(3);
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('softmax');
        expect(usePlaygroundStore.getState().training.lossType).toBe('categoricalCrossEntropy');
        expect(button).toHaveAttribute('aria-pressed', 'true');
    });

    it('returns to a scalar tuple when switching from three-class to a binary dataset', async () => {
        const user = userEvent.setup();
        usePlaygroundStore.getState().setDataset('three-class-clusters');

        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'XOR' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
        expect(usePlaygroundStore.getState().data.problemType).toBe('classification');
        expect(usePlaygroundStore.getState().network.outputSize).toBe(1);
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('sigmoid');
        expect(usePlaygroundStore.getState().training.lossType).toBe('crossEntropy');
    });

    it('normalizes hidden multiclass state when public dataset modes are selected', async () => {
        const user = userEvent.setup();
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...state.training,
                lossType: 'categoricalCrossEntropy',
            },
        }));

        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Regression' }));
        expect(usePlaygroundStore.getState().network.outputSize).toBe(1);
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('linear');
        expect(usePlaygroundStore.getState().training.lossType).toBe('mse');

        await user.click(screen.getByRole('button', { name: 'Classification' }));
        expect(usePlaygroundStore.getState().network.outputSize).toBe(1);
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('sigmoid');
        expect(usePlaygroundStore.getState().training.lossType).toBe('crossEntropy');
    });

    it('reshuffles by changing the data seed through the data config path', async () => {
        const user = userEvent.setup();

        render(<DataPanel onReset={vi.fn()} />);
        await user.click(screen.getByRole('button', { name: 'Reshuffle split' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
        expect(usePlaygroundStore.getState().data.seed).toBe(43);
    });

    it('disables reshuffle while data config is loading', () => {
        useTrainingStore.setState({ dataConfigLoading: true });

        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Reshuffle split' })).toBeDisabled();
        expect(screen.getByRole('button', { name: '600 samples' })).toBeDisabled();
    });
});
