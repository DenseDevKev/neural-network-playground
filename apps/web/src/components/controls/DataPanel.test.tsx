import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DataPanel } from './DataPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

describe('DataPanel loading feedback', () => {
    beforeEach(() => {
        usePlaygroundStore.setState((state) => ({
            data: {
                ...state.data,
                dataset: 'circle',
                problemType: 'classification',
                noise: 0,
                trainTestRatio: 0.5,
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

    it('shows accessible train/test split counts from runtime points', () => {
        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByLabelText('Train/test split: 2 train, 1 test')).toBeInTheDocument();
        expect(screen.getByText('Train 2')).toBeInTheDocument();
        expect(screen.getByText('Test 1')).toBeInTheDocument();
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
    });
});
