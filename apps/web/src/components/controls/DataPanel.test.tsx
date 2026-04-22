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
            },
        }));

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [],
            testPoints: [],
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
});
