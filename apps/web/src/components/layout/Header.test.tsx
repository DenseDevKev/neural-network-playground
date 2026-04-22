import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from './Header';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';

function createTrainingMock(): Pick<TrainingHook, 'play' | 'pause'> {
    return {
        play: vi.fn(),
        pause: vi.fn(),
    };
}

describe('Header', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
        });
    });

    it('renders training metrics from the store snapshot', () => {
        useTrainingStore.setState({
            snapshot: {
                epoch: 12,
                trainLoss: 0.1234,
                testLoss: 0.5678,
                trainMetrics: {
                    accuracy: 0.91,
                },
            } as any,
        });

        render(<Header training={createTrainingMock()} />);

        expect(screen.getByText('0012')).toBeInTheDocument();
        expect(screen.getByText('0.1234')).toBeInTheDocument();
        expect(screen.getByText('0.5678')).toBeInTheDocument();
        expect(screen.getByText('91.0%')).toBeInTheDocument();
    });

    it('uses the mobile play button to start and pause training', async () => {
        const user = userEvent.setup();
        const training = createTrainingMock();

        const { rerender } = render(<Header training={training} />);

        await user.click(screen.getByRole('button', { name: 'Start training' }));
        expect(training.play).toHaveBeenCalledTimes(1);

        act(() => {
            useTrainingStore.setState({ status: 'running' });
        });
        rerender(<Header training={training} />);

        await user.click(screen.getByRole('button', { name: 'Pause training' }));
        expect(training.pause).toHaveBeenCalledTimes(1);
    });
});
