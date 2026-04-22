import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from './Header';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
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
        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
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

    it('renders the layout picker with dock, grid, and split options', () => {
        render(<Header training={createTrainingMock()} />);

        const picker = screen.getByRole('group', { name: 'Layout variant' });
        expect(picker).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'dock' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'grid' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'split' })).toBeInTheDocument();
    });

    it('updates the layout store when a layout option is clicked', async () => {
        const user = userEvent.setup();
        render(<Header training={createTrainingMock()} />);

        await user.click(screen.getByRole('button', { name: 'grid' }));
        expect(useLayoutStore.getState().layout).toBe('grid');

        await user.click(screen.getByRole('button', { name: 'split' }));
        expect(useLayoutStore.getState().layout).toBe('split');
    });

    it('renders the phase switch with build and run options', () => {
        render(<Header training={createTrainingMock()} />);

        const phaseGroup = screen.getByRole('group', { name: 'Workspace phase' });
        expect(phaseGroup).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Build' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Run' })).toBeInTheDocument();
    });

    it('shows the NN·FORGE brand name', () => {
        render(<Header training={createTrainingMock()} />);
        expect(screen.getByText('NN·FORGE')).toBeInTheDocument();
    });
});
