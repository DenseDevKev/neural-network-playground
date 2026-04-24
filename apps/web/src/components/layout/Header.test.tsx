import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from './Header';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import type { LayoutVariant } from '../../store/useLayoutStore.ts';

function createTrainingMock(): Pick<TrainingHook, 'play' | 'pause'> {
    return {
        play: vi.fn(),
        pause: vi.fn(),
    };
}

function renderHeader({
    effectiveLayout = 'dock',
    isCompact = false,
    training = createTrainingMock(),
}: {
    effectiveLayout?: LayoutVariant;
    isCompact?: boolean;
    training?: Pick<TrainingHook, 'play' | 'pause'>;
} = {}) {
    return render(
        <Header
            training={training}
            effectiveLayout={effectiveLayout}
            isCompact={isCompact}
        />,
    );
}

describe('Header', () => {
    beforeEach(() => {
        window.localStorage.clear();
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

        renderHeader();

        expect(screen.getByText('0012')).toBeInTheDocument();
        expect(screen.getByText('0.1234')).toBeInTheDocument();
        expect(screen.getByText('0.5678')).toBeInTheDocument();
        expect(screen.getByText('91.0%')).toBeInTheDocument();
    });

    it('uses the mobile play button to start and pause training', async () => {
        const user = userEvent.setup();
        const training = createTrainingMock();

        const { rerender } = renderHeader({ training });

        await user.click(screen.getByRole('button', { name: 'Start training' }));
        expect(training.play).toHaveBeenCalledTimes(1);

        act(() => {
            useTrainingStore.setState({ status: 'running' });
        });
        rerender(
            <Header
                training={training}
                effectiveLayout="dock"
                isCompact={false}
            />,
        );

        await user.click(screen.getByRole('button', { name: 'Pause training' }));
        expect(training.pause).toHaveBeenCalledTimes(1);
    });

    it('renders the layout picker with dock, focus, grid, and split options', () => {
        renderHeader();

        const picker = screen.getByRole('group', { name: 'Layout variant' });
        expect(picker).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'dock' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'focus' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'grid' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'split' })).toBeInTheDocument();
    });

    it('updates the layout store when a layout option is clicked on desktop', async () => {
        const user = userEvent.setup();
        renderHeader();

        await user.click(screen.getByRole('button', { name: 'focus' }));
        expect(useLayoutStore.getState().layout).toBe('focus');

        await user.click(screen.getByRole('button', { name: 'grid' }));
        expect(useLayoutStore.getState().layout).toBe('grid');

        await user.click(screen.getByRole('button', { name: 'split' }));
        expect(useLayoutStore.getState().layout).toBe('split');
    });

    it('renders phase controls only for the split layout', () => {
        const { rerender } = renderHeader({ effectiveLayout: 'dock' });

        expect(screen.queryByRole('group', { name: 'Workspace phase' })).not.toBeInTheDocument();

        rerender(
            <Header
                training={createTrainingMock()}
                effectiveLayout="split"
                isCompact={false}
            />,
        );

        const phaseGroup = screen.getByRole('group', { name: 'Workspace phase' });
        expect(phaseGroup).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Build' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Run' })).toBeInTheDocument();
    });

    it('keeps the layout picker visible in compact mode and disables grid/split', () => {
        renderHeader({ effectiveLayout: 'dock', isCompact: true });

        expect(screen.getByRole('group', { name: 'Layout variant' })).toBeInTheDocument();
        expect(screen.queryByRole('group', { name: 'Workspace phase' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'dock' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'focus' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'grid' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'split' })).toBeDisabled();
    });

    it('shows the NN·FORGE brand name', () => {
        renderHeader();
        expect(screen.getByText('NN·FORGE')).toBeInTheDocument();
    });
});
