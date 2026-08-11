import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from './Header';
import { TrainingControls } from '../controls/TrainingControls';
import { Sidebar } from './Sidebar';
import { DataPanel } from '../controls/DataPanel';
import { useTrainingStore } from '../../store/useTrainingStore';
import { usePlaygroundStore } from '../../store/usePlaygroundStore';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
} from '@nn-playground/shared';
import { createScientificTrustFixtures } from '../../test/scientificTrustFixtures.ts';

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
};

describe('UI integration flows', () => {
    beforeEach(async () => {
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();
        trainingMock.restoreCheckpoint.mockReset();

        const restored = await usePlaygroundStore.getState()
            .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);

        useTrainingStore.getState().resetEvidence();
        const fixtures = await createScientificTrustFixtures();
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        useTrainingStore.setState({
            status: 'idle',
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
        });

    });

    it('supports preset selection flow with reset and highlighted selection', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();

        render(<Sidebar onReset={onReset} />);

        const presetButton = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        await user.click(presetButton);

        await waitFor(() => expect(onReset).toHaveBeenCalledTimes(1));
        await screen.findByText('Learning rate');
        await screen.findByRole('button', { name: /Export JSON/ });
        const { access } = usePlaygroundStore.getState();
        expect(access.status === 'ready' ? access.prepared.compiled.task.dataset : null).toBe('xor');
        expect(presetButton).toHaveClass('preset-card--selected');
    });

    it('supports training flow with progress indicator visibility', async () => {
        const user = userEvent.setup();

        const { container, rerender } = render(
            <>
                <Header
                    training={trainingMock}
                    openSurface={null}
                    onToggleSurface={vi.fn()}
                    advancedToolsOpen={false}
                    onToggleAdvancedTools={vi.fn()}
                />
                <TrainingControls training={trainingMock as any} />
            </>,
        );

        const bar = container.querySelector('.training-bar') as HTMLElement | null;
        expect(bar).toBeTruthy();

        await user.click(within(bar as HTMLElement).getByRole('button', { name: 'Start training' }));
        expect(trainingMock.play).toHaveBeenCalledTimes(1);

        act(() => {
            useTrainingStore.setState((state) => ({
                status: 'running',
                latestLiveSignal: {
                    ...state.latestLiveSignal!,
                    model: {
                        ...state.latestLiveSignal!.model,
                        epoch: 12,
                    },
                },
            }));
        });

        rerender(
            <>
                <Header
                    training={trainingMock}
                    openSurface={null}
                    onToggleSurface={vi.fn()}
                    advancedToolsOpen={false}
                    onToggleAdvancedTools={vi.fn()}
                />
                <TrainingControls training={trainingMock as any} />
            </>,
        );

        expect(screen.getByRole('progressbar', { name: 'Training progress' })).toBeInTheDocument();
        expect(screen.getByText('Training...')).toBeInTheDocument();
    });

    it('shows tooltips for interactive controls during hover', () => {
        vi.useFakeTimers();
        render(<DataPanel onReset={vi.fn()} />);

        const xorButton = screen.getByRole('button', { name: 'XOR' });
        const trigger = xorButton.parentElement;
        expect(trigger).toBeTruthy();

        fireEvent.mouseEnter(trigger!);
        act(() => {
            vi.advanceTimersByTime(500);
        });

        const tooltipId = trigger!.getAttribute('aria-describedby');
        const tooltip = tooltipId ? document.getElementById(tooltipId) : null;
        expect(tooltip).toBeInTheDocument();
        expect(tooltip).toHaveTextContent('Cause: XOR alternates labels by quadrant. Effect: a straight boundary fails, so hidden layers have something meaningful to learn.');

        vi.useRealTimers();
    });
});
