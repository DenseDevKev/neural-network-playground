import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from './Header';
import { TrainingControls } from '../controls/TrainingControls';
import { Sidebar } from './Sidebar';
import { DataPanel } from '../controls/DataPanel';
import { useTrainingStore } from '../../store/useTrainingStore';
import { usePlaygroundStore } from '../../store/usePlaygroundStore';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
};

describe('UI integration flows', () => {
    beforeEach(() => {
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();

        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: {
                step: 0,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                trainMetrics: { loss: 0.5, accuracy: 0.5 },
                testMetrics: { loss: 0.6, accuracy: 0.4 },
                weights: [],
                biases: [],
                outputGrid: [],
                gridSize: 40,
                historyPoint: { step: 0, trainLoss: 0.5, testLoss: 0.6 },
            } as any,
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

        expect(onReset).toHaveBeenCalledTimes(1);
        expect(usePlaygroundStore.getState().data.dataset).toBe('xor');
        expect(presetButton).toHaveClass('preset-card--selected');
    });

    it('supports training flow with progress indicator visibility', async () => {
        const user = userEvent.setup();

        const { container, rerender } = render(
            <>
                <Header training={trainingMock} effectiveLayout="dock" isCompact={false} />
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
                snapshot: {
                    ...(state.snapshot as any),
                    epoch: 12,
                },
            }));
        });

        rerender(
            <>
                <Header training={trainingMock} effectiveLayout="dock" isCompact={false} />
                <TrainingControls training={trainingMock as any} />
            </>,
        );

        expect(screen.getByRole('progressbar', { name: 'Training progress' })).toBeInTheDocument();
        expect(screen.getByText('Training...')).toBeInTheDocument();
    });

    it('shows tooltips for interactive controls during hover', () => {
        vi.useFakeTimers();
        try {
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
            expect(tooltip).toHaveTextContent('Use the XOR dataset');
        } finally {
            vi.useRealTimers();
        }
    });
});
