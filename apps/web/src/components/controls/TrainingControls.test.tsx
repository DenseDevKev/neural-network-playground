import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TrainingControls } from './TrainingControls';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';

function createTrainingMock(): TrainingHook {
  return {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
  };
}

describe('TrainingControls', () => {
  beforeEach(() => {
    useTrainingStore.getState().resetHistory();
    useTrainingStore.setState({
      status: 'idle',
      snapshot: null,
      trainPoints: [],
      testPoints: [],
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
      pauseReason: null,
      checkpointTimeline: {
        checkpoints: [],
        maxCheckpoints: 8,
        evictedCount: 0,
        liveCheckpointId: null,
        restoredCheckpointId: null,
      },
    });
  });

  it('should display visible keyboard shortcut hints on control buttons', () => {
    const training = createTrainingMock();

    render(<TrainingControls training={training} />);

    expect(screen.getByRole('region', { name: 'Timeline strip' })).toBeInTheDocument();
    const playButton = screen.getByRole('button', { name: 'Start training' });
    const stepButton = screen.getByRole('button', { name: 'Run one training step' });
    const resetButton = screen.getByRole('button', { name: 'Reset model and data' });

    expect(within(playButton).getByText('Space')).toBeInTheDocument();
    expect(within(stepButton).getAllByText('→')[1]).toBeInTheDocument();
    expect(within(resetButton).getByText('R')).toBeInTheDocument();
  });

  it('preserves the primary Start, Pause, and Resume action flow', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();

    const { rerender } = render(<TrainingControls training={training} />);

    await user.click(screen.getByRole('button', { name: 'Start training' }));
    expect(training.play).toHaveBeenCalledTimes(1);

    act(() => {
      useTrainingStore.setState({ status: 'running' });
    });
    rerender(<TrainingControls training={training} />);

    await user.click(screen.getByRole('button', { name: 'Pause training' }));
    expect(training.pause).toHaveBeenCalledTimes(1);

    act(() => {
      useTrainingStore.setState({ status: 'paused' });
    });
    rerender(<TrainingControls training={training} />);

    await user.click(screen.getByRole('button', { name: 'Resume training' }));
    expect(training.play).toHaveBeenCalledTimes(2);
  });

  it('should highlight the active speed button and update speed on click', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();
    useTrainingStore.setState({ stepsPerFrame: 10 });

    render(<TrainingControls training={training} />);

    const speed10 = screen.getByRole('button', { name: '10 steps per frame' });
    const speed25 = screen.getByRole('button', { name: '25 steps per frame' });

    expect(screen.getByText('Steps/frame:')).toBeInTheDocument();
    expect(speed10).toHaveClass('active');
    expect(speed25).not.toHaveClass('active');

    await user.click(speed25);

    expect(useTrainingStore.getState().stepsPerFrame).toBe(25);
    expect(speed25).toHaveClass('active');
    expect(speed10).not.toHaveClass('active');
  });

  it('should show the training status indicator only while training is running', () => {
    const training = createTrainingMock();

    const { rerender } = render(<TrainingControls training={training} />);
    expect(screen.queryByText('Training...')).not.toBeInTheDocument();

    act(() => {
      useTrainingStore.setState({
        status: 'running',
        snapshot: {
          step: 128,
          epoch: 4,
        } as any,
      });
    });

    rerender(<TrainingControls training={training} />);

    expect(screen.getByText('Training...')).toBeInTheDocument();
    expect(screen.getByText('Step 128')).toBeInTheDocument();
    expect(screen.getByText('Epoch 4')).toBeInTheDocument();
  });

  it('explains cause and effect in training tooltips', () => {
    const training = createTrainingMock();

    render(<TrainingControls training={training} />);

    expect(screen.getByText('Cause: play repeats weight updates continuously. Effect: the boundary and metrics evolve until you pause or reset.')).toBeInTheDocument();
    expect(screen.getByText('Cause: higher speed runs more updates per animation frame. Effect: learning completes sooner, but individual changes are harder to inspect.')).toBeInTheDocument();
  });

  it('blocks lifecycle-sensitive actions during config sync but keeps speed editable', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();
    useTrainingStore.setState({ pendingConfigSource: 'preset', presetConfigLoading: true });

    render(<TrainingControls training={training} />);

    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Run one training step' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reset model and data' })).toBeDisabled();
    expect(screen.getByText('Updating preset config...')).toBeInTheDocument();

    const speed25 = screen.getByRole('button', { name: '25 steps per frame' });
    expect(speed25).toBeEnabled();

    await user.click(speed25);

    expect(useTrainingStore.getState().stepsPerFrame).toBe(25);
  });

  it('renders checkpoint timeline controls and restores the selected checkpoint', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();
    useTrainingStore.setState({
      snapshot: { step: 10, epoch: 1 } as any,
      checkpointTimeline: {
        checkpoints: [
          { id: 1, step: 0, epoch: 0, trainLoss: 0.5, testLoss: 0.6, label: 'Step 0' },
          { id: 2, step: 10, epoch: 1, trainLoss: 0.3, testLoss: 0.4, label: 'Step 10' },
        ],
        maxCheckpoints: 8,
        evictedCount: 0,
        liveCheckpointId: 2,
        restoredCheckpointId: null,
      },
    });

    render(<TrainingControls training={training} />);

    const slider = screen.getByRole('slider', { name: 'Checkpoint timeline' });
    expect(slider).toHaveValue('1');
    const restoreButton = screen.getByRole('button', { name: 'Restore checkpoint Step 10' });
    expect(restoreButton).toBeInTheDocument();
    act(() => restoreButton.focus());
    expect(restoreButton).toHaveAccessibleDescription(
      'Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only.',
    );
    expect(screen.getByText('Restore in-session parameters and optimizer state')).toBeInTheDocument();
    expect(screen.getAllByText(
      'Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only.',
    ).length).toBeGreaterThan(0);

    act(() => slider.focus());
    await user.keyboard('{ArrowLeft}');
    expect(slider).toHaveValue('0');

    await user.click(screen.getByRole('button', { name: 'Restore checkpoint Step 0' }));

    expect(training.restoreCheckpoint).toHaveBeenCalledWith(1);
  });
});
