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
  };
}

describe('TrainingControls', () => {
  beforeEach(() => {
    useTrainingStore.setState({
      status: 'idle',
      snapshot: null,
      history: [],
      trainPoints: [],
      testPoints: [],
      stepsPerFrame: 5,
    });
  });

  it('should display visible keyboard shortcut hints on control buttons', () => {
    const training = createTrainingMock();

    render(<TrainingControls training={training} />);

    const playButton = screen.getByRole('button', { name: 'Start training' });
    const stepButton = screen.getByRole('button', { name: 'Run one training step' });
    const resetButton = screen.getByRole('button', { name: 'Reset model and data' });

    expect(within(playButton).getByText('Space')).toBeInTheDocument();
    expect(within(stepButton).getAllByText('→')[1]).toBeInTheDocument();
    expect(within(resetButton).getByText('R')).toBeInTheDocument();
  });

  it('should highlight the active speed button and update speed on click', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();
    useTrainingStore.setState({ stepsPerFrame: 10 });

    render(<TrainingControls training={training} />);

    const speed10 = screen.getByRole('button', { name: '10×' });
    const speed25 = screen.getByRole('button', { name: '25×' });

    expect(screen.getByText('Speed:')).toBeInTheDocument();
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
});
