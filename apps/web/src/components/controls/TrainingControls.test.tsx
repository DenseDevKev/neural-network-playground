import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TrainingControls } from './TrainingControls';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getConceptById } from '../../concepts/conceptCatalog.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { TRAINING_SHORTCUTS } from '../../shortcuts/trainingShortcuts.ts';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

function createTrainingMock(): TrainingHook {
  return {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
  };
}

function liveSignal(revision: number, step: number, epoch: number) {
  return {
    model: { generationId: 1, revision, step, epoch },
    dataset: {
      generatorVersion: 2,
      datasetKey: 'test-dataset',
      trainCount: 150,
      testCount: 150,
    },
    objectiveKey: 'test-objective',
    basis: {
      kind: 'mini-batch-ema' as const,
      alpha: 0.1,
      latestBatchSize: 10,
      throughStep: step,
    },
    dataLoss: 0.25,
  };
}

function fullEvaluation(revision: number, step: number, epoch: number) {
  return {
    evaluationId: 12,
    trigger: 'pause' as const,
    model: { generationId: 1, revision, step, epoch },
    dataset: {
      generatorVersion: 2,
      datasetKey: 'test-dataset',
      trainCount: 150,
      testCount: 150,
    },
    objectiveKey: 'test-objective',
    train: {
      basis: { kind: 'full-split' as const, split: 'train' as const, sampleCount: 150, populationCount: 150 },
      values: { dataLoss: 0.2, accuracy: 0.9 },
    },
    test: {
      basis: { kind: 'full-split' as const, split: 'test' as const, sampleCount: 150, populationCount: 150 },
      values: { dataLoss: 0.3, accuracy: 0.8 },
    },
    objective: { regularizationPenalty: 0, trainTotalObjective: 0.2 },
  };
}

describe('TrainingControls', () => {
  beforeEach(() => {
    useTrainingStore.getState().resetEvidence();
    useTrainingStore.setState({
      status: 'idle',
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
    useLayoutStore.setState({ audienceMode: 'beginner' });
  });

  it('should display visible keyboard shortcut hints on control buttons', () => {
    const training = createTrainingMock();

    render(<TrainingControls training={training} />);

    expect(screen.getByRole('region', { name: 'Timeline strip' })).toBeInTheDocument();
    const playButton = screen.getByRole('button', { name: 'Start training' });
    const stepButton = screen.getByRole('button', { name: 'Run one training step' });
    const resetButton = screen.getByRole('button', { name: 'Reset training' });

    expect(within(playButton).getByText('Space')).toBeInTheDocument();
    expect(within(stepButton).getAllByText('→')[1]).toBeInTheDocument();
    expect(within(resetButton).getByText('R')).toBeInTheDocument();
  });

  it('directly describes every visible Reset training action with unique non-live copy', () => {
    const firstTraining = createTrainingMock();
    const secondTraining = createTrainingMock();

    render(
      <>
        <TrainingControls training={firstTraining} />
        <TrainingControls training={secondTraining} />
      </>,
    );

    const resetButtons = screen.getAllByRole('button', { name: 'Reset training' });
    const descriptionIds = resetButtons.map((button) => {
      expect(within(button).getByText('Reset training', { exact: true })).toBeVisible();
      expect(button).toHaveAccessibleDescription(STATE_EFFECTS['training-reset']);

      const ids = (button.getAttribute('aria-describedby') ?? '').split(/\s+/u).filter(Boolean);
      expect(ids).toHaveLength(1);
      const description = document.getElementById(ids[0]);
      expect(description).toHaveTextContent(STATE_EFFECTS['training-reset']);
      expect(description?.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
      return ids[0];
    });

    expect(new Set(descriptionIds).size).toBe(resetButtons.length);
    expect(
      [...document.querySelectorAll('.tooltip__content')]
        .filter((content) => content.textContent === STATE_EFFECTS['training-reset']),
    ).toHaveLength(resetButtons.length);
  });

  it('exposes the shared shortcut registry in a default-closed native disclosure', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();

    render(<TrainingControls training={training} />);

    const trainingBar = screen.getByRole('region', { name: 'Timeline strip' });
    const controls = trainingBar.querySelector('.training-bar__controls');
    const details = screen.getByRole('group', { name: 'Keyboard shortcuts' });
    const summary = within(details).getByText('Keyboard shortcuts');

    expect(details.parentElement).toBe(trainingBar);
    expect(details.previousElementSibling).toBe(controls);
    expect(details).not.toHaveAttribute('open');
    await user.click(summary);
    expect(details).toHaveAttribute('open');

    const terms = within(details).getAllByRole('term');
    const definitions = within(details).getAllByRole('definition');
    expect(details.querySelectorAll('dl')).toHaveLength(1);
    expect(terms).toHaveLength(TRAINING_SHORTCUTS.length);
    expect(definitions).toHaveLength(TRAINING_SHORTCUTS.length);
    expect(terms.map((term) => term.textContent))
      .toEqual(TRAINING_SHORTCUTS.map(({ label }) => label));
    expect(definitions.map((definition) => definition.textContent))
      .toEqual(TRAINING_SHORTCUTS.map(({ description }) => description));
    expect(terms.every((term) => term.firstElementChild?.tagName === 'KBD')).toBe(true);
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
        evidenceGenerationId: 1,
        latestLiveSignal: {
          model: { generationId: 1, revision: 128, step: 128, epoch: 4 },
          dataset: {
            generatorVersion: 2,
            datasetKey: 'test-dataset',
            trainCount: 1,
            testCount: 1,
          },
          objectiveKey: 'test-objective',
          basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 1,
            throughStep: 128,
          },
          dataLoss: 0.25,
        },
      });
    });

    rerender(<TrainingControls training={training} />);

    expect(screen.getByText('Training...')).toBeInTheDocument();
    expect(screen.getByText('Step 128')).toBeInTheDocument();
    expect(screen.getByText('Epoch 4')).toBeInTheDocument();
  });

  it('explains epoch beside the exact progress text with audience-specific guidance', async () => {
    const user = userEvent.setup();
    const training = createTrainingMock();
    useTrainingStore.setState({ latestLiveSignal: liveSignal(4, 20, 4) });
    render(<TrainingControls training={training} />);

    const epochText = screen.getByText('Epoch 4', { selector: 'span' });
    const trigger = screen.getByRole('button', { name: 'Learn about Epoch' });
    expect(epochText.nextElementSibling).toBe(trigger.parentElement);
    expect(trigger.parentElement).toHaveClass(
      'concept-help--above',
      'concept-help--end',
      'concept-help--viewport-overlay',
    );

    await user.click(trigger);
    const region = screen.getByRole('region', { name: 'Epoch' });
    expect(region).toHaveTextContent('Epoch');
    expect(region).toHaveTextContent(
      'One epoch is one complete pass through the current training set.',
    );
    expect(region).toHaveTextContent(getConceptById('epoch')?.extendedExplanation ?? '');
    expect(region).toHaveTextContent(getConceptById('epoch')?.examples?.[0] ?? '');

    await user.click(trigger);
    act(() => useLayoutStore.setState({ audienceMode: 'lab' }));
    await user.click(screen.getByRole('button', { name: 'Learn about Epoch' }));
    const compactRegion = screen.getByRole('region', { name: 'Epoch' });
    expect(compactRegion).toHaveTextContent(
      'One epoch is one complete pass through the current training set.',
    );
    expect(compactRegion).not.toHaveTextContent(
      getConceptById('epoch')?.extendedExplanation ?? '',
    );
    expect(compactRegion).not.toHaveTextContent(
      getConceptById('epoch')?.examples?.[0] ?? '',
    );
  });

  it('shows the newer forced evaluation model instead of a stale live signal after pause', () => {
    const training = createTrainingMock();
    useTrainingStore.setState({
      status: 'paused',
      evidenceGenerationId: 1,
      latestLiveSignal: liveSignal(2_450, 2_450, 163),
      latestEvaluation: fullEvaluation(2_500, 2_500, 166),
    });

    render(<TrainingControls training={training} />);

    expect(screen.getByText('Step 2,500')).toBeInTheDocument();
    expect(screen.getByText('Epoch 166')).toBeInTheDocument();
    expect(screen.queryByText('Step 2,450')).not.toBeInTheDocument();
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
    expect(screen.getByRole('button', { name: 'Reset training' })).toBeDisabled();
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
      checkpointTimeline: {
        checkpoints: [
          { id: 1, step: 0, epoch: 0, trainDataLoss: 0.5, testDataLoss: 0.6, label: 'Step 0' },
          { id: 2, step: 10, epoch: 1, trainDataLoss: 0.3, testDataLoss: 0.4, label: 'Step 10' },
        ],
        maxCheckpoints: 8,
        evictedCount: 0,
        liveCheckpointId: 2,
        restoredCheckpointId: null,
      },
    });

    render(<TrainingControls training={training} />);

    expect(screen.getByText('Timeline')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Learn about Checkpoint' }));
    expect(screen.getByText(getConceptById('checkpoint')?.plainDefinition ?? ''))
      .toBeInTheDocument();
    expect(screen.getByText(getConceptById('checkpoint')?.examples?.[0] ?? ''))
      .toBeInTheDocument();

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
