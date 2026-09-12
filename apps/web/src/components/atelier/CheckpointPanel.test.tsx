import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { CheckpointPanel } from './CheckpointPanel.tsx';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';

function hook() {
    return { pause: vi.fn(() => useTrainingStore.setState({ status: 'paused' })), restoreCheckpoint: vi.fn().mockImplementation(async (id: number) => useTrainingStore.setState((state) => ({ checkpointTimeline: { ...state.checkpointTimeline, restoredCheckpointId: id } }))) } as unknown as TrainingHook;
}
const checkpoint = (id: number, step: number) => ({ id, step, epoch: 0, trainDataLoss: .15, testDataLoss: .18, label: `Step ${step}` });
beforeEach(() => {
    useTrainingStore.setState({ status: 'paused', pendingConfigSource: null, latestLiveSignal: null, latestEvaluation: null,
        checkpointTimeline: { checkpoints: [checkpoint(1, 0), checkpoint(2, 400)], maxCheckpoints: 8, evictedCount: 0, liveCheckpointId: 2, restoredCheckpointId: null } });
});
describe('Session checkpoints', () => {
    it('selection only previews; explicit restore calls the existing command once', async () => {
        const training = hook(); render(<CheckpointPanel training={training} />);
        fireEvent.change(screen.getByRole('slider'), { target: { value: '0' } });
        expect(screen.getByText('Not restored yet')).toBeInTheDocument();
        expect(training.restoreCheckpoint).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole('button', { name: 'Restore Step 0' }));
        await screen.findByText('Restored checkpoint');
        expect(training.restoreCheckpoint).toHaveBeenCalledExactlyOnceWith(1);
    });
    it('exposes Pause while running and blocks restore until paused', async () => {
        useTrainingStore.setState({ status: 'running' });
        const training = hook(); render(<CheckpointPanel training={training} />);
        expect(screen.getByRole('button', { name: 'Restore Step 400' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Pause training' }));
        expect(training.pause).toHaveBeenCalledOnce();
        expect(screen.getByRole('button', { name: 'Restore Step 400' })).toBeEnabled();
    });
    it('blocks restoration during configuration synchronization', () => {
        useTrainingStore.setState({ pendingConfigSource: 'setup' });
        render(<CheckpointPanel training={hook()} />);
        expect(screen.getByRole('button', { name: 'Restore Step 400' })).toBeDisabled();
    });
    it('retains actionable restore errors and unlocks a retry', async () => {
        const training = hook(); vi.mocked(training.restoreCheckpoint).mockRejectedValueOnce(new Error('Checkpoint is no longer available. Select another checkpoint.'));
        render(<CheckpointPanel training={training} />);
        fireEvent.click(screen.getByRole('button', { name: 'Restore Step 400' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Select another checkpoint');
        expect(screen.getByRole('button', { name: 'Restore Step 400' })).toBeEnabled();
        expect(screen.getByText('Not restored yet')).toBeInTheDocument();
    });
    it('reports the newest scientific model step independently of selected checkpoint', () => {
        const model = (step: number) => ({ generationId: 1, revision: step, step, epoch: 0 });
        const dataset = { generatorVersion: 1, datasetKey: 'dataset', trainCount: 10, testCount: 10 } as const;
        const live: LiveTrainingSignal = { model: model(10), dataset, objectiveKey: 'objective', basis: { kind: 'mini-batch-ema', alpha: .1, latestBatchSize: 2, throughStep: 10 }, dataLoss: .2 };
        const evaluation: PairedEvaluation = { model: model(20), dataset, objectiveKey: 'objective', evaluationId: 1, trigger: 'cadence',
            train: { basis: { kind: 'full-split', split: 'train', sampleCount: 10, populationCount: 10 }, values: { dataLoss: .2 } },
            test: { basis: { kind: 'full-split', split: 'test', sampleCount: 10, populationCount: 10 }, values: { dataLoss: .3 } }, objective: { regularizationPenalty: 0, trainTotalObjective: .2 } };
        useTrainingStore.setState({ latestLiveSignal: live, latestEvaluation: evaluation });
        render(<CheckpointPanel training={hook()} />);
        expect(screen.getByText('Step 20')).toBeInTheDocument();
        expect(screen.getByText('Step 400')).toBeInTheDocument();
    });
    it('blocks double submission and checkpoint selection until restore settles', async () => {
        let complete!: () => void;
        const training = hook(); vi.mocked(training.restoreCheckpoint).mockImplementation(() => new Promise<void>((resolve) => { complete = resolve; }));
        render(<CheckpointPanel training={training} />);
        const button = screen.getByRole('button', { name: 'Restore Step 400' });
        fireEvent.click(button); fireEvent.click(button);
        expect(screen.getByRole('slider')).toBeDisabled();
        expect(training.restoreCheckpoint).toHaveBeenCalledOnce();
        await act(async () => complete());
        await waitFor(() => expect(screen.getByRole('slider')).toBeEnabled());
    });
});
