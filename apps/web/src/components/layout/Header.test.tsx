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

function renderHeader({
    training = createTrainingMock(),
    openSurface = null,
    onToggleSurface = vi.fn(),
}: {
    training?: Pick<TrainingHook, 'play' | 'pause'>;
    openSurface?: 'presets' | 'lessons' | 'history' | 'more' | null;
    onToggleSurface?: (surface: 'presets' | 'lessons' | 'history' | 'more') => void;
} = {}) {
    return render(
        <Header
            training={training}
            openSurface={openSurface}
            onToggleSurface={onToggleSurface}
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
        });
        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('labels a newer batch trend separately from the paired full evaluation', () => {
        useTrainingStore.setState({
            snapshot: {
                epoch: 99,
                trainLoss: 9,
                testLoss: 8,
            } as any,
            latestLiveSignal: {
                model: { generationId: 4, revision: 1240, step: 1240, epoch: 12 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 1240 },
                dataLoss: 0.1234,
            },
            latestEvaluation: {
                evaluationId: 31,
                trigger: 'cadence',
                model: { generationId: 4, revision: 1230, step: 1230, epoch: 11 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 }, values: { dataLoss: 0.2345, accuracy: 0.527 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 }, values: { dataLoss: 0.5678, accuracy: 0.493 } },
                objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.2445 },
            },
        });

        renderHeader();

        expect(screen.getByText('0012')).toBeInTheDocument();
        expect(screen.getByText('0.1234')).toBeInTheDocument();
        expect(screen.getByText(/Batch trend \(EMA\).*step 1,240/i)).toBeInTheDocument();
        expect(screen.getByText(/Train data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
        expect(screen.getByText(/Test data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
        expect(screen.getByText('0.2345')).toBeInTheDocument();
        expect(screen.getByText('0.5678')).toBeInTheDocument();
        expect(screen.getByText('49.3%')).toBeInTheDocument();
        expect(screen.queryByText('9.0000')).not.toBeInTheDocument();
    });

    it('uses the primary header play button to start and pause training', async () => {
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
                openSurface={null}
                onToggleSurface={vi.fn()}
            />,
        );

        await user.click(screen.getByRole('button', { name: 'Pause training' }));
        expect(training.pause).toHaveBeenCalledTimes(1);
    });

    it('disables the primary start control while config sync is pending', async () => {
        const user = userEvent.setup();
        const training = createTrainingMock();
        useTrainingStore.setState({ pendingConfigSource: 'preset', presetConfigLoading: true });

        renderHeader({ training });

        const button = screen.getByRole('button', { name: 'Start training' });
        expect(button).toBeDisabled();

        await user.click(button);

        expect(training.play).not.toHaveBeenCalled();
    });

    it('renders only Build and Run as global workspace views', () => {
        renderHeader();

        const switcher = screen.getByRole('group', { name: 'Workspace view' });
        expect(switcher).toBeInTheDocument();

        expect(screen.getByRole('button', { name: /build/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'dock' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'focus' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'grid' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'split' })).not.toBeInTheDocument();
    });

    it('updates the workspace view when Build or Run is clicked', async () => {
        const user = userEvent.setup();
        renderHeader();

        await user.click(screen.getByRole('button', { name: /run/i }));
        expect(useLayoutStore.getState().view).toBe('run');

        await user.click(screen.getByRole('button', { name: /build/i }));
        expect(useLayoutStore.getState().view).toBe('build');
    });

    it('opens instrument menu surfaces through the top bar', async () => {
        const user = userEvent.setup();
        const onToggleSurface = vi.fn();
        renderHeader({ onToggleSurface, openSurface: 'history' });

        expect(screen.getByRole('button', { name: 'History' })).toHaveAttribute('aria-pressed', 'true');

        await user.click(screen.getByRole('button', { name: 'Presets' }));
        await user.click(screen.getByRole('button', { name: 'Lessons' }));
        await user.click(screen.getByRole('button', { name: 'More' }));

        expect(onToggleSurface).toHaveBeenNthCalledWith(1, 'presets');
        expect(onToggleSurface).toHaveBeenNthCalledWith(2, 'lessons');
        expect(onToggleSurface).toHaveBeenNthCalledWith(3, 'more');
    });

    it('shows the NN·FORGE brand name', () => {
        renderHeader();
        expect(screen.getByText('NN·FORGE')).toBeInTheDocument();
    });
});
