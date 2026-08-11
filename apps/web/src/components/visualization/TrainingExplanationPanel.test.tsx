import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { TrainingExplanationPanel } from './TrainingExplanationPanel.tsx';

function live(step = 20): LiveTrainingSignal {
    return {
        model: { generationId: 7, revision: step, step, epoch: 2 },
        dataset: {
            generatorVersion: 1,
            datasetKey: 'dataset-v2',
            trainCount: 70,
            testCount: 30,
        },
        objectiveKey: 'objective-v2',
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 10,
            throughStep: step,
        },
        dataLoss: 0.4,
    };
}

function evaluation(step = 20): PairedEvaluation {
    return {
        evaluationId: 4,
        trigger: 'cadence',
        model: { generationId: 7, revision: step, step, epoch: 2 },
        dataset: {
            generatorVersion: 1,
            datasetKey: 'dataset-v2',
            trainCount: 70,
            testCount: 30,
        },
        objectiveKey: 'objective-v2',
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 70, populationCount: 70 },
            values: { dataLoss: 0.2, accuracy: 0.8 },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 30, populationCount: 30 },
            values: { dataLoss: 0.6, accuracy: 0.7 },
        },
        objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.21 },
    };
}

describe('TrainingExplanationPanel', () => {
    beforeEach(() => {
        useTrainingStore.setState({
            latestLiveSignal: live(),
            latestEvaluation: evaluation(),
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
            activeLessonId: null,
            activeLessonStepIndex: null,
        });
    });

    it('renders the top stop-reason explanation', () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        render(<TrainingExplanationPanel />);

        expect(screen.getByText('Why did this happen?')).toBeInTheDocument();
        expect(screen.getByText('Training diverged')).toBeInTheDocument();
        expect(screen.getByText(/Try lowering the learning rate/i)).toBeInTheDocument();
    });

    it('renders exact evaluation age when there is no stop reason', () => {
        useTrainingStore.setState({
            latestLiveSignal: live(22),
            latestEvaluation: evaluation(20),
        });

        render(<TrainingExplanationPanel />);

        expect(screen.getByText('Full evaluation trails the batch trend')).toBeInTheDocument();
        expect(screen.getByText(/step 20 used all 70 train and 30 test samples.*step 22/i)).toBeInTheDocument();
    });

    it('does not infer a generalization gap without a paired full evaluation', () => {
        useTrainingStore.setState({
            latestLiveSignal: live(22),
            latestEvaluation: null,
        });

        const { container } = render(<TrainingExplanationPanel />);

        expect(container).toBeEmptyDOMElement();
    });

    it('renders semantic action buttons for the top explanation', () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        render(<TrainingExplanationPanel />);

        expect(screen.getByRole('group', { name: 'Suggested explanation actions' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Tune learning rate & clipping' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Read the loss spike' })).toBeInTheDocument();
    });

    it('has no obvious accessibility violations with action cards rendered', async () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        const { container } = render(<TrainingExplanationPanel />);

        const results = await axe(container);
        expect(results.violations).toHaveLength(0);
    });

    it('focuses the selected action target when an action button is clicked', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({ pauseReason: 'diverged' });
        document.body.innerHTML = '<button id="forge-left-tab-hyperparams">Hyperparams</button>';
        const host = document.createElement('div');
        document.body.append(host);

        render(<TrainingExplanationPanel />, { container: host });

        await user.click(screen.getByRole('button', { name: 'Tune learning rate & clipping' }));

        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        await waitFor(() => {
            expect(document.activeElement).toBe(document.getElementById('forge-left-tab-hyperparams'));
        });
    });

    it('activates from the keyboard without bubbling global shortcut keys', async () => {
        const user = userEvent.setup();
        const keydownEvents: string[] = [];
        useTrainingStore.setState({ pauseReason: 'diverged' });
        document.body.innerHTML = '<button id="forge-right-tab-loss">Loss</button>';
        const host = document.createElement('div');
        document.body.append(host);

        render(
            <div onKeyDown={(event) => keydownEvents.push(event.key)}>
                <TrainingExplanationPanel />
            </div>,
            { container: host },
        );

        const action = screen.getByRole('button', { name: 'Read the loss spike' });
        action.focus();
        await user.keyboard('{Enter}');

        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        await waitFor(() => {
            expect(document.activeElement).toBe(document.getElementById('forge-right-tab-loss'));
        });
        expect(keydownEvents).toEqual([]);
    });
});
