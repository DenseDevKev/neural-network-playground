import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import type { NetworkSnapshot } from '@nn-playground/engine';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { TrainingExplanationPanel } from './TrainingExplanationPanel.tsx';

function makeSnapshot(overrides: Partial<NetworkSnapshot> = {}): NetworkSnapshot {
    return {
        step: 20,
        epoch: 2,
        weights: [],
        biases: [],
        trainLoss: 0.2,
        testLoss: 0.6,
        trainMetrics: { loss: 0.2, accuracy: 0.8 },
        testMetrics: { loss: 0.6, accuracy: 0.7 },
        outputGrid: [],
        gridSize: 40,
        historyPoint: { step: 20, trainLoss: 0.2, testLoss: 0.6 },
        ...overrides,
    };
}

describe('TrainingExplanationPanel', () => {
    beforeEach(() => {
        useTrainingStore.setState({
            snapshot: makeSnapshot(),
            pauseReason: null,
            testMetricsStale: false,
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

    it('renders a diagnostic explanation when there is no stop reason', () => {
        useTrainingStore.setState({
            snapshot: makeSnapshot({ trainLoss: 0.4, testLoss: 0.5 }),
            testMetricsStale: true,
        });

        render(<TrainingExplanationPanel />);

        expect(screen.getByText('Test metrics are catching up')).toBeInTheDocument();
        expect(screen.getByText(/test set is evaluated less often/i)).toBeInTheDocument();
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
