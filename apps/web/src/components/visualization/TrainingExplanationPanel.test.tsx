import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { NetworkSnapshot } from '@nn-playground/engine';
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
});
