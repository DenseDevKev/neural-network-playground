import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MainArea } from './MainArea.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';
import type { TrainingHook } from '../../hooks/useTraining.ts';

vi.mock('../controls/TrainingControls.tsx', () => ({
    TrainingControls: () => <div>Training controls</div>,
}));

vi.mock('../visualization/NetworkGraph.tsx', () => ({
    NetworkGraph: () => <div>Network graph</div>,
}));

vi.mock('../visualization/DecisionBoundary.tsx', () => ({
    DecisionBoundary: () => <div>Decision boundary</div>,
}));

vi.mock('../visualization/LossChart.tsx', () => ({
    LossChart: () => <div>Loss chart</div>,
}));

vi.mock('../visualization/ConfusionMatrix.tsx', () => ({
    ConfusionMatrix: () => <div>Confusion matrix</div>,
}));

vi.mock('../common/ErrorBoundary.tsx', () => ({
    ErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

function createTrainingMock(): TrainingHook {
    return {
        play: vi.fn(),
        pause: vi.fn(),
        step: vi.fn(),
        reset: vi.fn(),
    };
}

describe('MainArea right-panel content', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed, hiddenLayers: [2] },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });

        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: {
                step: 5,
                epoch: 0,
                trainLoss: 0.5,
                testLoss: 0.6,
                trainMetrics: { loss: 0.5, accuracy: 0.5 },
                testMetrics: { loss: 0.6, accuracy: 0.4 },
                weights: [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]],
                biases: [[0.1, 0.2], [0.3]],
                outputGrid: [],
                gridSize: 40,
                historyPoint: { step: 5, trainLoss: 0.5, testLoss: 0.6 },
            } as any,
            frameVersion: 0,
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
            pauseReason: null,
            testMetricsStale: false,
        });
    });

    it('renders code export tools in the right panel without requiring a toggle', async () => {
        render(<MainArea training={createTrainingMock()} />);

        // Code Export panel is always visible — no collapse needed
        expect(await screen.findByRole('button', { name: 'Pseudocode' })).toBeInTheDocument();
        expect(document.querySelector('.code-export__code')?.textContent).toBeTruthy();
    });

    it('renders all main visualization sections', () => {
        render(<MainArea training={createTrainingMock()} />);

        expect(screen.getByText('Decision boundary')).toBeInTheDocument();
        expect(screen.getByText('Loss chart')).toBeInTheDocument();
        expect(screen.getByText('Confusion matrix')).toBeInTheDocument();
        expect(screen.getByText('Training controls')).toBeInTheDocument();
    });

    it('renders the training explanation surface with the loss chart', () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        render(<MainArea training={createTrainingMock()} />);

        expect(screen.getByText('Why did this happen?')).toBeInTheDocument();
        expect(screen.getByText('Training diverged')).toBeInTheDocument();
    });
});
