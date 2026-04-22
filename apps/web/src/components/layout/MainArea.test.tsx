import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

vi.mock('../common/Tooltip.tsx', () => ({
    Tooltip: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

function createTrainingMock(): TrainingHook {
    return {
        play: vi.fn(),
        pause: vi.fn(),
        step: vi.fn(),
        reset: vi.fn(),
    };
}

function getPanelToggle(label: RegExp): HTMLButtonElement {
    const matches = screen.getAllByRole('button', { name: label });
    const panelToggle = matches.find((button) => button.classList.contains('panel__header'));
    if (!panelToggle) {
        throw new Error(`Panel toggle not found for label: ${label.toString()}`);
    }
    return panelToggle as HTMLButtonElement;
}

describe('MainArea lazy panels', () => {
    beforeEach(() => {
        const store = new Map<string, string>();
        Object.defineProperty(window, 'localStorage', {
            configurable: true,
            value: {
                getItem: (key: string) => store.get(key) ?? null,
                setItem: (key: string, value: string) => {
                    store.set(key, value);
                },
                removeItem: (key: string) => {
                    store.delete(key);
                },
                clear: () => {
                    store.clear();
                },
            },
        });

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
            testMetricsStale: false,
        });
    });

    it('loads code export tools when the code export panel is expanded', async () => {
        const user = userEvent.setup();

        render(<MainArea training={createTrainingMock()} />);

        const codeExport = getPanelToggle(/Code Export/);

        expect(codeExport).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('button', { name: 'Pseudocode' })).not.toBeInTheDocument();

        await user.click(codeExport);

        expect(codeExport).toHaveAttribute('aria-expanded', 'true');
        expect(await screen.findByRole('button', { name: 'Pseudocode' })).toBeInTheDocument();
        expect(document.querySelector('.code-export__code')?.textContent).toBeTruthy();
    });
});
