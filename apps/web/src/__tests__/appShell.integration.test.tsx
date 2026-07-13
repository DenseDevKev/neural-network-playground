import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { NetworkSnapshot } from '@nn-playground/engine';
import App from '../App.tsx';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
};

const lazyPanelWait = { timeout: 10000 };

vi.mock('../hooks/useTraining.ts', () => ({
    useTraining: () => trainingMock,
}));

vi.mock('../components/controls/TrainingControls.tsx', () => ({
    TrainingControls: () => <div>Mock Transport</div>,
}));
vi.mock('../components/controls/PresetPanel.tsx', () => ({
    PresetPanel: () => <div>Mock Presets</div>,
}));
vi.mock('../components/controls/DataPanel.tsx', () => ({
    DataPanel: () => <div>Mock Data</div>,
}));
vi.mock('../components/controls/FeaturesPanel.tsx', () => ({
    FeaturesPanel: () => <div>Mock Features</div>,
}));
vi.mock('../components/controls/NetworkConfigPanel.tsx', () => ({
    NetworkConfigPanel: () => <div>Mock Network Config</div>,
}));
vi.mock('../components/controls/HyperparamPanel.tsx', () => ({
    HyperparamPanel: () => <div>Mock Hyperparameters</div>,
}));
vi.mock('../components/controls/ConfigPanel.tsx', () => ({
    ConfigPanel: () => <div>Mock Config Panel</div>,
}));
vi.mock('../components/controls/InspectionPanel.tsx', () => ({
    InspectionPanel: () => <div>Mock Inspection</div>,
}));
vi.mock('../components/controls/CodeExportPanel.tsx', () => ({
    CodeExportPanel: () => <div>Mock Code Export</div>,
}));
vi.mock('../components/visualization/NetworkGraph.tsx', () => ({
    NetworkGraph: () => <div>Mock Topology Graph</div>,
}));
vi.mock('../components/visualization/DecisionBoundary.tsx', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../components/visualization/DecisionBoundary.tsx')>()),
    DecisionBoundary: () => <div>Mock Boundary</div>,
}));
vi.mock('../components/visualization/LossChart.tsx', () => ({
    LossChart: () => <div>Mock Loss Chart</div>,
}));
vi.mock('../components/visualization/ConfusionMatrix.tsx', () => ({
    ConfusionMatrix: () => <div>Mock Confusion Matrix</div>,
}));

function setViewportWidth(width: number) {
    Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        writable: true,
        value: width,
    });
    window.dispatchEvent(new Event('resize'));
}

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

describe('App shell integration', () => {
    beforeEach(() => {
        window.localStorage.clear();
        setViewportWidth(1280);
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();

        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: {
                ...DEFAULT_NETWORK,
                inputSize: 2,
                hiddenLayers: [...DEFAULT_NETWORK.hiddenLayers],
                outputSize: 1,
                seed: DEFAULT_DATA.seed,
            },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });

        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainedRecipeConfig: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            configError: null,
            configErrorSource: null,
            workerError: null,
            testMetricsStale: false,
        });

        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('renders parity-complete controls in the grid layout', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'grid' }));

        expect(screen.getByText('Mock Presets')).toBeInTheDocument();
        expect(screen.getByText('Mock Data')).toBeInTheDocument();
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.getByText('Mock Config Panel')).toBeInTheDocument();
        expect(await screen.findByText('Mock Inspection', undefined, lazyPanelWait)).toBeInTheDocument();
        expect(await screen.findByText('Mock Code Export', undefined, lazyPanelWait)).toBeInTheDocument();
    });

    it('renders experiment context in dock and focus layouts', async () => {
        const user = userEvent.setup();
        render(<App />);

        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Current run' })).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'focus' }));

        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Current run' })).toBeInTheDocument();
    });

    it('renders parity-complete controls in the focus layout', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'focus' }));

        expect(screen.getByText('Mock Presets')).toBeInTheDocument();
        expect(screen.getByText('Mock Data')).toBeInTheDocument();
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.getByText('Mock Config Panel')).toBeInTheDocument();
        expect(screen.getByText('Mock Topology Graph')).toBeInTheDocument();
        expect(screen.getByText('Mock Boundary')).toBeInTheDocument();
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
        expect(screen.getByText('Mock Confusion Matrix')).toBeInTheDocument();
        expect(await screen.findByText('Mock Inspection')).toBeInTheDocument();
        expect(await screen.findByText('Mock Code Export')).toBeInTheDocument();
    });

    it('highlights the Features panel for feature-focused lesson steps', async () => {
        const user = userEvent.setup();
        const { container } = render(<App />);

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Guided lesson' }),
            'lesson-feature-engineering-circle',
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        expect(useLayoutStore.getState().activeTabLeft).toBe('features');
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(container.querySelector('[data-lesson-target="features"]')).toHaveClass('lesson-target--active');
    });

    it('uses explanation action cards to focus existing dock panels', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({
            snapshot: makeSnapshot(),
            pauseReason: 'diverged',
        });
        render(<App />);

        await user.click(screen.getByRole('tab', { name: 'Loss' }));
        await user.click(screen.getByRole('button', { name: 'Tune learning rate & clipping' }));

        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Read the loss spike' }));

        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
    });

    it('restores split build parity with network and config editors', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'split' }));

        expect(screen.getByRole('group', { name: 'Workspace phase' })).toBeInTheDocument();
        expect(screen.getByText('Mock Presets')).toBeInTheDocument();
        expect(screen.getByText('Mock Data')).toBeInTheDocument();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.getByText('Mock Config Panel')).toBeInTheDocument();
        expect(await screen.findByText('Mock Code Export')).toBeInTheDocument();
    });

    it('keeps both-phase config and code panels available in split run mode', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'split' }));
        await user.click(screen.getByRole('button', { name: 'Run' }));

        expect(screen.getByText('Mock Topology Graph')).toBeInTheDocument();
        expect(screen.getByText('Mock Boundary')).toBeInTheDocument();
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
        expect(screen.getByText('Mock Confusion Matrix')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.getByText('Mock Config Panel')).toBeInTheDocument();
        expect(await screen.findByText('Mock Inspection')).toBeInTheDocument();
        expect(await screen.findByText('Mock Code Export')).toBeInTheDocument();
    });

    it('falls back to compact dock mode without stranding persisted split users', () => {
        setViewportWidth(800);
        useLayoutStore.setState({
            layout: 'split',
            phase: 'run',
            activeTabLeft: 'network',
            activeTabRight: 'boundary',
        });

        render(<App />);

        expect(screen.queryByRole('group', { name: 'Workspace phase' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'dock' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'focus' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'grid' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'split' })).toBeDisabled();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Boundary')).toBeInTheDocument();

        const statusBar = screen.getByRole('status', { name: 'Status bar' });
        expect(statusBar).toHaveTextContent('LAYOUT: dock');
        expect(statusBar).not.toHaveTextContent('PHASE:');
    });
});
