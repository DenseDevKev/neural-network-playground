import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App.tsx';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
};

const INITIAL_ACCESS = usePlaygroundStore.getState().access;
if (INITIAL_ACCESS.status !== 'ready') throw new Error('missing app-shell prepared fixture');
const INITIAL_PREPARED = INITIAL_ACCESS.prepared;

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
vi.mock('../components/controls/RunHistoryPanel.tsx', () => ({
    RunHistoryPanel: () => <div>Mock Run History</div>,
}));

function setViewportWidth(width: number) {
    Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        writable: true,
        value: width,
    });
    window.dispatchEvent(new Event('resize'));
}

function mockMatchMedia(matches = false) {
    Object.defineProperty(window, 'matchMedia', {
        writable: true,
        configurable: true,
        value: vi.fn().mockImplementation((query: string) => ({
            matches,
            media: query,
            onchange: null,
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            addListener: vi.fn(),
            removeListener: vi.fn(),
            dispatchEvent: vi.fn(),
        })),
    });
}

function makeLiveSignal(step = 20): LiveTrainingSignal {
    return {
        model: { generationId: 7, revision: step, step, epoch: 2 },
        dataset: {
            generatorVersion: 2,
            datasetKey: INITIAL_PREPARED.identities.datasetKey,
            trainCount: 210,
            testCount: 90,
        },
        objectiveKey: INITIAL_PREPARED.identities.objectiveKey,
        basis: {
            kind: 'mini-batch-ema',
            alpha: 0.1,
            latestBatchSize: 10,
            throughStep: step,
        },
        dataLoss: 0.4,
    };
}

function makeEvaluation(step = 20): PairedEvaluation {
    const model = { generationId: 7, revision: step, step, epoch: 2 };
    const dataset = makeLiveSignal(step).dataset;
    return {
        evaluationId: 4,
        trigger: 'cadence',
        model,
        dataset,
        objectiveKey: INITIAL_PREPARED.identities.objectiveKey,
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 },
            values: { dataLoss: 0.2, accuracy: 0.8 },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 },
            values: { dataLoss: 0.6, accuracy: 0.7 },
        },
        objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.21 },
    };
}

describe('App shell integration', () => {
    beforeEach(() => {
        window.localStorage.clear();
        mockMatchMedia(false);
        setViewportWidth(1280);
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();
        trainingMock.restoreCheckpoint.mockReset();

        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: INITIAL_PREPARED },
            preparation: { status: 'ready', requestId: 0, issues: [] },
            incompatibleSource: null,
        });

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
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

    it('exposes only Build and Run as global workspace views', () => {
        render(<App />);

        const viewSwitcher = screen.getByRole('group', { name: 'Workspace view' });
        expect(within(viewSwitcher).getByRole('button', { name: /build/i })).toBeInTheDocument();
        expect(within(viewSwitcher).getByRole('button', { name: /run/i })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'dock' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'focus' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'grid' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'split' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Presets' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Lessons' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'History' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'More' })).toBeInTheDocument();
    });

    it('renders Build as recipe, topology, features, and hyperparameters without permanent drawers', () => {
        render(<App />);

        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByText('Mock Data')).toBeInTheDocument();
        expect(screen.getByText('Mock Topology Graph')).toBeInTheDocument();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.queryByText('Mock Presets')).not.toBeInTheDocument();
        expect(screen.queryByText('Mock Run History')).not.toBeInTheDocument();
    });

    it('opens Presets, Lessons, History, and More as drawer surfaces', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'Presets' }));
        expect(screen.getByRole('dialog', { name: 'Presets' })).toBeInTheDocument();
        expect(screen.getByText('Mock Presets')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Lessons' }));
        expect(screen.getByRole('dialog', { name: 'Lessons' })).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Guided lesson' })).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'History' }));
        expect(screen.getByRole('dialog', { name: 'History' })).toBeInTheDocument();
        expect(await screen.findByText('Mock Run History')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'More' }));
        expect(screen.getByRole('dialog', { name: 'More / Commands' })).toBeInTheDocument();
        expect(screen.getByText('Mock Config Panel')).toBeInTheDocument();
        expect(await screen.findByText('Mock Code Export')).toBeInTheDocument();
    });

    it('renders Run with transport and one active evidence view', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: /run/i }));

        expect(useLayoutStore.getState().view).toBe('run');
        expect(screen.getByRole('region', { name: 'Current run' })).toBeInTheDocument();
        expect(screen.getByText('Mock Transport')).toBeInTheDocument();
        expect(screen.getByText('Mock Boundary')).toBeInTheDocument();
        expect(screen.queryByText('Mock Loss Chart')).not.toBeInTheDocument();

        await user.click(screen.getByRole('tab', { name: 'Loss' }));

        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
        expect(screen.queryByText('Mock Boundary')).not.toBeInTheDocument();
    });

    it('starts a lesson from the lesson menu and focuses the Build/Run target section', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'Lessons' }));
        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Guided lesson' }),
            'lesson-feature-engineering-circle',
        );
        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('features');
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Feature Engineering Helps')).toBeInTheDocument();
    });

    it('uses explanation action cards to focus Build recipe sections and Run evidence', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({
            latestLiveSignal: makeLiveSignal(),
            latestEvaluation: makeEvaluation(),
            pauseReason: 'diverged',
        });
        render(<App />);

        await user.click(screen.getByRole('button', { name: /run/i }));
        await user.click(screen.getByRole('tab', { name: 'Loss' }));
        await user.click(screen.getByRole('button', { name: 'Tune learning rate & clipping' }));

        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /run/i }));
        await user.click(screen.getByRole('tab', { name: 'Loss' }));
        await user.click(screen.getByRole('button', { name: 'Read the loss spike' }));

        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
    });
});
