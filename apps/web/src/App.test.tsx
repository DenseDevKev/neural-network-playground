import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { axe } from 'jest-axe';
import type { ReactNode } from 'react';
import App from './App';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    prepareExperimentDocument,
} from '@nn-playground/shared';

const useTrainingMount = vi.hoisted(() => vi.fn());

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
    restoreCheckpoint: vi.fn(),
};

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

vi.mock('./hooks/useTraining.ts', () => ({
    useTraining: () => {
        useTrainingMount();
        return trainingMock;
    },
}));

vi.mock('./components/layout/Header.tsx', () => ({
    Header: () => <header>Header</header>,
}));

vi.mock('./components/layout/BuildRunShell.tsx', () => ({
    BuildRunShell: ({
        view,
        recipeContent,
        runContent,
        dataContent,
        networkContent,
        featuresContent,
        hyperparamContent,
        topologyContent,
        transportContent,
    }: {
        view: string;
        recipeContent?: ReactNode;
        runContent?: ReactNode;
        dataContent?: ReactNode;
        networkContent?: ReactNode;
        featuresContent?: ReactNode;
        hyperparamContent?: ReactNode;
        topologyContent?: ReactNode;
        transportContent?: ReactNode;
    }) => (
        <section aria-label={`${view} workspace`} data-testid="build-run-shell">
            <div data-forge-panel-targets="experiment">{recipeContent}</div>
            <div data-forge-panel-targets="run">{runContent}</div>
            <div data-forge-panel-targets="data">{dataContent}</div>
            <div data-forge-panel-targets="network">{networkContent}</div>
            <div data-forge-panel-targets="features">{featuresContent}</div>
            <div data-forge-panel-targets="hyperparams">{hyperparamContent}</div>
            <div data-forge-panel-targets="topology">{topologyContent}</div>
            <div data-forge-panel-targets="transport">{transportContent}</div>
        </section>
    ),
}));

vi.mock('./components/layout/MainArea.tsx', () => ({
    CanvasContent:   () => <div>Canvas</div>,
    BoundaryContent: () => <div>Boundary</div>,
    LossContent:     () => <div>Loss</div>,
    ConfusionContent:() => <div>Confusion</div>,
    InspectContent:  () => <div>Inspect</div>,
    CodeContent:     () => <div>Code</div>,
    HistoryContent:  () => <div>History</div>,
}));

vi.mock('./components/controls/TrainingControls.tsx', () => ({ TrainingControls: () => <div>Controls</div> }));
vi.mock('./components/visualization/NetworkGraph.tsx', () => ({ NetworkGraph: () => <div>Graph</div> }));
vi.mock('./components/controls/PresetPanel.tsx',       () => ({ PresetPanel: () => <div>Presets</div> }));
vi.mock('./components/controls/DataPanel.tsx',         () => ({ DataPanel: () => <div>Data</div> }));
vi.mock('./components/controls/FeaturesPanel.tsx',     () => ({ FeaturesPanel: () => <div>Features</div> }));
vi.mock('./components/controls/NetworkConfigPanel.tsx',() => ({ NetworkConfigPanel: () => <div>Network</div> }));
vi.mock('./components/controls/HyperparamPanel.tsx',   () => ({ HyperparamPanel: () => <div>Hyperparams</div> }));
vi.mock('./components/controls/ConfigPanel.tsx',       () => ({ ConfigPanel: () => <div>Config</div> }));
vi.mock('./components/controls/InspectionPanel.tsx',   () => ({ InspectionPanel: () => <div>Inspection</div> }));
vi.mock('./components/controls/CodeExportPanel.tsx',   () => ({ CodeExportPanel: () => <div>CodeExport</div> }));
vi.mock('./components/controls/RunHistoryPanel.tsx',   () => ({ RunHistoryPanel: () => <div>RunHistory</div> }));

describe('App accessibility shell', () => {
    beforeEach(async () => {
        window.localStorage.clear();
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();
        trainingMock.restoreCheckpoint.mockReset();
        useTrainingMount.mockClear();

        const prepared = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        if (!prepared.ok) throw new Error('default experiment fixture did not prepare');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: prepared.value },
            prepared: prepared.value,
            preparation: { status: 'ready', requestId: 0, issues: [] },
        });

        useTrainingStore.setState({
            status: 'idle',
            dataConfigLoading: false,
            networkConfigLoading: false,
            configError: null,
            configErrorSource: null,
            workerError: null,
            snapshot: null,
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

    it('renders durable compatibility recovery without mounting the training hook', () => {
        usePlaygroundStore.getState().markIncompatible(
            { kind: 'url', rawHash: '#d=xor&n=0.2' },
            [{
                code: 'legacy-state',
                path: 'schemaVersion',
                message: 'unversioned experiment documents are incompatible',
            }],
        );

        render(<App />);

        expect(useTrainingMount).not.toHaveBeenCalled();
        expect(screen.getByRole('main', { name: 'Experiment compatibility' }))
            .toBeInTheDocument();
        expect(screen.getByText('#d=xor&n=0.2')).toBeInTheDocument();
    });

    it('renders a skip link to the main content', () => {
        render(<App />);
        expect(screen.getByRole('link', { name: 'Skip to main content' })).toHaveAttribute('href', '#main-content');
    });

    it('exposes the redesigned workspace as the main landmark', () => {
        render(<App />);
        expect(screen.getByRole('main', { name: 'Neural network playground workspace' })).toHaveAttribute('id', 'main-content');
    });

    it('has no obvious accessibility violations in the shell', async () => {
        const { container } = render(<App />);
        const results = await axe(container);
        expect(results.violations).toHaveLength(0);
    });

    it('shows a recovery overlay when the worker connection fails', () => {
        useTrainingStore.setState({ workerError: 'Worker channel closed unexpectedly.' });

        render(<App />);

        expect(screen.getAllByText('Worker connection lost').length).toBeGreaterThan(0);
        expect(screen.getAllByText('Worker channel closed unexpectedly. Refresh the page to restart the playground.').length).toBeGreaterThan(0);
        expect(screen.getByRole('button', { name: 'Refresh page' })).toBeInTheDocument();
    });

    it('handles global keyboard shortcuts for training controls', () => {
        render(<App />);

        fireEvent.keyDown(window, { code: 'Space' });
        fireEvent.keyDown(window, { code: 'ArrowRight' });
        fireEvent.keyDown(window, { code: 'KeyR' });

        expect(trainingMock.play).toHaveBeenCalledTimes(1);
        expect(trainingMock.step).toHaveBeenCalledTimes(1);
        expect(trainingMock.reset).toHaveBeenCalledTimes(1);
    });

    it('handles global keyboard shortcuts from page background focus', () => {
        render(<App />);

        fireEvent.keyDown(document.body, { code: 'KeyR' });
        fireEvent.keyDown(window, { code: 'KeyR' });

        expect(trainingMock.reset).toHaveBeenCalledTimes(2);
    });

    it('ignores global keyboard shortcuts from focused interactive elements', () => {
        render(
            <>
                <App />
                <button type="button">Regular button</button>
                <button type="button" role="tab">Tab control</button>
                <a href="#example">Example link</a>
                <input aria-label="Typing field" />
                <select aria-label="Dataset select"><option>Circle</option></select>
                <textarea aria-label="Notes" />
                <div contentEditable role="textbox" aria-label="Editable notes" />
                <div role="button" tabIndex={0}>Div button</div>
                <div role="switch" tabIndex={0} aria-checked="false">Switch control</div>
                <div role="slider" tabIndex={0} aria-label="Slider control" aria-valuemin={0} aria-valuemax={1} aria-valuenow={0}>Slider control</div>
                <div tabIndex={0} aria-label="Graph node">Graph node</div>
            </>,
        );

        const interactiveTargets = [
            screen.getByRole('button', { name: 'Regular button' }),
            screen.getByRole('tab', { name: 'Tab control' }),
            screen.getByRole('link', { name: 'Example link' }),
            screen.getByRole('textbox', { name: 'Typing field' }),
            screen.getByRole('combobox', { name: 'Dataset select' }),
            screen.getByRole('textbox', { name: 'Notes' }),
            screen.getByRole('textbox', { name: 'Editable notes' }),
            screen.getByRole('button', { name: 'Div button' }),
            screen.getByRole('switch', { name: 'Switch control' }),
            screen.getByRole('slider', { name: 'Slider control' }),
            screen.getByLabelText('Graph node'),
        ];

        for (const target of interactiveTargets) {
            target.focus();
            fireEvent.keyDown(target, { code: 'KeyR' });
        }

        expect(trainingMock.reset).not.toHaveBeenCalled();
    });

    it('pauses training with Space when already running and ignores shortcuts in inputs', () => {
        useTrainingStore.setState({ status: 'running' });
        render(
            <>
                <App />
                <input aria-label="Typing field" />
            </>,
        );

        fireEvent.keyDown(window, { code: 'Space' });
        expect(trainingMock.pause).toHaveBeenCalledTimes(1);

        const input = screen.getByRole('textbox', { name: 'Typing field' });
        fireEvent.keyDown(input, { code: 'KeyR' });
        expect(trainingMock.reset).not.toHaveBeenCalled();
    });

    it('renders forge-shell with status bar', () => {
        const { container } = render(<App />);
        expect(container.querySelector('.forge-shell')).toBeTruthy();
        expect(screen.getByRole('status', { name: 'Status bar' })).toBeInTheDocument();
    });

    it('shows the newest scientific model in the status bar after a forced pause evaluation', () => {
        useTrainingStore.setState({
            status: 'paused',
            evidenceGenerationId: 1,
            latestLiveSignal: liveSignal(2_450, 2_450, 163),
            latestEvaluation: fullEvaluation(2_500, 2_500, 166),
        });

        render(<App />);

        const statusBar = screen.getByRole('status', { name: 'Status bar' });
        expect(statusBar).toHaveTextContent('STEP 2,500');
        expect(statusBar).not.toHaveTextContent('STEP 2,450');
    });

    it('switches Build and Run views through the store', () => {
        render(<App />);

        act(() => {
            useLayoutStore.getState().setView('run');
        });
        expect(useLayoutStore.getState().view).toBe('run');

        act(() => {
            useLayoutStore.getState().setView('build');
        });
        expect(useLayoutStore.getState().view).toBe('build');
    });

    it('passes target hooks to Build/Run workspace panels', () => {
        useLayoutStore.setState({ view: 'build', phase: 'build' });

        const { container } = render(<App />);
        const targets = Array.from(container.querySelectorAll('[data-forge-panel-targets]'))
            .map((panel) => panel.getAttribute('data-forge-panel-targets'));

        expect(targets).toEqual(expect.arrayContaining([
            'data',
            'topology',
            'network',
            'features',
            'hyperparams',
            'transport',
        ]));
    });
});
