import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import type { ReactNode } from 'react';
import App from './App';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    encodeExperimentUrl,
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

function dispatchGlobalKeyDown(code: string, options: KeyboardEventInit = {}) {
    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code, ...options });
    window.dispatchEvent(event);
    return event;
}

async function experimentHashWithNoise(noise: number) {
    const result = await prepareExperimentDocument({
        ...DEFAULT_EXPERIMENT_DOCUMENT,
        recipe: {
            ...DEFAULT_EXPERIMENT_DOCUMENT.recipe,
            data: {
                ...DEFAULT_EXPERIMENT_DOCUMENT.recipe.data,
                noise,
            },
        },
    });
    if (!result.ok) throw new Error('experiment hash fixture did not prepare');
    return encodeExperimentUrl(result.value.document);
}

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
    Header: ({
        openSurface,
        onToggleSurface,
        advancedToolsOpen,
        onToggleAdvancedTools,
    }: {
        openSurface: 'presets' | 'lessons' | 'history' | null;
        onToggleSurface: (surface: 'history') => void;
        advancedToolsOpen: boolean;
        onToggleAdvancedTools: () => void;
    }) => (
        <header>
            <button
                type="button"
                aria-pressed={openSurface === 'history'}
                onClick={() => onToggleSurface('history')}
            >
                History
            </button>
            <button
                type="button"
                aria-expanded={advancedToolsOpen}
                onClick={onToggleAdvancedTools}
            >
                Advanced Tools
            </button>
        </header>
    ),
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
    ConfigurationContent: () => <div>Config</div>,
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
        window.history.replaceState(null, '', '/');
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
            preparation: { status: 'ready', requestId: 0, issues: [] },
        });

        useTrainingStore.setState({
            status: 'idle',
            pauseReason: null,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            workerError: null,
            evidenceGenerationId: 1,
            trainedRecipe: prepared.value.document.recipe,
            trainedRecipeSource: 'initialize',
        });

        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('loads a valid experiment hash changed after mount through the URL loader', async () => {
        render(<App />);

        window.history.replaceState(null, '', await experimentHashWithNoise(7));
        act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));

        await waitFor(() => expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'ready',
            prepared: { document: { recipe: { data: { noise: 7 } } } },
        }));
    });

    it('shows compatibility recovery when a changed hash is incompatible', async () => {
        render(<App />);

        window.history.replaceState(null, '', '#v=3&r=AAAA');
        act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));

        await waitFor(() => expect(usePlaygroundStore.getState().access.status).toBe('incompatible'));
        expect(screen.getByRole('main', { name: 'Experiment compatibility' })).toBeInTheDocument();

        window.history.replaceState(null, '', await experimentHashWithNoise(5));
        act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));
        await waitFor(() => expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'ready',
            prepared: { document: { recipe: { data: { noise: 5 } } } },
        }));
    });

    it('loads each valid hashchange while navigating between two experiments', async () => {
        render(<App />);
        const firstHash = await experimentHashWithNoise(3);
        const secondHash = await experimentHashWithNoise(9);

        window.history.replaceState(null, '', firstHash);
        act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));
        await waitFor(() => expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'ready',
            prepared: { document: { recipe: { data: { noise: 3 } } } },
        }));

        window.history.replaceState(null, '', secondHash);
        act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));
        await waitFor(() => expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'ready',
            prepared: { document: { recipe: { data: { noise: 9 } } } },
        }));
    });

    it('stops loading hash changes after App unmounts', async () => {
        const store = usePlaygroundStore.getState();
        expect(vi.isMockFunction(store.loadFromUrl)).toBe(false);
        const loadFromUrl = vi.spyOn(store, 'loadFromUrl');
        try {
            const { unmount } = render(<App />);
            unmount();

            window.history.replaceState(null, '', await experimentHashWithNoise(5));
            act(() => window.dispatchEvent(new HashChangeEvent('hashchange')));

            expect(loadFromUrl).not.toHaveBeenCalled();
        } finally {
            loadFromUrl.mockRestore();
        }
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

    it('contains worker-error focus, background, Escape, and global shortcuts until the error clears', async () => {
        const user = userEvent.setup();
        const { container } = render(<App />);
        const skipLink = screen.getByRole('link', { name: 'Skip to main content' });
        const historyTrigger = screen.getByRole('button', { name: 'History' });
        const advancedTrigger = screen.getByRole('button', { name: 'Advanced Tools' });
        const shell = container.querySelector('.forge-shell');
        expect(shell).not.toBeNull();

        await user.click(advancedTrigger);
        await user.click(historyTrigger);
        skipLink.focus();
        act(() => useTrainingStore.setState({
            workerError: 'Worker channel closed unexpectedly.',
        }));

        const dialog = screen.getByRole('alertdialog', { name: 'Worker connection lost' });
        expect(dialog).toHaveAccessibleDescription(
            'Worker channel closed unexpectedly. Refresh the page to restart the playground.',
        );
        expect(dialog).toHaveFocus();
        expect(shell).toHaveAttribute('inert');
        expect(shell).toHaveAttribute('aria-hidden', 'true');
        expect(shell).not.toContainElement(dialog);
        expect(document.body).toContainElement(dialog);
        expect(screen.getByRole('button', { name: 'Refresh page' })).toBeInTheDocument();

        await user.keyboard('{Escape}');
        expect(useTrainingStore.getState().workerError).toBe('Worker channel closed unexpectedly.');
        expect(historyTrigger).toHaveAttribute('aria-pressed', 'true');
        expect(advancedTrigger).toHaveAttribute('aria-expanded', 'true');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);

        for (const code of ['Space', 'ArrowRight', 'KeyR']) {
            dispatchGlobalKeyDown(code);
        }
        expect(trainingMock.play).not.toHaveBeenCalled();
        expect(trainingMock.pause).not.toHaveBeenCalled();
        expect(trainingMock.step).not.toHaveBeenCalled();
        expect(trainingMock.reset).not.toHaveBeenCalled();

        act(() => useTrainingStore.setState({ workerError: null }));
        await waitFor(() => expect(skipLink).toHaveFocus());
        expect(shell).not.toHaveAttribute('inert');
        expect(shell).not.toHaveAttribute('aria-hidden');
    });

    it('cancels each resolved global shortcut and dispatches its matching training command once', () => {
        render(<App />);

        const shortcuts = [
            { code: 'Space', command: trainingMock.play },
            { code: 'ArrowRight', command: trainingMock.step },
            { code: 'KeyR', command: trainingMock.reset },
        ];

        for (const { code, command } of shortcuts) {
            const event = dispatchGlobalKeyDown(code);
            expect(event.defaultPrevented).toBe(true);
            expect(command).toHaveBeenCalledTimes(1);
        }

        expect(trainingMock.play).toHaveBeenCalledTimes(1);
        expect(trainingMock.pause).not.toHaveBeenCalled();
        expect(trainingMock.step).toHaveBeenCalledTimes(1);
        expect(trainingMock.reset).toHaveBeenCalledTimes(1);
    });

    it('leaves modified and repeated training shortcuts for the browser without dispatching commands', () => {
        render(<App />);

        const modifiers: KeyboardEventInit[] = [
            { metaKey: true },
            { ctrlKey: true },
            { altKey: true },
            { shiftKey: true },
            { repeat: true },
        ];

        for (const code of ['Space', 'ArrowRight', 'KeyR']) {
            for (const modifier of modifiers) {
                const event = dispatchGlobalKeyDown(code, modifier);
                expect(event.defaultPrevented).toBe(false);
            }
        }

        expect(trainingMock.play).not.toHaveBeenCalled();
        expect(trainingMock.pause).not.toHaveBeenCalled();
        expect(trainingMock.step).not.toHaveBeenCalled();
        expect(trainingMock.reset).not.toHaveBeenCalled();
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
        const statusBar = screen.getByRole('group', { name: 'Status bar' });
        expect(statusBar).toBeInTheDocument();
        expect(statusBar.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
    });

    it('shows the newest scientific model in the status bar after a forced pause evaluation', () => {
        useTrainingStore.setState({
            status: 'paused',
            evidenceGenerationId: 1,
            latestLiveSignal: liveSignal(2_450, 2_450, 163),
            latestEvaluation: fullEvaluation(2_500, 2_500, 166),
        });

        render(<App />);

        const statusBar = screen.getByRole('group', { name: 'Status bar' });
        expect(statusBar).toHaveTextContent('STEP 2,500');
        expect(statusBar).not.toHaveTextContent('STEP 2,450');
        expect(statusBar.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();

        act(() => useTrainingStore.setState({
            latestEvaluation: fullEvaluation(2_600, 2_600, 173),
        }));
        expect(statusBar).toHaveTextContent('STEP 2,600');
        expect(statusBar.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
    });

    it('announces a complete configuration transaction driven through the real store', () => {
        render(<App />);
        const announcements = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });

        act(() => useTrainingStore.getState().beginConfigChange('data'));
        expect(announcements).toHaveTextContent('Generating data');

        act(() => useTrainingStore.getState().finishConfigChange());
        expect(announcements).toHaveTextContent('Data update complete');

        act(() => useTrainingStore.getState().beginConfigChange('network'));
        expect(announcements).toHaveTextContent('Initializing network');
        act(() => useTrainingStore.getState().failConfigChange('network failed'));
        expect(announcements).toHaveTextContent('Network error: network failed');
        expect(announcements).not.toHaveTextContent('Network update complete');
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
