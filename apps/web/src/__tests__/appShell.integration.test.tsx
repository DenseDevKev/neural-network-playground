import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import App from '../App.tsx';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { useExperimentMemoryStore } from '../store/experimentMemoryStore.ts';
import {
    encodeExperimentUrl,
    prepareExperimentDocument,
    type LiveTrainingSignal,
    type PairedEvaluation,
    type PreparedExperimentDocumentV2,
} from '@nn-playground/shared';

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
    PresetPanel: ({ onApplied }: { onApplied?: () => void }) => (
        <div>
            Mock Presets
            {onApplied && <button type="button" onClick={onApplied}>Apply mock preset</button>}
        </div>
    ),
}));
vi.mock('../components/controls/DataPanel.tsx', async () => {
    const { Tooltip } = await import('../components/common/Tooltip.tsx');
    return {
        DataPanel: () => (
            <div>
                Mock Data
                <Tooltip content="Nested data help">
                    <button
                        type="button"
                        onKeyDown={(event) => {
                            if (event.key !== 'Escape') return;
                            event.currentTarget.dataset.escapeReceived = 'true';
                            event.currentTarget.dataset.escapeDefaultPrevented = String(event.defaultPrevented);
                        }}
                    >
                        Nested tooltip trigger
                    </button>
                </Tooltip>
            </div>
        ),
    };
});
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

async function makeAdvancedPrepared(): Promise<PreparedExperimentDocumentV2> {
    const result = await prepareExperimentDocument({
        ...INITIAL_PREPARED.document,
        recipe: {
            ...INITIAL_PREPARED.document.recipe,
            training: {
                ...INITIAL_PREPARED.document.recipe.training,
                schedule: { kind: 'step', interval: 17, gamma: 0.63 },
                optimizer: { kind: 'sgd-momentum', momentum: 0.81 },
                gradientClipping: {
                    kind: 'global-norm',
                    maximumNorm: 2.5,
                    scope: 'total-objective-gradient',
                },
            },
        },
    });
    if (!result.ok) throw new Error('advanced app-shell fixture must prepare');
    return result.value;
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
            audienceMode: 'explore',
            advancedToolsOpen: false,
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
            activeLessonId: null,
            activeLessonStepIndex: null,
            lessonCueDismissed: false,
            hasStartedLesson: false,
        });
        useExperimentMemoryStore.setState({
            hydrationStatus: 'ready',
            records: Object.freeze([]),
            rejectedRecords: Object.freeze([]),
            incompatibleEnvelope: null,
            legacyRaw: null,
            legacyNoticeDismissed: false,
            persistenceError: null,
            pendingSave: null,
        });
    });

    it.each([
        ['beginner', 'build', false],
        ['beginner', 'build', true],
        ['beginner', 'run', false],
        ['beginner', 'run', true],
        ['explore', 'build', false],
        ['explore', 'build', true],
        ['explore', 'run', false],
        ['explore', 'run', true],
        ['lab', 'build', false],
        ['lab', 'build', true],
        ['lab', 'run', false],
        ['lab', 'run', true],
    ] as const)(
        'has no axe violations in the %s %s shell with Advanced Tools %s',
        async (audienceMode, view, advancedToolsOpen) => {
            useLayoutStore.setState({
                audienceMode,
                advancedToolsOpen,
                view,
                phase: view,
                activeRecipeSection: 'data',
                activeTabLeft: 'data',
                activeEvidenceView: 'boundary',
                activeTabRight: 'boundary',
            });

            const { container } = render(<App />);
            if (view === 'run') {
                expect(screen.getByRole('tabpanel', { name: 'Boundary' })).toBeInTheDocument();
            } else if (advancedToolsOpen) {
                await screen.findByText('Mock Config Panel');
            } else {
                expect(screen.getByRole('region', { name: 'Workspace tools' })).toBeInTheDocument();
            }

            const results = await axe(container);
            expect(
                results.violations,
                results.violations
                    .map((violation) => `${violation.id}: ${violation.nodes.map((node) => node.html).join(' | ')}`)
                    .join('\n'),
            ).toHaveLength(0);
        },
    );

    it('keeps the global shell controls in a logical keyboard focus order', async () => {
        const user = userEvent.setup();
        render(<App />);

        const header = within(screen.getByRole('banner'));
        const expectedOrder = [
            screen.getByRole('link', { name: 'Skip to main content' }),
            header.getByRole('button', { name: 'build' }),
            header.getByRole('button', { name: 'run' }),
            header.getByRole('button', { name: 'About workspace views' }),
            header.getByRole('combobox', { name: 'Audience mode' }),
            header.getByRole('button', { name: 'Presets' }),
            header.getByRole('button', { name: 'Lessons' }),
            header.getByRole('button', { name: 'History' }),
            header.getByRole('button', { name: 'Advanced Tools' }),
            header.getByRole('button', { name: 'Start training' }),
        ];

        for (const control of expectedOrder) {
            await user.tab();
            expect(control).toHaveFocus();
        }

        await user.tab({ shift: true });
        expect(header.getByRole('button', { name: 'Advanced Tools' })).toHaveFocus();
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
        expect(screen.getByRole('button', { name: 'Advanced Tools' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'More' })).not.toBeInTheDocument();
    });

    it('renders Build as recipe, topology, features, and hyperparameters without permanent drawers', () => {
        render(<App />);

        expect(screen.getByRole('region', { name: 'Recipe summary' })).toBeInTheDocument();
        expect(screen.getByText('Mock Data')).toBeInTheDocument();
        expect(screen.getByText('Mock Topology Graph')).toBeInTheDocument();
        expect(screen.getByText('Mock Network Config')).toBeInTheDocument();
        expect(screen.getByText('Mock Features')).toBeInTheDocument();
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.queryByText('Mock Config Panel')).not.toBeInTheDocument();
        expect(screen.queryByText('Mock Presets')).not.toBeInTheDocument();
        expect(screen.queryByText('Mock Run History')).not.toBeInTheDocument();
    });

    it('opens Presets, Lessons, and History as drawer surfaces', async () => {
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
    });

    it('waits for accepted-run hydration, then opens Lessons without mutating app state', async () => {
        const user = userEvent.setup();
        useExperimentMemoryStore.setState({ hydrationStatus: 'loading', records: Object.freeze([]) });
        window.history.replaceState(null, '', '#first-visit-cue');

        const preparedRef = INITIAL_PREPARED;
        const recipeRef = INITIAL_PREPARED.document.recipe;
        const trainingBefore = useTrainingStore.getState();
        const audienceBefore = useLayoutStore.getState().audienceMode;
        const viewBefore = useLayoutStore.getState().view;
        const hashBefore = window.location.hash;

        render(<App />);
        expect(screen.queryByRole('button', { name: 'Start a 3-minute lesson' }))
            .not.toBeInTheDocument();

        act(() => {
            useExperimentMemoryStore.setState({
                hydrationStatus: 'ready',
                records: Object.freeze([]),
            });
        });
        const openLesson = await screen.findByRole('button', { name: 'Start a 3-minute lesson' });
        await user.click(openLesson);

        expect(screen.getByRole('dialog', { name: 'Lessons' })).toBeInTheDocument();
        const playground = usePlaygroundStore.getState();
        expect(playground.access.status).toBe('ready');
        if (playground.access.status === 'ready') {
            expect(playground.access.prepared).toBe(preparedRef);
            expect(playground.access.prepared.document.recipe).toBe(recipeRef);
        }
        expect(useTrainingStore.getState()).toBe(trainingBefore);
        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: audienceBefore,
            view: viewBefore,
            lessonCueDismissed: false,
            hasStartedLesson: false,
        });
        expect(window.location.hash).toBe(hashBefore);
        expect(trainingMock.play).not.toHaveBeenCalled();
        expect(trainingMock.pause).not.toHaveBeenCalled();
        expect(trainingMock.step).not.toHaveBeenCalled();
        expect(trainingMock.reset).not.toHaveBeenCalled();
        expect(trainingMock.restoreCheckpoint).not.toHaveBeenCalled();
    });

    it('dismisses and persists the cue without mutating recipe, training, workspace, or URL', async () => {
        const user = userEvent.setup();
        window.history.replaceState(null, '', '#dismiss-first-visit-cue');
        const preparedRef = INITIAL_PREPARED;
        const recipeRef = INITIAL_PREPARED.document.recipe;
        const trainingBefore = useTrainingStore.getState();
        const audienceBefore = useLayoutStore.getState().audienceMode;
        const viewBefore = useLayoutStore.getState().view;
        const hashBefore = window.location.hash;

        const mounted = render(<App />);
        await user.click(await screen.findByRole('button', { name: 'Dismiss lesson suggestion' }));

        expect(screen.queryByRole('region', { name: 'Getting started' }))
            .not.toBeInTheDocument();
        const playground = usePlaygroundStore.getState();
        expect(playground.access.status).toBe('ready');
        if (playground.access.status === 'ready') {
            expect(playground.access.prepared).toBe(preparedRef);
            expect(playground.access.prepared.document.recipe).toBe(recipeRef);
        }
        expect(useTrainingStore.getState()).toBe(trainingBefore);
        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: audienceBefore,
            view: viewBefore,
            lessonCueDismissed: true,
            hasStartedLesson: false,
        });
        expect(window.location.hash).toBe(hashBefore);
        const stored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(stored.state?.lessonCueDismissed).toBe(true);

        mounted.unmount();
        render(<App />);
        expect(screen.queryByRole('button', { name: 'Start a 3-minute lesson' }))
            .not.toBeInTheDocument();
    });

    it('does not show the cue to an established user with an accepted saved run', () => {
        useExperimentMemoryStore.setState({
            hydrationStatus: 'ready',
            records: Object.freeze([{ id: 'accepted-run' }]) as never,
        });

        render(<App />);

        expect(screen.queryByRole('region', { name: 'Getting started' }))
            .not.toBeInTheDocument();
    });

    it('opens Advanced Tools without requesting diagnostics and collapses hidden Build targets atomically', async () => {
        const user = userEvent.setup();
        render(<App />);

        const trigger = screen.getByRole('button', { name: 'Advanced Tools' });
        expect(trigger).toHaveAttribute('aria-expanded', 'false');

        await user.click(trigger);

        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
        expect(await screen.findByText('Mock Config Panel')).toBeInTheDocument();
        expect(screen.getByText(/advanced tools are visible/i)).toBeInTheDocument();
        expect(usePlaygroundStore.getState().demand).toMatchObject({
            needLayerStats: false,
            needActivationHistograms: false,
            needConfusionMatrix: false,
        });

        act(() => {
            useLayoutStore.getState().setActiveRecipeSection('config');
        });
        await user.click(trigger);

        expect(trigger).toHaveFocus();
        expect(useLayoutStore.getState()).toMatchObject({
            advancedToolsOpen: false,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
        });
        expect(screen.queryByText('Mock Config Panel')).not.toBeInTheDocument();
    });

    it('restores drawer focus and closes a drawer before Advanced Tools on Escape', async () => {
        const user = userEvent.setup();
        render(<App />);

        const advancedTrigger = screen.getByRole('button', { name: 'Advanced Tools' });
        await user.click(advancedTrigger);
        const historyTrigger = screen.getByRole('button', { name: 'History' });
        await user.click(historyTrigger);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Close History' })).toHaveFocus();
        });
        await user.keyboard('{Escape}');

        expect(screen.queryByRole('dialog', { name: 'History' })).not.toBeInTheDocument();
        expect(historyTrigger).toHaveFocus();
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);

        await user.keyboard('{Escape}');

        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
        expect(advancedTrigger).toHaveFocus();
    });

    it('dismisses a visible nested tooltip before collapsing Advanced Tools on Escape', async () => {
        const user = userEvent.setup();
        render(<App />);

        const advancedTrigger = screen.getByRole('button', { name: 'Advanced Tools' });
        await user.click(advancedTrigger);
        const nestedTrigger = screen.getByRole('button', { name: 'Nested tooltip trigger' });
        await user.click(nestedTrigger);
        const tooltip = screen.getByRole('tooltip', { name: 'Nested data help' });
        expect(tooltip).toHaveStyle({ visibility: 'visible' });

        await user.keyboard('{Escape}');

        expect(nestedTrigger).toHaveAttribute('data-escape-received', 'true');
        expect(nestedTrigger).toHaveAttribute('data-escape-default-prevented', 'false');
        expect(tooltip).toHaveStyle({ visibility: 'hidden' });
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);

        await user.keyboard('{Escape}');

        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
        expect(advancedTrigger).toHaveFocus();
    });

    it('restores drawer-trigger focus after applying a preset', async () => {
        const user = userEvent.setup();
        render(<App />);

        const presetTrigger = screen.getByRole('button', { name: 'Presets' });
        await user.click(presetTrigger);
        await user.click(screen.getByRole('button', { name: 'Apply mock preset' }));

        expect(screen.queryByRole('dialog', { name: 'Presets' })).not.toBeInTheDocument();
        expect(presetTrigger).toHaveFocus();
    });

    it('does not steal focus when persisted Advanced Tools state is already open', () => {
        useLayoutStore.setState({ advancedToolsOpen: true });
        const outside = document.createElement('button');
        outside.textContent = 'Outside focus';
        document.body.append(outside);
        try {
            outside.focus();

            render(<App />);

            expect(outside).toHaveFocus();
            expect(screen.getByRole('button', { name: 'Advanced Tools' }))
                .toHaveAttribute('aria-expanded', 'true');
        } finally {
            outside.remove();
        }
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

    it('renders the same visible evidence view that drives demand for a hidden target', async () => {
        useLayoutStore.setState({
            view: 'run',
            phase: 'run',
            audienceMode: 'beginner',
            advancedToolsOpen: false,
            activeEvidenceView: 'inspection',
            activeTabRight: 'inspection',
        });

        render(<App />);

        expect(screen.getByText('Mock Boundary')).toBeInTheDocument();
        expect(screen.queryByText('Mock Inspection')).not.toBeInTheDocument();
        expect(screen.queryByRole('tab', { name: 'Inspect' })).not.toBeInTheDocument();
        await waitFor(() => {
            expect(usePlaygroundStore.getState().demand).toMatchObject({
                needDecisionBoundary: true,
                needLayerStats: false,
                needActivationHistograms: false,
            });
        });
    });

    it('starts a lesson from the lesson menu and focuses the Build/Run target section', async () => {
        const user = userEvent.setup();
        render(<App />);

        await user.click(screen.getByRole('button', { name: 'Lessons' }));
        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Guided lesson' }),
            'lesson-feature-engineering-circle',
        );
        await user.click(screen.getByRole('button', { name: 'Start lesson and reset' }));

        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('features');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
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
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /run/i }));
        await user.click(screen.getByRole('tab', { name: 'Loss' }));
        await user.click(screen.getByRole('button', { name: 'Read the loss spike' }));

        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(screen.getByText('Mock Loss Chart')).toBeInTheDocument();
    });

    it('cycles every profile and disclosure state without mutating paused scientific state', async () => {
        const user = userEvent.setup();
        const prepared = await makeAdvancedPrepared();
        const liveSignal: LiveTrainingSignal = {
            ...makeLiveSignal(20),
            dataset: {
                ...makeLiveSignal(20).dataset,
                datasetKey: prepared.identities.datasetKey,
            },
            objectiveKey: prepared.identities.objectiveKey,
        };
        const latestEvaluation: PairedEvaluation = {
            ...makeEvaluation(20),
            dataset: liveSignal.dataset,
            objectiveKey: prepared.identities.objectiveKey,
        };
        const checkpointTimeline = {
            checkpoints: [{
                id: 11,
                step: 20,
                epoch: 2,
                trainDataLoss: 0.2,
                testDataLoss: 0.6,
                label: 'Step 20',
            }],
            maxCheckpoints: 12,
            evictedCount: 0,
            liveCheckpointId: 11,
            restoredCheckpointId: 11,
        };
        const savedRecords = Object.freeze([{ id: 'saved-run-sentinel' }]) as never;

        usePlaygroundStore.setState({
            access: { status: 'ready', prepared },
            preparation: { status: 'ready', requestId: 0, issues: [] },
        });
        useTrainingStore.setState({
            status: 'paused',
            pauseReason: 'manual',
            trainedRecipe: prepared.document.recipe,
            trainedRecipeFingerprint: prepared.identities.recipeFingerprint,
            trainedRecipeRecordedAt: 1,
            trainedRecipeSource: 'initialize',
            latestLiveSignal: liveSignal,
            latestEvaluation,
            checkpointTimeline,
        });
        useLayoutStore.setState({
            view: 'build',
            phase: 'build',
            audienceMode: 'beginner',
            advancedToolsOpen: false,
            activeRecipeSection: 'data',
            activeTabLeft: 'data',
            activeEvidenceView: 'boundary',
            activeTabRight: 'boundary',
            codeExportTab: 'numpy',
        });
        useExperimentMemoryStore.setState({ records: savedRecords });
        window.history.replaceState(null, '', encodeExperimentUrl(prepared.document));

        const preparedRef = prepared;
        const recipeRef = prepared.document.recipe;
        const liveRef = liveSignal;
        const evaluationRef = latestEvaluation;
        const timelineRef = checkpointTimeline;
        const recordsRef = savedRecords;
        const hash = window.location.hash;

        const assertScientificState = () => {
            const playground = usePlaygroundStore.getState();
            expect(playground.access.status).toBe('ready');
            if (playground.access.status === 'ready') {
                expect(playground.access.prepared).toBe(preparedRef);
                expect(playground.access.prepared.document.recipe).toBe(recipeRef);
            }
            const training = useTrainingStore.getState();
            expect(training.status).toBe('paused');
            expect(training.pauseReason).toBe('manual');
            expect(training.latestLiveSignal).toBe(liveRef);
            expect(training.latestEvaluation).toBe(evaluationRef);
            expect(training.latestLiveSignal?.model.step).toBe(20);
            expect(training.trainedRecipe).toBe(recipeRef);
            expect(training.trainedRecipeFingerprint).toBe(prepared.identities.recipeFingerprint);
            expect(training.checkpointTimeline).toBe(timelineRef);
            expect(useExperimentMemoryStore.getState().records).toBe(recordsRef);
            expect(useExperimentMemoryStore.getState().records).toHaveLength(1);
            expect(useLayoutStore.getState().codeExportTab).toBe('numpy');
            expect(window.location.hash).toBe(hash);
            expect(trainingMock.play).not.toHaveBeenCalled();
            expect(trainingMock.pause).not.toHaveBeenCalled();
            expect(trainingMock.step).not.toHaveBeenCalled();
            expect(trainingMock.reset).not.toHaveBeenCalled();
            expect(trainingMock.restoreCheckpoint).not.toHaveBeenCalled();
        };

        render(<App />);
        expect(screen.getByRole('note', { name: 'Advanced settings active' }))
            .toBeInTheDocument();
        assertScientificState();

        const disclosure = screen.getByRole('button', { name: 'Advanced Tools' });
        await user.click(disclosure);
        expect(screen.getByText('Mock Hyperparameters')).toBeInTheDocument();
        expect(screen.queryByRole('note', { name: 'Advanced settings active' }))
            .not.toBeInTheDocument();
        assertScientificState();
        await user.click(disclosure);
        assertScientificState();

        const mode = screen.getByRole('combobox', { name: 'Audience mode' });
        await user.selectOptions(mode, 'explore');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
        expect(screen.queryByRole('note', { name: 'Advanced settings active' }))
            .not.toBeInTheDocument();
        assertScientificState();
        await user.click(disclosure);
        assertScientificState();
        await user.click(disclosure);
        assertScientificState();

        await user.selectOptions(mode, 'lab');
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(true);
        assertScientificState();
        await user.click(disclosure);
        expect(useLayoutStore.getState().advancedToolsOpen).toBe(false);
        assertScientificState();
        await user.click(disclosure);
        assertScientificState();
    });
});
