import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
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
vi.mock('../components/visualization/DecisionBoundaryCanvas.tsx', () => ({
    DecisionBoundaryCanvas: () => <div data-decision-boundary-canvas>Mock Boundary</div>,
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
            pendingConfigSource: null, featuresConfigLoading: false, trainingConfigLoading: false, presetConfigLoading: false,
            dataConfigLoading: false,
            networkConfigLoading: false,
            configError: null,
            configErrorSource: null,
            workerError: null,
            pauseReason: null,
        });

        useLayoutStore.setState({
            destination: 'playground', workspaceTab: 'network', setupTab: 'dataset', resultsTab: 'boundary', inspectTab: 'trace',
            view: 'build',
            buildContextOpen: false,
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

    it.each(['setup', 'network', 'results', 'inspect'] as const)('has no axe violations in the %s workspace', async (workspaceTab) => {
        useLayoutStore.setState({ workspaceTab, lessonCueDismissed: true });
        const { container } = render(<App />);
        if (workspaceTab === 'inspect') await screen.findByText('Mock Inspection');
        expect((await axe(container)).violations).toEqual([]);
    });

    it('keeps header controls in logical keyboard order', async () => {
        const user = userEvent.setup(); render(<App />);
        for (const element of [
            screen.getByRole('link', {name:'Skip to main content'}),
            screen.getByRole('button', {name:'NN FORGE playground'}),
            screen.getByRole('button', {name:'Playground'}),
            screen.getByRole('button', {name:'Saved runs'}),
            screen.getByRole('combobox', {name:'Color theme'}),
            screen.getByRole('button', {name:'Lessons'}),
            screen.getByRole('button', {name:'Utilities'}),
        ]) { await user.tab(); expect(element).toHaveFocus(); }
    });

    it('opens utilities with keyboard navigation and restores focus after closing', async () => {
        const user = userEvent.setup(); render(<App />);
        const trigger = screen.getByRole('button', {name:'Utilities'});
        await user.click(trigger);
        const menu = screen.getByRole('menu');
        await user.click(within(menu).getByRole('menuitem', {name:'Guidance'}));
        const dialog = screen.getByRole('dialog', {name:'Guidance'});
        expect(dialog).toHaveFocus();
        await user.click(within(dialog).getByRole('button', {name:'Close Guidance'}));
        expect(trigger).toHaveFocus();
    });

    it('mounts only visible evidence and requests the same capabilities', async () => {
        const user = userEvent.setup(); render(<App />);
        expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(true);
        await user.click(screen.getByRole('tab', {name:'Results'}));
        await user.click(screen.getByRole('tab', {name:'Learning progress'}));
        await screen.findByText('Mock Loss Chart');
        expect(screen.queryByText('Mock Boundary')).not.toBeInTheDocument();
        expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(false);
        await user.click(screen.getByRole('tab', {name:'Inspect'}));
        await screen.findByText('Mock Inspection');
        expect(usePlaygroundStore.getState().demand).toMatchObject({needLayerStats:true,needActivationHistograms:false});
        act(() => useLayoutStore.getState().setInspectTab('activations'));
        expect(usePlaygroundStore.getState().demand.needActivationHistograms).toBe(true);
        await user.click(screen.getByRole('button', {name:'Saved runs'}));
        await screen.findByText('Mock Run History');
        expect(usePlaygroundStore.getState().demand).toMatchObject({needDecisionBoundary:false,needNeuronGrids:false,needLayerStats:false,needActivationHistograms:false,needConfusionMatrix:false});
    });

    it('offers first-visit lessons after hydration and persists dismissal without starting training', async () => {
        useExperimentMemoryStore.setState({hydrationStatus:'loading'});
        const user = userEvent.setup(); render(<App />);
        expect(screen.queryByRole('complementary', {name:'Getting started'})).not.toBeInTheDocument();
        act(() => useExperimentMemoryStore.setState({hydrationStatus:'ready'}));
        expect(screen.getByRole('complementary', {name:'Getting started'})).toBeInTheDocument();
        const access = usePlaygroundStore.getState().access;
        await user.click(screen.getByRole('button', {name:'Dismiss lesson suggestion'}));
        expect(useLayoutStore.getState().lessonCueDismissed).toBe(true);
        expect(usePlaygroundStore.getState().access).toBe(access);
        expect(trainingMock.play).not.toHaveBeenCalled();
        expect(trainingMock.reset).not.toHaveBeenCalled();
    });

    it('guards a shared draft on navigation and cancels all unsubmitted edits together', async () => {
        const user = userEvent.setup(); render(<App />);
        const access = usePlaygroundStore.getState().access;
        await user.click(screen.getByRole('tab', {name:'Setup'}));
        await user.clear(screen.getByLabelText('Samples')); await user.type(screen.getByLabelText('Samples'),'600');
        await user.click(screen.getByRole('button', {name:'Training'}));
        await user.click(screen.getByRole('tab', {name:'Network'}));
        const dialog = screen.getByRole('alertdialog', {name:'Apply your setup changes?'});
        await user.click(within(dialog).getByRole('button', {name:'Stay'}));
        expect(useLayoutStore.getState().workspaceTab).toBe('setup');
        await user.click(screen.getByRole('button', {name:'Data'}));
        expect(screen.getByLabelText('Samples')).toHaveValue('600');
        await user.click(screen.getByRole('tab', {name:'Network'}));
        await user.click(within(screen.getByRole('alertdialog')).getByRole('button', {name:'Discard changes'}));
        expect(useLayoutStore.getState().workspaceTab).toBe('network');
        expect(usePlaygroundStore.getState().access).toBe(access);
        expect(trainingMock.reset).not.toHaveBeenCalled();
    });

    it('mounts and requests only the focused mobile network region and restores desktop composition on resize', async () => {
        setViewportWidth(390);
        const user = userEvent.setup(); render(<App />);
        const access = usePlaygroundStore.getState().access;
        const regions = screen.getByRole('tablist', {name:'Network regions'});
        expect(screen.getByText('Mock Topology Graph')).toBeVisible();
        expect(screen.queryByText('Mock Boundary')).not.toBeInTheDocument();
        expect(usePlaygroundStore.getState().demand).toMatchObject({needNeuronGrids:true,needDecisionBoundary:false});
        await user.click(within(regions).getByRole('tab',{name:'Prediction'}));
        expect(screen.queryByText('Mock Topology Graph')).not.toBeInTheDocument();
        expect(screen.getByText('Mock Boundary')).toBeVisible();
        expect(usePlaygroundStore.getState().demand).toMatchObject({needNeuronGrids:false,needDecisionBoundary:true});
        await user.click(within(regions).getByRole('tab',{name:'Data'}));
        expect(screen.queryByText('Mock Boundary')).not.toBeInTheDocument();
        expect(usePlaygroundStore.getState().demand).toMatchObject({needNeuronGrids:false,needDecisionBoundary:false});
        act(() => setViewportWidth(1440));
        expect(screen.getByText('Mock Topology Graph')).toBeVisible();
        expect(screen.getByText('Mock Boundary')).toBeVisible();
        expect(usePlaygroundStore.getState().access).toBe(access);
        for (const command of Object.values(trainingMock)) expect(command).not.toHaveBeenCalled();
    });

    it('preserves all scientific identities, checkpoints, saved records and URL across themes, guidance and navigation', async () => {
        const user = userEvent.setup();
        const prepared = await makeAdvancedPrepared();
        const liveSignal = {...makeLiveSignal(), dataset:{...makeLiveSignal().dataset,datasetKey:prepared.identities.datasetKey}, objectiveKey:prepared.identities.objectiveKey};
        const latestEvaluation = {...makeEvaluation(), dataset:liveSignal.dataset,objectiveKey:prepared.identities.objectiveKey};
        const timeline = {checkpoints:[{id:11,step:20,epoch:2,trainDataLoss:.2,testDataLoss:.6,label:'Step20'}],maxCheckpoints:12,evictedCount:0,liveCheckpointId:11,restoredCheckpointId:11};
        const records = Object.freeze([{id:'saved-run-sentinel'}]) as never;
        usePlaygroundStore.setState({access:{status:'ready',prepared}});
        useTrainingStore.setState({status:'paused',pauseReason:'manual',trainedRecipe:prepared.document.recipe,trainedRecipeFingerprint:prepared.identities.recipeFingerprint,latestLiveSignal:liveSignal,latestEvaluation,checkpointTimeline:timeline});
        useExperimentMemoryStore.setState({records});
        useLayoutStore.setState({codeExportTab:'numpy',lessonCueDismissed:true});
        window.history.replaceState(null,'',encodeExperimentUrl(prepared.document));
        const hash = window.location.hash;
        render(<App />);
        for (const preference of ['dark','light','system']) {
            await user.selectOptions(screen.getByRole('combobox',{name:'Color theme'}),preference);
            for (const workspaceTab of ['setup','network','results','inspect'] as const) {
                act(() => useLayoutStore.getState().navigate('playground',workspaceTab));
                for (const guidance of ['beginner','explore','lab'] as const) act(() => useLayoutStore.getState().setAudienceMode(guidance));
                const state = useTrainingStore.getState();
                expect(state.latestLiveSignal).toBe(liveSignal); expect(state.latestEvaluation).toBe(latestEvaluation);
                expect(state.checkpointTimeline).toBe(timeline); expect(state.trainedRecipe).toBe(prepared.document.recipe);
                expect(state.status).toBe('paused'); expect(state.pauseReason).toBe('manual');
                expect(usePlaygroundStore.getState().access).toEqual({status:'ready',prepared});
                expect(useExperimentMemoryStore.getState().records).toBe(records);
                expect(useLayoutStore.getState().codeExportTab).toBe('numpy'); expect(window.location.hash).toBe(hash);
            }
        }
        for (const command of Object.values(trainingMock)) expect(command).not.toHaveBeenCalled();
    });
});
