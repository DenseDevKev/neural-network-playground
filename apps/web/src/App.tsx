// App owns model lifetime, recipe drafts and visualization demand across navigation.
import { lazy, Suspense, useEffect, useRef, useCallback, useState } from 'react';
import { DEFAULT_DEMAND } from '@nn-playground/shared';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useExperimentMemoryStore } from './store/experimentMemoryStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useThemeEffect } from './store/theme.ts';
import { useSaveCurrentRun } from './hooks/useSaveCurrentRun.ts';
import { useTraining } from './hooks/useTraining.ts';
import { useRecipeDraft } from './hooks/useRecipeDraft.ts';
import { useExperimentMemoryStorageSync } from './hooks/useExperimentMemoryStorageSync.ts';
import { useNetworkSelectionController } from './components/visualization/useNetworkSelectionController.ts';
import { useDecisionBoundaryController } from './components/visualization/useDecisionBoundaryController.ts';
import { DecisionBoundaryCanvas } from './components/visualization/DecisionBoundaryCanvas.tsx';
import { BoundaryEvidencePanel } from './components/visualization/BoundaryEvidencePanel.tsx';
import { AtelierHeader } from './components/atelier/AtelierHeader.tsx';
import { AtelierNetwork } from './components/atelier/AtelierNetwork.tsx';
import { AtelierTransport, EvidenceMetrics } from './components/atelier/AtelierTransport.tsx';
import { SetupEditor } from './components/atelier/setup/SetupEditor.tsx';
import { CheckpointPanel } from './components/atelier/CheckpointPanel.tsx';
import { Dialog, Tabs } from './components/atelier/ui.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { CompatibilityState } from './components/common/CompatibilityState.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { CurrentRunCard } from './components/controls/CurrentRunCard.tsx';
import type { LessonTarget } from './lessons/lessonRegistry.ts';
import type { Destination, WorkspaceTab, SetupTab, ResultsTab, UtilitySurface } from './productShell/atelierTypes.ts';
import { resolveTrainingShortcut, TRAINING_SHORTCUTS } from './shortcuts/trainingShortcuts.ts';

const GuidedLessonPanel = lazy(() => import('./components/EducationContent.ts').then((module) => ({ default: module.GuidedLessonPanel })));
const TrainingExplanationPanel = lazy(() => import('./components/EducationContent.ts').then((module) => ({ default: module.TrainingExplanationPanel })));
const RunHistoryPanel = lazy(() => import('./components/WorkspaceUtilities.ts').then((module) => ({ default: module.RunHistoryPanel })));
const ConfigPanel = lazy(() => import('./components/WorkspaceUtilities.ts').then((module) => ({ default: module.ConfigPanel })));
const CodeExportPanel = lazy(() => import('./components/controls/CodeExportPanel.tsx').then((module) => ({ default: module.CodeExportPanel })));
const InspectionPanel = lazy(() => import('./components/controls/InspectionPanel.tsx').then((module) => ({ default: module.InspectionPanel })));
const LossChart = lazy(() => import('./components/visualization/LossChart.tsx').then((module) => ({ default: module.LossChart })));
const ConfusionMatrix = lazy(() => import('./components/visualization/ConfusionMatrix.tsx').then((module) => ({ default: module.ConfusionMatrix })));
const WORKSPACE_TABS: { id: WorkspaceTab; label: string }[] = [{ id:'setup',label:'Setup' },{ id:'network',label:'Network' },{ id:'results',label:'Results' },{ id:'inspect',label:'Inspect' }];
const RESULTS_TABS: { id: ResultsTab; label: string }[] = [{ id:'boundary',label:'Prediction' },{ id:'learning',label:'Learning progress' },{ id:'errors',label:'Errors & confusion' }];
const loading = <p role="status">Loading workspace…</p>;

export default function App() {
    useThemeEffect();
    useExperimentMemoryStorageSync();
    const access = usePlaygroundStore((state) => state.access);
    const startFresh = usePlaygroundStore((state) => state.startFresh);
    useEffect(() => {
        const loadLocation = () => { void usePlaygroundStore.getState().loadFromUrl(); };
        window.addEventListener('hashchange', loadLocation);
        return () => window.removeEventListener('hashchange', loadLocation);
    }, []);
    const recoverWithDefault = useCallback(async () => {
        const result = await startFresh();
        if (!result.ok) throw new Error(result.issues.map((issue) => issue.message).join(' '));
    }, [startFresh]);
    if (access.status === 'incompatible') return <CompatibilityState access={access} onStartFresh={recoverWithDefault} />;
    return <AtelierShell />;
}

function AtelierShell() {
    const training = useTraining();
    const saveController = useSaveCurrentRun();
    const draft = useRecipeDraft();
    const boundary = useDecisionBoundaryController();
    const selection = useNetworkSelectionController();
    const destination = useLayoutStore((s) => s.destination);
    const workspaceTab = useLayoutStore((s) => s.workspaceTab);
    const setupTab = useLayoutStore((s) => s.setupTab);
    const resultsTab = useLayoutStore((s) => s.resultsTab);
    const inspectTab = useLayoutStore((s) => s.inspectTab);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const activeLessonId = useLayoutStore((s) => s.activeLessonId);
    const cueDismissed = useLayoutStore((s) => s.lessonCueDismissed);
    const hasStartedLesson = useLayoutStore((s) => s.hasStartedLesson);
    const recordsReady = useExperimentMemoryStore((s) => s.hydrationStatus === 'ready');
    const hasRecords = useExperimentMemoryStore((s) => s.records.length > 0);
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const evidenceGenerationId = useTrainingStore((s) => s.evidenceGenerationId);
    const trainedRecipe = useTrainingStore((s) => s.trainedRecipe);
    const trainedRecipeSource = useTrainingStore((s) => s.trainedRecipeSource);
    const prepared = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared : null);
    const [utility, setUtility] = useState<UtilitySurface>(null);
    const [pendingNavigation, setPendingNavigation] = useState<(() => void) | null>(null);
    const [savedVisited, setSavedVisited] = useState(destination === 'saved-runs');
    const [lessonsVisited, setLessonsVisited] = useState(destination === 'lessons' || activeLessonId !== null);
    const [lessonHighlight, setLessonHighlight] = useState<LessonTarget | null>(null);
    const [exportTab, setExportTab] = useState<'setup' | 'code'>('setup');
    const exportRequest = useLayoutStore((state) => state.exportRequest);
    const backgroundRef = useRef<HTMLDivElement>(null);
    const trainingRef = useRef(training);
    useEffect(() => { trainingRef.current = training; }, [training]);
    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const guard = (action: () => void) => {
        if (draft.dirty) { setUtility(null); setPendingNavigation(() => action); }
        else action();
    };
    const guardRef = useRef(guard);
    guardRef.current = guard;
    const requestGuardedNavigation = useCallback((action: () => void) => guardRef.current(action), []);
    useEffect(() => {
        if (!exportRequest) return;
        useLayoutStore.getState().clearExportRequest();
        requestGuardedNavigation(() => { setExportTab(exportRequest.mode); setUtility('exports'); });
    }, [exportRequest, requestGuardedNavigation]);
    const navigate = (next: Destination, tab?: WorkspaceTab) => {
        if (next === destination && (!tab || tab === workspaceTab)) return;
        const action = () => { setUtility(null); useLayoutStore.getState().navigate(next,tab); };
        if (destination === 'playground' && workspaceTab === 'setup') requestGuardedNavigation(action); else action();
    };
    const openSetup = (tab: SetupTab) => { setUtility(null); useLayoutStore.getState().openSetup(tab); };
    const openUtility = (surface: UtilitySurface) => {
        const action = () => setUtility(surface);
        if (surface === 'exports') guard(action); else action();
    };
    useEffect(() => { if (destination === 'saved-runs') setSavedVisited(true); if (destination === 'lessons') setLessonsVisited(true); }, [destination]);

    const regression = prepared?.document.recipe.task.kind === 'regression';
    const effectiveResultsTab = regression && resultsTab === 'errors' ? 'boundary' : resultsTab;

    // Only mounted evidence demands expensive worker projections. This is the sole writer.
    useEffect(() => {
        const playground = destination === 'playground';
        const network = playground && workspaceTab === 'network';
        const inspect = playground && workspaceTab === 'inspect';
        const results = playground && workspaceTab === 'results';
        const next = { ...DEFAULT_DEMAND,
            needDecisionBoundary: network || (results && effectiveResultsTab === 'boundary'),
            needNeuronGrids: network,
            needLayerStats: inspect,
            needActivationHistograms: inspect && inspectTab === 'activations',
            needConfusionMatrix: results && effectiveResultsTab === 'errors',
        };
        const store = usePlaygroundStore.getState();
        if (Object.entries(next).some(([key,value]) => store.demand[key as keyof typeof next] !== value)) store.setDemand(next);
    }, [destination,workspaceTab,effectiveResultsTab,inspectTab]);

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (workerError || event.isComposing || document.querySelector('dialog[open], [role="menu"]')) return;
            const action = resolveTrainingShortcut(event);
            if (!action) return;
            event.preventDefault();
            const current = useTrainingStore.getState();
            if (action === 'play-pause') { if (current.status === 'running') trainingRef.current.pause(); else trainingRef.current.play(); }
            else if (action === 'step') { if (current.status !== 'running') trainingRef.current.step(); }
            else trainingRef.current.reset();
        };
        window.addEventListener('keydown',handler);
        return () => window.removeEventListener('keydown',handler);
    }, [workerError]);

    useEffect(() => {
        if (pendingNavigation && !draft.dirty && !draft.submitted && !draft.error) {
            const action = pendingNavigation; setPendingNavigation(null); action();
        }
    }, [pendingNavigation, draft.dirty, draft.submitted, draft.error]);

    const dataset = prepared?.document.recipe.task.dataset ?? 'Experiment';
    const heading = destination === 'saved-runs' ? 'Saved runs' : destination === 'lessons' ? 'Learn by experimenting' : `${dataset.charAt(0).toUpperCase()}${dataset.slice(1)} experiment`;
    const description = destination === 'saved-runs' ? 'Keep the evidence. Compare what changed.' : destination === 'lessons' ? 'Small experiments. Ideas that become visible.' : 'Shape a network. Watch it learn. Understand what changes.';
    return <div className="atelier" ref={backgroundRef} data-lesson-highlight={lessonHighlight ?? undefined}>
        <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to main content</a>
        <AccessibilityAnnouncer status={status} pauseReason={pauseReason} workerError={workerError} pendingConfigSource={pendingConfigSource} configError={configError} configErrorSource={configErrorSource} evidenceGenerationId={evidenceGenerationId} trainedRecipe={trainedRecipe} trainedRecipeSource={trainedRecipeSource} />
        <AtelierHeader destination={destination} onNavigate={(next) => navigate(next)} onUtility={openUtility} />
        <main data-guided={activeLessonId && destination === 'playground' ? 'true' : undefined} id="main-content" tabIndex={-1} className="atelier-main" aria-label="Neural network playground workspace">
            <div className="atelier-heading"><div><h1>{heading}</h1><p>{description}</p></div>{destination === 'playground' && <div className="atelier-heading-actions"><button type="button" onClick={() => openUtility('exports')}>Share setup</button><button type="button" onClick={() => navigate('saved-runs')}>Save run</button></div>}</div>
            {configError && <div className="atelier-notice" role="alert"><p>{configError}</p><button type="button" onClick={() => useTrainingStore.getState().retryConfigSync()}>Retry configuration</button></div>}
            {destination === 'playground' && <>
                <Tabs panelPrefix="workspace" label="Experiment workspace" items={WORKSPACE_TABS} value={workspaceTab} onChange={(tab) => navigate('playground',tab)} />
                {recordsReady && !hasRecords && !cueDismissed && !hasStartedLesson && !activeLessonId && <aside className="atelier-notice" aria-label="Getting started"><p>New to neural networks? Start with one neuron and build your intuition.</p><button type="button" onClick={() => navigate('lessons')}>Explore lessons</button><button type="button" aria-label="Dismiss lesson suggestion" onClick={() => useLayoutStore.getState().dismissLessonCue()}>Not now</button></aside>}
                <div className="atelier-tab-content" role="tabpanel" id={`workspace-${workspaceTab}`} aria-labelledby={`workspace-tab-${workspaceTab}`}>
                    <ErrorBoundary key={workspaceTab} title="Workspace unavailable" description="This view could not render. Your experiment remains available in the other views.">
                    {workspaceTab === 'setup' && <SetupEditor controller={draft} tab={setupTab} onTabChange={(tab) => useLayoutStore.getState().setSetupTab(tab)} />}
                    {workspaceTab === 'network' && <><AtelierNetwork boundary={boundary} selection={selection} onSetup={openSetup} onResults={() => { useLayoutStore.getState().setResultsTab('boundary'); navigate('playground','results'); }} /><EvidenceMetrics /></>}
                    {workspaceTab === 'results' && <section className="atelier-evidence">
                        <Tabs label="Results views" items={regression ? RESULTS_TABS.filter((tab) => tab.id !== 'errors') : RESULTS_TABS} value={effectiveResultsTab} onChange={(tab) => useLayoutStore.getState().setResultsTab(tab)} />
                        <Suspense fallback={loading}>
                            {effectiveResultsTab === 'boundary' && <div className="atelier-results-boundary"><DecisionBoundaryCanvas model={boundary.model} /><BoundaryEvidencePanel controller={boundary} /></div>}
                            {effectiveResultsTab === 'learning' && <><LossChart /><EvidenceMetrics /><TrainingExplanationPanel /><CurrentRunCard /></>}
                            {effectiveResultsTab === 'errors' && <ConfusionMatrix />}
                        </Suspense>
                    </section>}
                    {workspaceTab === 'inspect' && <section className="atelier-evidence"><Suspense fallback={loading}><InspectionPanel onPause={training.pause} /></Suspense></section>}
                    </ErrorBoundary>
                </div>
            </>}
            {savedVisited && <div hidden={destination !== 'saved-runs'}><Suspense fallback={loading}><RunHistoryPanel saveController={saveController} /></Suspense></div>}
            {lessonsVisited && <div className="atelier-lesson-host" hidden={destination !== 'lessons' && !activeLessonId}><Suspense fallback={loading}><GuidedLessonPanel onNavigate={requestGuardedNavigation} onReset={stableReset} onHighlightChange={setLessonHighlight} /></Suspense></div>}
            <AtelierTransport training={training} onCheckpoints={() => openUtility('checkpoints')} />
        </main>
        <footer className="atelier-footer"><span>NN·FORGE · Experiments stay in your browser</span><span>Data → network → prediction</span></footer>
        {!workerError && !pendingNavigation && utility && <Dialog title={utility === 'exports' ? 'Export / import' : utility === 'checkpoints' ? 'Session checkpoints' : utility === 'preferences' ? 'Guidance' : 'Shortcuts & help'} onClose={() => setUtility(null)}>
            {utility === 'checkpoints' && <CheckpointPanel training={training} />}
            {utility === 'exports' && <><Tabs<'setup' | 'code'> label="Export type" items={[{id:'setup',label:'Setup & sharing'},{id:'code',label:'Code'}]} value={exportTab} onChange={setExportTab} /><Suspense fallback={loading}>{exportTab === 'setup' ? <ConfigPanel onReset={stableReset} /> : <CodeExportPanel />}</Suspense></>}
            {utility === 'preferences' && <><p>Choose how much explanation appears alongside the controls. Every feature remains available.</p><label htmlFor="atelier-guidance">Explanation density</label><select id="atelier-guidance" value={audienceMode} onChange={(event) => useLayoutStore.getState().setAudienceMode(event.target.value as 'beginner'|'explore'|'lab')}><option value="beginner">More</option><option value="explore">Standard</option><option value="lab">Compact</option></select></>}
            {utility === 'help' && <div className="atelier-help"><p>Build a dataset and network in Setup, apply the recipe, then train and inspect the results. Training continues when you change views.</p><dl>{TRAINING_SHORTCUTS.map((shortcut) => <div key={shortcut.code}><dt><kbd>{shortcut.label}</kbd></dt><dd>{shortcut.description}</dd></div>)}</dl><p>Shortcuts are inactive while editing fields or using a menu or dialog. Step is available while paused.</p><p>Saved runs hold evaluated evidence. Applying a saved recipe starts a fresh model; session checkpoints restore parameters and optimizer state.</p></div>}
        </Dialog>}
        {!workerError && pendingNavigation && <Dialog title="Apply your setup changes?" onClose={() => { if (!draft.busy) setPendingNavigation(null); }} alert>
            <p>You have one shared draft across Dataset, Network, and Training. Apply it before leaving, discard it, or stay to keep editing.</p>
            {draft.error && <div role="alert" className="atelier-error"><p>{draft.error}</p>{draft.submitted && <button type="button" onClick={draft.commands.retry}>Retry configuration</button>}</div>}
            <div className="atelier-dialog-actions"><button type="button" disabled={draft.busy || draft.submitted} onClick={() => setPendingNavigation(null)}>Stay</button><button type="button" disabled={draft.busy || draft.submitted} onClick={() => { draft.commands.cancel(); const action = pendingNavigation; setPendingNavigation(null); action(); }}>Discard changes</button><button type="button" className="atelier-primary" disabled={!draft.valid || draft.busy || draft.submitted} onClick={() => { void draft.commands.apply(); }}>Apply changes</button></div>
        </Dialog>}
        {workerError && <Dialog title="Worker connection lost" persistent backgroundRef={backgroundRef} description={`${workerError} Refresh the page to restart the playground.`} onClose={() => { /* Persistent recovery requires reload. */ }} alert><p>{workerError}</p><p>Refresh the page to restart the playground.</p><button type="button" className="atelier-primary" onClick={() => window.location.reload()}>Refresh page</button></Dialog>}
    </div>;
}
