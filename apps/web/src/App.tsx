// ── Root App Component ──
// Wires the Build / Run instrument shell. The app keeps training, worker,
// URL, persistence, and saved-run contracts separate from UI placement.

import { lazy, Suspense, useEffect, useRef, useCallback, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useExperimentMemoryStore } from './store/experimentMemoryStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { selectScientificEvidence } from './store/evidenceSelectors.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useSaveCurrentRun } from './hooks/useSaveCurrentRun.ts';
import { useTraining } from './hooks/useTraining.ts';
import { useModalFocusContainment } from './hooks/useModalFocusContainment.ts';
import { useExperimentMemoryStorageSync } from './hooks/useExperimentMemoryStorageSync.ts';
import { Header } from './components/layout/Header.tsx';
import { PrecisionLabShell } from './components/layout/precisionLab/PrecisionLabShell.tsx';
import { PrecisionLabRecipeStrip } from './components/layout/precisionLab/PrecisionLabRecipeStrip.tsx';
import { usePrecisionLabRecipeModel } from './components/layout/precisionLab/usePrecisionLabRecipeModel.ts';
import { useNetworkSelectionController } from './components/visualization/useNetworkSelectionController.ts';
import { NetworkSelectionDeck } from './components/visualization/NetworkSelectionDeck.tsx';
import { useDecisionBoundaryController } from './components/visualization/useDecisionBoundaryController.ts';
import { PinnedBoundaryRail } from './components/visualization/PinnedBoundaryRail.tsx';
import { BoundaryEvidencePanel } from './components/visualization/BoundaryEvidencePanel.tsx';
import { AdvancedRecipeNotice } from './components/controls/AdvancedRecipeNotice.tsx';
import {
    TopologyContent,
    LossContent,
    ConfusionContent,
    InspectContent,
    CodeContent,
    HistoryContent,
    ConfigurationContent,
} from './components/layout/PrecisionLabContent.tsx';
import { TrainingControls } from './components/controls/TrainingControls.tsx';
import { PresetPanel } from './components/controls/PresetPanel.tsx';
const GuidedLessonPanel = lazy(() => import('./components/EducationContent.ts').then((module) => ({ default: module.GuidedLessonPanel })));
import { FirstVisitLessonCue } from './components/controls/FirstVisitLessonCue.tsx';
import { CurrentRunCard } from './components/controls/CurrentRunCard.tsx';
import type { LessonTarget } from './lessons/lessonRegistry.ts';
import { DataPanel } from './components/controls/DataPanel.tsx';
import { FeaturesPanel } from './components/controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from './components/controls/NetworkConfigPanel.tsx';
import { HyperparamPanel } from './components/controls/HyperparamPanel.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';
import { CompatibilityState } from './components/common/CompatibilityState.tsx';
import { deriveVisualizationDemand } from './components/layout/deriveVisualizationDemand.ts';
import { shouldReportSlowInteraction } from './performance/interactionMeasures.ts';
import {
    ADVANCED_TOOLS_TRIGGER_ID,
    type DrawerSurfaceId,
} from './productShell/shellTypes.ts';
import { resolveTrainingShortcut } from './shortcuts/trainingShortcuts.ts';

export default function App() {
    useExperimentMemoryStorageSync();
    const access = usePlaygroundStore((state) => state.access);
    const startFresh = usePlaygroundStore((state) => state.startFresh);
    useEffect(() => {
        const loadLocation = () => {
            void usePlaygroundStore.getState().loadFromUrl();
        };
        window.addEventListener('hashchange', loadLocation);
        return () => window.removeEventListener('hashchange', loadLocation);
    }, []);
    const recoverWithDefault = useCallback(async () => {
        const result = await startFresh();
        if (!result.ok) {
            throw new Error(result.issues.map((issue) => issue.message).join(' '));
        }
    }, [startFresh]);

    if (access.status === 'incompatible') {
        return <CompatibilityState access={access} onStartFresh={recoverWithDefault} />;
    }

    return <CompatiblePlayground />;
}

function CompatiblePlayground() {
    const training = useTraining();
    const saveController = useSaveCurrentRun();
    const recipeModel = usePrecisionLabRecipeModel();
    const boundary = useDecisionBoundaryController();
    const selection = useNetworkSelectionController();
    const activeRecipeSection = useLayoutStore((s) => s.activeRecipeSection);
    const buildContextOpen = useLayoutStore((s) => s.buildContextOpen);
    const selectBuildContext = useLayoutStore((s) => s.selectBuildContext);
    const setBuildContextOpen = useLayoutStore((s) => s.setBuildContextOpen);
    const view = useLayoutStore((s) => s.view);
    const activeEvidenceView = useLayoutStore((s) => s.activeEvidenceView);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const advancedToolsOpen = useLayoutStore((s) => s.advancedToolsOpen);
    const setActiveEvidenceView = useLayoutStore((s) => s.setActiveEvidenceView);
    const setAdvancedToolsOpen = useLayoutStore((s) => s.setAdvancedToolsOpen);
    const lessonCueDismissed = useLayoutStore((s) => s.lessonCueDismissed);
    const hasStartedLesson = useLayoutStore((s) => s.hasStartedLesson);
    const hasActiveLesson = useLayoutStore((s) => s.activeLessonId !== null);
    const dismissLessonCue = useLayoutStore((s) => s.dismissLessonCue);
    const historyReady = useExperimentMemoryStore((s) => s.hydrationStatus === 'ready');
    const hasSavedRuns = useExperimentMemoryStore((s) => (
        s.hydrationStatus === 'ready' && s.records.length > 0
    ));
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const evidenceGenerationId = useTrainingStore((s) => s.evidenceGenerationId);
    const trainedRecipe = useTrainingStore((s) => s.trainedRecipe);
    const trainedRecipeSource = useTrainingStore((s) => s.trainedRecipeSource);
    const demand = usePlaygroundStore((s) => s.demand);
    const canvasNetworkGraph = usePlaygroundStore((s) => s.featuresUI.canvasNetworkGraph);
    const setDemand = usePlaygroundStore((s) => s.setDemand);
    const [lessonHighlight, setLessonHighlight] = useState<LessonTarget | null>(null);
    const [openSurface, setOpenSurface] = useState<DrawerSurfaceId | null>(null);

    // Stable refs so keyboard handler never goes stale
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    const workerErrorDialogRef = useRef<HTMLDivElement>(null);
    const backgroundRef = useRef<HTMLDivElement>(null);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);
    useModalFocusContainment(Boolean(workerError), workerErrorDialogRef, backgroundRef);

    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const handleLessonHighlightChange = useCallback((target: LessonTarget | null) => {
        setLessonHighlight(target);
    }, []);
    const toggleSurface = useCallback((surface: DrawerSurfaceId) => {
        setOpenSurface((current) => current === surface ? null : surface);
    }, []);
    const openLessons = useCallback(() => {
        setOpenSurface('lessons');
    }, []);
    const closeSurface = useCallback(() => {
        if (openSurface) {
            document.getElementById(`forge-surface-trigger-${openSurface}`)?.focus();
        }
        setOpenSurface(null);
    }, [openSurface]);
    const toggleAdvancedTools = useCallback(() => {
        if (advancedToolsOpen) {
            document.getElementById(ADVANCED_TOOLS_TRIGGER_ID)?.focus();
        }
        setAdvancedToolsOpen(!advancedToolsOpen);
    }, [advancedToolsOpen, setAdvancedToolsOpen]);
    const lessonTargetClass = useCallback(
        (target: LessonTarget) => `lesson-target ${lessonHighlight === target ? 'lesson-target--active' : ''}`,
        [lessonHighlight],
    );

    // Performance observer (dev only)
    useEffect(() => {
        if (!import.meta.env.DEV || typeof PerformanceObserver === 'undefined') return;
        const obs = new PerformanceObserver((list) => {
            for (const e of list.getEntriesByType('measure')) {
                if (shouldReportSlowInteraction(e)) {
                    console.warn(`[perf] Slow interaction: ${e.name} (${e.duration.toFixed(2)}ms)`);
                }
            }
        });
        obs.observe({ entryTypes: ['measure'] });
        return () => obs.disconnect();
    }, []);

    useEffect(() => {
        const nextDemand = deriveVisualizationDemand({
            view,
            activeEvidenceView,
            audienceMode,
            advancedToolsOpen,
            graphRenderer: canvasNetworkGraph ? 'canvas' : 'svg',
            boundaryRailMounted: true,
        });
        if (
            demand.needDecisionBoundary === nextDemand.needDecisionBoundary &&
            demand.needNeuronGrids === nextDemand.needNeuronGrids &&
            demand.needLayerStats === nextDemand.needLayerStats &&
            demand.needActivationHistograms === nextDemand.needActivationHistograms &&
            demand.needConfusionMatrix === nextDemand.needConfusionMatrix
        ) {
            return;
        }
        setDemand(nextDemand);
    }, [
        view,
        activeEvidenceView,
        audienceMode,
        advancedToolsOpen,
        canvasNetworkGraph,
        demand,
        setDemand,
    ]);

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (workerError) return;
            if (event.key !== 'Escape') return;
            if (openSurface) {
                event.preventDefault();
                document.getElementById(`forge-surface-trigger-${openSurface}`)?.focus();
                setOpenSurface(null);
                return;
            }
            if (advancedToolsOpen) {
                event.preventDefault();
                document.getElementById(ADVANCED_TOOLS_TRIGGER_ID)?.focus();
                setAdvancedToolsOpen(false);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [advancedToolsOpen, openSurface, setAdvancedToolsOpen, workerError]);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (workerError) return;
            const action = resolveTrainingShortcut(e);
            if (!action) return;

            e.preventDefault();
            if (action === 'play-pause') {
                if (statusRef.current === 'running') {
                    trainingRef.current.pause();
                } else {
                    trainingRef.current.play();
                }
            } else if (action === 'step') {
                trainingRef.current.step();
            } else {
                trainingRef.current.reset();
            }
        };

        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [workerError]);

    const leftTabContent = {
        data: <div className={lessonTargetClass('data')} data-lesson-target="data"><DataPanel onReset={stableReset} /></div>,
        features: <div className={lessonTargetClass('features')} data-lesson-target="features"><FeaturesPanel /></div>,
        network: <div className={lessonTargetClass('network')} data-lesson-target="network"><NetworkConfigPanel /></div>,
        hyperparams: <div className={lessonTargetClass('hyperparams')} data-lesson-target="hyperparams"><HyperparamPanel /></div>,
        config: <ConfigurationContent onReset={stableReset} />,
    };

    const rightTabContent = {
        boundary: <BoundaryEvidencePanel controller={boundary} />,
        loss: <LossContent />,
        confusion: <ConfusionContent />,
        inspection: <InspectContent />,
        code: <CodeContent />,
    };

    const transport = (
        <div className="forge-transport-cluster">
            <div className={lessonTargetClass('transport')} data-lesson-target="transport">
                <TrainingControls training={training} saveController={saveController} />
            </div>
        </div>
    );

    const topologyContent = (
        <div className="forge-buildrun__topology-stage">
            <TopologyContent selectionController={selection} />
        </div>
    );

    const historyContent = <HistoryContent saveController={saveController} />;

    const workerErrorDescription = workerError
        ? `${workerError} Refresh the page to restart the playground.`
        : '';

    const workerErrorModal = workerError && typeof document !== 'undefined'
        ? createPortal(
            <div
                className="error-overlay"
                role="alertdialog"
                aria-modal="true"
                aria-labelledby="worker-error-title"
                aria-describedby="worker-error-description"
                tabIndex={-1}
                ref={workerErrorDialogRef}
            >
                <div className="error-overlay__content">
                    <EmptyState
                        icon="⚠"
                        title="Worker connection lost"
                        titleId="worker-error-title"
                        description={workerErrorDescription}
                        descriptionId="worker-error-description"
                        action={{ label: 'Refresh page', onClick: () => window.location.reload() }}
                    />
                </div>
            </div>,
            document.body,
        )
        : null;

    return (
        <>
        <div className="forge-shell" ref={backgroundRef}>
            <a
                className="skip-link"
                href="#main-content"
                onClick={(event) => {
                    // The URL fragment is the experiment document, not shell navigation.
                    event.preventDefault();
                    document.getElementById('main-content')?.focus();
                }}
            >
                Skip to main content
            </a>

            <AccessibilityAnnouncer
                status={status}
                pauseReason={pauseReason}
                workerError={workerError}
                pendingConfigSource={pendingConfigSource}
                configError={configError}
                configErrorSource={configErrorSource}
                evidenceGenerationId={evidenceGenerationId}
                trainedRecipe={trainedRecipe}
                trainedRecipeSource={trainedRecipeSource}
            />

            <ErrorBoundary title="Header unavailable" description="Header render failed." actionLabel="Reload" onRetry={stableReset}>
                <Header
                    training={training}
                    openSurface={openSurface}
                    onToggleSurface={toggleSurface}
                    advancedToolsOpen={advancedToolsOpen}
                    onToggleAdvancedTools={toggleAdvancedTools}
                />
            </ErrorBoundary>

            <main
                className="forge-workspace"
                id="main-content"
                tabIndex={-1}
                aria-label="Neural network playground workspace"
            >
                <ErrorBoundary title="Workspace unavailable" description="Layout shell failed." actionLabel="Reload" onRetry={stableReset}>
                    <PrecisionLabShell
                        view={view}
                        status={status}
                        activeEvidenceView={activeEvidenceView}
                        audienceMode={audienceMode}
                        advancedToolsOpen={advancedToolsOpen}
                        onSelectEvidence={setActiveEvidenceView}
                        openSurface={openSurface}
                        onCloseSurface={closeSurface}
                        firstVisitLessonCue={(
                            <FirstVisitLessonCue
                                historyReady={historyReady}
                                lessonCueDismissed={lessonCueDismissed}
                                hasSavedRuns={hasSavedRuns}
                                hasStartedLesson={hasStartedLesson}
                                hasActiveLesson={hasActiveLesson}
                                onOpenLessons={openLessons}
                                onDismiss={dismissLessonCue}
                            />
                        )}
                        recipeStripContent={<div className="precision-recipe-slot">
                            <PrecisionLabRecipeStrip model={recipeModel} onEditRecipe={() => selectBuildContext('data')} />
                            <AdvancedRecipeNotice />
                        </div>}
                        runSummaryContent={<CurrentRunCard />}
                        activeRecipeSection={activeRecipeSection}
                        buildContextOpen={buildContextOpen}
                        onSelectRecipeSection={selectBuildContext}
                        onCloseRecipeSection={() => setBuildContextOpen(false)}
                        buildContent={leftTabContent}
                        selectionContent={<NetworkSelectionDeck model={selection.model} onClear={selection.commands.clearSelection} />}
                        boundaryRailContent={<PinnedBoundaryRail controller={boundary} onExpand={() => {
                            setActiveEvidenceView('boundary');
                            requestAnimationFrame(() => document.getElementById('precision-evidence-tab-boundary')?.focus());
                        }} />}
                        topologyContent={topologyContent}
                        transportContent={transport}
                        evidenceContent={rightTabContent}
                        presetContent={<PresetPanel onReset={stableReset} onApplied={closeSurface} />}
                        lessonContent={<Suspense fallback={<p role="status">Loading guided lessons…</p>}><GuidedLessonPanel onReset={stableReset} onHighlightChange={handleLessonHighlightChange} /></Suspense>}
                        historyContent={historyContent}
                    />
                </ErrorBoundary>
            </main>

            <StatusBar />
        </div>
        {workerErrorModal}
        </>
    );
}

function StatusBar() {
    const status = useTrainingStore((s) => s.status);
    const view = useLayoutStore((s) => s.view);
    const dataset = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared.compiled.data.dataset : 'unavailable'
    ));
    const latestLiveSignal = useTrainingStore((s) => s.latestLiveSignal);
    const latestEvaluation = useTrainingStore((s) => s.latestEvaluation);
    const step = selectScientificEvidence({ latestLiveSignal, latestEvaluation })
        .currentModel?.step ?? 0;

    return (
        <div
            className="forge-statusbar"
            role="group"
            aria-label="Status bar"
            data-status={status}
        >
            <span>
                <span className="forge-statusbar__dot" aria-hidden />
                {status.toUpperCase()}
            </span>
            <span>VIEW: <span className="forge-statusbar__accent">{view}</span></span>
            <span>DATA: <span className="forge-statusbar__accent">{dataset}</span></span>
            <span className="forge-statusbar__spacer" />
            <span>STEP <span className="forge-statusbar__accent">{step.toLocaleString()}</span></span>
            <span>
                Inspired by{' '}
                <a
                    href="https://playground.tensorflow.org"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: 'var(--color-primary)', textDecoration: 'none' }}
                >
                    TensorFlow Playground
                </a>
            </span>
        </div>
    );
}
