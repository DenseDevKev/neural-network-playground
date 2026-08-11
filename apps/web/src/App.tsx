// ── Root App Component ──
// Wires the Build / Run instrument shell. The app keeps training, worker,
// URL, persistence, and saved-run contracts separate from UI placement.

import { useEffect, useRef, useCallback, useState } from 'react';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { selectScientificEvidence } from './store/evidenceSelectors.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useTraining } from './hooks/useTraining.ts';
import { Header } from './components/layout/Header.tsx';
import { BuildRunShell } from './components/layout/BuildRunShell.tsx';
import {
    CanvasContent,
    BoundaryContent,
    LossContent,
    ConfusionContent,
    InspectContent,
    CodeContent,
    HistoryContent,
    ConfigurationContent,
} from './components/layout/MainArea.tsx';
import { TrainingControls } from './components/controls/TrainingControls.tsx';
import { PresetPanel } from './components/controls/PresetPanel.tsx';
import { GuidedLessonPanel } from './components/controls/GuidedLessonPanel.tsx';
import { RecipeSummaryCard } from './components/controls/RecipeSummaryCard.tsx';
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
    const view = useLayoutStore((s) => s.view);
    const activeEvidenceView = useLayoutStore((s) => s.activeEvidenceView);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const advancedToolsOpen = useLayoutStore((s) => s.advancedToolsOpen);
    const setActiveEvidenceView = useLayoutStore((s) => s.setActiveEvidenceView);
    const setAdvancedToolsOpen = useLayoutStore((s) => s.setAdvancedToolsOpen);
    const status = useTrainingStore((s) => s.status);
    const dataConfigLoading = useTrainingStore((s) => s.dataConfigLoading);
    const networkConfigLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const demand = usePlaygroundStore((s) => s.demand);
    const canvasNetworkGraph = usePlaygroundStore((s) => s.featuresUI.canvasNetworkGraph);
    const setDemand = usePlaygroundStore((s) => s.setDemand);
    const [lessonHighlight, setLessonHighlight] = useState<LessonTarget | null>(null);
    const [openSurface, setOpenSurface] = useState<DrawerSurfaceId | null>(null);

    // Stable refs so keyboard handler never goes stale
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    const workerErrorDialogRef = useRef<HTMLDivElement>(null);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);

    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const handleLessonHighlightChange = useCallback((target: LessonTarget | null) => {
        setLessonHighlight(target);
    }, []);
    const toggleSurface = useCallback((surface: DrawerSurfaceId) => {
        setOpenSurface((current) => current === surface ? null : surface);
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
        if (workerError) {
            workerErrorDialogRef.current?.focus();
        }
    }, [workerError]);

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
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
    }, [advancedToolsOpen, openSurface, setAdvancedToolsOpen]);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
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
    }, []);

    const leftTabContent = {
        presets: <PresetPanel onReset={stableReset} />,
        data: <div className={lessonTargetClass('data')} data-lesson-target="data"><DataPanel onReset={stableReset} /></div>,
        features: <div className={lessonTargetClass('features')} data-lesson-target="features"><FeaturesPanel /></div>,
        network: <div className={lessonTargetClass('network')} data-lesson-target="network"><NetworkConfigPanel /></div>,
        hyperparams: <div className={lessonTargetClass('hyperparams')} data-lesson-target="hyperparams"><HyperparamPanel /></div>,
        config: <ConfigurationContent onReset={stableReset} />,
    };

    const rightTabContent = {
        boundary: <BoundaryContent />,
        loss: <LossContent />,
        confusion: <ConfusionContent />,
        inspection: <InspectContent />,
        code: <CodeContent />,
        history: <HistoryContent />,
    };

    const transport = (
        <div className="forge-transport-cluster">
            <div className={lessonTargetClass('transport')} data-lesson-target="transport">
                <TrainingControls training={training} />
            </div>
        </div>
    );

    const topologyContent = (
        <div className="forge-buildrun__topology-stage">
            <CanvasContent />
        </div>
    );

    const historyContent = <HistoryContent />;

    const workerErrorDescription = workerError
        ? `${workerError} Refresh the page to restart the playground.`
        : '';

    return (
        <div className="forge-shell">
            <a className="skip-link" href="#main-content">Skip to main content</a>

            <AccessibilityAnnouncer
                status={status}
                dataConfigLoading={dataConfigLoading}
                networkConfigLoading={networkConfigLoading}
                configError={configError}
                configErrorSource={configErrorSource}
            />

            {/* Worker crash overlay */}
            {workerError && (
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
                </div>
            )}

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
                    <BuildRunShell
                        view={view}
                        status={status}
                        activeEvidenceView={activeEvidenceView}
                        audienceMode={audienceMode}
                        advancedToolsOpen={advancedToolsOpen}
                        onSelectEvidence={setActiveEvidenceView}
                        openSurface={openSurface}
                        onCloseSurface={closeSurface}
                        recipeContent={<RecipeSummaryCard />}
                        runContent={<CurrentRunCard />}
                        dataContent={leftTabContent.data}
                        networkContent={leftTabContent.network}
                        featuresContent={leftTabContent.features}
                        hyperparamContent={leftTabContent.hyperparams}
                        configurationContent={leftTabContent.config}
                        topologyContent={topologyContent}
                        transportContent={transport}
                        evidenceContent={rightTabContent}
                        presetContent={<PresetPanel onReset={stableReset} onApplied={closeSurface} />}
                        lessonContent={<GuidedLessonPanel onReset={stableReset} onHighlightChange={handleLessonHighlightChange} />}
                        historyContent={historyContent}
                    />
                </ErrorBoundary>
            </main>

            <StatusBar />
        </div>
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
            role="status"
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
