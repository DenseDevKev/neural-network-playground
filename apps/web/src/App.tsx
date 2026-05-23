// ── Root App Component ──
// Wires the Build / Run instrument shell. The app keeps training, worker,
// URL, persistence, and saved-run contracts separate from UI placement.

import { useEffect, useRef, useCallback, useState } from 'react';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useTraining, type LiveArenaModelInput } from './hooks/useTraining.ts';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import { Header } from './components/layout/Header.tsx';
import { Panel } from './components/common/Panel.tsx';
import { BuildRunShell } from './components/layout/BuildRunShell.tsx';
import {
    CanvasContent,
    BoundaryContent,
    LossContent,
    ConfusionContent,
    InspectContent,
    CodeContent,
    HistoryContent,
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
import { ConfigPanel } from './components/controls/ConfigPanel.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';
import { deriveVisualizationDemand } from './components/layout/deriveVisualizationDemand.ts';

const SHORTCUT_BLOCKED_ROLES = new Set(['button', 'tab', 'switch', 'slider']);
type SurfaceId = 'presets' | 'lessons' | 'history' | 'more';

function createLiveArenaModel(record: ExperimentRunRecordV1): LiveArenaModelInput {
    return {
        label: record.title ?? record.id,
        network: record.config.network,
        training: record.config.training,
        data: record.config.data,
        features: record.config.features,
    };
}

function shouldIgnoreGlobalShortcut(target: EventTarget | null) {
    if (!(target instanceof Element)) return false;
    if (target === document.body || target === document.documentElement) return false;

    let el: Element | null = target;
    while (el) {
        if (
            el instanceof HTMLButtonElement ||
            el instanceof HTMLInputElement ||
            el instanceof HTMLSelectElement ||
            el instanceof HTMLTextAreaElement ||
            el instanceof HTMLAnchorElement
        ) {
            return true;
        }

        const role = el.getAttribute('role');
        if (role && SHORTCUT_BLOCKED_ROLES.has(role)) return true;

        const tabIndex = el.getAttribute('tabindex');
        if (tabIndex !== null && tabIndex !== '-1') return true;

        const contentEditable = el.getAttribute('contenteditable');
        if (contentEditable !== null && contentEditable.toLowerCase() !== 'false') return true;

        el = el.parentElement;
    }

    return false;
}

export default function App() {
    const training = useTraining();
    const view = useLayoutStore((s) => s.view);
    const activeEvidenceView = useLayoutStore((s) => s.activeEvidenceView);
    const setActiveEvidenceView = useLayoutStore((s) => s.setActiveEvidenceView);
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
    const [openSurface, setOpenSurface] = useState<SurfaceId | null>(null);

    // Stable refs so keyboard handler never goes stale
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    const workerErrorDialogRef = useRef<HTMLDivElement>(null);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);

    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const stableInitializeArena = useCallback((modelA: ExperimentRunRecordV1, modelB: ExperimentRunRecordV1) => (
        trainingRef.current.initializeArena(createLiveArenaModel(modelA), createLiveArenaModel(modelB))
    ), []);
    const stableStepArena = useCallback(() => trainingRef.current.stepArena(1), []);
    const handleLessonHighlightChange = useCallback((target: LessonTarget | null) => {
        setLessonHighlight(target);
    }, []);
    const toggleSurface = useCallback((surface: SurfaceId) => {
        setOpenSurface((current) => current === surface ? null : surface);
    }, []);
    const lessonTargetClass = useCallback(
        (target: LessonTarget) => `lesson-target ${lessonHighlight === target ? 'lesson-target--active' : ''}`,
        [lessonHighlight],
    );

    // Performance observer (dev only)
    useEffect(() => {
        if (!import.meta.env.DEV || typeof PerformanceObserver === 'undefined') return;
        const obs = new PerformanceObserver((list) => {
            for (const e of list.getEntriesByType('measure')) {
                if (e.duration > 16) {
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
            historyDrawerOpen: openSurface === 'history',
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
    }, [view, activeEvidenceView, openSurface, canvasNetworkGraph, demand, setDemand]);

    useEffect(() => {
        if (workerError) {
            workerErrorDialogRef.current?.focus();
        }
    }, [workerError]);

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setOpenSurface(null);
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (shouldIgnoreGlobalShortcut(e.target)) return;

            if (e.code === 'Space') {
                e.preventDefault();
                if (statusRef.current === 'running') {
                    trainingRef.current.pause();
                } else {
                    trainingRef.current.play();
                }
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                trainingRef.current.step();
            } else if (e.code === 'KeyR') {
                e.preventDefault();
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
        config: <ConfigPanel onReset={stableReset} />,
    };

    const rightTabContent = {
        boundary: <BoundaryContent />,
        loss: <LossContent />,
        confusion: <ConfusionContent />,
        inspection: <InspectContent />,
        code: <CodeContent />,
        history: (
            <HistoryContent
                onRestore={stableReset}
                onInitializeArena={stableInitializeArena}
                onStepArena={stableStepArena}
            />
        ),
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

    const historyContent = (
        <HistoryContent
            onRestore={stableReset}
            onInitializeArena={stableInitializeArena}
            onStepArena={stableStepArena}
        />
    );

    const moreContent = (
        <div className="forge-drawer-stack">
            <Panel title="Configuration" phase="both" panelTargets="config">
                <ConfigPanel onReset={stableReset} />
            </Panel>
            <Panel title="Code Export" phase="both" panelTargets="code">
                <CodeContent />
            </Panel>
        </div>
    );

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
                        onSelectEvidence={setActiveEvidenceView}
                        openSurface={openSurface}
                        onCloseSurface={() => setOpenSurface(null)}
                        recipeContent={<RecipeSummaryCard />}
                        runContent={<CurrentRunCard />}
                        dataContent={leftTabContent.data}
                        networkContent={leftTabContent.network}
                        featuresContent={leftTabContent.features}
                        hyperparamContent={leftTabContent.hyperparams}
                        topologyContent={topologyContent}
                        transportContent={transport}
                        evidenceContent={rightTabContent}
                        presetContent={<PresetPanel onReset={stableReset} onApplied={() => setOpenSurface(null)} />}
                        lessonContent={<GuidedLessonPanel onReset={stableReset} onHighlightChange={handleLessonHighlightChange} />}
                        historyContent={historyContent}
                        moreContent={moreContent}
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
    const dataset = usePlaygroundStore((s) => s.data.dataset);
    const snapshot = useTrainingStore((s) => s.snapshot);

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
            <span>STEP <span className="forge-statusbar__accent">{(snapshot?.step ?? 0).toLocaleString()}</span></span>
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
