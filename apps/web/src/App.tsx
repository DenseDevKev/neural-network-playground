// ── Root App Component ──
// Wires the three RegionShell layout variants (Dock / Grid / Split).
// Layout selection lives in useLayoutStore (persisted to localStorage).
// Each shell receives typed content props — adding a new panel means
// adding it to the relevant content map, not editing any layout code.

import { useEffect, useRef, useCallback } from 'react';
import { useLayoutStore } from './store/useLayoutStore.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { usePlaygroundStore } from './store/usePlaygroundStore.ts';
import { useTraining } from './hooks/useTraining.ts';
import { Header } from './components/layout/Header.tsx';
import { Panel } from './components/common/Panel.tsx';
import {
    DockShell,
    GridShell,
    SplitShell,
} from './components/layout/RegionShell.tsx';
import {
    CanvasContent,
    BoundaryContent,
    LossContent,
    ConfusionContent,
    InspectContent,
    CodeContent,
} from './components/layout/MainArea.tsx';
import { TrainingControls } from './components/controls/TrainingControls.tsx';
import { NetworkGraph } from './components/visualization/NetworkGraph.tsx';
import { PresetPanel } from './components/controls/PresetPanel.tsx';
import { DataPanel } from './components/controls/DataPanel.tsx';
import { FeaturesPanel } from './components/controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from './components/controls/NetworkConfigPanel.tsx';
import { HyperparamPanel } from './components/controls/HyperparamPanel.tsx';
import { ConfigPanel } from './components/controls/ConfigPanel.tsx';
import { InspectionPanel } from './components/controls/InspectionPanel.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';
import { CodeExportPanel } from './components/controls/CodeExportPanel.tsx';

export default function App() {
    const training = useTraining();
    const layout   = useLayoutStore((s) => s.layout);
    const status   = useTrainingStore((s) => s.status);
    const dataConfigLoading    = useTrainingStore((s) => s.dataConfigLoading);
    const networkConfigLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError          = useTrainingStore((s) => s.configError);
    const configErrorSource    = useTrainingStore((s) => s.configErrorSource);
    const workerError          = useTrainingStore((s) => s.workerError);

    // Stable refs so keyboard handler never goes stale
    const trainingRef = useRef(training);
    const statusRef   = useRef(status);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current   = status;   }, [status]);

    const stableReset = useCallback(() => trainingRef.current.reset(), []);

    // Performance observer (dev only)
    useEffect(() => {
        if (!import.meta.env.DEV || typeof PerformanceObserver === 'undefined') return;
        const obs = new PerformanceObserver((list) => {
            for (const e of list.getEntriesByType('measure')) {
                if (e.duration > 16)
                    console.warn(`[perf] Slow interaction: ${e.name} (${e.duration.toFixed(2)}ms)`);
            }
        });
        obs.observe({ entryTypes: ['measure'] });
        return () => obs.disconnect();
    }, []);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement ||
                e.target instanceof HTMLSelectElement ||
                e.target instanceof HTMLTextAreaElement) return;
            if (e.code === 'Space') {
                e.preventDefault();
                statusRef.current === 'running'
                    ? trainingRef.current.pause()
                    : trainingRef.current.play();
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

    // ── Content maps ──────────────────────────────────────────────────────
    const leftTabContent = {
        presets:     <PresetPanel onReset={stableReset} />,
        data:        <DataPanel onReset={stableReset} />,
        features:    <FeaturesPanel />,
        network:     <NetworkConfigPanel />,
        hyperparams: <HyperparamPanel />,
        config:      <ConfigPanel onReset={stableReset} />,
    };

    const rightTabContent = {
        boundary:   <BoundaryContent />,
        loss:       <LossContent />,
        confusion:  <ConfusionContent />,
        inspection: <InspectContent />,
        code:       <CodeContent />,
    };

    const transport = <TrainingControls training={training} />;

    const canvasPanel = (
        <Panel title="Network Topology" phase="build" fill>
            <div className="network-graph-wrapper" style={{ flex: 1, height: '100%', minHeight: 200, borderRadius: 'var(--radius-sm)' }}>
                <NetworkGraph />
            </div>
        </Panel>
    );

    const boundaryPanel = (
        <Panel title="Decision Boundary" phase="run" fill>
            <BoundaryContent />
        </Panel>
    );

    const lossPanel = (
        <Panel title="Loss / Accuracy" phase="run" fill>
            <LossContent />
        </Panel>
    );

    const confusionPanel = (
        <Panel title="Confusion Matrix" phase="run">
            <ConfusionContent />
        </Panel>
    );

    const inspectPanel = (
        <Panel title="Layer Inspection" phase="run" fill>
            <InspectContent />
        </Panel>
    );

    const configPanels = (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
            <Panel title="Data" phase="build"><DataPanel onReset={stableReset} /></Panel>
            <Panel title="Features" phase="build"><FeaturesPanel /></Panel>
        </div>
    );

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
                <div className="error-overlay" role="alertdialog" aria-modal="true">
                    <div className="error-overlay__content">
                        <EmptyState
                            icon="⚠"
                            title="Worker connection lost"
                            description={`${workerError} Refresh the page to restart the playground.`}
                            action={{ label: 'Refresh page', onClick: () => window.location.reload() }}
                        />
                    </div>
                </div>
            )}

            {/* Top bar */}
            <ErrorBoundary title="Header unavailable" description="Header render failed." actionLabel="Reload" onRetry={stableReset}>
                <Header training={training} />
            </ErrorBoundary>

            {/* Workspace */}
            <div className="forge-workspace" id="main-content" tabIndex={-1}>
                <ErrorBoundary title="Workspace unavailable" description="Layout shell failed." actionLabel="Reload" onRetry={stableReset}>
                    {layout === 'dock' && (
                        <DockShell
                            leftTabContent={leftTabContent}
                            rightTabContent={rightTabContent}
                            canvasContent={canvasPanel}
                            transportContent={transport}
                        />
                    )}

                    {layout === 'grid' && (
                        <GridShell
                            topologyContent={canvasPanel}
                            boundaryContent={boundaryPanel}
                            configContent={configPanels}
                            lossContent={lossPanel}
                            confusionContent={confusionPanel}
                            inspectContent={inspectPanel}
                            transportContent={transport}
                        />
                    )}

                    {layout === 'split' && (
                        <SplitShell
                            buildLeft={
                                <>
                                    <Panel title="Presets" phase="build"><PresetPanel onReset={stableReset} /></Panel>
                                    <Panel title="Data" phase="build"><DataPanel onReset={stableReset} /></Panel>
                                </>
                            }
                            buildCenter={
                                <Panel title="Network Topology" phase="build" fill>
                                    <div className="network-graph-wrapper" style={{ flex: 1, height: '100%', minHeight: 280 }}>
                                        <NetworkGraph />
                                    </div>
                                </Panel>
                            }
                            buildRight={
                                <>
                                    <Panel title="Features" phase="build"><FeaturesPanel /></Panel>
                                    <Panel title="Hyperparameters" phase="both"><HyperparamPanel /></Panel>
                                </>
                            }
                            runLeft={
                                <>
                                    <Panel title="Network Topology" phase="build" fill>
                                        <div className="network-graph-wrapper" style={{ height: 240 }}>
                                            <NetworkGraph />
                                        </div>
                                    </Panel>
                                    <Panel title="Layer Inspection" phase="run">
                                        <InspectionPanel />
                                    </Panel>
                                </>
                            }
                            runCenter={
                                <>
                                    <Panel title="Decision Boundary" phase="run"><BoundaryContent /></Panel>
                                    <Panel title="Loss / Accuracy" phase="run"><LossContent /></Panel>
                                </>
                            }
                            runRight={
                                <>
                                    <Panel title="Confusion Matrix" phase="run"><ConfusionContent /></Panel>
                                    <Panel title="Code Export" phase="both"><CodeExportPanel /></Panel>
                                </>
                            }
                            transportContent={transport}
                        />
                    )}
                </ErrorBoundary>
            </div>

            {/* Status bar */}
            <StatusBar />
        </div>
    );
}

function StatusBar() {
    const status   = useTrainingStore((s) => s.status);
    const layout   = useLayoutStore((s) => s.layout);
    const phase    = useLayoutStore((s) => s.phase);
    const dataset  = usePlaygroundStore((s) => s.data.dataset);
    const hl       = usePlaygroundStore((s) => s.network.hiddenLayers);
    const snapshot = useTrainingStore((s) => s.snapshot);

    return (
        <div className="forge-statusbar" role="status" aria-label="Status bar">
            <span>
                <span className="forge-statusbar__accent">●</span>{' '}
                {status.toUpperCase()}
            </span>
            <span>LAYOUT: <span className="forge-statusbar__accent">{layout}</span></span>
            <span>PHASE: <span className="forge-statusbar__accent">{phase}</span></span>
            <span>DATA: {dataset}</span>
            <span>ARCH: [{hl.join(', ')}]</span>
            <span className="forge-statusbar__spacer" />
            <span>step {(snapshot?.step ?? 0).toLocaleString()}</span>
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
