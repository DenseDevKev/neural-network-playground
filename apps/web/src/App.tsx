// ── Root App Component ──
// Wires the three RegionShell layout variants (Dock / Grid / Split).
// Layout selection lives in useLayoutStore (persisted to localStorage).
// Each shell receives typed content props — adding a new panel means
// adding it to the relevant content map, not editing any layout code.

import { useEffect, useRef, useCallback, useState } from 'react';
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
import { PresetPanel } from './components/controls/PresetPanel.tsx';
import { DataPanel } from './components/controls/DataPanel.tsx';
import { FeaturesPanel } from './components/controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from './components/controls/NetworkConfigPanel.tsx';
import { HyperparamPanel } from './components/controls/HyperparamPanel.tsx';
import { ConfigPanel } from './components/controls/ConfigPanel.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';

const COMPACT_BREAKPOINT = 900;

export default function App() {
    const training = useTraining();
    const persistedLayout = useLayoutStore((s) => s.layout);
    const status = useTrainingStore((s) => s.status);
    const dataConfigLoading = useTrainingStore((s) => s.dataConfigLoading);
    const networkConfigLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const [isCompact, setIsCompact] = useState(() => window.innerWidth < COMPACT_BREAKPOINT);

    // Stable refs so keyboard handler never goes stale
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);

    const stableReset = useCallback(() => trainingRef.current.reset(), []);
    const effectiveLayout = isCompact ? 'dock' : persistedLayout;

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
        const updateCompactMode = () => {
            setIsCompact(window.innerWidth < COMPACT_BREAKPOINT);
        };

        window.addEventListener('resize', updateCompactMode);
        return () => window.removeEventListener('resize', updateCompactMode);
    }, []);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (
                e.target instanceof HTMLInputElement ||
                e.target instanceof HTMLSelectElement ||
                e.target instanceof HTMLTextAreaElement
            ) return;

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
        data: <DataPanel onReset={stableReset} />,
        features: <FeaturesPanel />,
        network: <NetworkConfigPanel />,
        hyperparams: <HyperparamPanel />,
        config: <ConfigPanel onReset={stableReset} />,
    };

    const rightTabContent = {
        boundary: <BoundaryContent />,
        loss: <LossContent />,
        confusion: <ConfusionContent />,
        inspection: <InspectContent />,
        code: <CodeContent />,
    };

    const transport = <TrainingControls training={training} />;

    const canvasPanel = (
        <Panel title="Network Topology" phase="build" fill>
            <div style={{ display: 'flex', minHeight: 200, height: '100%' }}>
                <CanvasContent />
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

    const codePanel = (
        <Panel title="Code Export" phase="both">
            <CodeContent />
        </Panel>
    );

    const gridConfigPanels = (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
            <Panel title="Presets" phase="build"><PresetPanel onReset={stableReset} /></Panel>
            <Panel title="Data" phase="build"><DataPanel onReset={stableReset} /></Panel>
            <Panel title="Features" phase="build"><FeaturesPanel /></Panel>
            <Panel title="Network" phase="build"><NetworkConfigPanel /></Panel>
            <Panel title="Hyperparameters" phase="both"><HyperparamPanel /></Panel>
            <Panel title="Config" phase="both"><ConfigPanel onReset={stableReset} /></Panel>
        </div>
    );

    const gridInspectPanels = (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
            {inspectPanel}
            {codePanel}
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

            <ErrorBoundary title="Header unavailable" description="Header render failed." actionLabel="Reload" onRetry={stableReset}>
                <Header training={training} effectiveLayout={effectiveLayout} isCompact={isCompact} />
            </ErrorBoundary>

            <div className="forge-workspace" id="main-content" tabIndex={-1}>
                <ErrorBoundary title="Workspace unavailable" description="Layout shell failed." actionLabel="Reload" onRetry={stableReset}>
                    {effectiveLayout === 'dock' && (
                        <DockShell
                            leftTabContent={leftTabContent}
                            rightTabContent={rightTabContent}
                            canvasContent={canvasPanel}
                            transportContent={transport}
                            compact={isCompact}
                        />
                    )}

                    {effectiveLayout === 'grid' && (
                        <GridShell
                            topologyContent={canvasPanel}
                            boundaryContent={boundaryPanel}
                            configContent={gridConfigPanels}
                            lossContent={lossPanel}
                            confusionContent={confusionPanel}
                            inspectContent={gridInspectPanels}
                            transportContent={transport}
                        />
                    )}

                    {effectiveLayout === 'split' && (
                        <SplitShell
                            buildLeft={
                                <>
                                    <Panel title="Presets" phase="build"><PresetPanel onReset={stableReset} /></Panel>
                                    <Panel title="Data" phase="build"><DataPanel onReset={stableReset} /></Panel>
                                </>
                            }
                            buildCenter={
                                <>
                                    <Panel title="Network Topology" phase="build" fill>
                                        <div style={{ display: 'flex', minHeight: 280, height: '100%' }}>
                                            <CanvasContent />
                                        </div>
                                    </Panel>
                                    <Panel title="Network" phase="build"><NetworkConfigPanel /></Panel>
                                </>
                            }
                            buildRight={
                                <>
                                    <Panel title="Features" phase="build"><FeaturesPanel /></Panel>
                                    <Panel title="Hyperparameters" phase="both"><HyperparamPanel /></Panel>
                                    <Panel title="Config" phase="both"><ConfigPanel onReset={stableReset} /></Panel>
                                </>
                            }
                            runLeft={
                                <>
                                    <Panel title="Network Topology" phase="build" fill>
                                        <div style={{ display: 'flex', minHeight: 240, height: '100%' }}>
                                            <CanvasContent />
                                        </div>
                                    </Panel>
                                    {inspectPanel}
                                </>
                            }
                            runCenter={
                                <>
                                    {boundaryPanel}
                                    {lossPanel}
                                </>
                            }
                            runRight={
                                <>
                                    {confusionPanel}
                                    <Panel title="Hyperparameters" phase="both"><HyperparamPanel /></Panel>
                                    {codePanel}
                                </>
                            }
                            transportContent={transport}
                        />
                    )}
                </ErrorBoundary>
            </div>

            <StatusBar effectiveLayout={effectiveLayout} />
        </div>
    );
}

function StatusBar({ effectiveLayout }: { effectiveLayout: 'dock' | 'grid' | 'split' }) {
    const status = useTrainingStore((s) => s.status);
    const phase = useLayoutStore((s) => s.phase);
    const dataset = usePlaygroundStore((s) => s.data.dataset);
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const snapshot = useTrainingStore((s) => s.snapshot);

    return (
        <div className="forge-statusbar" role="status" aria-label="Status bar">
            <span>
                <span className="forge-statusbar__accent">●</span>{' '}
                {status.toUpperCase()}
            </span>
            <span>LAYOUT: <span className="forge-statusbar__accent">{effectiveLayout}</span></span>
            {effectiveLayout === 'split' && (
                <span>PHASE: <span className="forge-statusbar__accent">{phase}</span></span>
            )}
            <span>DATA: {dataset}</span>
            <span>ARCH: [{hiddenLayers.join(', ')}]</span>
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
