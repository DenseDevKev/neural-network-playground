// ── Root App Component ──
import { useEffect, useRef, useCallback, useState } from 'react';
import { Header } from './components/layout/Header.tsx';
import { Sidebar } from './components/layout/Sidebar.tsx';
import { MainArea } from './components/layout/MainArea.tsx';
import { AccessibilityAnnouncer } from './components/layout/AccessibilityAnnouncer.tsx';
import { ErrorBoundary } from './components/common/ErrorBoundary.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';
import { useTraining } from './hooks/useTraining.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';

export default function App() {
    const training = useTraining();
    const status = useTrainingStore((s) => s.status);
    const dataConfigLoading = useTrainingStore((s) => s.dataConfigLoading);
    const networkConfigLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const [viewport, setViewport] = useState<'mobile' | 'tablet' | 'desktop' | 'wide'>(() => {
        if (window.innerWidth < 540) return 'mobile';
        if (window.innerWidth < 860) return 'tablet';
        if (window.innerWidth < 1200) return 'desktop';
        return 'wide';
    });

    // Stable ref so the keydown handler never goes stale between renders.
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);

    useEffect(() => {
        if (!import.meta.env.DEV || typeof PerformanceObserver === 'undefined') {
            return;
        }

        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntriesByType('measure')) {
                if (entry.duration > 16) {
                    console.warn(`[perf] Slow interaction: ${entry.name} (${entry.duration.toFixed(2)}ms)`);
                }
            }
        });

        observer.observe({ entryTypes: ['measure'] });
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        const updateViewport = () => {
            if (window.innerWidth < 540) {
                setViewport('mobile');
            } else if (window.innerWidth < 860) {
                setViewport('tablet');
            } else if (window.innerWidth < 1200) {
                setViewport('desktop');
            } else {
                setViewport('wide');
            }
        };

        window.addEventListener('resize', updateViewport);
        return () => window.removeEventListener('resize', updateViewport);
    }, []);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Don't intercept when user is typing in a form element
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
    }, []); // empty deps — handler reads always-current refs
    // Stable callback for Sidebar — delegates to the always-current ref
    // so React.memo on Sidebar actually prevents re-renders.
    const stableReset = useCallback(() => trainingRef.current.reset(), []);

    return (
        <div className={`app-shell app-shell--${viewport}`} data-viewport={viewport}>
            <a className="skip-link" href="#main-content">Skip to main content</a>
            <AccessibilityAnnouncer
                status={status}
                dataConfigLoading={dataConfigLoading}
                networkConfigLoading={networkConfigLoading}
                configError={configError}
                configErrorSource={configErrorSource}
            />
            <Header training={training} />
            {workerError && (
                <div className="error-overlay" role="alertdialog" aria-modal="true">
                    <div className="error-overlay__content">
                        <EmptyState
                            icon="⚠"
                            title="Worker connection lost"
                            description={`${workerError} Refresh the page to restart the playground.`}
                            action={{
                                label: 'Refresh page',
                                onClick: () => window.location.reload(),
                            }}
                        />
                    </div>
                </div>
            )}
            <div className="main-layout">
                <ErrorBoundary
                    title="Controls unavailable"
                    description="The configuration sidebar hit an error."
                    actionLabel="Reload controls"
                    onRetry={stableReset}
                >
                    <Sidebar onReset={stableReset} />
                </ErrorBoundary>
                <ErrorBoundary
                    title="Workspace unavailable"
                    description="The main playground area failed to render."
                    actionLabel="Reload workspace"
                    onRetry={stableReset}
                >
                    <MainArea training={training} />
                </ErrorBoundary>
            </div>
            <footer className="footer">
                Inspired by{' '}
                <a
                    href="https://playground.tensorflow.org"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    TensorFlow Playground
                </a>{' '}
                by Daniel Smilkov &amp; Shan Carter
            </footer>
        </div>
    );
}
