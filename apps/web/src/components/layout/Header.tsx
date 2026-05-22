// ── Header ── brand + Build/Run switch + live metrics + instrument menus
import { memo, useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { TrainingProgressBar } from './TrainingProgressBar.tsx';
import { getTrainingLifecycleUi } from '../controls/trainingLifecycle.ts';

interface HeaderProps {
    training: Pick<TrainingHook, 'play' | 'pause'>;
    openSurface: 'presets' | 'lessons' | 'history' | 'more' | null;
    onToggleSurface: (surface: 'presets' | 'lessons' | 'history' | 'more') => void;
}

const SURFACE_LABELS = {
    presets: 'Presets',
    lessons: 'Lessons',
    history: 'History',
    more: 'More',
} as const;

function useFlash(value: string) {
    const prev = useRef(value);
    const [flash, setFlash] = useState(false);

    useEffect(() => {
        if (prev.current === value) return;
        prev.current = value;
        setFlash(true);
        const t = setTimeout(() => setFlash(false), 200);
        return () => clearTimeout(t);
    }, [value]);

    return flash;
}

export const Header = memo(function Header({ training, openSurface, onToggleSurface }: HeaderProps) {
    const snapshot = useTrainingStore((s) => s.snapshot);
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const stale = useTrainingStore((s) => s.testMetricsStale);
    const view = useLayoutStore((s) => s.view);
    const setView = useLayoutStore((s) => s.setView);

    const epoch = snapshot?.epoch ?? 0;
    const trainLoss = (snapshot?.trainLoss ?? 0).toFixed(4);
    const testLoss = (snapshot?.testLoss ?? 0).toFixed(4);
    const accuracy = snapshot?.trainMetrics?.accuracy;
    const accStr = accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—';

    const flashEpoch = useFlash(String(epoch));
    const flashTrain = useFlash(trainLoss);
    const flashTest = useFlash(testLoss);
    const flashAcc = useFlash(accStr);
    const isRunning = status === 'running';
    const lifecycle = getTrainingLifecycleUi({ status, pauseReason, pendingConfigSource });

    return (
        <header className="forge-topbar" role="banner">
            <div className="forge-topbar__brand">
                <span className="forge-topbar__brand-dot" aria-hidden />
                <span>NN·FORGE</span>
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            <div className="forge-phase" role="group" aria-label="Workspace view">
                {(['build', 'run'] as const).map((nextView) => (
                    <button
                        key={nextView}
                        type="button"
                        className={`forge-phase__opt ${view === nextView ? 'forge-phase__opt--active' : ''}`}
                        onClick={() => setView(nextView)}
                        aria-pressed={view === nextView}
                    >
                        <i aria-hidden /><span>{nextView}</span>
                    </button>
                ))}
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            <div className="forge-topbar__metrics" role="status" aria-live="polite" aria-label="Training metrics">
                <div className="forge-metric">
                    <span className="forge-metric__label">Epoch</span>
                    <span className={`forge-metric__value ${flashEpoch ? 'forge-metric__value--updated' : ''}`}>
                        {String(epoch).padStart(4, '0')}
                    </span>
                </div>
                <div className="forge-metric">
                    <span className="forge-metric__label">Train Loss</span>
                    <span className={`forge-metric__value forge-metric__value--accent ${flashTrain ? 'forge-metric__value--updated' : ''}`}>
                        {trainLoss}
                    </span>
                </div>
                <div className="forge-metric">
                    <span className="forge-metric__label">
                        Test Loss{stale && <span title="Stale / cached" aria-label="Stale metric"> ~</span>}
                    </span>
                    <span className={`forge-metric__value forge-metric__value--primary ${flashTest ? 'forge-metric__value--updated' : ''} ${stale ? 'header__metric-value--stale' : ''}`}>
                        {testLoss}
                    </span>
                </div>
                {accuracy != null && (
                    <div className="forge-metric">
                        <span className="forge-metric__label">Accuracy</span>
                        <span className={`forge-metric__value ${flashAcc ? 'forge-metric__value--updated' : ''}`}>
                            {accStr}
                        </span>
                    </div>
                )}
            </div>

            <span className="forge-topbar__spacer" />

            <div className="forge-topbar__kit">
                {(['presets', 'lessons', 'history', 'more'] as const).map((surface) => (
                    <button
                        key={surface}
                        type="button"
                        className={`forge-menu-button ${openSurface === surface ? 'forge-menu-button--active' : ''}`}
                        aria-pressed={openSurface === surface}
                        aria-haspopup="dialog"
                        onClick={() => onToggleSurface(surface)}
                    >
                        {SURFACE_LABELS[surface]}
                    </button>
                ))}

                <button
                    type="button"
                    className={`btn btn--play header__mobile-play ${isRunning ? 'running' : ''}`}
                    onClick={isRunning ? training.pause : training.play}
                    aria-label={lifecycle.primaryAriaLabel}
                    disabled={!isRunning && lifecycle.isBlocked}
                >
                    {isRunning ? '⏸' : '▶'}
                </button>
            </div>

            <TrainingProgressBar isTraining={isRunning} currentEpoch={epoch} />
        </header>
    );
});
