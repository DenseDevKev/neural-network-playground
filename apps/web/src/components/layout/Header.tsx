// ── Header ── brand + phase switch + live metrics + layout picker
import { memo, useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { TrainingProgressBar } from './TrainingProgressBar.tsx';

interface HeaderProps {
    training: Pick<TrainingHook, 'play' | 'pause'>;
}

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

export const Header = memo(function Header({ training }: HeaderProps) {
    const snapshot  = useTrainingStore((s) => s.snapshot);
    const status    = useTrainingStore((s) => s.status);
    const stale     = useTrainingStore((s) => s.testMetricsStale);
    const layout    = useLayoutStore((s) => s.layout);
    const phase     = useLayoutStore((s) => s.phase);
    const setLayout = useLayoutStore((s) => s.setLayout);
    const setPhase  = useLayoutStore((s) => s.setPhase);

    const epoch     = snapshot?.epoch ?? 0;
    const trainLoss = (snapshot?.trainLoss ?? 0).toFixed(4);
    const testLoss  = (snapshot?.testLoss ?? 0).toFixed(4);
    const accuracy  = snapshot?.trainMetrics?.accuracy;
    const accStr    = accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—';

    const flashEpoch = useFlash(String(epoch));
    const flashTrain = useFlash(trainLoss);
    const flashTest  = useFlash(testLoss);
    const flashAcc   = useFlash(accStr);
    const isRunning  = status === 'running';

    return (
        <header className="forge-topbar" role="banner">
            {/* Brand */}
            <div className="forge-topbar__brand">
                <span className="forge-topbar__brand-dot" aria-hidden />
                <span>NN·FORGE</span>
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            {/* Phase switch — only meaningful in split layout, hidden on mobile */}
            <div className="forge-phase" role="group" aria-label="Workspace phase">
                <button
                    className={`forge-phase__opt ${phase === 'build' ? 'forge-phase__opt--active' : ''}`}
                    onClick={() => setPhase('build')}
                    aria-pressed={phase === 'build'}
                >
                    <i aria-hidden /><span>Build</span>
                </button>
                <button
                    className={`forge-phase__opt ${phase === 'run' ? 'forge-phase__opt--active' : ''}`}
                    onClick={() => setPhase('run')}
                    aria-pressed={phase === 'run'}
                >
                    <i aria-hidden /><span>Run</span>
                </button>
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            {/* Live metrics */}
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

            {/* Layout picker + mobile play */}
            <div className="forge-topbar__kit">
                <div className="forge-segmented" role="group" aria-label="Layout variant">
                    {(['dock', 'grid', 'split'] as const).map((v) => (
                        <button
                            key={v}
                            className={`forge-segmented__opt ${layout === v ? 'forge-segmented__opt--active forge-segmented__opt--primary' : ''}`}
                            onClick={() => setLayout(v)}
                            aria-pressed={layout === v}
                        >
                            {v}
                        </button>
                    ))}
                </div>

                {/* Mobile play — shown only on small screens via CSS */}
                <button
                    className={`btn btn--play header__mobile-play ${isRunning ? 'running' : ''}`}
                    onClick={isRunning ? training.pause : training.play}
                    aria-label={isRunning ? 'Pause training' : 'Start training'}
                >
                    {isRunning ? '⏸' : '▶'}
                </button>
            </div>

            <TrainingProgressBar isTraining={isRunning} currentEpoch={epoch} />
        </header>
    );
});
