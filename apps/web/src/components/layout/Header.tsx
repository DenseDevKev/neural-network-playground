// ── Header ── brand + Build/Run switch + live metrics + instrument menus
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { TrainingProgressBar } from './TrainingProgressBar.tsx';
import { getTrainingLifecycleUi } from '../controls/trainingLifecycle.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import {
    ADVANCED_TOOLS_REGION_ID,
    ADVANCED_TOOLS_TRIGGER_ID,
    type DrawerSurfaceId,
} from '../../productShell/shellTypes.ts';
import {
    AUDIENCE_MODES,
    getAudienceProfile,
    isAudienceMode,
} from '../../productShell/audienceProfiles.ts';

interface HeaderProps {
    training: Pick<TrainingHook, 'play' | 'pause'>;
    openSurface: DrawerSurfaceId | null;
    onToggleSurface: (surface: DrawerSurfaceId) => void;
    advancedToolsOpen: boolean;
    onToggleAdvancedTools: () => void;
}

const SURFACE_LABELS = {
    presets: 'Presets',
    lessons: 'Lessons',
    history: 'History',
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

export const Header = memo(function Header({
    training,
    openSurface,
    onToggleSurface,
    advancedToolsOpen,
    onToggleAdvancedTools,
}: HeaderProps) {
    const latestLiveSignal = useTrainingStore((s) => s.latestLiveSignal);
    const latestEvaluation = useTrainingStore((s) => s.latestEvaluation);
    const status = useTrainingStore((s) => s.status);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const view = useLayoutStore((s) => s.view);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const setView = useLayoutStore((s) => s.setView);
    const setAudienceMode = useLayoutStore((s) => s.setAudienceMode);
    const [modeAnnouncement, setModeAnnouncement] = useState('');

    const evidence = useMemo(() => selectScientificEvidence({
        latestLiveSignal,
        latestEvaluation,
    }), [latestEvaluation, latestLiveSignal]);
    const epoch = evidence.currentModel?.epoch ?? 0;
    const batchLoss = evidence.batchTrend?.dataLoss.toFixed(4) ?? '—';
    const trainLoss = evidence.fullEvaluation?.trainDataLoss.toFixed(4) ?? '—';
    const testLoss = evidence.fullEvaluation?.testDataLoss.toFixed(4) ?? '—';
    const accuracy = evidence.fullEvaluation?.testAccuracy;
    const accStr = accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—';

    const flashEpoch = useFlash(String(epoch));
    const flashBatch = useFlash(batchLoss);
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

            <div className="forge-audience-mode">
                <label className="forge-audience-mode__control">
                    <span>Mode</span>
                    <select
                        className="select forge-audience-mode__select"
                        aria-label="Audience mode"
                        aria-describedby="forge-audience-mode-description"
                        value={audienceMode}
                        onChange={(event) => {
                            const nextMode = event.currentTarget.value;
                            if (!isAudienceMode(nextMode)) return;
                            setAudienceMode(nextMode);
                            setModeAnnouncement(
                                `Mode: ${getAudienceProfile(nextMode).label}. Mode changes visible tools only.`,
                            );
                        }}
                    >
                        {AUDIENCE_MODES.map((mode) => (
                            <option key={mode} value={mode}>
                                {getAudienceProfile(mode).label}
                            </option>
                        ))}
                    </select>
                </label>
                <span id="forge-audience-mode-description" className="forge-audience-mode__note">
                    Mode changes visible tools only.
                </span>
                <span
                    className="sr-only"
                    role="status"
                    aria-label="Audience mode change"
                    aria-live="polite"
                    aria-atomic="true"
                >
                    {modeAnnouncement}
                </span>
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
                    <span className="forge-metric__label">
                        {`Batch trend (EMA)${evidence.batchTrend ? ` · step ${evidence.batchTrend.step.toLocaleString()}` : ''}`}
                    </span>
                    <span className={`forge-metric__value forge-metric__value--accent ${flashBatch ? 'forge-metric__value--updated' : ''}`}>
                        {batchLoss}
                    </span>
                </div>
                <div className="forge-metric">
                    <span className="forge-metric__label">
                        {`Train data loss (full split)${evidence.fullEvaluation ? ` · step ${evidence.fullEvaluation.step.toLocaleString()}` : ''}`}
                    </span>
                    <span className={`forge-metric__value forge-metric__value--accent ${flashTrain ? 'forge-metric__value--updated' : ''}`}>
                        {trainLoss}
                    </span>
                </div>
                <div className="forge-metric">
                    <span className="forge-metric__label">
                        {`Test data loss (full split)${evidence.fullEvaluation ? ` · step ${evidence.fullEvaluation.step.toLocaleString()}` : ''}`}
                    </span>
                    <span className={`forge-metric__value forge-metric__value--primary ${flashTest ? 'forge-metric__value--updated' : ''}`}>
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
                {(['presets', 'lessons', 'history'] as const).map((surface) => (
                    <button
                        key={surface}
                        type="button"
                        id={`forge-surface-trigger-${surface}`}
                        className={`forge-menu-button ${openSurface === surface ? 'forge-menu-button--active' : ''}`}
                        aria-pressed={openSurface === surface}
                        aria-haspopup="dialog"
                        aria-controls={`forge-surface-${surface}`}
                        onClick={() => onToggleSurface(surface)}
                    >
                        {SURFACE_LABELS[surface]}
                    </button>
                ))}

                <span id="forge-advanced-tools-description" className="sr-only">
                    Shows configuration and diagnostic tools without changing the experiment.
                </span>
                <button
                    type="button"
                    id={ADVANCED_TOOLS_TRIGGER_ID}
                    className={`forge-menu-button forge-menu-button--advanced ${advancedToolsOpen ? 'forge-menu-button--active' : ''}`}
                    aria-expanded={advancedToolsOpen}
                    aria-controls={ADVANCED_TOOLS_REGION_ID}
                    aria-describedby="forge-advanced-tools-description"
                    onClick={onToggleAdvancedTools}
                >
                    Advanced Tools
                </button>

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
