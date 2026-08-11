// ── Header ── brand + Build/Run switch + live metrics + instrument menus
import {
    memo,
    useCallback,
    useEffect,
    useId,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import { createPortal } from 'react-dom';
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

const WORKSPACE_VIEW_DESCRIPTION =
    'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.';
const WORKSPACE_HELP_GAP = 8;
const WORKSPACE_HELP_VIEWPORT_PADDING = 8;

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
    const [workspaceHelpOpen, setWorkspaceHelpOpen] = useState(false);
    const [workspaceHelpPosition, setWorkspaceHelpPosition] = useState({ left: 0, top: 0 });
    const workspaceHelpPinnedRef = useRef(false);
    const workspaceHelpTriggerRef = useRef<HTMLButtonElement>(null);
    const workspaceHelpDescriptionRef = useRef<HTMLDivElement>(null);
    const workspaceDescriptionId = `forge-workspace-view-description-${useId()}`;
    const audienceProfile = getAudienceProfile(audienceMode);

    const positionWorkspaceHelp = useCallback(() => {
        const trigger = workspaceHelpTriggerRef.current;
        const description = workspaceHelpDescriptionRef.current;
        if (!trigger || !description) return;

        const triggerRect = trigger.getBoundingClientRect();
        const descriptionRect = description.getBoundingClientRect();
        const descriptionWidth = descriptionRect.width || Math.min(320, window.innerWidth - 16);
        const descriptionHeight = descriptionRect.height || 80;
        const maxLeft = Math.max(
            WORKSPACE_HELP_VIEWPORT_PADDING,
            window.innerWidth - descriptionWidth - WORKSPACE_HELP_VIEWPORT_PADDING,
        );
        const left = Math.max(
            WORKSPACE_HELP_VIEWPORT_PADDING,
            Math.min(triggerRect.left, maxLeft),
        );
        const belowTop = triggerRect.bottom + WORKSPACE_HELP_GAP;
        const top = belowTop + descriptionHeight <= window.innerHeight - WORKSPACE_HELP_VIEWPORT_PADDING
            ? belowTop
            : Math.max(
                WORKSPACE_HELP_VIEWPORT_PADDING,
                triggerRect.top - descriptionHeight - WORKSPACE_HELP_GAP,
            );

        setWorkspaceHelpPosition((current) => (
            current.left === left && current.top === top ? current : { left, top }
        ));
    }, []);

    useLayoutEffect(() => {
        if (!workspaceHelpOpen) return;

        positionWorkspaceHelp();
        window.addEventListener('resize', positionWorkspaceHelp);
        window.addEventListener('scroll', positionWorkspaceHelp, true);
        return () => {
            window.removeEventListener('resize', positionWorkspaceHelp);
            window.removeEventListener('scroll', positionWorkspaceHelp, true);
        };
    }, [positionWorkspaceHelp, workspaceHelpOpen]);

    const evidence = useMemo(() => selectScientificEvidence({
        latestLiveSignal,
        latestEvaluation,
    }), [latestEvaluation, latestLiveSignal]);
    const fullEvaluation = evidence.fullEvaluation;
    const epoch = evidence.currentModel?.epoch ?? 0;
    const batchLoss = evidence.batchTrend?.dataLoss.toFixed(4) ?? '—';
    const trainLoss = fullEvaluation?.trainDataLoss.toFixed(4) ?? '—';
    const testLoss = fullEvaluation?.testDataLoss.toFixed(4) ?? '—';
    const accuracy = fullEvaluation?.testAccuracy;
    const accStr = accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—';
    const compactPrimaryOutcome = fullEvaluation === null
        ? 'Not evaluated yet'
        : accuracy != null
            ? `Test accuracy ${accStr}`
            : `Test data loss ${testLoss}`;
    const compactOutcomeSummary = fullEvaluation === null
        ? compactPrimaryOutcome
        : `Step ${fullEvaluation.step.toLocaleString()} · ${compactPrimaryOutcome}`;

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

            <div
                className="forge-phase"
                role="group"
                aria-label="Workspace view"
                aria-describedby={workspaceDescriptionId}
            >
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
                <span
                    className="concept-help"
                    onMouseEnter={() => setWorkspaceHelpOpen(true)}
                    onMouseLeave={() => {
                        if (!workspaceHelpPinnedRef.current) setWorkspaceHelpOpen(false);
                    }}
                    onFocus={() => setWorkspaceHelpOpen(true)}
                    onBlur={() => {
                        workspaceHelpPinnedRef.current = false;
                        setWorkspaceHelpOpen(false);
                    }}
                    onKeyDown={(event) => {
                        if (event.key !== 'Escape' || !workspaceHelpOpen) return;
                        event.preventDefault();
                        event.stopPropagation();
                        workspaceHelpPinnedRef.current = false;
                        setWorkspaceHelpOpen(false);
                        workspaceHelpTriggerRef.current?.focus();
                    }}
                >
                    <button
                        ref={workspaceHelpTriggerRef}
                        className="concept-help__trigger"
                        type="button"
                        aria-label="About workspace views"
                        aria-expanded={workspaceHelpOpen}
                        aria-controls={workspaceDescriptionId}
                        onClick={() => {
                            workspaceHelpPinnedRef.current = !workspaceHelpPinnedRef.current;
                            setWorkspaceHelpOpen(workspaceHelpPinnedRef.current);
                        }}
                    >
                        <span aria-hidden="true">?</span>
                    </button>
                </span>
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            <div className="forge-audience-mode">
                <label className="forge-audience-mode__control">
                    <span>Workspace</span>
                    <select
                        className="select forge-audience-mode__select"
                        aria-label="Workspace profile"
                        aria-describedby="forge-audience-mode-profile-description forge-audience-mode-description"
                        value={audienceMode}
                        onChange={(event) => {
                            const nextMode = event.currentTarget.value;
                            if (!isAudienceMode(nextMode)) return;
                            setAudienceMode(nextMode);
                            setModeAnnouncement(
                                `Workspace profile: ${getAudienceProfile(nextMode).label}. Profiles change visible tools and guidance only.`,
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
                <span
                    id="forge-audience-mode-profile-description"
                    className="forge-audience-mode__note"
                >
                    {audienceProfile.description}
                </span>
                <span id="forge-audience-mode-description" className="forge-audience-mode__note">
                    Profiles change visible tools and guidance only.
                </span>
                <span
                    className="sr-only"
                    role="status"
                    aria-label="Workspace profile change"
                    aria-live="polite"
                    aria-atomic="true"
                >
                    {modeAnnouncement}
                </span>
            </div>

            <div className="forge-topbar__divider" aria-hidden />

            <div className="forge-topbar__metrics" role="group" aria-label="Training metrics">
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

            <details className="forge-compact-outcome" aria-label="Evaluation outcome">
                <summary>{compactOutcomeSummary}</summary>
                {fullEvaluation && (
                    <div className="forge-compact-outcome__body">
                        <span>{`Full evaluation at step ${fullEvaluation.step.toLocaleString()}`}</span>
                        <span>{`Train data loss (full split) ${trainLoss}`}</span>
                        <span>{`Test data loss (full split) ${testLoss}`}</span>
                        {accuracy != null && <span>{`Test accuracy ${accStr}`}</span>}
                    </div>
                )}
            </details>

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
            {typeof document !== 'undefined' ? createPortal(
                <div
                    ref={workspaceHelpDescriptionRef}
                    id={workspaceDescriptionId}
                    className={workspaceHelpOpen ? 'tooltip tooltip__content' : 'sr-only'}
                    role={workspaceHelpOpen ? 'region' : undefined}
                    aria-label={workspaceHelpOpen ? 'Build and Run views' : undefined}
                    style={workspaceHelpOpen ? {
                        position: 'fixed',
                        left: `${workspaceHelpPosition.left}px`,
                        top: `${workspaceHelpPosition.top}px`,
                        width: 'max-content',
                        maxWidth: 'min(320px, calc(100vw - 16px))',
                        boxSizing: 'border-box',
                        pointerEvents: 'none',
                        zIndex: 1200,
                    } : undefined}
                >
                    {WORKSPACE_VIEW_DESCRIPTION}
                </div>,
                document.body,
            ) : null}
        </header>
    );
});
