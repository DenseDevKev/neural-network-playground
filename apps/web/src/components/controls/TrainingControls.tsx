// ── Training Controls ──
import { Fragment, memo, useEffect, useId, useMemo, useState } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';
import type { SaveCurrentRunController } from '../../hooks/useSaveCurrentRun.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { Tooltip } from '../common/Tooltip.tsx';
import { getTrainingLifecycleUi } from './trainingLifecycle.ts';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';
import { TRAINING_SHORTCUTS, type TrainingShortcutAction } from '../../shortcuts/trainingShortcuts.ts';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

interface Props {
    training: TrainingHook;
    saveController?: SaveCurrentRunController;
}

const SPEED_OPTIONS: { value: number; label: string }[] = [
    { value: 1, label: '1' },
    { value: 5, label: '5' },
    { value: 10, label: '10' },
    { value: 25, label: '25' },
    { value: 50, label: '50' },
];
const RESTORE_GUARANTEE = 'Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only.';
const trainingShortcutLabel = (action: TrainingShortcutAction) => (
    TRAINING_SHORTCUTS.find((shortcut) => shortcut.action === action)?.label
);

export const TrainingControls = memo(function TrainingControls({ training, saveController }: Props) {
    const saveReasonId = useId();
    const guidanceLevel = useAudienceGuidanceLevel();
    const status = useTrainingStore((s) => s.status);
    const currentModel = useTrainingStore((s) => (
        selectScientificEvidence({
            latestLiveSignal: s.latestLiveSignal,
            latestEvaluation: s.latestEvaluation,
        }).currentModel
    ));
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const setStepsPerFrame = useTrainingStore((s) => s.setStepsPerFrame);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const checkpointTimeline = useTrainingStore((s) => s.checkpointTimeline);
    const controlId = useId();
    const restoreGuaranteeId = `${controlId}-restore-guarantee`;
    const trainingResetEffectsId = `${controlId}-training-reset-effects`;
    const isRunning = status === 'running';
    const lifecycle = getTrainingLifecycleUi({ status, pauseReason, pendingConfigSource });
    const blockConfigAction = lifecycle.isBlocked;
    const checkpoints = checkpointTimeline.checkpoints;
    const defaultCheckpointIndex = useMemo(() => {
        if (checkpoints.length === 0) return 0;
        const highlightedId = checkpointTimeline.restoredCheckpointId ?? checkpointTimeline.liveCheckpointId;
        const highlightedIndex = checkpoints.findIndex((checkpoint) => checkpoint.id === highlightedId);
        return highlightedIndex >= 0 ? highlightedIndex : checkpoints.length - 1;
    }, [checkpointTimeline.liveCheckpointId, checkpointTimeline.restoredCheckpointId, checkpoints]);
    const [selectedCheckpointIndex, setSelectedCheckpointIndex] = useState(defaultCheckpointIndex);

    useEffect(() => {
        setSelectedCheckpointIndex(defaultCheckpointIndex);
    }, [defaultCheckpointIndex]);

    const selectedCheckpoint = checkpoints[selectedCheckpointIndex] ?? checkpoints[defaultCheckpointIndex];
    const checkpointControlsDisabled = blockConfigAction || isRunning || !selectedCheckpoint;
    const clampCheckpointIndex = (index: number) => Math.min(Math.max(index, 0), Math.max(0, checkpoints.length - 1));

    return (
        <div className="training-bar" role="region" aria-label="Timeline strip">
            <div className="training-bar__controls">
                <Tooltip
                    content={
                        isRunning
                            ? 'Cause: pause stops the update loop. Effect: the current boundary stays frozen so you can inspect metrics and weights.'
                            : lifecycle.disabledReason
                                ? 'Configuration is syncing. Training can resume when the current settings reach the worker.'
                                : 'Cause: play repeats weight updates continuously. Effect: the boundary and metrics evolve until you pause or reset.'
                    }
                    shortcut={trainingShortcutLabel('play-pause')}
                >
                    <button
                        type="button"
                        className={`btn btn--play btn--control ${isRunning ? 'running' : ''}`}
                        onClick={isRunning ? training.pause : training.play}
                        aria-label={lifecycle.primaryAriaLabel}
                        disabled={!isRunning && blockConfigAction}
                    >
                        <span className="btn__icon" aria-hidden="true">{isRunning ? '⏸' : '▶'}</span>
                        <span className="btn__label">{lifecycle.primaryLabel}</span>
                        <span className="btn__shortcut">{trainingShortcutLabel('play-pause')}</span>
                    </button>
                </Tooltip>
                <Tooltip content="Cause: step applies one update. Effect: you can connect a single weight change to the next boundary or loss movement." shortcut={trainingShortcutLabel('step')}>
                    <button
                        type="button"
                        className="btn btn--ghost btn--control"
                        onClick={training.step}
                        aria-label="Run one training step"
                        disabled={blockConfigAction}
                    >
                        <span className="btn__icon" aria-hidden="true">→</span>
                        <span className="btn__label">Step</span>
                        <span className="btn__shortcut">{trainingShortcutLabel('step')}</span>
                    </button>
                </Tooltip>
                <Tooltip content={STATE_EFFECTS['training-reset']} shortcut={trainingShortcutLabel('reset')}>
                    <button
                        type="button"
                        className="btn btn--ghost btn--control"
                        onClick={training.reset}
                        aria-label="Reset training"
                        aria-describedby={trainingResetEffectsId}
                        disabled={blockConfigAction}
                    >
                        <span className="btn__icon" aria-hidden="true">↺</span>
                        <span className="btn__label">Reset training</span>
                        <span className="btn__shortcut">{trainingShortcutLabel('reset')}</span>
                    </button>
                </Tooltip>
                <span id={trainingResetEffectsId} className="sr-only">
                    {STATE_EFFECTS['training-reset']}
                </span>
            </div>

            {saveController && (
                <div className="training-bar__save">
                    <button type="button" className="btn btn--ghost btn--control"
                        aria-label="Save run"
                        aria-describedby={saveController.disabledReason ? saveReasonId : undefined}
                        disabled={saveController.disabledReason !== null}
                        onClick={() => { void saveController.commands.save(); }}>
                        {saveController.busy ? 'Saving…' : 'Save run'}
                    </button>
                    {saveController.disabledReason && <span className="sr-only" id={saveReasonId}>{saveController.disabledReason}</span>}
                    {saveController.pending && <>
                        <button type="button" className="btn btn--ghost btn--control" disabled={saveController.busy}
                            onClick={() => { void saveController.commands.retry(); }}>Retry pending artifact</button>
                        <button type="button" className="btn btn--ghost btn--control" disabled={saveController.busy}
                            onClick={() => { void saveController.commands.discard(); }}>Discard pending artifact</button>
                    </>}
                    {saveController.error && <span role="alert" className="training-bar__save-error">{saveController.error}</span>}
                </div>
            )}

            <details className="training-shortcuts" aria-label="Keyboard shortcuts">
                <summary>Keyboard shortcuts</summary>
                <dl>
                    {TRAINING_SHORTCUTS.map(({ code, label, description }) => (
                        <Fragment key={code}>
                            <dt><kbd>{label}</kbd></dt>
                            <dd>{description}</dd>
                        </Fragment>
                    ))}
                </dl>
            </details>

            <div className="training-bar__speed" aria-label="Training speed">
                <span className="training-bar__speed-label">Steps/frame:</span>
                {SPEED_OPTIONS.map((opt) => {
                    const stepLabel = `${opt.value} ${opt.value === 1 ? 'step' : 'steps'} per frame`;
                    const speedTooltip = opt.value === 5
                        ? 'Cause: higher speed runs more updates per animation frame. Effect: learning completes sooner, but individual changes are harder to inspect.'
                        : `Cause: ${stepLabel} controls how many updates happen before each redraw. Effect: lower values are easier to inspect, while higher values finish faster.`;
                    return (
                        <Tooltip key={opt.value} content={speedTooltip}>
                            <button
                                type="button"
                                className={`speed-btn ${stepsPerFrame === opt.value ? 'active' : ''}`}
                                onClick={() => setStepsPerFrame(opt.value)}
                                aria-pressed={stepsPerFrame === opt.value}
                                aria-label={stepLabel}
                            >
                                {opt.label}
                            </button>
                        </Tooltip>
                    );
                })}
            </div>

            {selectedCheckpoint && (
                <div
                    className="training-bar__timeline"
                    aria-label="Checkpoint timeline controls"
                >
                    <span className="training-bar__timeline-label training-bar__timeline-label--concept">
                        <span>Timeline</span>
                        <ConceptHelp
                            conceptId="checkpoint"
                            guidanceLevel={guidanceLevel}
                            className="concept-help--above concept-help--end"
                        />
                    </span>
                    <input
                        className="training-bar__timeline-range"
                        type="range"
                        min={0}
                        max={Math.max(0, checkpoints.length - 1)}
                        step={1}
                        value={selectedCheckpointIndex}
                        onChange={(event) => setSelectedCheckpointIndex(Number(event.currentTarget.value))}
                        onKeyDown={(event) => {
                            if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
                                event.preventDefault();
                                setSelectedCheckpointIndex((index) => clampCheckpointIndex(index - 1));
                            } else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
                                event.preventDefault();
                                setSelectedCheckpointIndex((index) => clampCheckpointIndex(index + 1));
                            } else if (event.key === 'Home') {
                                event.preventDefault();
                                setSelectedCheckpointIndex(0);
                            } else if (event.key === 'End') {
                                event.preventDefault();
                                setSelectedCheckpointIndex(Math.max(0, checkpoints.length - 1));
                            }
                        }}
                        aria-label="Checkpoint timeline"
                        aria-valuetext={selectedCheckpoint.label}
                        disabled={checkpointControlsDisabled}
                    />
                    <span className="training-bar__timeline-meta">
                        <strong>{selectedCheckpoint.label}</strong>
                        <span>
                            train {selectedCheckpoint.trainDataLoss.toFixed(3)} / test {selectedCheckpoint.testDataLoss.toFixed(3)}
                        </span>
                    </span>
                    <Tooltip
                        content={(
                            <span>
                                <span>Restore in-session parameters and optimizer state</span>
                                <br />
                                <span>
                                    {RESTORE_GUARANTEE}
                                </span>
                            </span>
                        )}
                    >
                        <button
                            type="button"
                            className="btn btn--ghost btn--control training-bar__timeline-restore"
                            onClick={() => void training.restoreCheckpoint(selectedCheckpoint.id)}
                            aria-label={`Restore checkpoint ${selectedCheckpoint.label}`}
                            aria-describedby={restoreGuaranteeId}
                            disabled={checkpointControlsDisabled}
                        >
                            Restore
                        </button>
                    </Tooltip>
                    <span id={restoreGuaranteeId} className="sr-only">
                        {RESTORE_GUARANTEE}
                    </span>
                </div>
            )}

            <div className="training-bar__info">
                {isRunning && (
                    <span className="training-status">
                        <span className="training-status__dot" aria-hidden="true" />
                        Training...
                    </span>
                )}
                {!isRunning && lifecycle.statusText && (
                    <span className="training-status training-status--muted">
                        {lifecycle.statusText}
                    </span>
                )}
                {currentModel && (
                    <>
                        <span>Step {currentModel.step.toLocaleString()}</span>
                        <span>
                            <span>Epoch {currentModel.epoch}</span>
                            <ConceptHelp
                                conceptId="epoch"
                                guidanceLevel={guidanceLevel}
                                className="concept-help--above concept-help--end concept-help--viewport-overlay"
                            />
                        </span>
                    </>
                )}
            </div>
        </div>
    );
});
