// ── Training Controls ──
import { memo } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { Tooltip } from '../common/Tooltip.tsx';

interface Props {
    training: TrainingHook;
}

const SPEED_OPTIONS: { value: number; label: string }[] = [
    { value: 1, label: '1' },
    { value: 5, label: '5' },
    { value: 10, label: '10' },
    { value: 25, label: '25' },
    { value: 50, label: '50' },
];

export const TrainingControls = memo(function TrainingControls({ training }: Props) {
    const status = useTrainingStore((s) => s.status);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const stepsPerFrame = useTrainingStore((s) => s.stepsPerFrame);
    const setStepsPerFrame = useTrainingStore((s) => s.setStepsPerFrame);
    const isRunning = status === 'running';

    return (
        <div className="training-bar">
            <div className="training-bar__controls">
                <Tooltip content={isRunning ? 'Pause training' : 'Start training'} shortcut="Space">
                    <button
                        className={`btn btn--play btn--control ${isRunning ? 'running' : ''}`}
                        onClick={isRunning ? training.pause : training.play}
                        aria-label={isRunning ? 'Pause training' : 'Start training'}
                    >
                        <span className="btn__icon" aria-hidden="true">{isRunning ? '⏸' : '▶'}</span>
                        <span className="btn__label">{isRunning ? 'Pause' : 'Play'}</span>
                        <span className="btn__shortcut">Space</span>
                    </button>
                </Tooltip>
                <Tooltip content="Run one training step" shortcut="→">
                    <button
                        className="btn btn--ghost btn--control"
                        onClick={training.step}
                        aria-label="Run one training step"
                    >
                        <span className="btn__icon" aria-hidden="true">→</span>
                        <span className="btn__label">Step</span>
                        <span className="btn__shortcut">→</span>
                    </button>
                </Tooltip>
                <Tooltip content="Reset model and regenerate data" shortcut="R">
                    <button
                        className="btn btn--ghost btn--control"
                        onClick={training.reset}
                        aria-label="Reset model and data"
                    >
                        <span className="btn__icon" aria-hidden="true">↺</span>
                        <span className="btn__label">Reset</span>
                        <span className="btn__shortcut">R</span>
                    </button>
                </Tooltip>
            </div>

            <div className="training-bar__speed" aria-label="Training speed">
                <span className="training-bar__speed-label">Steps/frame:</span>
                {SPEED_OPTIONS.map((opt) => {
                    const stepLabel = `${opt.value} ${opt.value === 1 ? 'step' : 'steps'} per frame`;
                    return (
                        <Tooltip key={opt.value} content={stepLabel}>
                            <button
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

            <div className="training-bar__info">
                {isRunning && (
                    <span className="training-status">
                        <span className="training-status__dot" aria-hidden="true" />
                        Training...
                    </span>
                )}
                {snapshot && (
                    <>
                        <span>Step {snapshot.step.toLocaleString()}</span>
                        <span>Epoch {snapshot.epoch}</span>
                    </>
                )}
            </div>
        </div>
    );
});
