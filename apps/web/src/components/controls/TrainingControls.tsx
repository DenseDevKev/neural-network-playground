// ── Training Controls ──
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';

interface Props {
    training: TrainingHook;
}

const SPEED_OPTIONS: { value: number; label: string }[] = [
    { value: 1, label: '1×' },
    { value: 5, label: '5×' },
    { value: 10, label: '10×' },
    { value: 25, label: '25×' },
    { value: 50, label: '50×' },
];

export function TrainingControls({ training }: Props) {
    const status = usePlaygroundStore((s) => s.status);
    const snapshot = usePlaygroundStore((s) => s.snapshot);
    const stepsPerFrame = usePlaygroundStore((s) => s.stepsPerFrame);
    const setStepsPerFrame = usePlaygroundStore((s) => s.setStepsPerFrame);
    const isRunning = status === 'running';

    return (
        <div className="training-bar">
            <div className="training-bar__controls">
                <button
                    className={`btn btn--play ${isRunning ? 'running' : ''}`}
                    onClick={isRunning ? training.pause : training.play}
                    aria-label={isRunning ? 'Pause training' : 'Start training'}
                    title={isRunning ? 'Pause (Space)' : 'Play (Space)'}
                >
                    {isRunning ? '⏸' : '▶'}
                </button>
                <button
                    className="btn btn--ghost"
                    onClick={training.step}
                    aria-label="Run one training step"
                    title="Step (→)"
                >
                    Step
                </button>
                <button
                    className="btn btn--ghost"
                    onClick={training.reset}
                    aria-label="Reset model and data"
                    title="Reset (R)"
                >
                    Reset
                </button>
            </div>

            {/* Speed selector */}
            <div className="training-bar__speed" aria-label="Training speed">
                {SPEED_OPTIONS.map((opt) => (
                    <button
                        key={opt.value}
                        className={`speed-btn ${stepsPerFrame === opt.value ? 'active' : ''}`}
                        onClick={() => setStepsPerFrame(opt.value)}
                        title={`${opt.value} steps per frame`}
                        aria-pressed={stepsPerFrame === opt.value}
                    >
                        {opt.label}
                    </button>
                ))}
            </div>

            <div className="training-bar__info">
                {snapshot && (
                    <>
                        <span>Step {snapshot.step}</span>
                        <span>Epoch {snapshot.epoch}</span>
                    </>
                )}
            </div>
        </div>
    );
}
