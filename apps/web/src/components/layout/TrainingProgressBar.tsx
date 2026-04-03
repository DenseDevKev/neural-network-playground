interface TrainingProgressBarProps {
    isTraining: boolean;
    currentEpoch: number;
    targetEpoch?: number | null;
}

export function TrainingProgressBar({ isTraining, currentEpoch, targetEpoch = null }: TrainingProgressBarProps) {
    if (!isTraining) {
        return null;
    }

    const isDeterminate = targetEpoch != null && targetEpoch > 0;
    const progress = isDeterminate
        ? Math.max(0, Math.min(100, (currentEpoch / targetEpoch) * 100))
        : null;

    return (
        <div
            className={`training-progress ${isDeterminate ? '' : 'training-progress--indeterminate'}`.trim()}
            role="progressbar"
            aria-label="Training progress"
            aria-valuemin={isDeterminate ? 0 : undefined}
            aria-valuemax={isDeterminate ? 100 : undefined}
            aria-valuenow={isDeterminate && progress != null ? Math.round(progress) : undefined}
        >
            <div
                className="training-progress__fill"
                style={progress != null ? { width: `${progress}%` } : undefined}
            />
        </div>
    );
}
