// ── Header Component ──
import { memo, useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { TrainingProgressBar } from './TrainingProgressBar.tsx';

interface HeaderProps {
    training: Pick<TrainingHook, 'play' | 'pause'>;
}

function useMetricHighlight(value: string) {
    const previousValueRef = useRef(value);
    const [isHighlighted, setIsHighlighted] = useState(false);

    useEffect(() => {
        if (previousValueRef.current === value) {
            return;
        }

        previousValueRef.current = value;
        setIsHighlighted(true);

        const timeoutId = window.setTimeout(() => {
            setIsHighlighted(false);
        }, 200);

        return () => window.clearTimeout(timeoutId);
    }, [value]);

    return isHighlighted;
}

export const Header = memo(function Header({ training }: HeaderProps) {
    const snapshot = useTrainingStore((s) => s.snapshot);
    const status = useTrainingStore((s) => s.status);

    const epoch = snapshot?.epoch ?? 0;
    const trainLoss = snapshot?.trainLoss ?? 0;
    const testLoss = snapshot?.testLoss ?? 0;
    const accuracy = snapshot?.trainMetrics?.accuracy;
    const epochValue = String(epoch).padStart(4, '0');
    const trainLossValue = trainLoss.toFixed(4);
    const testLossValue = testLoss.toFixed(4);
    const accuracyValue = accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : null;

    const epochHighlighted = useMetricHighlight(epochValue);
    const trainLossHighlighted = useMetricHighlight(trainLossValue);
    const testLossHighlighted = useMetricHighlight(testLossValue);
    const accuracyHighlighted = useMetricHighlight(accuracyValue ?? '');
    const isRunning = status === 'running';

    return (
        <header className="header">
            <div className="header__brand">

                <h1 className="header__title">
                    <span>Neural</span> Architect
                </h1>
            </div>
            <div className="header__metrics">
                <div className="header__metric">
                    <span className="header__metric-label">Epoch</span>
                    <span className={`header__metric-value header__metric-value--epoch ${epochHighlighted ? 'header__metric-value--updated' : ''}`}>
                        {epochValue}
                    </span>
                </div>
                <div className="header__metric">
                    <span className="header__metric-label">Train Loss</span>
                    <span className={`header__metric-value header__metric-value--train ${trainLossHighlighted ? 'header__metric-value--updated' : ''}`}>
                        {trainLossValue}
                    </span>
                </div>
                <div className="header__metric">
                    <span className="header__metric-label">Test Loss</span>
                    <span className={`header__metric-value header__metric-value--test ${testLossHighlighted ? 'header__metric-value--updated' : ''}`}>
                        {testLossValue}
                    </span>
                </div>
                {accuracy != null && (
                    <div className="header__metric">
                        <span className="header__metric-label">Accuracy</span>
                        <span className={`header__metric-value header__metric-value--train ${accuracyHighlighted ? 'header__metric-value--updated' : ''}`}>
                            {accuracyValue}
                        </span>
                    </div>
                )}
            </div>
            <button
                className={`header__mobile-play btn btn--play ${isRunning ? 'running' : ''}`}
                onClick={isRunning ? training.pause : training.play}
                aria-label={isRunning ? 'Pause training' : 'Start training'}
            >
                {isRunning ? '⏸' : '▶'}
            </button>
            <TrainingProgressBar isTraining={isRunning} currentEpoch={epoch} />
        </header>
    );
});
