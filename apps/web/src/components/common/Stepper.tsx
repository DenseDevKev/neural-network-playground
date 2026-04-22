// ── Stepper ── +/- with value display, replaces ad-hoc layer-controls
import { memo } from 'react';

interface StepperProps {
    value: number;
    min?: number;
    max?: number;
    onChange: (v: number) => void;
    'aria-label'?: string;
}

export const Stepper = memo(function Stepper({
    value, min = 0, max = Infinity, onChange, 'aria-label': ariaLabel,
}: StepperProps) {
    return (
        <div className="forge-stepper" role="group" aria-label={ariaLabel}>
            <button
                className="forge-stepper__btn"
                onClick={() => onChange(Math.max(min, value - 1))}
                disabled={value <= min}
                aria-label="Decrease"
                type="button"
            >
                −
            </button>
            <span className="forge-stepper__value" aria-live="polite">{value}</span>
            <button
                className="forge-stepper__btn"
                onClick={() => onChange(Math.min(max, value + 1))}
                disabled={value >= max}
                aria-label="Increase"
                type="button"
            >
                +
            </button>
        </div>
    );
});
