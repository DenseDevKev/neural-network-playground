// ── Segmented ── replaces chip-group in selectors
import { type ReactNode } from 'react';

interface Option<T extends string> {
    value: T;
    label: ReactNode;
    title?: string;
}

interface SegmentedProps<T extends string> {
    value: T;
    onChange: (v: T) => void;
    options: Option<T>[];
    full?: boolean;
    variant?: 'default' | 'primary' | 'accent';
    disabled?: boolean;
    'aria-label'?: string;
}

export function Segmented<T extends string>({
    value, onChange, options, full, variant = 'default', disabled, 'aria-label': ariaLabel,
}: SegmentedProps<T>) {
    const variantClass = variant !== 'default' ? `forge-segmented__opt--${variant}` : '';
    return (
        <div
            className={`forge-segmented ${full ? 'forge-segmented--full' : ''}`}
            role="group"
            aria-label={ariaLabel}
        >
            {options.map((o) => (
                <button
                    key={o.value}
                    className={`forge-segmented__opt ${variantClass} ${value === o.value ? 'forge-segmented__opt--active' : ''}`}
                    onClick={() => !disabled && onChange(o.value)}
                    aria-pressed={value === o.value}
                    title={o.title}
                    disabled={disabled}
                    type="button"
                >
                    {o.label}
                </button>
            ))}
        </div>
    );
}
