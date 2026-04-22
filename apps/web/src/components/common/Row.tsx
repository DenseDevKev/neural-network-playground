// ── Row ── label + control + optional value, grid-aligned
import { memo, type ReactNode } from 'react';

interface RowProps {
    label: string;
    value?: ReactNode;
    children?: ReactNode;
}

export const Row = memo(function Row({ label, value, children }: RowProps) {
    return (
        <div className="forge-row">
            <span className="forge-row__label">{label}</span>
            {children}
            {value !== undefined && <span className="forge-row__value">{value}</span>}
        </div>
    );
});
