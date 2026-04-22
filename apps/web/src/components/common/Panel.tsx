// ── Panel ── top-level layout panel with phase tag
// Replaces CollapsiblePanel at the layout region level.
// phase='both' shows no tag; 'build' / 'run' shows a colored badge.

import { memo, type ReactNode } from 'react';

type Phase = 'build' | 'run' | 'both';

interface PanelProps {
    title: string;
    phase?: Phase;
    fill?: boolean;
    children?: ReactNode;
    className?: string;
    bodyClassName?: string;
    tight?: boolean;
    flush?: boolean;
}

export const Panel = memo(function Panel({
    title, phase = 'both', fill, children, className = '',
    bodyClassName = '', tight, flush,
}: PanelProps) {
    const bodyMod = flush ? 'forge-panel__body--flush'
        : tight ? 'forge-panel__body--tight'
        : '';
    return (
        <div className={`forge-panel ${fill ? 'forge-panel--fill' : ''} ${className}`}>
            <div className="forge-panel__head">
                <span className="forge-panel__grip" aria-hidden />
                <span className="forge-panel__title">{title}</span>
                {phase !== 'both' && (
                    <span className={`forge-panel__phase-tag forge-panel__phase-tag--${phase}`}>
                        {phase}
                    </span>
                )}
            </div>
            <div className={`forge-panel__body ${bodyMod} ${bodyClassName}`}>
                {children}
            </div>
        </div>
    );
});
