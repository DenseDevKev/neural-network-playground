import { useEffect, useId, useRef, useState } from 'react';
import { Tooltip } from './Tooltip.tsx';

interface CollapsiblePanelProps {
    title: string;
    children: React.ReactNode;
    defaultExpanded?: boolean;
    badge?: string | number;
    className?: string;
    tooltipContent?: string;
}

function getPanelStorageKey(title: string): string {
    return `panel-${title.toLowerCase().replace(/\s+/g, '-')}`;
}

export function CollapsiblePanel({
    title,
    children,
    defaultExpanded = true,
    badge,
    className,
    tooltipContent,
}: CollapsiblePanelProps) {
    const [isExpanded, setIsExpanded] = useState(defaultExpanded);
    const [announcement, setAnnouncement] = useState('');
    const [contentHeight, setContentHeight] = useState(defaultExpanded ? 'none' : '0px');
    const contentRef = useRef<HTMLDivElement>(null);
    const contentId = useId();
    const titleId = useId();
    const animationFrameRef = useRef<number | null>(null);
    const measurementIdRef = useRef<string | null>(null);

    useEffect(() => {
        try {
            const storageKey = getPanelStorageKey(title);
            const savedState = window.localStorage.getItem(storageKey);
            if (savedState === 'true') {
                setIsExpanded(true);
            } else if (savedState === 'false') {
                setIsExpanded(false);
            } else if (savedState !== null) {
                window.localStorage.removeItem(storageKey);
            }
        } catch {
            // Ignore localStorage read failures and fall back to defaults.
        }
    }, [title]);

    useEffect(() => {
        const contentNode = contentRef.current;
        if (!contentNode) {
            return;
        }

        if (!isExpanded) {
            setContentHeight('0px');
            return;
        }

        const updateHeight = () => {
            if (animationFrameRef.current !== null) {
                cancelAnimationFrame(animationFrameRef.current);
            }

            animationFrameRef.current = requestAnimationFrame(() => {
                setContentHeight(`${contentNode.scrollHeight}px`);
                animationFrameRef.current = null;
            });
        };

        updateHeight();

        if (typeof ResizeObserver === 'undefined') {
            return;
        }

        const observer = new ResizeObserver(updateHeight);
        observer.observe(contentNode);
        return () => {
            observer.disconnect();
            if (animationFrameRef.current !== null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [children, isExpanded]);

    const toggle = () => {
        setIsExpanded((previous) => {
            const next = !previous;

            if (import.meta.env.DEV && typeof performance !== 'undefined') {
                const measurementId = `panel-toggle:${title}:${performance.now()}`;
                measurementIdRef.current = measurementId;
                performance.mark(`${measurementId}:start`);
                requestAnimationFrame(() => {
                    performance.mark(`${measurementId}:end`);
                    performance.measure(measurementId, `${measurementId}:start`, `${measurementId}:end`);
                    performance.clearMarks(`${measurementId}:start`);
                    performance.clearMarks(`${measurementId}:end`);
                });
            }

            try {
                window.localStorage.setItem(getPanelStorageKey(title), String(next));
            } catch {
                // Ignore localStorage write failures and continue with in-memory state.
            }
            setAnnouncement(`${title} ${next ? 'expanded' : 'collapsed'}`);
            return next;
        });
    };

    return (
        <section className={`panel collapsible-panel${className ? ` ${className}` : ''}`}>
            <Tooltip
                content={tooltipContent ?? `${isExpanded ? 'Collapse' : 'Expand'} the ${title.toLowerCase()} panel`}
                block
            >
                <button
                    className="panel__header"
                    type="button"
                    onClick={toggle}
                    aria-expanded={isExpanded}
                    aria-controls={contentId}
                >
                    <span className="panel__icon" aria-hidden="true">{isExpanded ? '▾' : '▸'}</span>
                    <span className="panel__title" id={titleId}>{title}</span>
                    {badge != null && <span className="panel__badge">{badge}</span>}
                </button>
            </Tooltip>
            <div
                id={contentId}
                className="panel__content"
                role="region"
                aria-labelledby={titleId}
                aria-hidden={!isExpanded}
                style={{
                    maxHeight: contentHeight,
                    transition: 'max-height 300ms cubic-bezier(0.16, 1, 0.3, 1)',
                }}
            >
                <div ref={contentRef} className="panel__content-inner">
                    {children}
                </div>
            </div>
            <span className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</span>
        </section>
    );
}
