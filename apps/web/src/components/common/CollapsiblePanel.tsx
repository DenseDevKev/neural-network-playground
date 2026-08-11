import { Suspense, useEffect, useId, useRef, useState } from 'react';
import { Tooltip } from './Tooltip.tsx';
import { createAppMeasureName } from '../../performance/interactionMeasures.ts';

interface CollapsiblePanelProps {
    storageId: string;
    title: string;
    children: React.ReactNode;
    defaultExpanded?: boolean;
    badge?: string | number;
    className?: string;
    tooltipContent?: string;
    lazyMount?: boolean;
    fallback?: React.ReactNode;
}

const STORAGE_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

type PendingStorageOperation =
    | { type: 'remove'; key: string }
    | { type: 'migrate'; legacyKey: string; value: 'true' | 'false'; v2Key: string };

interface InitialPanelState {
    isExpanded: boolean;
    pendingStorageOperation: PendingStorageOperation | null;
}

function getLegacyPanelStorageKey(title: string): string {
    return `panel-${title.toLowerCase().replace(/\s+/g, '-')}`;
}

function getV2PanelStorageKey(storageId: string): string {
    return `panel-v2-${storageId}`;
}

function getStorage(): Storage | null {
    try {
        return window.localStorage ?? null;
    } catch {
        return null;
    }
}

function readStorageValue(storage: Storage, key: string): { ok: true; value: string | null } | { ok: false } {
    try {
        return { ok: true, value: storage.getItem(key) };
    } catch {
        return { ok: false };
    }
}

function writeStorageValue(storage: Storage, key: string, value: string): boolean {
    try {
        storage.setItem(key, value);
        return true;
    } catch {
        return false;
    }
}

function removeStorageValue(storage: Storage, key: string): void {
    try {
        storage.removeItem(key);
    } catch {
        // Storage cleanup is best effort.
    }
}

function readInitialPanelState(
    storageId: string,
    initialTitle: string,
    defaultExpanded: boolean,
): InitialPanelState {
    const storage = getStorage();
    if (!storage) {
        return { isExpanded: defaultExpanded, pendingStorageOperation: null };
    }

    const v2Key = getV2PanelStorageKey(storageId);
    const v2Read = readStorageValue(storage, v2Key);
    if (!v2Read.ok) {
        return { isExpanded: defaultExpanded, pendingStorageOperation: null };
    }
    if (v2Read.value === 'true' || v2Read.value === 'false') {
        return {
            isExpanded: v2Read.value === 'true',
            pendingStorageOperation: null,
        };
    }
    if (v2Read.value !== null) {
        return {
            isExpanded: defaultExpanded,
            pendingStorageOperation: { type: 'remove', key: v2Key },
        };
    }

    const legacyKey = getLegacyPanelStorageKey(initialTitle);
    const legacyRead = readStorageValue(storage, legacyKey);
    if (!legacyRead.ok) {
        return { isExpanded: defaultExpanded, pendingStorageOperation: null };
    }
    if (legacyRead.value === 'true' || legacyRead.value === 'false') {
        return {
            isExpanded: legacyRead.value === 'true',
            pendingStorageOperation: {
                type: 'migrate',
                legacyKey,
                value: legacyRead.value,
                v2Key,
            },
        };
    }
    if (legacyRead.value !== null) {
        return {
            isExpanded: defaultExpanded,
            pendingStorageOperation: { type: 'remove', key: legacyKey },
        };
    }

    return { isExpanded: defaultExpanded, pendingStorageOperation: null };
}

export function CollapsiblePanel({
    storageId,
    title,
    children,
    defaultExpanded = true,
    badge,
    className,
    tooltipContent,
    lazyMount = false,
    fallback = null,
}: CollapsiblePanelProps) {
    if (import.meta.env.DEV && !STORAGE_ID_PATTERN.test(storageId)) {
        throw new Error(
            `Invalid CollapsiblePanel storageId "${storageId}". Expected lowercase kebab-case matching /^[a-z0-9]+(?:-[a-z0-9]+)*$/.`,
        );
    }

    const initialStorageIdRef = useRef(storageId);
    if (import.meta.env.DEV && initialStorageIdRef.current !== storageId) {
        throw new Error(
            `CollapsiblePanel storageId cannot change while mounted. Remount with key={storageId}. Initial: "${initialStorageIdRef.current}"; received: "${storageId}".`,
        );
    }

    const initialTitleRef = useRef(title);
    const initialStateRef = useRef<InitialPanelState | null>(null);
    if (initialStateRef.current === null) {
        initialStateRef.current = readInitialPanelState(
            initialStorageIdRef.current,
            initialTitleRef.current,
            defaultExpanded,
        );
    }

    const [isExpanded, setIsExpanded] = useState(initialStateRef.current.isExpanded);
    const [hasMountedContent, setHasMountedContent] = useState(
        !lazyMount || initialStateRef.current.isExpanded,
    );
    const [announcement, setAnnouncement] = useState('');
    const [contentHeight, setContentHeight] = useState(
        initialStateRef.current.isExpanded ? 'none' : '0px',
    );
    const contentRef = useRef<HTMLDivElement>(null);
    const contentId = useId();
    const titleId = useId();
    const animationFrameRef = useRef<number | null>(null);
    const measurementIdRef = useRef<string | null>(null);

    useEffect(() => {
        const pendingOperation = initialStateRef.current?.pendingStorageOperation ?? null;
        if (!pendingOperation) {
            return;
        }

        initialStateRef.current!.pendingStorageOperation = null;
        const storage = getStorage();
        if (!storage) {
            return;
        }

        if (pendingOperation.type === 'remove') {
            removeStorageValue(storage, pendingOperation.key);
            return;
        }

        if (writeStorageValue(storage, pendingOperation.v2Key, pendingOperation.value)) {
            removeStorageValue(storage, pendingOperation.legacyKey);
        }
    }, []);

    useEffect(() => {
        if (isExpanded) {
            setHasMountedContent(true);
        }
    }, [isExpanded]);

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
                const measurementId = createAppMeasureName(
                    `panel-toggle:${title}:${performance.now()}`,
                );
                measurementIdRef.current = measurementId;
                performance.mark(`${measurementId}:start`);
                requestAnimationFrame(() => {
                    performance.mark(`${measurementId}:end`);
                    performance.measure(measurementId, `${measurementId}:start`, `${measurementId}:end`);
                    performance.clearMarks(`${measurementId}:start`);
                    performance.clearMarks(`${measurementId}:end`);
                });
            }

            const storage = getStorage();
            if (storage) {
                writeStorageValue(
                    storage,
                    getV2PanelStorageKey(initialStorageIdRef.current),
                    String(next),
                );
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
                    {hasMountedContent ? <Suspense fallback={fallback}>{children}</Suspense> : null}
                </div>
            </div>
            <span className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</span>
        </section>
    );
}
