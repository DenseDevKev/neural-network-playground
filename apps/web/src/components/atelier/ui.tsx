import { useEffect, useId, useRef, type ReactNode, type KeyboardEvent, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { useModalFocusContainment } from '../../hooks/useModalFocusContainment.ts';
import { X, Play, Pause, StepForward, RotateCcw, Settings2, Sun, Moon, Monitor, Ellipsis, ArrowUpRight, ChevronRight, History, BookOpen, type LucideIcon } from 'lucide-react';

const icons = { close: X, play: Play, pause: Pause, step: StepForward, reset: RotateCcw, settings: Settings2, sun: Sun, moon: Moon, system: Monitor, more: Ellipsis, expand: ArrowUpRight, next: ChevronRight, history: History, book: BookOpen } satisfies Record<string, LucideIcon>;
export function Icon({ name, size = 18 }: { name: keyof typeof icons; size?: number }) { const Component = icons[name]; return <Component size={size} strokeWidth={1.6} aria-hidden="true" focusable="false" />; }

export function Tabs<T extends string>({ label, items, value, onChange, panelPrefix }: { label: string; items: readonly { id: T; label: string }[]; value: T; onChange(value: T): void; panelPrefix?: string }) {
    const refs = useRef<Array<HTMLButtonElement | null>>([]);
    const key = (event: KeyboardEvent, index: number) => {
        const next = event.key === 'ArrowRight' ? (index + 1) % items.length : event.key === 'ArrowLeft' ? (index + items.length - 1) % items.length : event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : -1;
        if (next < 0) return;
        event.preventDefault(); event.stopPropagation(); refs.current[next]?.focus(); onChange(items[next].id);
    };
    return <div className="atelier-tabs" role="tablist" aria-label={label}>{items.map((item, index) => <button key={item.id} ref={(node) => { refs.current[index] = node; }} type="button" role="tab" id={panelPrefix ? `${panelPrefix}-tab-${item.id}` : undefined} aria-controls={panelPrefix ? `${panelPrefix}-${item.id}` : undefined} aria-selected={value === item.id} tabIndex={value === item.id ? 0 : -1} onKeyDown={(event) => key(event, index)} onClick={() => onChange(item.id)}>{item.label}</button>)}</div>;
}

export function Dialog({ title, children, onClose, alert = false, description, persistent = false, backgroundRef }: { title: string; children: ReactNode; onClose(): void; alert?: boolean; description?: string; persistent?: boolean; backgroundRef?: RefObject<HTMLElement | null> }) {
    const ref = useRef<HTMLDialogElement>(null);
    const titleId = useId();
    const descriptionId = useId();
    const fallbackBackground = useRef<HTMLElement | null>(null);
    useModalFocusContainment(persistent, ref, backgroundRef ?? fallbackBackground);
    useEffect(() => {
        const dialog = ref.current;
        const prior = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        if (dialog && !dialog.open) {
            if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
            dialog.focus({ preventScroll:true });
        }
        return () => { if (dialog?.open) dialog.close?.(); if (prior?.isConnected) prior.focus(); };
    }, []);
    return createPortal(<dialog ref={ref} className="atelier-dialog" role={alert ? 'alertdialog' : 'dialog'} aria-labelledby={titleId} aria-describedby={description ? descriptionId : undefined} tabIndex={-1} onCancel={(event) => { event.preventDefault(); onClose(); }} onKeyDown={(event) => event.stopPropagation()}>
        <header><h2 id={titleId}>{title}</h2>{!persistent && <button className="atelier-icon-button" type="button" onClick={onClose} aria-label={`Close ${title}`}><Icon name="close" /></button>}</header>
        <div className="atelier-dialog-body">{description && <p className="sr-only" id={descriptionId}>{description}</p>}{children}</div>
    </dialog>, document.body);
}
