import { useEffect, useRef, useState } from 'react';
import type { Destination, UtilitySurface } from '../../productShell/atelierTypes.ts';
import { useThemeStore } from '../../store/theme.ts';
import { Icon } from './ui.tsx';

export function AtelierHeader({ destination, onNavigate, onUtility }: { destination: Destination; onNavigate(value: Destination): void; onUtility(value: UtilitySurface): void }) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);
    const trigger = useRef<HTMLButtonElement>(null);
    const preference = useThemeStore((s) => s.preference);
    const setPreference = useThemeStore((s) => s.setPreference);
    useEffect(() => {
        if (!open) return;
        ref.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus();
        const outside = (event: PointerEvent) => { if (event.target instanceof Node && !ref.current?.contains(event.target)) setOpen(false); };
        document.addEventListener('pointerdown', outside);
        return () => document.removeEventListener('pointerdown', outside);
    }, [open]);
    return <header className="atelier-header">
        <button className="atelier-wordmark" type="button" onClick={() => onNavigate('playground')} aria-label="NN FORGE playground">NN·FORGE</button>
        <nav aria-label="Main navigation"><button type="button" aria-current={destination === 'playground' ? 'page' : undefined} onClick={() => onNavigate('playground')}>Playground</button><button type="button" aria-current={destination === 'saved-runs' ? 'page' : undefined} onClick={() => onNavigate('saved-runs')}>Saved runs</button></nav>
        <div className="atelier-header-end">
            <label className="atelier-theme"><Icon name={preference === 'system' ? 'system' : preference === 'light' ? 'sun' : 'moon'} /><span className="sr-only">Color theme</span><select aria-label="Color theme" value={preference} onChange={(e) => setPreference(e.target.value as 'system' | 'light' | 'dark')}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
            <button className="atelier-lessons-link" type="button" aria-current={destination === 'lessons' ? 'page' : undefined} onClick={() => onNavigate('lessons')}><Icon name="book" /><span>Lessons</span></button>
            <div className="atelier-menu" ref={ref}><button ref={trigger} className="atelier-icon-button" type="button" aria-label="Utilities" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}><Icon name="more" /></button>
                {open && <div role="menu" aria-label="Utilities" className="atelier-menu-items" onKeyDown={(event) => {
                    const items = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('button'));
                    const index = items.indexOf(document.activeElement as HTMLButtonElement);
                    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
                    const next = event.key === 'ArrowDown' ? (index + 1) % items.length : event.key === 'ArrowUp' ? (index + items.length - 1) % items.length : event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : -1;
                    if (next >= 0) { event.preventDefault(); event.stopPropagation(); items[next].focus(); }
                    if (event.key === 'Tab') setOpen(false);
                }}>{([{ id: 'exports', label: 'Export / import' }, { id: 'checkpoints', label: 'Session checkpoints' }, { id: 'preferences', label: 'Guidance' }, { id: 'help', label: 'Shortcuts & help' }] as const).map((item) => <button key={item.id} type="button" role="menuitem" onClick={() => { setOpen(false); trigger.current?.focus(); onUtility(item.id); }}>{item.label}</button>)}</div>}
            </div>
        </div>
    </header>;
}
