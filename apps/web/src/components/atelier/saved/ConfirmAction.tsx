import { useState, type ReactNode } from 'react';
import { Dialog } from '../ui.tsx';
export function ConfirmAction({ label, title, children, onConfirm, descriptionId, disabled = false }: { label: string; title: string; children: ReactNode; onConfirm(): Promise<boolean | void>; descriptionId?: string; disabled?: boolean }) {
    const [open, setOpen] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    async function confirm() {
        if (busy || disabled) return;
        setBusy(true); setError(null);
        try {
            const result = await onConfirm();
            if (result === false) setError(`${title} failed. Your saved evidence has been retained. Try again after resolving the storage error.`);
            else setOpen(false);
        } catch (reason) { setError(`${title} failed: ${reason instanceof Error ? reason.message : String(reason)}`); }
        finally { setBusy(false); }
    }
    return <><button type="button" aria-describedby={descriptionId} disabled={disabled} onClick={() => setOpen(true)}>{label}</button>{open && <Dialog title={title} alert onClose={() => { if (!busy) setOpen(false); }}><p>{children}</p>{error && <p role="alert">{error}</p>}<div className="saved-actions"><button type="button" disabled={busy || disabled} onClick={() => setOpen(false)}>Cancel</button><button className="saved-primary" type="button" disabled={busy || disabled} onClick={() => { void confirm(); }}>{busy ? 'Working…' : 'Confirm'}</button></div></Dialog>}</>;
}
