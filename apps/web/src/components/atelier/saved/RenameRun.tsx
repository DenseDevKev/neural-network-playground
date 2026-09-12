import { useState } from 'react';
import type { ExperimentRunRecordV2 } from '@nn-playground/shared';
import { Dialog } from '../ui.tsx';
import { useExperimentMemoryStore } from '../../../store/experimentMemoryStore.ts';
export function RenameRun({ record }: { record: ExperimentRunRecordV2 }) {
    const [open, setOpen] = useState(false); const [title, setTitle] = useState(record.title ?? '');
    const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null);
    const invalid = !title.trim() || Array.from(title.trim()).length > 120;
    async function save() {
        if (invalid || busy) return;
        setBusy(true); setError(null);
        try {
            if (await useExperimentMemoryStore.getState().renameRecord(record.id, title.trim())) setOpen(false);
            else setError('Rename failed. Resolve the storage error and try again.');
        } catch (reason) { setError(`Rename failed: ${String(reason)}`); }
        finally { setBusy(false); }
    }
    return <><button type="button" aria-label={`Rename ${record.title ?? record.id}`} onClick={() => { setTitle(record.title ?? ''); setOpen(true); }}>Rename</button>{open && <Dialog title="Rename run" onClose={() => { if (!busy) setOpen(false); }}><label className="saved-name">Run name<input value={title} aria-invalid={invalid} onChange={(event) => setTitle(event.target.value)} /></label><p>Use 1–120 Unicode code points.</p>{error && <p role="alert">{error}</p>}<button type="button" className="saved-primary" disabled={invalid || busy} onClick={() => { void save(); }}>Save name</button></Dialog>}</>;
}
