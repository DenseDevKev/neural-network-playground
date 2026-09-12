import { useEffect, useState } from 'react';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

export function CheckpointPanel({ training }: { training: TrainingHook }) {
    const timeline = useTrainingStore((s) => s.checkpointTimeline);
    const status = useTrainingStore((s) => s.status);
    const pending = useTrainingStore((s) => s.pendingConfigSource);
    const current = useTrainingStore((s) => s.latestLiveSignal?.model.step ?? s.latestEvaluation?.model.step ?? 0);
    const [id, setId] = useState<number | null>(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const selected = timeline.checkpoints.find((point) => point.id === id) ?? timeline.checkpoints.at(-1);
    useEffect(() => { if (id !== null && !timeline.checkpoints.some((point) => point.id === id)) setId(null); }, [id, timeline.checkpoints]);
    if (!selected) return <p>No checkpoints yet. Start training to build a session timeline.</p>;
    const index = timeline.checkpoints.findIndex((point) => point.id === selected.id);
    return <section className="atelier-checkpoints">
        <p>Explore this session’s timeline. Selecting a checkpoint previews its details; restore applies its parameters and optimizer state.</p>
        <dl><div><dt>Current model</dt><dd>Step {current}</dd></div><div><dt>Selected checkpoint</dt><dd>{selected.label}</dd></div></dl>
        <label htmlFor="atelier-checkpoint-range">Checkpoint timeline</label>
        <input id="atelier-checkpoint-range" type="range" min={0} max={timeline.checkpoints.length - 1} step={1} value={index} aria-valuetext={selected.label} onChange={(event) => setId(timeline.checkpoints[Number(event.target.value)].id)} />
        <p>Training loss {selected.trainDataLoss.toFixed(4)} · test loss {selected.testDataLoss.toFixed(4)}</p>
        <p>Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only. Checkpoints last for this session and are separate from saved evidence.</p>
        {error && <p role="alert" className="atelier-error">{error}</p>}
        <div className="atelier-dialog-actions">{status === 'running' && <button type="button" onClick={training.pause}>Pause training</button>}<button type="button" className="atelier-primary" disabled={status === 'running' || pending !== null || busy} onClick={async () => { setBusy(true); setError(null); try { await training.restoreCheckpoint(selected.id); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Checkpoint restoration failed.'); } finally { setBusy(false); } }}>{busy ? 'Restoring…' : `Restore ${selected.label}`}</button></div>
    </section>;
}
