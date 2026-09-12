import { useEffect, useRef, useState } from 'react';
import { validateExperimentDocument, type PreparedExperimentDocumentV2, type StandardExperimentRecipeV2 } from '@nn-playground/shared';
import { getDatasetContract, type DatasetId } from '@nn-playground/engine';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { commitRecipeEditPrepared } from '../store/commitRecipeEdit.ts';

export type SetupTab = 'dataset' | 'network' | 'training';
type Draft = { base: PreparedExperimentDocumentV2; recipe: StandardExperimentRecipeV2; text: Record<string, string> };
export interface RecipeDraftController {
    recipe: StandardExperimentRecipeV2 | null;
    dirty: boolean; valid: boolean; busy: boolean; submitted: boolean; error: string | null;
    issues: readonly string[];
    value(path: string): unknown;
    commands: {
        set(path: string, value: unknown): void;
        number(path: string, text: string): void;
        dataset(dataset: DatasetId): void;
        preset(recipe: StandardExperimentRecipeV2): void;
        apply(): Promise<boolean>;
        cancel(): void;
        retry(): void;
    };
}
function clone<T>(value: T): T { return structuredClone(value); }
function read(value: unknown, path: string): unknown {
    return path.split('.').reduce<unknown>((part, key) => (part as Record<string, unknown>)?.[key], value);
}
function write(value: unknown, path: string, next: unknown) {
    const keys = path.split('.');
    const last = keys.pop()!;
    const target = keys.reduce<unknown>((part, key) => (part as Record<string, unknown>)[key], value);
    (target as Record<string, unknown>)[last] = next;
}

/** One session-only candidate; publication and worker acknowledgement are distinct. */
export function useRecipeDraft(): RecipeDraftController {
    const prepared = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared : null);
    const pending = useTrainingStore((s) => s.pendingConfigSource);
    const configError = useTrainingStore((s) => s.configError);
    const [draft, setDraft] = useState<Draft | null>(null);
    const draftRef = useRef(draft); draftRef.current = draft;
    const [error, setError] = useState<string | null>(null);
    const [submitted, setSubmitted] = useState(false);
    const lock = useRef(false);
    const target = useRef<PreparedExperimentDocumentV2 | null>(null);
    const resolve = useRef<((success: boolean) => void) | null>(null);
    const recipe = draft?.recipe ?? prepared?.document.recipe ?? null;
    const validation = recipe && prepared ? validateExperimentDocument({ ...prepared.document, recipe }) : null;
    const issues = validation && !validation.ok ? validation.issues.map((i) => `${i.path}: ${i.message}`) : [];
    const dirty = draft !== null && (JSON.stringify(draft.recipe) !== JSON.stringify(draft.base.document.recipe)
        || Object.entries(draft.text).some(([path, text]) => text.trim() === '' || !Number.isFinite(Number(text)) || Number(text) !== read(draft.base.document.recipe, path)));
    const valid = validation?.ok === true;
    const busy = submitted || pending !== null;

    useEffect(() => {
        const check = () => {
            const expected = target.current;
            if (!expected) return;
            const ts = useTrainingStore.getState();
            const ps = usePlaygroundStore.getState();
            if (ps.access.status !== 'ready' || ps.access.prepared !== expected) {
                target.current = null; lock.current = false; setSubmitted(false);
                setError('The active experiment changed while setup was applying. Discard this draft to reload it.');
                resolve.current?.(false); resolve.current = null;
                return;
            }
            if (ts.configError) {
                setError(ts.configError);
                resolve.current?.(false); resolve.current = null;
                return; // Retain exact submitted candidate for existing sync recovery.
            }
            if (ts.pendingConfigSource === null && ts.trainedRecipeFingerprint === expected.identities.recipeFingerprint
                && ts.trainedRecipeSource === 'config-sync') {
                target.current = null; lock.current = false;
                setDraft(null); setSubmitted(false); setError(null);
                resolve.current?.(true); resolve.current = null;
            }
        };
        const unsubscribe = useTrainingStore.subscribe(check);
        const unsubscribePlayground = usePlaygroundStore.subscribe(check);
        return () => { unsubscribe(); unsubscribePlayground(); resolve.current?.(false); };
    }, []);
    useEffect(() => {
        if (!dirty) return;
        const guard = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
        window.addEventListener('beforeunload', guard);
        return () => window.removeEventListener('beforeunload', guard);
    }, [dirty]);
    const edit = (path: string, value: unknown, text?: string) => {
        if (lock.current || !prepared) return;
        const current = draftRef.current;
        const next: Draft = current ? { ...current, recipe: clone(current.recipe), text: { ...current.text } }
            : { base: prepared, recipe: clone(prepared.document.recipe), text: {} };
        write(next.recipe, path, value);
        for (const key of Object.keys(next.text)) if (key === path || key.startsWith(`${path}.`)) delete next.text[key];
        if (text !== undefined) next.text[path] = text;
        draftRef.current = next; setDraft(next); setError(null);
    };
    return {
        recipe, dirty, valid, busy, submitted, error: error ?? configError, issues,
        value: (path) => draft?.text[path] ?? read(recipe, path),
        commands: {
            set: edit,
            preset: (recipe) => {
                if (lock.current || pending !== null || !prepared) return;
                const next: Draft = {
                    base: draftRef.current?.base ?? prepared,
                    recipe: clone(recipe),
                    text: {},
                };
                draftRef.current = next;
                setDraft(next);
                setError(null);
            },
            number: (path, text) => edit(path, text.trim() === '' ? NaN : Number(text), text),
            dataset: (dataset) => {
                if ((draftRef.current?.recipe ?? prepared?.document.recipe)?.task.dataset === dataset) return;
                const kind = getDatasetContract(dataset).taskKind;
                edit('task', { kind, dataset });
                edit('objective.dataLoss', { kind: kind === 'regression' ? 'mean-squared-error' : kind === 'multiclass-classification' ? 'categorical-cross-entropy-with-logits' : 'binary-cross-entropy-with-logits' });
            },
            cancel: () => {
                if (lock.current) return;
                draftRef.current = null; setDraft(null); setError(null);
            },
            retry: () => useTrainingStore.getState().retryConfigSync(),
            apply: async () => {
                const candidate = draftRef.current;
                if (lock.current || !candidate || !dirty || !valid || pending !== null) return false;
                const ps = usePlaygroundStore.getState();
                if (ps.access.status !== 'ready' || ps.access.prepared.identities.canonicalRecipeKey !== candidate.base.identities.canonicalRecipeKey) {
                    setError('The active recipe changed. Cancel to reload it before applying setup.'); return false;
                }
                lock.current = true; setSubmitted(true); setError(null);
                const published = await commitRecipeEditPrepared('setup', () => ({ ok: true, recipe: candidate.recipe }), candidate.base.identities.canonicalRecipeKey);
                if (!published) {
                    lock.current = false; setSubmitted(false);
                    setError(useTrainingStore.getState().configError ?? 'Setup could not be applied. Your draft is preserved.');
                    return false;
                }
                const current = usePlaygroundStore.getState().access;
                if (current.status !== 'ready' || current.prepared !== published) {
                    lock.current = false; setSubmitted(false);
                    setError('The active experiment changed while setup was applying. Discard this draft to reload it.');
                    return false;
                }
                target.current = published;
                return new Promise<boolean>((done) => {
                    resolve.current = done;
                    const ts = useTrainingStore.getState();
                    if (ts.pendingConfigSource === null && !ts.configError && ts.trainedRecipeSource === 'config-sync'
                        && ts.trainedRecipeFingerprint === published.identities.recipeFingerprint) {
                        target.current = null; lock.current = false; setDraft(null); setSubmitted(false); resolve.current = null; done(true);
                    } else if (ts.configError) { setError(ts.configError); resolve.current = null; done(false); }
                });
            },
        },
    };
}
