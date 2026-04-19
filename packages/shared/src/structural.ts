// ── Structural equality (shallow-recursive, primitives + arrays + plain objects) ──
// Used to short-circuit config-diff sync paths without the allocation cost
// of JSON.stringify. Matches the semantics previously open-coded in
// apps/web/src/worker/training.worker.ts so that main thread and worker
// agree on when a rebuild is required.

export function structuralEqual(a: unknown, b: unknown): boolean {
    if (a === b) return true;
    if (a == null || b == null) return a === b;
    if (typeof a !== 'object' || typeof b !== 'object') return false;
    if (Array.isArray(a) || Array.isArray(b)) {
        if (!Array.isArray(a) || !Array.isArray(b)) return false;
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (!structuralEqual(a[i], b[i])) return false;
        }
        return true;
    }
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const aKeys = Object.keys(ao);
    const bKeys = Object.keys(bo);
    if (aKeys.length !== bKeys.length) return false;
    for (const k of aKeys) {
        if (!structuralEqual(ao[k], bo[k])) return false;
    }
    return true;
}
