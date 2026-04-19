// ── Shared-memory snapshot transport ────────────────────────────────────────
// The worker's heavy per-frame buffers (decision-boundary output grid and
// per-neuron activation grids) can be published through SharedArrayBuffers
// instead of postMessage-with-transferables. That saves both the per-frame
// reallocation in the worker (transferred ArrayBuffers are detached) and the
// structured-clone overhead of the message envelope, and lets the main thread
// read the latest frame at rAF cadence without coordinating an ack.
//
// A classic seqlock synchronises writer and reader:
//   Writer:  seqStart++  →  write data  →  seqEnd = seqStart
//   Reader:  e = seqEnd  →  read data   →  s = seqStart
//            if (s !== e) retry            (concurrent write tore the read)
//
// Because ECMAScript Atomics have seq-cst ordering w.r.t. other Atomics
// operations in the same agent cluster, and non-atomic typed-array accesses
// are ordered relative to surrounding Atomics in program order, this pattern
// is race-free without explicit fences.
//
// Feature detection: SharedArrayBuffer is only available when the document is
// cross-origin isolated (COOP: same-origin + COEP: require-corp). Outside
// that (tests, GH Pages, plain dev without headers), `canUseSharedBuffers()`
// returns false and callers keep using the existing postMessage path.

/** Index of seqStart (Int32) in the control buffer. */
export const CTL_SEQ_START = 0;
/** Index of seqEnd (Int32) in the control buffer. */
export const CTL_SEQ_END = 1;
/** Index of the validity-flags word in the control buffer. */
export const CTL_FLAGS = 2;
/** Number of Int32 slots used in the control buffer (rounded up for alignment). */
export const CTL_LENGTH = 8;

export const FLAG_OUTPUT_GRID = 1 << 0;
export const FLAG_NEURON_GRIDS = 1 << 1;

/**
 * Does this global have a functional SharedArrayBuffer?
 * - SAB must exist on the global.
 * - `crossOriginIsolated` must be true (or unavailable, which indicates an
 *   environment where the flag isn't the gate — e.g. Node/test runners that
 *   still expose SAB). On the Web, a missing flag means no SAB anyway.
 */
export function canUseSharedBuffers(): boolean {
    if (typeof SharedArrayBuffer === 'undefined') return false;
    const g = globalThis as typeof globalThis & { crossOriginIsolated?: boolean };
    if (typeof g.crossOriginIsolated === 'boolean' && !g.crossOriginIsolated) {
        return false;
    }
    return true;
}

/** Permanent views over a set of shared buffers. */
export interface SharedSnapshotViews {
    /** [seqStart, seqEnd, flags] Int32 cells. */
    control: Int32Array;
    /** Float32 view of the output grid (gridSize × gridSize). */
    outputGrid: Float32Array;
    /** Float32 view of the concatenated neuron activation grids. */
    neuronGrids: Float32Array;
    /** Backing SABs — kept around so the handshake message can transfer them. */
    controlSAB: SharedArrayBuffer;
    outputGridSAB: SharedArrayBuffer;
    neuronGridsSAB: SharedArrayBuffer;
    /** Cached layout for reader-side validation + UI consumption. */
    gridSize: number;
    neuronCount: number;
}

/** Allocate a fresh set of shared buffers sized for the given network shape. */
export function allocSharedSnapshotViews(
    gridSize: number,
    neuronCount: number,
): SharedSnapshotViews {
    const SAB = SharedArrayBuffer;
    // Int32Array view: 4 bytes per slot × CTL_LENGTH.
    const controlSAB = new SAB(CTL_LENGTH * 4);
    const gridLen = gridSize * gridSize;
    // Always allocate at least 1 byte so view construction doesn't throw
    // when the network has no hidden neurons to visualise.
    const outputGridSAB = new SAB(Math.max(1, gridLen * 4));
    const neuronGridsSAB = new SAB(Math.max(1, neuronCount * gridLen * 4));
    return {
        control: new Int32Array(controlSAB),
        outputGrid: new Float32Array(outputGridSAB, 0, gridLen),
        neuronGrids: new Float32Array(neuronGridsSAB, 0, neuronCount * gridLen),
        controlSAB,
        outputGridSAB,
        neuronGridsSAB,
        gridSize,
        neuronCount,
    };
}

/**
 * Install views over already-allocated SharedArrayBuffers on the consumer
 * side. Used by the main thread after receiving the `sharedBuffers`
 * handshake message.
 */
export function attachSharedSnapshotViews(args: {
    control: SharedArrayBuffer;
    outputGrid: SharedArrayBuffer;
    neuronGrids: SharedArrayBuffer;
    gridSize: number;
    neuronCount: number;
}): SharedSnapshotViews {
    const gridLen = args.gridSize * args.gridSize;
    return {
        control: new Int32Array(args.control),
        outputGrid: new Float32Array(args.outputGrid, 0, gridLen),
        neuronGrids: new Float32Array(args.neuronGrids, 0, args.neuronCount * gridLen),
        controlSAB: args.control,
        outputGridSAB: args.outputGrid,
        neuronGridsSAB: args.neuronGrids,
        gridSize: args.gridSize,
        neuronCount: args.neuronCount,
    };
}

/**
 * Writer: publish `output` and/or `neurons` into the shared buffers under
 * a seqlock. `flags` marks which payloads were updated this frame (so the
 * reader knows whether a stale-reuse is acceptable or the new data replaces
 * the previous). Returns the seqEnd counter for the published frame.
 */
export function publishSharedSnapshot(
    views: SharedSnapshotViews,
    output: Float32Array | null,
    neurons: Float32Array | null,
    flags: number,
): number {
    // Compute next seq number from the previous seqEnd — a monotonically
    // increasing count across all frames on this run.
    const prevEnd = Atomics.load(views.control, CTL_SEQ_END);
    const nextSeq = prevEnd + 1;

    // Mark "write in progress" — readers observing seqStart !== seqEnd
    // will either retry or skip this frame.
    Atomics.store(views.control, CTL_SEQ_START, nextSeq);

    if (output && (flags & FLAG_OUTPUT_GRID) !== 0) {
        // Both views are Float32 and same-length; .set is a simple memcpy.
        views.outputGrid.set(output);
    }
    if (neurons && (flags & FLAG_NEURON_GRIDS) !== 0) {
        views.neuronGrids.set(neurons);
    }

    // Flags are seqlock-protected too — updated inside the critical section.
    Atomics.store(views.control, CTL_FLAGS, flags);

    // Close the write: seqEnd = seqStart. Any reader seeing seqEnd = seqStart
    // after reading the data is guaranteed the read was consistent.
    Atomics.store(views.control, CTL_SEQ_END, nextSeq);
    return nextSeq;
}

/**
 * Reader: attempt to read a consistent snapshot. Copies the latest published
 * data into the provided destination views (which are the existing frame
 * buffer Float32Arrays — SAB reads are fast but the downstream renderer
 * expects stable, non-SAB-backed arrays so painters don't observe torn state
 * later in the frame). Returns the observed seq on success, or -1 if the
 * write tore on every attempt in the retry budget.
 *
 * Callers can also pass `null` for a destination to skip copying that payload.
 */
export function readSharedSnapshot(
    views: SharedSnapshotViews,
    outputDst: Float32Array | null,
    neuronDst: Float32Array | null,
    maxRetries = 4,
): { seq: number; flags: number } | null {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        const endBefore = Atomics.load(views.control, CTL_SEQ_END);
        const flags = Atomics.load(views.control, CTL_FLAGS);

        if (outputDst && (flags & FLAG_OUTPUT_GRID) !== 0) {
            outputDst.set(views.outputGrid);
        }
        if (neuronDst && (flags & FLAG_NEURON_GRIDS) !== 0) {
            neuronDst.set(views.neuronGrids);
        }

        const startAfter = Atomics.load(views.control, CTL_SEQ_START);
        if (startAfter === endBefore) {
            return { seq: endBefore, flags };
        }
        // Torn read: a concurrent publish happened. Retry.
    }
    return null;
}
