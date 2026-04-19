// ── Training-History Ring Buffer ──
// Module-level typed arrays holding the per-step loss/accuracy trace.
//
// Previously the history lived as `HistoryPoint[]` inside the Zustand
// training store. Every append allocated a fresh array (`[...s.history,
// point]`) and triggered every subscriber, which meant the LossChart
// re-rendered end-to-end on every training frame — the dominant paint
// cost at long runs.
//
// The buffer is a monotonically growing series with an in-place halving
// step when it hits the capacity ceiling; same log-spaced density profile
// as the old implementation, but zero GC pressure and O(1) append.
// Readers subscribe to a single `historyVersion` scalar in useTrainingStore
// and pull typed arrays out of this module on demand.

import type { HistoryPoint } from '@nn-playground/engine';

export const HISTORY_CAPACITY = 4096;

// Typed-array backing store — fixed capacity, resized in place via the
// halving compaction when full. `Float64Array` because our loss values can
// be very small and we want to avoid float32 denormal slowdowns, while
// still ditching the HistoryPoint object allocations.
let _step = new Float64Array(HISTORY_CAPACITY);
let _trainLoss = new Float64Array(HISTORY_CAPACITY);
let _testLoss = new Float64Array(HISTORY_CAPACITY);
let _trainAccuracy = new Float64Array(HISTORY_CAPACITY);
let _testAccuracy = new Float64Array(HISTORY_CAPACITY);
// Which cells in the accuracy arrays hold a real value (vs. the default 0).
// Kept as Uint8Array so "is defined" queries stay branch-free.
let _hasTrainAcc = new Uint8Array(HISTORY_CAPACITY);
let _hasTestAcc = new Uint8Array(HISTORY_CAPACITY);

let _count = 0;
let _version = 0;
// Tracks compaction events so consumers (e.g. LossChart) can tell whether
// the last version bump was a pure append (draw incremental segment) or a
// structural rewrite (force full redraw).
let _compactionCount = 0;

/** Number of valid entries currently stored (0..HISTORY_CAPACITY). */
export function getHistoryCount(): number {
    return _count;
}

/** Monotonic version counter — bumped on append, reset, compact. */
export function getHistoryVersion(): number {
    return _version;
}

/** Monotonic counter of compaction events. Consumers that do incremental
 *  drawing should invalidate their cached offset if this advances. */
export function getHistoryCompactionCount(): number {
    return _compactionCount;
}

export interface HistoryArrays {
    step: Float64Array;
    trainLoss: Float64Array;
    testLoss: Float64Array;
    trainAccuracy: Float64Array;
    testAccuracy: Float64Array;
    hasTrainAccuracy: Uint8Array;
    hasTestAccuracy: Uint8Array;
    count: number;
}

/** Read-only view onto the current buffers. Returned arrays are shared
 *  references — callers must not mutate them. */
export function readHistory(): HistoryArrays {
    return {
        step: _step,
        trainLoss: _trainLoss,
        testLoss: _testLoss,
        trainAccuracy: _trainAccuracy,
        testAccuracy: _testAccuracy,
        hasTrainAccuracy: _hasTrainAcc,
        hasTestAccuracy: _hasTestAcc,
        count: _count,
    };
}

/** Append one training-history point. O(1) amortized; O(capacity) only
 *  on the rare compaction (halving) pass when the buffer fills up. */
export function appendHistoryPoint(point: HistoryPoint): number {
    if (_count >= HISTORY_CAPACITY) {
        compact();
    }
    const i = _count;
    _step[i] = point.step;
    _trainLoss[i] = point.trainLoss;
    _testLoss[i] = point.testLoss;
    if (point.trainAccuracy !== undefined) {
        _trainAccuracy[i] = point.trainAccuracy;
        _hasTrainAcc[i] = 1;
    } else {
        _trainAccuracy[i] = 0;
        _hasTrainAcc[i] = 0;
    }
    if (point.testAccuracy !== undefined) {
        _testAccuracy[i] = point.testAccuracy;
        _hasTestAcc[i] = 1;
    } else {
        _testAccuracy[i] = 0;
        _hasTestAcc[i] = 0;
    }
    _count++;
    _version++;
    return _version;
}

/** Drop everything. Used on network reset / new run. */
export function resetHistoryBuffer(): number {
    _count = 0;
    _version++;
    _compactionCount++;
    return _version;
}

/** Replace the buffer contents wholesale. Used by tests that seeded the
 *  old `history: HistoryPoint[]` directly. */
export function seedHistory(points: HistoryPoint[]): number {
    _count = 0;
    for (const p of points) {
        if (_count >= HISTORY_CAPACITY) break;
        appendHistoryPointRaw(p);
    }
    _version++;
    _compactionCount++;
    return _version;
}

function appendHistoryPointRaw(point: HistoryPoint): void {
    const i = _count;
    _step[i] = point.step;
    _trainLoss[i] = point.trainLoss;
    _testLoss[i] = point.testLoss;
    if (point.trainAccuracy !== undefined) {
        _trainAccuracy[i] = point.trainAccuracy;
        _hasTrainAcc[i] = 1;
    } else {
        _trainAccuracy[i] = 0;
        _hasTrainAcc[i] = 0;
    }
    if (point.testAccuracy !== undefined) {
        _testAccuracy[i] = point.testAccuracy;
        _hasTestAcc[i] = 1;
    } else {
        _testAccuracy[i] = 0;
        _hasTestAcc[i] = 0;
    }
    _count++;
}

/** In-place halving: keep every other point. Matches the old store's
 *  log-spaced density behaviour once the array crossed 2000 points. */
function compact(): void {
    const kept = Math.ceil(_count / 2);
    for (let dst = 0; dst < kept; dst++) {
        const src = dst * 2;
        _step[dst] = _step[src];
        _trainLoss[dst] = _trainLoss[src];
        _testLoss[dst] = _testLoss[src];
        _trainAccuracy[dst] = _trainAccuracy[src];
        _testAccuracy[dst] = _testAccuracy[src];
        _hasTrainAcc[dst] = _hasTrainAcc[src];
        _hasTestAcc[dst] = _hasTestAcc[src];
    }
    _count = kept;
    _compactionCount++;
}

/** Reallocate internal buffers. Exported only so tests can reset module
 *  state between runs; no production code should call this. */
export function __resetHistoryBufferForTests(): void {
    _step = new Float64Array(HISTORY_CAPACITY);
    _trainLoss = new Float64Array(HISTORY_CAPACITY);
    _testLoss = new Float64Array(HISTORY_CAPACITY);
    _trainAccuracy = new Float64Array(HISTORY_CAPACITY);
    _testAccuracy = new Float64Array(HISTORY_CAPACITY);
    _hasTrainAcc = new Uint8Array(HISTORY_CAPACITY);
    _hasTestAcc = new Uint8Array(HISTORY_CAPACITY);
    _count = 0;
    _version = 0;
    _compactionCount = 0;
}
