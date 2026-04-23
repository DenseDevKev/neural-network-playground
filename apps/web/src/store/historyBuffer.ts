// ── Training-History Ring Buffer ──
// Module-level typed arrays holding the per-step loss/accuracy trace.
//
// Previously the history lived as `HistoryPoint[]` inside the Zustand
// training store. Every append allocated a fresh array and triggered
// every subscriber, which meant the LossChart re-rendered end-to-end on
// every training frame — the dominant paint cost at long runs.
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
const _step = new Float64Array(HISTORY_CAPACITY);
const _trainLoss = new Float64Array(HISTORY_CAPACITY);
const _testLoss = new Float64Array(HISTORY_CAPACITY);
const _trainAccuracy = new Float64Array(HISTORY_CAPACITY);
const _testAccuracy = new Float64Array(HISTORY_CAPACITY);
// Which cells in the accuracy arrays hold a real value (vs. the default 0).
// Kept as Uint8Array so "is defined" queries stay branch-free.
const _hasTrainAcc = new Uint8Array(HISTORY_CAPACITY);
const _hasTestAcc = new Uint8Array(HISTORY_CAPACITY);

let _count = 0;
let _version = 0;
// Tracks compaction events so consumers (e.g. LossChart) can tell whether
// the last version bump was a pure append (draw incremental segment) or a
// structural rewrite (force full redraw).
let _compactionCount = 0;

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
