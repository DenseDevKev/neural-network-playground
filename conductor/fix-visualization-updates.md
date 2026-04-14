# Plan: Fix Real-Time Visualization Updates

## Background & Motivation
The user reported that when they click "Play", the UI shows "Running..." but the visualizations (Decision Boundary, Network Graph, Loss Chart) do not update. The visualizations only update when the user clicks "Pause", then "Play" again.

This indicates a breakdown in the streaming communication between the web worker and the main thread, specifically during the continuous training loop.

## Scope & Impact
The issue is isolated to the real-time update mechanism driven by `requestAnimationFrame` and `MessageChannel` communication. It impacts all dynamic visualizations. The fix will involve modifying `workerBridge.ts`, `training.worker.ts`, and potentially the `NetworkSnapshot` type to ensure reliable streaming and optimal performance.

## Proposed Solution

Our investigation revealed several root causes combining to create this behavior:

1.  **Race Condition in `training.worker.ts`**: The `trainTick` loop checks `if (!state.running) return;`. If `state.streamPort` isn't set yet, it skips sending the snapshot, but more importantly, the `setTimeout(trainTick, 0)` is *inside* an `if (state.running)` check that was previously combined with the `state.streamPort` check. If it exits early due to missing `streamPort`, the loop permanently stops.
2.  **Unreliable `MessagePort` Initialization**: The worker bridge and the worker rely on assigning `onmessage` to start the port implicitly. Explicitly calling `start()` and using `addEventListener` is much more robust for `MessageChannel` communication.
3.  **`requestAnimationFrame` (rAF) Loop Sync**: In `workerBridge.ts`, the `rafLoop` schedules itself indefinitely, even if `_rafId` is cleared by `stopRenderLoop`. This can lead to multiple interleaved loops or dead loops if state gets desynchronized.
4.  **Performance Bottleneck**: In `training.worker.ts`, `computeSnapshot` uses `Array.from(outputGrid)` if `outputGrid` is a `Float32Array`. This is a blocking, synchronous copy of a large array on the worker thread, causing unnecessary overhead.

### Implementation Steps

1.  **Fix `rafLoop` in `workerBridge.ts`**:
    *   Add an explicit check: `if (_rafId === null) return;` at the beginning of `rafLoop` to ensure it only runs when expected.

2.  **Improve `MessagePort` Reliability**:
    *   In `workerBridge.ts` (`setupStreamChannel`), switch from `.onmessage` to `.addEventListener('message', ...)` and add an explicit `_streamPort.start()`.
    *   In `training.worker.ts` (`workerApi.setStreamPort`), switch from `.onmessage` to `.addEventListener('message', ...)` and add an explicit `port.start()`.

3.  **Fix Race Condition in `training.worker.ts`**:
    *   Separate the `state.streamPort` check from the loop continuation logic. Even if `streamPort` is temporarily unavailable, the `setTimeout(trainTick, 0)` must be called if `state.running` is true so the loop doesn't die.

4.  **Optimize Snapshot Array Handling**:
    *   Modify `NetworkSnapshot` in `packages/engine/src/types.ts` to type `outputGrid` and `neuronGrids` as `ArrayLike<number>` instead of `number[] | Float32Array`.
    *   Update `Network.getSnapshot` signature in `packages/engine/src/network.ts` to match.
    *   In `training.worker.ts` (`computeSnapshot`), pass the `Float32Array` directly to `getSnapshot` without using `Array.from()`. The `packSnapshotMessage` function already handles extracting the underlying buffer for transferable posting.

## Verification
- Run `pnpm test:engine` and `pnpm --filter @nn-playground/web test` to ensure no regressions.
- The user will need to verify that clicking "Play" immediately results in smooth, real-time updates to the Decision Boundary, Loss Chart, and Network Graph.
