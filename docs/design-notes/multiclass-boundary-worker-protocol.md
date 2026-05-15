# Design Note: Multiclass Boundary Worker Protocol

## Problem

Multiclass Classification Mode now has hidden engine, shared-config, worker-target,
dataset-contract, code-export, persistence-eligibility, and public non-exposure
guards. The remaining blocker before visible multiclass controls is visualization:
the worker currently clears scalar decision-boundary grids for multiclass snapshots
so the UI cannot accidentally render binary regions, legends, or confusion copy.

The next slice needs to teach class regions and confidence without sending raw
per-cell probability vectors, changing persistence, or making multiclass public
before downstream UI and QA are ready.

## Proposed Change

Implement multiclass boundary transport as a staged worker/protocol slice.

First code slice:

1. Add a bounded snapshot payload for exactly the approved three-class worker
   config.
2. Reuse existing `needDecisionBoundary` and `gridInterval` demand gating.
3. Stream only two flattened grids:
   - `multiclassClassGrid?: Uint8Array`, length `GRID_SIZE * GRID_SIZE`,
     values `0..2`.
   - `multiclassConfidenceGrid?: Float32Array`, length `GRID_SIZE * GRID_SIZE`,
     values `0..1`, where each value is the winning softmax probability.
4. Include compact layout metadata:
   - `multiclassBoundaryLayout?: { gridSize: number; classCount: 3;
     classLabels: readonly [0, 1, 2] }`
   - `multiclassBoundaryVersion?: number`
5. Store the arrays in the frame buffer with a dedicated version counter.
6. Keep scalar `outputGrid` and `neuronGrids` explicitly empty for multiclass
   snapshots so existing binary/scalar renderers stay cleared.
7. Do not use SharedArrayBuffer for the first multiclass boundary payload. The
   payload is bounded and cadence-gated; SAB can be considered later only after
   evidence shows postMessage transfer is a problem.

Exact `WorkerSnapshotMessage` additions:

```ts
multiclassClassGrid?: Uint8Array;
multiclassConfidenceGrid?: Float32Array;
multiclassBoundaryLayout?: {
  gridSize: number;
  classCount: 3;
  classLabels: readonly [0, 1, 2];
};
multiclassBoundaryVersion?: number;
```

These fields are an all-or-nothing protocol group. If any one field is present,
all four must be present and runtime-valid.

Version semantics:

- Worker `multiclassBoundaryVersion` increments only when a fresh valid
  multiclass boundary payload is computed and posted.
- Omitted multiclass fields on cadence-reuse snapshots do not bump the worker
  version and must not bump the main-thread frame-buffer counter.
- Main-thread `multiclassBoundaryVersion` in `FrameVersions` increments only
  when the frame buffer stores or clears multiclass boundary arrays/layout.
- Fresh scalar snapshots, explicit scalar grid clears, `resetFrameBuffer()`, and
  mode/config rebuilds clear stale multiclass boundary arrays and increment the
  main-thread multiclass boundary counter.

Later slices:

1. Render class regions and confidence summaries in `DecisionBoundary`.
2. Add a bounded 3x3 confusion-matrix payload and UI if needed.
3. Expose public dataset/preset controls only after visualization, URL/config,
   persistence eligibility, import/export, and Browser QA all have explicit
   coverage.

## User Value

This makes the first public multiclass visualization possible: learners can see
which class wins in each region and how confident the model is without exposing a
heavy probability tensor or misusing binary decision-boundary copy.

## Affected Files

First code slice:

- `packages/engine/src/network.ts`
- `packages/engine/src/__tests__/network.test.ts`
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/__tests__/workerProtocol.test.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/training.worker.test.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/__tests__/frameBuffer.test.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/worker/workerBridge.test.ts`
- `docs/worker-protocol.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

Later visible UI slice:

- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.test.tsx`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/*`

## Affected Runtime/Data Flow

1. The worker already validates the only supported multiclass runtime shape:
   `classification + outputSize: 3 + softmax + categoricalCrossEntropy`.
2. When `needDecisionBoundary` is false, no multiclass boundary arrays are
   computed or posted. Omitted fields from boundary-demand-off snapshots retain
   any cached multiclass boundary arrays with no version bump, matching existing
   cadence-retention behavior. Mode/config changes and fresh scalar snapshots
   are responsible for clearing stale arrays.
3. When `needDecisionBoundary` is true and the existing grid cadence is due, the
   worker evaluates each grid input once on the CPU, computes the winning class
   index and confidence, and posts bounded typed arrays.
4. Cadence-reuse snapshots omit the multiclass boundary fields, letting the main
   thread retain the previous frame-buffer arrays.
5. Fresh scalar snapshots never include multiclass boundary fields. If a fresh
   scalar snapshot is applied, the frame-buffer patch clears any stale
   multiclass boundary arrays.
6. Fresh multiclass payloads continue sending explicit empty scalar `outputGrid`
   and `neuronGrids` arrays so the bridge clears stale binary/scalar grids and
   neuron grids.
7. The main thread stores arrays in `frameBuffer.ts`, not React or Zustand state.
   React receives only version counters through existing frame-version plumbing.

The worker must not reuse scalar grid helpers for multiclass. Existing helpers
like `predictGridInto(...)` write output index `0` only, which is correct for
binary/regression scalar boundaries but wrong for class-region rendering. Add a
small engine helper such as:

```ts
predictMulticlassBoundaryInto(
  gridInputs: number[][],
  classTarget: Uint8Array,
  confidenceTarget: Float32Array,
): void
```

The helper should assert `outputSize === 3`, target lengths match grid length,
and every emitted class index/confidence is bounded and finite. If multiple
classes share the maximum confidence, choose the lowest class index to keep
tests and visuals deterministic.

The first transport slice must also bypass current WebGPU and SharedArrayBuffer
grid paths. Multiclass boundary arrays are always inline transferables in the
first slice, and no new shared snapshot flags or SAB buffers are added.

## Compatibility Risks

High-risk surfaces:

- Worker snapshot protocol.
- Runtime guards in `isWorkerToMainMessage`.
- Frame-buffer shape and version counters.
- Decision-boundary cadence behavior.
- Interaction with existing scalar grid clearing for hidden multiclass snapshots.

Runtime guard rules:

- If any multiclass boundary field is present, all companion fields must be
  present.
- `multiclassClassGrid` must be a `Uint8Array`.
- `multiclassConfidenceGrid` must be a `Float32Array`.
- `multiclassBoundaryLayout.gridSize` must be a positive integer that matches
  `scalars.gridSize`.
- `multiclassBoundaryLayout.classCount` must be exactly `3`.
- `multiclassBoundaryLayout.classLabels` must be exactly `[0, 1, 2]`.
- Both grids must have length `gridSize * gridSize`.
- Every class index must be an integer in `0..2`.
- Every confidence must be finite and in `[0, 1]`.
- `multiclassBoundaryVersion` must be a non-negative integer.

The guard may scan both grids because the current grid is bounded (`40x40`) and
field-presence gated. If a future slice grows the grid or class count, revisit
guard cost before widening the contract.

Explicitly forbidden in this slice:

- URL/config serialization changes.
- Persistence/run-history schema changes.
- Public config shape changes.
- Public dataset or preset exposure.
- UI controls that make multiclass selectable.
- Raw per-cell probability vectors.
- Raw activations, targets, samples, weights, biases, or checkpoints in React
  state.
- SharedArrayBuffer redesign.
- New dependencies.

## Accessibility Impact

The first transport slice has no visible UI. The later UI slice must provide:

- A visible text legend with class labels.
- A text summary of dominant class and confidence/ambiguity.
- Non-color confidence indication.
- Keyboard-reachable visualization controls if any are added.
- Compact viewport checks for legend and summary wrapping.

## Performance Impact

The first transport adds at most:

- `GRID_SIZE * GRID_SIZE` bytes for class indices.
- `GRID_SIZE * GRID_SIZE * 4` bytes for confidence.

For the current `40x40` grid, that is about `1.6 kB + 6.4 kB` per fresh
demand-gated boundary payload before message overhead. This is smaller than a
three-probability grid and avoids full probability transport.

Expected runtime work is one forward pass per grid cell when the existing
decision-boundary cadence is due. The worker already performs one scalar grid
forward pass per cell for binary decision boundaries, so the main added cost is
argmax/confidence extraction from three outputs.

Compare `pnpm test:perf` and production bundle sizes against
`docs/perf/PERFORMANCE_BASELINE.md`. Browser QA is not required for the
transport-only slice unless visible UI changes are included.

## Test Plan

TDD tests before production changes:

1. Engine test for a helper that fills class-index and confidence grids from
   multi-output predictions, including class bounds and confidence range.
2. Shared protocol guard tests accepting valid bounded multiclass boundary
   payloads.
3. Shared protocol guard tests rejecting missing companion fields, wrong typed
   arrays, mismatched lengths, non-finite confidence, and out-of-range classes.
4. Frame-buffer tests for storage, clearing, and version-counter behavior.
5. Worker bridge tests proving multiclass payloads update the frame buffer,
   omitted cadence fields retain cached arrays, and fresh scalar snapshots clear
   stale multiclass arrays.
6. Worker tests proving multiclass boundary payloads are omitted when
   `needDecisionBoundary` is false and emitted only when the grid cadence is
   due.
7. Protocol docs tests are not needed, but `docs/worker-protocol.md` must be
   updated with the exact snapshot fields and the new frame-buffer version
   counter.

Required verification for the first code slice:

- Targeted affected tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

## Browser QA Plan

Transport-only code does not require Browser QA because no visible UI changes.

Later visible UI slice must record Mode B QA under `docs/qa/browser-qa/`:

1. App loads with no console errors.
2. Existing binary and regression datasets still train, pause, step, reset.
3. Multiclass path renders class regions and confidence text.
4. Boundary summary is reachable and readable by keyboard.
5. Compact viewport has no legend or summary overlap.
6. Existing scalar decision-boundary and loss panels still render.

## Rollback Plan

Keep rollback scoped by slice:

1. Protocol/frame-buffer transport slice.
2. Worker compute slice if split from protocol/frame-buffer.
3. Decision-boundary UI slice.
4. QA/performance/state evidence slice.

If protocol or frame-buffer tests fail after three targeted fixes, revert only
the current slice files and restore `docs/worker-protocol.md`,
`docs/perf/PERFORMANCE_BASELINE.md`, and `docs/roadmap/ROADMAP_STATE.md` to the
previous verified state.

## Approval Required

Yes. This crosses worker protocol and frame-buffer contracts. The user has
provided broad approval to continue, but implementation must remain inside this
bounded class-index/confidence design. Any SharedArrayBuffer redesign,
probability-tensor streaming, public UI exposure, URL/config change,
persistence/run-history schema change, or dependency addition requires a new
approval gate.

## Decision

Proceed with this design note first, run two approval-gate review agents, then
implement the first protocol/frame-buffer transport slice only if review finds
no blockers.
