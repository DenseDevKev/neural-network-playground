# Design Note: Worker-Authored Multiclass Confusion Matrix

## Problem

The app now has a public, bounded three-class softmax path through the
`three-class-clusters` dataset, the `Three-Class Softmax Lab` preset, a Data
panel chip, a guided lesson, bounded multiclass decision-boundary transport,
and a UI-derived 3x3 confusion readout.

The current readout is intentionally local to `ConfusionMatrix.tsx`: it derives
counts from frame-buffer parameters and test points while training is paused.
That was useful as a safe first slice, but it is not an official worker metric
and cannot update with the same demand-gated cadence as scalar test metrics.

## Proposed Change

Add a small official worker-authored 3x3 confusion metric for exactly the
approved three-class tuple:

- `classification`
- dataset `three-class-clusters`
- `network.outputSize === 3`
- `network.outputActivation === 'softmax'`
- `training.lossType === 'categoricalCrossEntropy'`

The metric should be a bounded 9-cell count payload, not raw predictions,
probability arrays, or per-sample activations. It should travel only with fresh
test metrics when `VisualizationDemand.needConfusionMatrix` is true, using the
existing test-evaluation cadence.

The implementation should be split into small commits:

1. Engine metric shape and deterministic evaluation tests.
2. Shared protocol guard for the bounded payload.
3. Worker snapshot packing and demand/cadence tests.
4. Frame-buffer/bridge/store plumbing tests.
5. Confusion Matrix UI preference for worker-authored data, with UI-derived
   fallback retained only when the worker metric is unavailable.
6. Docs, performance evidence, Browser QA, and roadmap state.

## User Value

Learners get the same always-fresh mental model for multiclass errors that they
already get for binary metrics: rows are actual classes, columns are predicted
classes, and diagonal cells show correct predictions. The metric becomes part
of the runtime evaluation path instead of a paused-panel reconstruction.

## Affected Files

Expected implementation files:

- `packages/engine/src/types.ts`
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
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/hooks/useTraining.test.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.test.tsx`
- `docs/worker-protocol.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/qa/browser-qa/*`
- `docs/roadmap/ROADMAP_STATE.md`

Explicitly out of scope:

- URL/config serialization changes.
- Persistence/run-history schema changes.
- Public config shape changes beyond the already-approved tuple.
- Arbitrary class counts.
- Raw probability-grid transport.
- Gradient-flow overlay.
- New dependencies.
- Training behavior changes.

First engine-only slice invariants:

- Do not change or widen `ConfusionMatrixData`.
- Do not put multiclass data in `Metrics.confusionMatrix`; that field remains
  binary-only.
- Add a distinct bounded field such as
  `Metrics.multiclassConfusionMatrix?: MulticlassConfusionMatrixData`.
- Keep `Network.evaluate(...).confusionMatrix` undefined for multiclass
  evaluations.
- Do not touch `packages/shared`, worker files, frame-buffer files, bridge
  files, `useTraining`, UI files, URL/config, persistence, or
  `docs/worker-protocol.md` in the engine-only commit.
- The engine cannot enforce the `three-class-clusters` dataset gate because
  `Network.evaluate` does not receive dataset metadata. The engine metric gate
  is therefore limited to bounded 3-output classification evaluation; the
  full approved tuple gate remains the responsibility of later shared/worker
  and app slices.

## Affected Runtime/Data Flow

Current fresh binary metric path:

1. Worker demand requests confusion metrics.
2. `Network.evaluate` computes binary `{ tn, fp, fn, tp }` on the test set.
3. Fresh streamed snapshots include `confusionMatrix` plus the existing
   `confusionMatrixVersion`.
4. The bridge writes the small metric to the frame buffer and React state keeps
   only scalar snapshot metadata.

Proposed full worker-authored multiclass path:

1. `Network.evaluate` computes a bounded 3x3 count payload in a distinct
   `multiclassConfusionMatrix` field when the engine is evaluating a
   three-output classification model. It keeps binary `confusionMatrix`
   undefined for that multiclass evaluation.
2. Later shared/worker slices may emit the bounded multiclass matrix only when
   the full approved tuple is active, the test metrics are fresh, and
   `needConfusionMatrix` is true.
3. The shared runtime guard rejects malformed payloads: wrong class count,
   wrong label set, wrong count length, negative/non-integer counts, missing
   version, or partial field sets.
4. The frame buffer stores the small 9-cell metric outside React state, with a
   version counter tied to fresh worker data.
5. `ConfusionMatrix` renders worker-authored data when available. The existing
   UI-derived readout remains a fallback for missing worker data and should keep
   its note honest.

## Compatibility Risks

- **Protocol shape:** New snapshot fields require shared guard tests and
  `docs/worker-protocol.md` updates.
- **Frame-buffer semantics:** A new metric slot and version counter must not
  bump unrelated grid, parameter, or boundary counters.
- **Binary metric compatibility:** Existing `ConfusionMatrixData` and
  `Metrics.confusionMatrix` must remain binary-only so existing worker,
  frame-buffer, run-history, and UI assumptions do not accidentally accept a
  3x3 payload.
- **Persistence leakage:** The first engine-only slice must not expand saved
  run or run-history payloads. Later worker/UI slices must add negative
  persistence tests proving the worker-authored 3x3 metric is not stored unless
  a separate persistence schema approval is granted.
- **Stale data:** Scalar binary and multiclass matrices must clear each other
  on fresh incompatible snapshots, while cadence-omitted stale snapshots can
  retain the last compatible metric.
- **React state:** The 9-cell metric is small, but the implementation should
  keep it in the frame buffer so runtime visualization data stays out of
  Zustand/React state.
- **Engine behavior:** Training, gradients, losses, and determinism must not
  change. This slice adds evaluation metadata only.
- **Public product scope:** Do not use this as a gateway to arbitrary class
  controls or persistence migrations.

## Accessibility Impact

The existing 3x3 table semantics should remain:

- Row headers name actual classes.
- Column headers name predicted classes.
- Each cell has an accessible label with count, percentage, actual class, and
  predicted class.
- Summary text announces diagonal accuracy.
- The UI must not rely on color alone.

The worker-authored version should update the note from "derived from current
frame-buffer parameters" to a worker-authored/fallback-specific message.

## Performance Impact

Expected impact is small and bounded:

- The metric is computed only during existing fresh test evaluations.
- The payload is 9 integer counts plus compact layout metadata.
- No raw predictions or per-class probability grids are transported.
- No new hot-path diagnostics or dependencies are added.

Record:

- `pnpm test:perf`
- production build chunk sizes
- whether the values exceed existing roadmap warning thresholds

If the worker cadence or perf benchmark regresses by more than the existing
thresholds, revert or split before broadening.

## Test Plan

Use TDD for each code slice:

- Engine tests:
  - Red test that a 3-output classification evaluation returns a distinct 3x3
    `multiclassConfusionMatrix`.
  - Counts are actual-row/predicted-column ordered.
  - The 3x3 counts sum to the sample count.
  - Multiclass evaluation leaves `confusionMatrix` undefined.
  - Binary `{ tn, fp, fn, tp }` behavior is unchanged.
  - Regression evaluation still has no confusion matrix.
- Shared protocol tests:
  - Accept a complete bounded multiclass matrix payload.
  - Reject partial payloads, wrong label set, wrong count length, non-integer
    counts, negative counts, and missing version.
- Worker tests:
  - Fresh multiclass snapshots include the metric only when
    the approved tuple is active and `needConfusionMatrix` is true.
  - Non-approved multiclass-like configs with wrong dataset, loss, activation,
    output size, or problem type do not emit the metric.
  - Cadence-omitted snapshots omit a fresh payload.
  - Scalar binary snapshots still emit binary matrices.
  - Multiclass snapshots do not emit stale binary matrices.
- Frame-buffer/bridge tests:
  - New metric updates only the relevant frame versions.
  - Fresh scalar snapshots clear multiclass matrix data.
  - Fresh multiclass snapshots clear binary matrix data.
  - Cadence omissions retain the last compatible metric.
- UI tests:
  - Confusion Matrix prefers worker-authored 3x3 data.
  - Fallback derived readout still works when worker data is unavailable.
  - Accessible names and totals remain stable.
- Persistence/run-history tests for later non-engine slices:
  - Saved run capture does not persist worker-authored multiclass matrix data.
  - Existing experiment-memory schema remains unchanged.

Required verification for code commits:

- Targeted affected tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

## Browser QA Plan

Use Browser QA Mode B when the UI slice lands:

1. Open the app.
2. Select the `Three-Class Softmax Lab` preset or Data-panel `Three-Class`
   chip.
3. Start training, wait for fresh test metrics, pause.
4. Open the Confusion Matrix panel.
5. Confirm a labeled 3x3 matrix appears.
6. Confirm the UI identifies worker-authored data or accurately identifies the
   fallback state.
7. Check compact viewport around `390x800`.
8. Check console errors.
9. Capture screenshots under `docs/qa/browser-qa/`.

## Rollback Plan

Rollback by slice:

1. Revert engine metric shape/tests.
2. Revert shared protocol fields/guards/docs.
3. Revert worker packing/tests.
4. Revert frame-buffer/bridge/store plumbing/tests.
5. Revert UI preference and QA evidence.

After rollback, run targeted affected tests plus the required full verification
commands. If rollback overlaps with user work, stop and write a block report.

## Approval Required

Yes. This crosses protected worker protocol and frame-buffer surfaces.

The user has provided broad Wave 7 continuation approval, but implementation
still requires the requested approval-gate review loop: one agent reviews the
design/work, and one agent decides whether to proceed or prompts fixes.

## Decision

Proceed only after two approval-gate review agents find no blockers for this
bounded design. The first implementation slice after this note should be
engine-only metric support with tests; do not touch worker, protocol,
frame-buffer, bridge, `useTraining`, UI, URL/config, persistence, public
config, dependencies, `docs/worker-protocol.md`, or training behavior in that
first code commit.
