# Design Note: Multiclass Public UI Controls and Browser QA

## Problem

Multiclass Classification Mode now has hidden foundations across the engine,
shared validation, worker target encoding, bounded boundary transport,
read-only visualization, code export, and experiment-memory eligibility. The
feature is still intentionally unavailable through public app controls.

The next risk is not another isolated helper. It is the first user-visible path
that lets someone select, train, share, export, save, restore, and inspect a
3-class model. That path crosses public controls, URL/config behavior,
runtime validation, run history, report export, visualization copy,
accessibility, Browser QA, and performance evidence.

This note is a design-only gate. It defines the exact public flow and QA bar
before any visible controls are implemented.

## Proposed Change

Expose only one public multiclass path in a later code slice:

- Problem type: `classification`.
- Dataset entry point: one bounded 3-class dataset/preset.
- Class labels: exactly `0`, `1`, and `2`.
- Network output: exactly `outputSize: 3`.
- Output activation: `softmax`.
- Loss: `categoricalCrossEntropy`.
- Decision boundary: existing bounded class-index plus confidence frame-buffer
  transport.
- Confusion readout: existing UI-derived 3x3 test-set readout until an official
  worker-authored metric gets its own design/protocol gate.

Do not add arbitrary class counts, free-form output-size controls, raw
probability-grid transport, persistence schema changes, new dependencies, or
new worker/frame-buffer fields as part of the first public-control slice.

## User Value

Learners get a complete first multiclass workflow instead of a hidden demo:
choose a 3-class dataset, watch class regions emerge, inspect confidence,
compare actual-vs-predicted class counts, and round-trip the run through the
same share/export/history tools used by binary and regression lessons.

The bounded scope keeps the current product approachable while teaching that
softmax outputs compete and that a 2x2 confusion matrix does not describe all
classification problems.

## Affected Files

Likely first public-control implementation files:

- `packages/engine/src/datasets.ts`
- `packages/engine/src/__tests__/datasets.test.ts`
- `packages/shared/src/presets.ts`
- `packages/shared/src/__tests__/presets.test.ts`
- `packages/shared/src/serialization.ts`
- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/codeExport.ts`
- `apps/web/src/store/usePlaygroundStore.ts`
- `apps/web/src/store/experimentRunCapture.ts`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/components/controls/DataPanel.tsx`
- `apps/web/src/components/controls/HyperparamPanel.tsx`
- `apps/web/src/components/controls/NetworkConfigPanel.tsx`
- `apps/web/src/components/controls/ConfigPanel.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- `apps/web/src/data/datasetInsights.ts`
- `docs/qa/browser-qa/*`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

Explicitly out of scope for the first public-control slice:

- `packages/shared/src/workerProtocol.ts` unless a later design gate approves a
  new worker-authored metric.
- `apps/web/src/worker/frameBuffer.ts` unless a later design gate approves new
  payloads beyond the already implemented multiclass boundary fields.
- URL/config format changes beyond the already implemented opt-in `outputSize:
  3` contract.
- Persistence/run-history schema changes.
- New dependencies.

## Affected Runtime/Data Flow

The public flow should use one reusable approved-config predicate. That rule
must require all of these at once:

1. `data.problemType === "classification"`.
2. Dataset is the approved public 3-class dataset or preset.
3. `network.outputSize === 3`.
4. Final activation is `softmax`.
5. Loss is `categoricalCrossEntropy`.
6. Class labels are bounded integer values in `0..2`.
7. Shared serialization and experiment-memory validation accept the config only
   through explicit multiclass opt-in.

Expected user flow:

1. User selects the public 3-class dataset or multiclass preset.
2. Store transition applies the compatible network, loss, and activation
   together.
3. Public controls prevent invalid partial pairings, such as softmax with MSE,
   categorical loss with sigmoid, or `outputSize` values other than `1` or `3`.
4. `useTraining` accepts the approved multiclass config before touching the
   worker and still rejects malformed hidden states.
5. Worker training uses existing one-hot target encoding for the approved
   3-class config.
6. Decision-boundary demand uses existing class-index plus confidence arrays in
   the frame buffer.
7. Confusion Matrix renders the existing UI-derived 3x3 test-set readout from
   bounded test data and frame-buffer params.
8. URL share and JSON import/export use the already approved opt-in
   `outputSize: 3` contract.
9. Run history either saves/restores/report-exports approved multiclass records
   with explicit opt-in validation or shows visible feedback explaining why a
   save is unavailable.

## Compatibility Risks

- Public defaults must remain binary classification with one output.
- Old binary and regression URLs must decode unchanged.
- Old JSON imports and saved runs must remain valid.
- Partial multiclass URLs must fail closed or fall back to scalar defaults.
- Public controls must not leave the store in a half-multiclass state.
- Run-history capture must not silently return `null` for a visible public
  config.
- Decision-boundary and confusion readouts must not present stale binary copy
  for 3-class state.
- The current UI-derived confusion readout has a local forward pass; future
  engine activation changes must preserve parity coverage or move the readout
  behind an approved worker metric.

## Accessibility Impact

The first visible multiclass path must include:

- Keyboard-reachable dataset/preset controls and focus-visible training
  controls.
- Visible text class labels for `Class 0`, `Class 1`, and `Class 2`.
- Decision-boundary text summary naming dominant class and confidence.
- Non-color confidence cues in legends or summaries.
- Confusion readout labels for actual and predicted classes, plus row/column
  totals.
- Screen-reader text that distinguishes the UI-derived test-set readout from a
  worker-persisted metric.
- Compact viewport layout around `390x844` with no clipped legends, matrix
  headers, or buttons.
- Reduced-motion safety if any transition or animation is added.

## Performance Impact

The public-control slice should mostly wire existing bounded work:

- Worker output grows to three logits only for the approved dataset/preset.
- Decision-boundary payloads reuse class-index and confidence grids.
- Raw per-cell probability vectors remain forbidden.
- Large arrays remain in the worker/frame-buffer path, not React state.
- The 3x3 readout iterates bounded train/test split data only while the
  Confusion panel is visible.

Before marking the slice complete, compare against
`docs/perf/PERFORMANCE_BASELINE.md`:

- `pnpm test:perf` values.
- Production bundle sizes.
- Worker bundle size.
- Main UI chunk size.
- Any visible interaction or Browser QA uncertainty.

## Test Plan

Design-only slice:

- Docs consistency review.
- `git diff --check`.

First public-control implementation slice:

- Engine dataset tests proving deterministic labels `0`, `1`, and `2` and
  bounded sample counts.
- Shared preset tests allowing exactly one public multiclass preset while
  rejecting unsupported public dataset/preset combinations.
- Shared serialization tests for old scalar URLs unchanged, explicit
  `outputSize: 3` round-trip, and malformed multiclass rejection.
- Store tests for selecting the multiclass dataset/preset and preventing
  partial invalid pairings.
- Hyperparam and network-control tests proving compatible controls are locked,
  hidden, or updated together.
- Config Panel tests for URL copy, JSON export, JSON import, and failure copy.
- `useTraining` tests proving approved multiclass configs are accepted before
  worker calls and malformed states are rejected before worker calls.
- Worker tests proving the approved public dataset path produces one-hot
  targets and demand-gated multiclass boundary payloads.
- Run History tests for save, restore, report export, and visible rejected-save
  feedback if any path remains unavailable.
- Decision Boundary tests for class legend, confidence summary, scalar
  fallback, and no stale binary copy.
- Confusion Matrix tests for public 3x3 labels, totals, invalid/missing data,
  and scalar fallback.
- Accessibility assertions with Testing Library and existing axe setup where
  nearby tests already use it.

Required commands for a code slice:

- Targeted affected tests first.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

## Browser QA Plan

Browser QA Mode B is mandatory before public controls are marked complete.

Required scenarios:

1. Load the app at the local dev URL with no console errors.
2. Verify existing binary classification train, step, pause, reset, URL copy,
   JSON export/import, and run-history save still work.
3. Verify existing regression train, step, pause, reset, URL copy, JSON
   export/import, and run-history save still work.
4. Select the public multiclass dataset or preset.
5. Confirm the UI applies `outputSize: 3`, `softmax`, and
   `categoricalCrossEntropy` together.
6. Train, step, pause, and reset the multiclass run.
7. Confirm Decision Boundary renders class regions, a class legend, and a
   confidence summary.
8. Confirm Confusion Matrix renders a labeled 3x3 readout with row and column
   totals.
9. Confirm URL share/reload round-trips the multiclass config.
10. Confirm JSON export/import round-trips the multiclass config.
11. Confirm run-history save/restore/report export works or visibly explains a
    deliberate unavailable state.
12. Check keyboard navigation through dataset/preset controls, training
    controls, Boundary, and Confusion.
13. Check compact viewport around `390x844` for no overlap or clipped controls.
14. Capture screenshots and console status under `docs/qa/browser-qa/`.

## Rollback Plan

Keep implementation commits small:

1. Public dataset/preset contract.
2. Store/control state transition.
3. Runtime hook acceptance.
4. Config Panel import/export/share behavior.
5. Run-history save/restore/report behavior.
6. Visualization copy/accessibility polish.
7. Browser QA and roadmap evidence.

If a slice fails after the rollback protocol, revert only that slice and its
evidence docs. The existing scalar public path remains the fallback. Because
the first public-control slice should avoid schema migrations and new protocol
fields, rollback should not require data migrations.

## Approval Required

Yes. Public Multiclass UI controls are a Wave 7 product-direction change and
touch public config flow, URL/config behavior, public datasets/presets, runtime
validation, visualization, and run-history UX.

The user has broadly approved continuing, but code slices must still use this
note, spawn the requested approval/review subagents at gate boundaries, and
stay within the constraints above. Any worker-authored 3x3 metric, new
frame-buffer field, persistence schema migration, arbitrary class count, or raw
probability transport requires a separate design note and approval gate.

## Exact Approval Question

Do you approve the first public multiclass implementation slice to expose only
one bounded 3-class dataset/preset plus the minimum store/control transition
needed to apply the approved `classification`, `outputSize: 3`, `softmax`, and
`categoricalCrossEntropy` pairing together, while preserving scalar defaults,
avoiding new worker/protocol/frame-buffer fields, avoiding persistence schema
changes, avoiding arbitrary class counts, adding targeted tests first, and
leaving official worker-authored 3x3 metrics for a separate design gate?

## Decision

Proceed with this design-only slice. The next code slice may start only after
this note is committed and reviewed, and it should be the smallest public
dataset/preset plus state-transition slice that can preserve scalar defaults
and fail closed on invalid multiclass pairings.
