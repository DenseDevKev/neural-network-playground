# Design Note: Advanced Architecture Comparison

## Problem

Neural Network Playground already lets learners save runs, compare scalar
metrics, inspect loss thumbnails, export reports, and initialize a scalar live
arena from saved runs. Those comparisons answer "which run performed better,"
but they do not yet make the architectural differences easy to scan.

Learners still need to mentally connect architecture choices such as hidden
layer depth, neuron count, activation, output activation, optimizer, learning
rate, regularization, and enabled features with the resulting loss and
generalization behavior. The next safe Wave 7 slice should make those
differences explicit without changing runtime behavior.

## Proposed Change

Add an existing-data architecture comparison to the saved-run History panel.

The approved first implementation slice is UI-only:

- Use only existing `ExperimentRunRecordV1.config`, `summary`, and `history`
  data already loaded by the run-history store.
- Compare the two selected saved runs in the existing side-by-side model arena.
- Render a compact architecture diff group below the existing metric
  comparison.
- Include architecture and tuning fields that already exist in saved records:
  hidden layers, total hidden units, activation, output activation, loss,
  optimizer, learning rate, regularization, batch size, enabled features,
  dataset, sample count, and noise.
- Use text summaries and native semantic markup; do not rely on color alone.
- Keep all comparison state local to `RunHistoryPanel`.

Out of scope for this slice:

- No worker protocol changes.
- No frame-buffer changes.
- No engine math changes.
- No persistence schema changes.
- No URL/config serialization changes.
- No public config shape changes.
- No new dependencies.
- No continuous live arena or paired heavy visualization changes.

## User Value

- Helps learners see what changed between two saved experiments before they
  interpret metric differences.
- Makes model capacity and tuning tradeoffs more concrete.
- Gives teachers a quick "what changed?" explanation surface for saved runs.
- Builds on the existing History panel instead of creating a new workflow.

## Affected Files

Design-only slice:

- `docs/design-notes/advanced-architecture-comparison.md`
- `docs/roadmap/ROADMAP_STATE.md`

Approved UI-only implementation slice:

- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.test.tsx`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/wave-7-architecture-comparison.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

The UI-only slice has no runtime data-flow impact.

- Existing saved-run records remain the only source of comparison data.
- Derived architecture rows are computed at render time from bounded config
  objects already present in React state.
- No large arrays are stored or streamed.
- No worker RPCs are added.
- No frame-buffer version counters are added.
- No training behavior changes.

## Compatibility Risks

Low for the approved UI-only slice:

- Existing saved records already contain the fields being compared.
- Missing or unsupported values should render as `n/a`, not throw.
- No schema migration is needed.
- Deleting, restoring, exporting, and saving runs must keep existing behavior.

High-risk future extensions remain gated:

- Persisted comparison preferences.
- Shareable comparison URLs.
- Live dual architecture training.
- Additional run-history schema fields.
- Worker-authored architecture diagnostics.

## Accessibility Impact

The comparison must be readable without visual-only styling.

Requirements:

- Wrap the architecture comparison in a semantic group labelled
  `Architecture comparison`.
- Expose each diff row as text with both model values and a concise
  interpretation where useful.
- Keep values legible at compact widths through wrapping, not horizontal
  scrolling.
- Use native selects and existing buttons.
- Do not add animation.
- Preserve keyboard order: model selectors, live arena controls, model panes,
  metric comparison, architecture comparison.

## Performance Impact

Expected impact is minimal:

- The comparison derives short strings from two saved-run configs.
- No large arrays or worker data are involved.
- The lazy Run History chunk may grow slightly.
- No default training or visualization hot path changes.

Performance evidence should record:

- `pnpm test:perf` values.
- `pnpm build` chunk output, especially `RunHistoryPanel`.
- Whether the lazy chunk changes by more than 10%.

## Test Plan

Design-only slice:

- `git diff --check`.

UI implementation slice:

- Add a failing component test that selects two saved runs with different
  hidden layers and expects an accessible `Architecture comparison` group.
- Add assertions for hidden layers, total hidden units, activation/loss,
  optimizer/learning-rate, regularization, dataset, and features.
- Add a regression assertion that selecting different model options updates the
  architecture comparison.
- Run the targeted Run History test.
- Run `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` before
  commit.

## Browser QA Plan

Use Browser QA Mode B if available.

1. Start the local web app.
2. Save or use existing saved runs so at least two records are visible.
3. Open History.
4. Verify the side-by-side arena shows `Architecture comparison`.
5. Change the Model A or Model B selector and verify the architecture rows
   update.
6. Verify keyboard navigation reaches the selectors and existing action
   buttons.
7. Verify compact viewport wrapping does not overlap text.
8. Check console errors.
9. Capture desktop and compact screenshots.

## Rollback Plan

For the design-only slice, revert this file and the roadmap-state update.

For the UI-only implementation slice, revert the Run History component, test,
CSS, Browser QA evidence, performance evidence, and roadmap-state updates. No
schema, worker, URL/config, persistence, engine, or dependency rollback is
needed because those areas are out of scope.

## Approval Required

The design note is for a Wave 7 product-bet slice, so approval is required
before implementation. The user approved continuing with this next safe slice
on 2026-05-15 after reviewing the structured next-work list.

Any future expansion that changes worker protocol, frame-buffer semantics,
persistence schema, URL/config serialization, public config shape, dependencies,
or training behavior requires a separate approval gate.

## Decision

Approved for a first UI-only saved-run architecture comparison using existing
run-history data only. Proceed with TDD and stop if implementation requires any
gated contract change.
