# Design Note: Side-by-Side Model Arena

## Problem

Neural Network Playground 2.0 currently teaches experimentation through one live model at a time, saved run history, comparison summaries, thumbnails, reports, lessons, explanation action cards, and checkpoint timeline controls. Those features help learners inspect a single run, but they still require users to mentally compare how two different choices behave over time.

A side-by-side arena would make comparison explicit: learners could place two model configurations next to each other, run bounded training scenarios, and see how architecture, features, regularization, learning rate, and dataset choices affect boundary shape, loss, accuracy, and generalization.

## Proposed Change

Implement Wave 7 as a staged initiative, not one large feature. The recommended first implementation phase is a UI-first arena prototype that reuses existing data and saved-run concepts before introducing additional runtime complexity.

Proposed implementation phases:

1. Design-only phase, this note:
   - Define target UX, risk boundaries, tests, performance plan, and approval gates.
   - No product code changes.

2. Phase 1, low-to-medium-risk arena shell:
   - Add a web-local arena panel or mode that compares two existing saved runs or current run summaries using already available data.
   - Reuse existing run-history records, loss thumbnails, diff summaries, and report data.
   - Do not run two live models.
   - Do not change worker protocol, frame buffer, engine math, persistence schema, URL/config serialization, or public config shape.

3. Phase 2, live arena design note:
   - Before live dual-model training, write a second design note covering worker ownership, runtime state, memory bounds, demand cadence, frame-buffer semantics, and browser QA.
   - Stop for approval if any worker protocol, frame-buffer, public config, URL/config, or persistence changes are required.

4. Phase 3, approved live arena implementation:
   - Prefer one isolated runtime owner per model only if lifecycle and memory bounds are proven.
   - Prefer sequential/time-sliced training before multiple always-hot workers.
   - Keep large arrays out of React state and preserve the existing single-model flow.

## User Value

- Compare two models without losing the educational thread.
- See how one changed variable affects loss, accuracy, generalization gap, and boundary shape.
- Give teachers and learners a clearer discussion artifact than a single run history list.
- Build on existing saved-run comparison, thumbnails, report export, and checkpoint work rather than replacing them.

## Affected Files

Design-only slice:

- `docs/design-notes/side-by-side-model-arena.md`
- `docs/roadmap/ROADMAP_STATE.md`

Likely Phase 1 files if separately approved:

- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.test.tsx`
- `apps/web/src/components/layout/*`
- `apps/web/src/store/experimentMemoryStore.ts`
- `apps/web/src/store/experimentRunCapture.ts`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/*`
- `docs/perf/PERFORMANCE_BASELINE.md`

Likely Phase 2 or Phase 3 files only after separate approval:

- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/worker/*`
- `packages/shared/src/workerProtocol.ts`
- `packages/engine/src/network.ts`
- `docs/worker-protocol.md`

## Affected Runtime/Data Flow

Design-only phase has no runtime impact.

Recommended Phase 1 should use existing saved-run data only:

- Existing run records remain the data source.
- Existing thumbnails and diff summaries remain render-time artifacts.
- No new worker messages.
- No new frame-buffer payloads.
- No new engine state.
- No live second model.

Live arena phases would likely affect runtime data flow and must be treated as high-risk:

- Two model states need bounded ownership.
- Training cadence and visualization demand must avoid starving the main app.
- Decision-boundary and loss data must remain cadence-gated.
- Large arrays must stay in frame buffers or worker-owned state, not React state.
- Existing single-model training behavior must remain unchanged.

## Compatibility Risks

Phase 1 can avoid compatibility risk by not changing schemas or shared contracts.

High-risk compatibility areas for later phases:

- Public config shape.
- URL/config serialization.
- Run-history persistence schema.
- Worker protocol.
- Frame-buffer semantics.
- Runtime snapshot format.
- Training determinism.

Any change in those areas requires explicit approval before implementation.

## Accessibility Impact

Arena UI must be usable without visual-only comparison.

Requirements:

- Each model pane needs a semantic region label, for example `Model A` and `Model B`.
- Diff summaries must not rely on color alone.
- Controls must be native buttons, selects, sliders, checkboxes, or tabs where possible.
- Keyboard order should move predictably through model A controls, model B controls, then shared comparison controls.
- Screen-reader summaries should describe which model improved, regressed, or diverged.
- Compact layout must avoid overlapping panels and oversized hero-style composition.
- Any animation must honor reduced motion.

## Performance Impact

Design-only phase has no performance impact.

Phase 1 should have minimal performance impact because it reuses saved-run summaries and render-time thumbnails.

Live arena phases require a performance baseline comparison before implementation:

- Build size.
- Fixed-step training time.
- Worker message cadence.
- Decision-boundary responsiveness.
- Memory use or bounded proxy measurement where available.
- Main-thread responsiveness while two panes are visible.

Warning thresholds should follow `docs/perf/PERFORMANCE_BASELINE.md`.

## Test Plan

Design-only phase:

- `git diff --check`.

Phase 1 tests if separately approved:

- Component tests for selecting two saved runs.
- Component tests for accessible model-region labels.
- Tests for comparison summaries that do not rely on color alone.
- Store tests if web-local selection state is introduced.
- Regression tests proving existing run history save/delete/restore/export behavior still works.
- Browser QA for desktop and compact layout.

Live arena tests if separately approved:

- Worker lifecycle tests for two model owners or time-sliced execution.
- Demand/cadence tests for paired visualizations.
- Frame-buffer/version-counter tests if any paired heavy data is transported.
- Engine determinism tests if model-state cloning or branching is introduced.
- Accessibility tests for keyboard traversal and screen-reader summaries.
- Performance comparison against baseline.

## Browser QA Plan

Phase 1:

1. Load app with no console errors.
2. Save or select two runs.
3. Open arena comparison.
4. Verify both model regions are labelled.
5. Verify diff summaries and thumbnails render.
6. Navigate comparison controls by keyboard.
7. Check compact viewport layout.
8. Confirm existing run-history restore/export/delete still works.

Live phases:

1. Start, pause, reset, and step both models according to the approved runtime design.
2. Verify no console errors.
3. Verify single-model workflow still works.
4. Capture screenshots and performance notes.

## Rollback Plan

Phase 1 rollback:

- Revert arena UI files and tests.
- Leave existing run-history, saved-run comparison, thumbnails, reports, training, URL/config, and worker behavior untouched.

Live phase rollback:

- Revert the live arena slice as a single feature boundary.
- Preserve existing single-model worker path.
- Re-run `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and browser regression QA after rollback.

## Approval Required

This design note was approved to be created by the user on 2026-05-12.

Implementation is not approved by this note.

Separate approval is required before:

- Adding any arena product code.
- Running two live models.
- Adding or changing worker protocol.
- Changing frame-buffer semantics.
- Changing public config shape.
- Changing URL/config serialization.
- Changing persistence or run-history schema.
- Adding dependencies.
- Changing engine math or determinism-sensitive behavior.

## Decision

Proceed no further than this design note until the user explicitly approves a specific implementation phase.

Recommended next approval question:

Do you approve implementing Phase 1 of the Side-by-Side Model Arena as a saved-run comparison UI using existing saved-run data only, with no worker, protocol, frame-buffer, engine, persistence, URL/config, public config, or dependency changes?
