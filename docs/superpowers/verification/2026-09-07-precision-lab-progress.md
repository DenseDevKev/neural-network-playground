# Precision Lab continuation — verified task ledger

The production baseline is `98f29b86e469a2a545be75928ae6f32309fd1582` on main.
Work continues on `codex/nn-forge-precision-lab`; new component work is not a release.

## P2 navigation compatibility correction

- [x] Reproduce the two failing existing tests: layout persistence and the lesson that asks the learner to select Build.
- [x] Trace both failures to the early Precision Lab change making the generic recipe-section setters also change the workspace view.
- [x] Preserve `setActiveRecipeSection` and legacy `setActiveTabLeft` view semantics; they reveal the selected context without forcing Build.
- [x] Keep explicit `selectBuildContext` and `openAdvancedRecipeSection` atomic Build navigation.
- [x] Retain the original layout and guided-lesson assertions unchanged.
- [x] Correct the newer, contradictory Precision Lab expectation to distinguish selecting a section from navigating to it.
- [x] Run the three affected files: 55 tests pass after the correction, compared with 53 passes and two failures before it.

This is a narrow reconciliation of the July task-3 wording against the working
lesson contract. It does not change the recipe, model, worker, URL, persistence
format, or scientific evaluation cadence. The full web test run that identified
the problem had 1,142 passes and these two failures; that was not a passing run.

## Integration still pending

- [ ] Connect selection to both production graph renderers and the visible selection deck.
- [ ] Integrate the always-visible live boundary and evidence-only detail controls.
- [ ] Activate and style the sole Precision Lab shell.
- [ ] Complete responsive, accessibility, browser, and performance acceptance.
- [ ] Merge only a qualified replacement into main.
- [ ] Publish and verify a live deployment after the repository Pages blocker is resolved.

“Pinned boundary” means a live boundary that stays mounted and visible, not a new
saved snapshot or pin/replace/clear feature. The living checklist must not invent
such a workflow. The unavailable historical prototype is not a visual-parity reference.
