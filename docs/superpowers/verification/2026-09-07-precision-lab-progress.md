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

## Integration status — reconciled September 8

- [x] Connect selection to both production graph renderers and the visible selection deck — committed `6e06b678...` / `3e7da9a...`; focused/full package CI passed.
- [x] Integrate one live boundary and evidence-only detail controls — committed `3e7da9a...`; component tests/full package CI passed; browser geometry still pending.
- [x] Activate PrecisionLabShell in App — committed `3e7da9a...`; legacy compatibility source retained pending browser acceptance.
- [ ] Complete responsive, accessibility, browser, and performance acceptance.
- [ ] Merge only a qualified replacement into main.
- [ ] Publish and verify a live deployment after the repository Pages blocker is resolved.

“Pinned boundary” means a live boundary that stays mounted and visible, not a new
saved snapshot or pin/replace/clear feature. The living checklist must not invent
such a workflow. The unavailable historical prototype is not a visual-parity reference.

## P3 renderer integration slice

- [x] Give both graph renderers the same optional selection controller; the standalone graph also keeps one controller across renderer switches.
- [x] Select neurons through pointer, Enter, and Space; clear through Escape or the deck action without changing the experiment URL.
- [x] Preserve selection across parameter/grid updates, hover/blur, workspace profiles, and disclosure changes; invalidate on model generation or architecture change.
- [x] Keep parameter and activation-grid steps distinct in the visible details.
- [x] Show an unavailable activation grid as unavailable, not an invented numeric grid.
- [x] Paint selected signed influences at actual weight magnitude; negative selected paths are dashed, ordinary paths stay visible, and filters remain effective.
- [x] Use bounded rounded Canvas tiles, padded hit targets, and coalesced ResizeObserver updates; fit the first real measurement rather than the placeholder size.
- [x] Preserve the later user zoom across container resizes and skip Canvas repaint on unrelated output-grid frames.
- [x] Preserve zoom/Fit, edge filters, mode controls, architecture summary, and lesson context in the SVG fallback.
- [x] Run all four graph files: 47 tests pass, including the retained original Canvas/SVG tests.
- [x] Run the complete web suite after the navigation correction: 101 files, 1,145 tests pass.
- [x] Run lint, complete typechecks, production build, and fixed JavaScript gzip limits: all pass in the isolated Node 22.16.0 / pnpm 9.15.9 workspace.

TDD evidence: the initial new geometry/selection expectations failed (13 failures,
2 passes) before implementation. Missing-grid labels failed in both renderers
before their correction. First-real-size fit failed at 100% versus the required
89% before removing the placeholder-size latch. Final graph verification has
zero failures. Unit tests mock only browser drawing and known numerical artifacts;
these are not a claim of real-browser pointer/geometry acceptance.

The local build measured entry 151,777 / 152,245 gzip bytes, InspectionPanel
5,414 / 7,373, and total JavaScript 231,253 / 234,161. Caps are unchanged. Main's
engine and shared implementation are unchanged. At that earlier checkpoint the App still used BuildRunShell. This was superseded
by `3e7da9a...`: App now uses PrecisionLabShell, a shared selection deck and one
live boundary rail. Do not repeat or overwrite that integrated work. The branch workflow now also runs
the full package suite so a passing changed-file subset cannot hide old failures.

## September 8 local continuation — not pushed

The exact GitHub source tree at `3e7da9a...` was recovered and checksum/tree-verified.
The canonical root `NN-FORGE-LIVING-EXECUTION-PLAN.md` is restored from its original
Library copy. See `2026-09-08-precision-lab-local-continuation.md` for source and test evidence.

Local changes add one App-owned save controller shared by transport/History, exact
pending-artifact retries, responsive evidence/transport fixes, portal help/focus
regressions and targeted education lazy loading. Local final tests: engine 497,
shared 334, web 1,170; helpers 91; lint/types/build/bundle pass. Gzip bytes:
entry 151,131, InspectionPanel 5,431, total 234,068; caps unchanged.

These changes are NOT an upstream commit or browser-qualified release. The available
GitHub tools are read-only; Chromium/WebKit executables are absent and both launch attempts failed before the app ran.
The new recovery and paired-reference workflow jobs have not executed. The accepted
baseline policy is resolved and merged; Pages still has a settings blocker.
