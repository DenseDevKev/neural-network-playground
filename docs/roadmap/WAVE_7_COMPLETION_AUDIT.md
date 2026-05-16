# Wave 7 Completion Audit

Date: 2026-05-15

## Scope

This audit checks Wave 7 status after the worker-authored multiclass confusion
metric path landed through UI preference and persistence non-leakage guards.

Source of truth:

- `docs/roadmap/ROADMAP_STATE.md`
- `docs/roadmap/WAVE_7_PROPOSAL.md`
- Wave 7 design notes under `docs/design-notes/`
- Browser QA evidence under `docs/qa/browser-qa/`
- Performance evidence in `docs/perf/PERFORMANCE_BASELINE.md`
- Git history through `b275bc9`

## Completed Bounded Wave 7 Slices

- Side-by-Side Model Arena:
  - Saved-run comparison UI.
  - Scalar live arena runtime prototype.
  - Scalar live arena UI prototype and QA evidence.
- Slow-Motion Backprop Explanation Mode:
  - Engine dry-run backprop summary foundation.
  - Worker one-shot RPC.
  - Inspection-panel preview UI and QA evidence.
- Loss Landscape Probe:
  - Engine dry-run loss grid foundation.
  - Worker one-shot RPC.
  - Inspection-panel probe UI and QA evidence.
- Multiclass Classification Mode:
  - Softmax/categorical-loss helpers and private `Network` path.
  - Shared config/URL/persistence eligibility guards for the approved tuple.
  - Worker/runtime acceptance for the approved tuple.
  - Public `Three-Class Softmax Lab` preset.
  - Public `Three-Class` Data-panel chip.
  - Guided Three-Class Softmax lesson.
  - Multiclass decision-boundary transport and rendering.
  - Worker-authored bounded 3x3 confusion metrics, frame-buffer storage, UI
    preference, Browser QA evidence, and persistence non-leakage guards.

## Latest Verification

- `pnpm --filter @nn-playground/shared exec vitest run src/__tests__/experimentMemory.test.ts --pool=forks --reporter=dot`
  passed with 1 file and 35 tests.
- `pnpm --filter @nn-playground/web exec vitest run src/store/experimentRunCapture.test.ts src/store/experimentMemoryStore.test.ts src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot`
  passed with 3 files and 29 tests.
- `pnpm test` passed with engine 13 files/320 tests, shared 6 files/125 tests,
  and web 51 files/474 tests.
- `pnpm lint` passed with no ESLint output.
- `pnpm build` passed with the existing Vite large-chunk warning.
- `pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

## Browser QA Status

Browser QA Mode B has been recorded for the latest user-visible Wave 7
multiclass surfaces:

- `docs/qa/browser-qa/wave-7-public-multiclass-preset.md`
- `docs/qa/browser-qa/wave-7-public-multiclass-data-chip.md`
- `docs/qa/browser-qa/wave-7-three-class-lesson.md`
- `docs/qa/browser-qa/wave-7-code-export-guard.md`
- `docs/qa/browser-qa/wave-7-multiclass-run-history.md`
- `docs/qa/browser-qa/wave-7-multiclass-boundary-renderer.md`
- `docs/qa/browser-qa/wave-7-confusion-ui-worker-authored.md`

Several earlier hidden or guard-only slices remain correctly marked as no
Browser QA required. Historical partial QA notes for Code Export, public
multiclass run history, and public multiclass boundary rendering now include
follow-up Mode B checks after public three-class controls landed.

## Remaining Deferred Product Bets

These are not incomplete defects in the landed slices. They are larger Wave 7
product bets that still need their own design notes, tests, Browser QA plans,
and rollback plans before implementation:

- Advanced architecture comparison.
- Interactive gradient explanation mode beyond the current slow-motion preview.
- Paired heavy live-arena visualizations.
- Continuous arena streaming or multiple-worker arena execution.
- Advanced Loss Landscape controls beyond the bounded one-shot probe.
- Arbitrary multiclass class counts.
- Additional public multiclass controls beyond the approved tuple surfaces.
- URL/config, persistence schema, or public config changes for any of the above.

## Next Safe Slice

The next lowest-risk Wave 7 slice is a design note for Advanced Architecture
Comparison, followed by a small UI-only saved-run architecture diff if the
design stays within existing run-history data.

Constraints for that slice:

- Use existing saved-run config and summary data only.
- Do not change worker protocol, frame-buffer semantics, engine math,
  persistence schema, URL/config serialization, public config shape, or
  dependencies.
- Keep the implementation in `RunHistoryPanel` or a small local helper.
- Add component tests for architecture diff summaries and accessible labels.
- Run Browser QA Mode B if the UI changes.

## Audit Result

Wave 7 is complete for the bounded slices already implemented and recorded in
`ROADMAP_STATE.md`. Continuing toward "all product bets" should proceed through
the next small design-and-implementation slice, not through broad refactors.
