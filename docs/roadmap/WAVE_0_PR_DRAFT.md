# Wave 0 PR Draft

## Summary

- Record the Neural Network Playground 2.0 roadmap baseline in persistent project docs.
- Preserve the existing discovery report as Wave 0 evidence.
- Ignore local generated `.claude/worktrees/` agent state.
- Add baseline Browser QA and performance evidence files for repeatable follow-up verification.

## What Changed

- Added `docs/roadmap/ROADMAP_STATE.md` as the persistent roadmap source of truth.
- Added `docs/qa/browser-qa/wave-0-baseline.md` for Mode B browser QA evidence.
- Added `docs/perf/PERFORMANCE_BASELINE.md` for lightweight performance baseline evidence.
- Added `docs/roadmap/WAVE_0_PR_DRAFT.md`.
- Updated `docs/agent-discovery-report.md` from a temporary handoff to retained Wave 0 discovery evidence.
- Added `.claude/worktrees/` to `.gitignore`.

## Why

- Wave 0 establishes a clean, reviewable baseline before more product work.
- Future roadmap slices need persistent state, verification history, QA evidence, and performance notes.
- Local agent worktrees should remain untracked and out of commits.

## Test Plan

- [x] `pnpm test`
- [x] `pnpm lint`
- [x] `pnpm build`
- [ ] Targeted tests: not applicable for docs-only Wave 0
- [x] `pnpm test:perf`
- [x] Browser QA: `docs/qa/browser-qa/wave-0-baseline.md`

## Screenshots / QA Evidence

- Browser QA evidence: `docs/qa/browser-qa/wave-0-baseline.md`
- Compact screenshot: `/private/tmp/nn-playground-wave0-compact.png`
- Final compact loss/lesson screenshot: `/private/tmp/nn-playground-wave0-final.png`

## Accessibility Notes

- Wave 0 has no product UI change.
- Browser QA inspected the app shell, lesson panel, tabs, tabpanels, training controls, dataset controls, and visualization reachability.

## Performance Notes

- Wave 0 has no runtime change.
- Baseline evidence is recorded in `docs/perf/PERFORMANCE_BASELINE.md`.

## Backward Compatibility / Migration Notes

- No worker protocol, serialization, URL/config, persistence, public config, engine, or deployment behavior changes.

## Risks

- Existing `main` is ahead of `origin/main`; this PR branch is stacked on local commits.
- Vite may continue to report its existing large chunk warning during build.

## Follow-up Backlog

- Wave 1: contextual explanation action cards.
- Wave 2: lesson and explanation depth.
- Wave 3: repeatable QA infrastructure.
- Wave 4: visualization inspection improvements.
- Wave 5: runtime and performance hardening.
- Wave 6A-6D: experiment workflows.
- Wave 6E and Wave 7 remain approval-gated.
