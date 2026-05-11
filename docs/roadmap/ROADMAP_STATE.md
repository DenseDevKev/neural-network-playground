# Roadmap State

## Last Updated

2026-05-11

## Repository

- Branch: `codex/wave-0-review-packaging`
- Last verified commit: `da480c8` before Wave 0 verification
- Remote: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- PR: Not created yet
- Package manager: pnpm with `pnpm-lock.yaml` and `pnpm-workspace.yaml`
- Verification commands: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`
- CI workflows: `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

## Current Position

- Wave: Wave 0
- Slice: Review, package, baseline, and stabilize
- Risk: Low
- Status: In progress

## Completed Slices

| Date | Wave | Slice | Commit | Verification | Evidence |
|---|---|---|---|---|---|
| 2026-05-11 | Pre-roadmap | Expanded guided lesson library | `56a4565` | `pnpm test`, `pnpm lint`, `pnpm build` previously recorded as passing in discovery evidence | `docs/agent-discovery-report.md` |
| 2026-05-11 | Pre-roadmap | Decision-boundary overlay explanations | `97d9e9f` | Targeted web tests previously recorded as passing in discovery evidence | `docs/agent-discovery-report.md` |
| 2026-05-11 | Pre-roadmap | Decision-boundary browser QA evidence | `da480c8` | Browser QA previously recorded with no console errors | `docs/agent-discovery-report.md` |

## Current Verification Status

- Tests: `pnpm test` passed on 2026-05-11 with engine 270 tests, shared 62 tests, and web 308 tests.
- Lint: `pnpm lint` passed on 2026-05-11.
- Build: `pnpm build` passed on 2026-05-11 with the existing Vite chunk-size warning.
- Browser QA: Wave 0 Mode B passed on 2026-05-11 against `http://127.0.0.1:5173/` with no console errors.
- Accessibility: Wave 0 Browser QA confirmed semantic buttons, tabs, tabpanels, lesson combobox, and decision-boundary `img` role were reachable; no product UI changed in Wave 0.
- Performance: `pnpm test:perf` passed on 2026-05-11 with 2 benchmark files and 4 benchmark tests.

## Browser QA Evidence

- `docs/qa/browser-qa/wave-0-baseline.md`
- Prior decision-boundary screenshot: `/private/tmp/nn-playground-decision-overlay-errors.png`
- Wave 0 compact screenshot: `/private/tmp/nn-playground-wave0-compact.png`
- Wave 0 final screenshot: `/private/tmp/nn-playground-wave0-final.png`

## Performance Evidence

- `docs/perf/PERFORMANCE_BASELINE.md`

## Design Decisions

| Date | Decision | Reason | Source / Design Note |
|---|---|---|---|
| 2026-05-11 | Keep Wave 0 docs-only and ignore `.claude/worktrees/` | `.claude/worktrees/` is local generated agent state and should not be committed | `/goal` implementation plan |
| 2026-05-11 | Treat `docs/agent-discovery-report.md` as retained Wave 0 evidence | It contains architecture, risk, and prior verification context useful to reviewers | `/goal` implementation plan |
| 2026-05-11 | Keep Wave 1 action cards web-local and navigation-only | Avoids protected worker, schema, persistence, URL/config, runtime, and training behavior contracts | `/goal` implementation plan |

## Known Issues

- `main` is ahead of `origin/main` by local roadmap/pre-roadmap commits.
- `.claude/worktrees/` existed before Wave 0 as untracked local agent state and is intentionally ignored.
- Vite build has an existing large chunk warning; this is not a Wave 0 regression unless the warning changes materially.

## Blocked Items

- None currently.

## Deferred Items

- Wave 2 lesson depth.
- Wave 3 QA infrastructure beyond the Wave 0/Wave 1 records.
- Wave 4 visualization inspection improvements.
- Wave 5 runtime and performance hardening.
- Wave 6A-6D experiment workflows.
- Wave 6E checkpoints and timeline scrubber, pending mandatory approval.
- Wave 7 large product bets, pending mandatory approval.

## Approval Gates Reached

- None.

## Next Recommended Slice

Commit Wave 0 with `docs(wave0): record roadmap baseline`, then begin Wave 1 Slice 1: action registry.

## Handoff Notes

Required preflight was run on 2026-05-11:

- `pwd`: `/Users/kevincontreras/CascadeProjects/neural-network-playground`
- `git status --short`: `?? .claude/worktrees/`
- `git branch --show-current`: `main`
- `git remote -v`: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- `git log --oneline -5`: `da480c8`, `97d9e9f`, `56a4565`, `0fe2196`, `dc7160a`

The requested branch `codex/wave-0-review-packaging` was created after preflight. The first branch creation attempt hit a sandboxed Git ref write issue and succeeded when Git branch creation was approved.
