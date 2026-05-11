# Roadmap State

## Last Updated

2026-05-11

## Repository

- Branch: `codex/wave-0-review-packaging`
- Last verified commit: `d9cc432` before Wave 4 activation histogram design note
- Remote: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- PR: Not created yet
- Package manager: pnpm with `pnpm-lock.yaml` and `pnpm-workspace.yaml`
- Verification commands: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`
- CI workflows: `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

## Current Position

- Wave: Wave 4
- Slice: Activation histogram design note
- Risk: High if implemented
- Status: Approval gate reached

## Completed Slices

| Date | Wave | Slice | Commit | Verification | Evidence |
|---|---|---|---|---|---|
| 2026-05-11 | Pre-roadmap | Expanded guided lesson library | `56a4565` | `pnpm test`, `pnpm lint`, `pnpm build` previously recorded as passing in discovery evidence | `docs/agent-discovery-report.md` |
| 2026-05-11 | Pre-roadmap | Decision-boundary overlay explanations | `97d9e9f` | Targeted web tests previously recorded as passing in discovery evidence | `docs/agent-discovery-report.md` |
| 2026-05-11 | Pre-roadmap | Decision-boundary browser QA evidence | `da480c8` | Browser QA previously recorded with no console errors | `docs/agent-discovery-report.md` |
| 2026-05-11 | Wave 0 | Roadmap baseline packaging | `217a37a` | `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, Browser QA Mode B all passed | `docs/qa/browser-qa/wave-0-baseline.md`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-11 | Wave 1 | Explanation action registry | `4cca660` | `pnpm --filter @nn-playground/web test -- src/explanations/trainingExplanations.test.ts` passed with 50 files and 310 tests in the web run | `apps/web/src/explanations/trainingExplanations.test.ts` |
| 2026-05-11 | Wave 1 | Explanation action focus targets | `b8ff638` | `pnpm --filter @nn-playground/web test -- src/explanations/explanationActionFocus.test.ts` passed with 51 files and 314 tests in the web run | `apps/web/src/explanations/explanationActionFocus.test.ts` |
| 2026-05-11 | Wave 1 | Explanation action card UI | `1a3c734` | Targeted web run passed with 51 files and 319 tests, including component, integration, focus helper, action metadata, and axe coverage | `apps/web/src/components/visualization/TrainingExplanationPanel.test.tsx`, `apps/web/src/__tests__/appShell.integration.test.tsx` |
| 2026-05-11 | Wave 1 | Explanation action QA/state | `c901130` | `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, Browser QA Mode B all passed | `docs/qa/browser-qa/wave-1-explanation-actions.md` |
| 2026-05-11 | Wave 2 | Tuning/failure-mode lesson content | `e822187` | `pnpm test`, `pnpm lint`, `pnpm build`, Browser QA Mode B all passed | `docs/qa/browser-qa/wave-2-lesson-depth.md` |
| 2026-05-11 | Wave 3 | QA checklist and browser evidence template | `d9cc432` | Docs-only commit after prior `pnpm test`, `pnpm lint`, and `pnpm build` passed | `docs/qa/QA_CHECKLIST.md`, `docs/qa/browser-qa/TEMPLATE.md` |

## Current Verification Status

- Tests: `pnpm test` passed on 2026-05-11 after Wave 2 content with engine 270 tests, shared 62 tests, and web 319 tests.
- Lint: `pnpm lint` passed on 2026-05-11 after Wave 2 content.
- Build: `pnpm build` passed on 2026-05-11 after Wave 2 content with the existing Vite chunk-size warning.
- Browser QA: Wave 0, Wave 1, and Wave 2 Mode B passed on 2026-05-11 against `http://127.0.0.1:5173/` with no console errors.
- Accessibility: Wave 1 component `jest-axe` coverage passed for the rendered action-card panel; Browser QA verified native button keyboard activation.
- Performance: `pnpm test:perf` passed on 2026-05-11 with 2 benchmark files and 4 benchmark tests after Wave 1.

## Browser QA Evidence

- `docs/qa/browser-qa/wave-0-baseline.md`
- `docs/qa/browser-qa/wave-1-explanation-actions.md`
- `docs/qa/browser-qa/wave-2-lesson-depth.md`
- Prior decision-boundary screenshot: `/private/tmp/nn-playground-decision-overlay-errors.png`
- Wave 0 compact screenshot: `/private/tmp/nn-playground-wave0-compact.png`
- Wave 0 final screenshot: `/private/tmp/nn-playground-wave0-final.png`
- Wave 1 action-card screenshot: `/private/tmp/nn-playground-wave1-action-card.png`
- Wave 1 compact action-card screenshot: `/private/tmp/nn-playground-wave1-action-card-compact.png`
- Wave 2 learning-rate lesson screenshot: `/private/tmp/nn-playground-wave2-learning-rate-lesson.png`

## Performance Evidence

- `docs/perf/PERFORMANCE_BASELINE.md`
- Wave 1 post-change `pnpm test:perf` passed. Observed values: `predictGrid` 1299.7205 ms, `predictGridInto` 1277.3120 ms, `predictGridWithNeurons` 820.7586 ms, `predictGridWithNeuronsInto` 719.5801 ms, Adam/L2/Clip applyGradients 4.6340 ms, SGD applyGradients 1.7477 ms.
- Wave 1 build main bundle changed from 351.56 kB to 354.89 kB, below the 10% build-size warning threshold.

## Design Decisions

| Date | Decision | Reason | Source / Design Note |
|---|---|---|---|
| 2026-05-11 | Keep Wave 0 docs-only and ignore `.claude/worktrees/` | `.claude/worktrees/` is local generated agent state and should not be committed | `/goal` implementation plan |
| 2026-05-11 | Treat `docs/agent-discovery-report.md` as retained Wave 0 evidence | It contains architecture, risk, and prior verification context useful to reviewers | `/goal` implementation plan |
| 2026-05-11 | Keep Wave 1 action cards web-local and navigation-only | Avoids protected worker, schema, persistence, URL/config, runtime, and training behavior contracts | `/goal` implementation plan |
| 2026-05-11 | Keep action cards as native buttons with `aria-describedby` reasons | Preserves keyboard semantics and keeps action labels concise while retaining educational context | Wave 1 implementation |
| 2026-05-11 | Use existing layout store tabs/phase plus DOM focus targets for action cards | Navigates existing UI only and avoids persistence/schema changes | Wave 1 implementation |
| 2026-05-11 | Add Wave 2 lesson depth as content-only registry entries first | Reuses existing presets and lesson engine without schema, persistence, or runtime changes | Wave 2 implementation |

## Known Issues

- `main` is ahead of `origin/main` by local roadmap/pre-roadmap commits.
- `.claude/worktrees/` existed before Wave 0 as untracked local agent state and is intentionally ignored.
- Vite build has an existing large chunk warning; this is not a Wave 0 regression unless the warning changes materially.
- Browser QA exercised the live `test-metrics-stale` action card. Hyperparams/loss action targeting is covered by component and app integration tests.

## Blocked Items

- Wave 4 activation histogram implementation is blocked pending approval because it likely requires worker/protocol, frame-buffer, runtime snapshot, or engine activation data changes.

## Deferred Items

- Wave 2 lesson depth.
- Wave 3 QA infrastructure beyond the Wave 0/Wave 1 records.
- Wave 4 visualization inspection improvements.
- Wave 5 runtime and performance hardening.
- Wave 6A-6D experiment workflows.
- Wave 6E checkpoints and timeline scrubber, pending mandatory approval.
- Wave 7 large product bets, pending mandatory approval.

## Approval Gates Reached

- Wave 4 activation histogram explorer requires approval before implementation if it changes worker protocol, frame-buffer domains, runtime snapshot fields, or engine activation collection.

## Next Recommended Slice

Review `docs/design-notes/activation-histogram-explorer.md` and decide whether to approve a high-risk activation histogram data path, or defer it and continue with another existing-data-only visualization/accessibility slice.

## Handoff Notes

Required preflight was run on 2026-05-11:

- `pwd`: `/Users/kevincontreras/CascadeProjects/neural-network-playground`
- `git status --short`: `?? .claude/worktrees/`
- `git branch --show-current`: `main`
- `git remote -v`: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- `git log --oneline -5`: `da480c8`, `97d9e9f`, `56a4565`, `0fe2196`, `dc7160a`

The requested branch `codex/wave-0-review-packaging` was created after preflight. The first branch creation attempt hit a sandboxed Git ref write issue and succeeded when Git branch creation was approved.
