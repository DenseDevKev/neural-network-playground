# Roadmap State

## Last Updated

2026-05-11

## Repository

- Branch: `codex/wave-0-review-packaging`
- Last verified commit: `d1b6349` for Wave 5 runtime diagnostics design note; final Wave 5 verification passed after that commit before evidence packaging
- Remote: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- PR: Not created yet
- Package manager: pnpm with `pnpm-lock.yaml` and `pnpm-workspace.yaml`
- Verification commands: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`
- CI workflows: `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

## Current Position

- Wave: Wave 5
- Slice: Runtime and performance hardening completion
- Risk: Low for completed Wave 5 work; tests/docs only
- Status: Wave 5 complete through the approved non-gated scope; stop before Wave 6 pending explicit approval

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
| 2026-05-11 | Wave 4 | Activation histogram explorer | `590114d` | `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, targeted worker/shared/web tests, Browser QA Mode B all passed with dev perf warnings only | `docs/qa/browser-qa/wave-4-activation-histogram.md`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/worker-protocol.md` |
| 2026-05-11 | Wave 5 | Worker lifecycle and demand cadence regression tests | `6854f25` | Targeted web worker run passed with 51 files and 331 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit | `apps/web/src/worker/training.worker.test.ts`, `apps/web/src/worker/workerBridge.test.ts` |
| 2026-05-11 | Wave 5 | Frame-buffer, SharedArrayBuffer, WebGPU, and demand fallback tests | `847aebd` | Targeted web fallback run passed with 51 files and 336 tests; targeted engine WebGPU run passed with 13 files and 275 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit | `apps/web/src/__tests__/frameBuffer.test.ts`, `apps/web/src/worker/sharedSnapshot.test.ts`, `packages/engine/src/webgpu/__tests__/webgpu_detect.test.ts` |
| 2026-05-11 | Wave 5 | Runtime performance diagnostics design note | `d1b6349` | Docs/status review with `git diff --check` passed before commit; final Wave 5 `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed after commit | `docs/design-notes/runtime-performance-diagnostics.md` |

## Current Verification Status

- Tests: `pnpm test` passed on 2026-05-11 after Wave 5 diagnostics note with engine 275 tests, shared 65 tests, and web 336 tests.
- Lint: `pnpm lint` passed on 2026-05-11 after Wave 5 diagnostics note.
- Build: `pnpm build` passed on 2026-05-11 after Wave 5 diagnostics note with the existing Vite chunk-size warning. Relevant unchanged chunks: `training.worker--UxrksoV.js` 71.53 kB, `index-Do23QsAR.js` 360.52 kB gzip 109.51 kB.
- Browser QA: Wave 0, Wave 1, Wave 2, and Wave 4 Mode B passed on 2026-05-11. Wave 4 used `http://127.0.0.1:5176/`; no console errors were reported for the fresh URL, only dev perf warnings.
- Accessibility: Wave 1 component `jest-axe` coverage passed for the rendered action-card panel; Wave 4 histogram UI uses a native labelled select and `role="img"` text alternative covered by Testing Library assertions and Browser QA.
- Performance: `pnpm test:perf` passed on 2026-05-11 after Wave 5 diagnostics note with 2 benchmark files and 4 benchmark tests.

## Browser QA Evidence

- `docs/qa/browser-qa/wave-0-baseline.md`
- `docs/qa/browser-qa/wave-1-explanation-actions.md`
- `docs/qa/browser-qa/wave-2-lesson-depth.md`
- `docs/qa/browser-qa/wave-4-activation-histogram.md`
- Prior decision-boundary screenshot: `/private/tmp/nn-playground-decision-overlay-errors.png`
- Wave 0 compact screenshot: `/private/tmp/nn-playground-wave0-compact.png`
- Wave 0 final screenshot: `/private/tmp/nn-playground-wave0-final.png`
- Wave 1 action-card screenshot: `/private/tmp/nn-playground-wave1-action-card.png`
- Wave 1 compact action-card screenshot: `/private/tmp/nn-playground-wave1-action-card-compact.png`
- Wave 2 learning-rate lesson screenshot: `/private/tmp/nn-playground-wave2-learning-rate-lesson.png`
- Wave 4 activation histogram screenshot: `/private/tmp/nn-playground-wave4-activation-histogram.png`

## Performance Evidence

- `docs/perf/PERFORMANCE_BASELINE.md`
- Wave 1 post-change `pnpm test:perf` passed. Observed values: `predictGrid` 1299.7205 ms, `predictGridInto` 1277.3120 ms, `predictGridWithNeurons` 820.7586 ms, `predictGridWithNeuronsInto` 719.5801 ms, Adam/L2/Clip applyGradients 4.6340 ms, SGD applyGradients 1.7477 ms.
- Wave 1 build main bundle changed from 351.56 kB to 354.89 kB, below the 10% build-size warning threshold.
- Wave 4 post-change `pnpm test:perf` passed. Observed values: `predictGrid` 1083.7991 ms, `predictGridInto` 1060.3090 ms, `predictGridWithNeurons` 673.5523 ms, `predictGridWithNeuronsInto` 576.1459 ms, Adam/L2/Clip applyGradients 3.7961 ms, SGD applyGradients 1.4921 ms.
- Wave 4 build main bundle changed from 351.56 kB at Wave 0 to 360.52 kB, about 2.5%, and worker changed from 68.66 kB to 71.53 kB, about 4.2%.
- Wave 5 worker lifecycle tests changed test files only. `pnpm test:perf` first run was noisy (`predictGrid` 1715.3990 ms, `predictGridInto` 1407.6730 ms, `predictGridWithNeurons` 1168.0452 ms, `predictGridWithNeuronsInto` 802.7233 ms, Adam 6.8347 ms, SGD 1.9094 ms). Repeat run passed with `predictGrid` 1161.8212 ms, `predictGridInto` 1112.8496 ms, `predictGridWithNeurons` 707.6835 ms, `predictGridWithNeuronsInto` 601.9497 ms, Adam 4.9256 ms, SGD 1.4850 ms.
- Wave 5 fallback tests changed test files only. `pnpm test:perf` passed with `predictGrid` 1178.1976 ms, `predictGridInto` 1105.2287 ms, `predictGridWithNeurons` 699.9329 ms, `predictGridWithNeuronsInto` 610.8358 ms, Adam/L2/Clip applyGradients 5.1271 ms, SGD applyGradients 1.5424 ms.
- Wave 5 final verification changed tests/docs only. `pnpm test:perf` passed with `predictGrid` 1148.0024 ms, `predictGridInto` 1148.9127 ms, `predictGridWithNeurons` 719.4125 ms, `predictGridWithNeuronsInto` 604.6402 ms, Adam/L2/Clip applyGradients 4.8508 ms, SGD applyGradients 1.6078 ms.

## Design Decisions

| Date | Decision | Reason | Source / Design Note |
|---|---|---|---|
| 2026-05-11 | Keep Wave 0 docs-only and ignore `.claude/worktrees/` | `.claude/worktrees/` is local generated agent state and should not be committed | `/goal` implementation plan |
| 2026-05-11 | Treat `docs/agent-discovery-report.md` as retained Wave 0 evidence | It contains architecture, risk, and prior verification context useful to reviewers | `/goal` implementation plan |
| 2026-05-11 | Keep Wave 1 action cards web-local and navigation-only | Avoids protected worker, schema, persistence, URL/config, runtime, and training behavior contracts | `/goal` implementation plan |
| 2026-05-11 | Keep action cards as native buttons with `aria-describedby` reasons | Preserves keyboard semantics and keeps action labels concise while retaining educational context | Wave 1 implementation |
| 2026-05-11 | Use existing layout store tabs/phase plus DOM focus targets for action cards | Navigates existing UI only and avoids persistence/schema changes | Wave 1 implementation |
| 2026-05-11 | Add Wave 2 lesson depth as content-only registry entries first | Reuses existing presets and lesson engine without schema, persistence, or runtime changes | Wave 2 implementation |
| 2026-05-11 | Ship activation histograms as bounded layer-level bins in the frame buffer | Satisfies the approved high-risk slice without raw activation streaming, React large-array state, persistence, URL/config, or dependency changes | `docs/design-notes/activation-histogram-explorer.md`, `590114d` |
| 2026-05-11 | Gate histogram computation only on `needActivationHistograms` | Spec review found the layer-stat fallback too broad; explicit demand preserves the approved compute gate | Wave 4 review |
| 2026-05-11 | Keep Wave 5 performance diagnostics docs/test-based | Avoids hot-path runtime telemetry, protocol changes, user-visible diagnostic UI, and dependencies while preserving an evidence trail | `docs/design-notes/runtime-performance-diagnostics.md`, `d1b6349` |

## Known Issues

- `main` is ahead of `origin/main` by local roadmap/pre-roadmap commits.
- `.claude/worktrees/` existed before Wave 0 as untracked local agent state and is intentionally ignored.
- Vite build has an existing large chunk warning; this is not a Wave 0 regression unless the warning changes materially.
- Browser QA exercised the live `test-metrics-stale` action card. Hyperparams/loss action targeting is covered by component and app integration tests.
- Wave 4 Browser QA on the fresh `5176` URL reported development-only perf warnings for slow interactions, but no console errors.

## Blocked Items

- No current blocker for completed Waves 0-5.

## Deferred Items

- Additional Wave 2 lesson topics beyond the completed tuning/failure-mode lesson batch in `e822187`.
- Additional Wave 3 QA infrastructure beyond `d9cc432` and the recorded Wave 0/Wave 1/Wave 2/Wave 4 browser evidence.
- Wave 4 follow-up visualization inspection improvements, including any gradient-flow overlay design.
- Wave 4 existing-data text alternative/accessibility polish beyond the completed activation histogram explorer is explicitly deferred. No additional Wave 4 runtime, protocol, or frame-buffer data path is approved in this run.
- Wave 6A-6D experiment workflows.
- Wave 6E checkpoints and timeline scrubber, pending mandatory approval.
- Wave 7 large product bets, pending mandatory approval.

## Approval Gates Reached

- Wave 4 activation histogram explorer was approved by the user on 2026-05-11 and implemented in `590114d`.

## Next Recommended Slice

Stop at the Wave 5 boundary. Next recommended wave is Wave 6A improved experiment comparison, but do not continue into Wave 6 without explicit user approval.

## Handoff Notes

Required preflight was run on 2026-05-11:

- `pwd`: `/Users/kevincontreras/CascadeProjects/neural-network-playground`
- `git status --short`: `?? .claude/worktrees/`
- `git branch --show-current`: `main`
- `git remote -v`: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- `git log --oneline -5`: `da480c8`, `97d9e9f`, `56a4565`, `0fe2196`, `dc7160a`

The requested branch `codex/wave-0-review-packaging` was created after preflight. The first branch creation attempt hit a sandboxed Git ref write issue and succeeded when Git branch creation was approved.
