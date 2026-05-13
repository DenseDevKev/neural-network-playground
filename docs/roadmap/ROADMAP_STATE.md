# Roadmap State

## Last Updated

2026-05-13

## Repository

- Branch: `codex/wave-0-review-packaging`
- Last verified commit: `775d515`; `pnpm --filter @nn-playground/web test -- src/components/controls/InspectionPanel.test.tsx`, `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, Browser QA Mode B, and `git diff --check` passed before the Wave 7 slow-motion backprop preview UI and QA/state commits.
- Remote: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- PR: Not created yet
- Package manager: pnpm with `pnpm-lock.yaml` and `pnpm-workspace.yaml`
- Verification commands: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`
- CI workflows: `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`

## Current Position

- Wave: Wave 7
- Slice: Loss Landscape Probe design note
- Risk: High
- Status: Engine dry-run bounded backprop summaries, a worker-only one-shot Comlink RPC, and the Inspection-panel preview UI are implemented and verified. Loss Landscape Probe has reached a mandatory approval gate; design note is prepared and implementation must wait for explicit approval of the engine-only first slice.

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
| 2026-05-11 | Wave 5 | Runtime hardening completion evidence | `eacfa0a` | Clean-head `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed after commit | `docs/roadmap/ROADMAP_STATE.md`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-11 | Wave 6A | Saved-run comparison summaries | `aa5fff1` | Red targeted test failed before implementation; targeted web run passed with 51 files and 337 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and Browser QA Mode B passed before evidence packaging | `apps/web/src/components/controls/RunHistoryPanel.tsx`, `apps/web/src/components/controls/RunHistoryPanel.test.tsx`, `docs/qa/browser-qa/wave-6a-run-comparison.md` |
| 2026-05-11 | Wave 6A | Saved-run comparison QA/state | `9da19bf` | Docs-only evidence commit after Wave 6A full verification and Browser QA passed | `docs/qa/browser-qa/wave-6a-run-comparison.md`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |
| 2026-05-11 | Wave 6B | Saved-run thumbnail design note | `f8cf160` | `git diff --check` passed before commit | `docs/design-notes/saved-run-thumbnails.md` |
| 2026-05-11 | Wave 6B | Generated saved-run loss thumbnails | `96ee7a4` | Red targeted test failed before implementation; targeted web run passed with 51 files and 338 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and Browser QA Mode B passed before evidence packaging | `apps/web/src/components/controls/RunHistoryPanel.tsx`, `apps/web/src/components/controls/RunHistoryPanel.test.tsx`, `docs/qa/browser-qa/wave-6b-saved-run-thumbnails.md` |
| 2026-05-11 | Wave 6B | Saved-run thumbnail QA/state | `c0c4943` | Docs-only evidence commit after Wave 6B full verification and Browser QA passed | `docs/qa/browser-qa/wave-6b-saved-run-thumbnails.md`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |
| 2026-05-11 | Wave 6C | Richer saved-run markdown report export | `2d3ad9d` | Red targeted test failed before implementation; targeted web run passed with 51 files and 338 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before evidence packaging | `apps/web/src/components/controls/RunHistoryPanel.tsx`, `apps/web/src/components/controls/RunHistoryPanel.test.tsx` |
| 2026-05-11 | Wave 6C | Report export evidence | `89f84f4` | Docs-only evidence commit after Wave 6C full verification passed | `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |
| 2026-05-11 | Wave 6D | Bounded dataset sample presets | `d6eeffa` | Red targeted tests failed before implementation; targeted web run passed with 51 files and 340 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and Browser QA Mode B passed before evidence packaging | `apps/web/src/components/controls/DataPanel.tsx`, `apps/web/src/components/controls/DataPanel.test.tsx`, `docs/qa/browser-qa/wave-6d-dataset-lab.md` |
| 2026-05-11 | Wave 6D | Dataset lab QA/state | `2f5dffe` | Docs-only evidence commit after Wave 6D full verification and Browser QA passed; `git diff --check` passed before commit | `docs/qa/browser-qa/wave-6d-dataset-lab.md`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |
| 2026-05-11 | Wave 6E | Checkpoints/timeline design note | `c2896a2` | Docs-only mandatory-approval design note; `git diff --check` passed before commit | `docs/design-notes/training-checkpoints-timeline.md` |
| 2026-05-12 | Wave 6E | Engine checkpoint snapshot/restore semantics | `47a94d5` | Red targeted engine test failed before implementation; targeted engine run passed with 13 files and 276 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit | `packages/engine/src/network.ts`, `packages/engine/src/types.ts`, `packages/engine/src/__tests__/network.test.ts`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-12 | Wave 6E | Checkpoint timeline protocol metadata | `36b5e60` | Red targeted shared protocol test failed before implementation; targeted shared run passed with 5 files and 67 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit | `packages/shared/src/workerProtocol.ts`, `packages/shared/src/__tests__/workerProtocol.test.ts`, `docs/worker-protocol.md`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-12 | Wave 6E | Worker checkpoint ring buffer and restore RPC | `69d4ffe` | Red targeted web worker tests failed before implementation; targeted web run passed with 51 files and 342 tests; after fixing a shared type export caught by `pnpm build`, `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit | `apps/web/src/worker/training.worker.ts`, `apps/web/src/worker/training.worker.test.ts`, `packages/shared/src/index.ts`, `docs/worker-protocol.md`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-12 | Wave 6E | Protocol guard for omitted activation histograms | `972ec04` | Red targeted shared protocol test failed before implementation; targeted shared run passed with 5 files and 68 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit as part of the Wave 6E verification sweep | `packages/shared/src/workerProtocol.ts`, `packages/shared/src/__tests__/workerProtocol.test.ts` |
| 2026-05-12 | Wave 6E | Checkpoint timeline UI and QA evidence | `67b7c9f` | Red targeted web hook/control tests failed before implementation; targeted web run passed with 51 files and 344 tests; `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` passed before commit; Browser QA Mode B passed at desktop size | `apps/web/src/hooks/useTraining.ts`, `apps/web/src/components/controls/TrainingControls.tsx`, `apps/web/src/store/useTrainingStore.ts`, `docs/qa/browser-qa/wave-6e-checkpoint-timeline.md` |
| 2026-05-12 | Wave 7 | Side-by-Side Model Arena design note | `64ed867` | Design-note-only commit; `git diff --check` passed before commit | `docs/design-notes/side-by-side-model-arena.md` |
| 2026-05-12 | Wave 7 | Saved-run side-by-side arena comparison | `76c37ae` | Red targeted test failed before implementation; targeted web run passed with 51 files and 345 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and Browser QA Mode B passed before evidence packaging | `apps/web/src/components/controls/RunHistoryPanel.tsx`, `apps/web/src/components/controls/RunHistoryPanel.test.tsx`, `docs/qa/browser-qa/wave-7-side-by-side-arena.md` |
| 2026-05-12 | Wave 7 | Live Side-by-Side Model Arena runtime design note | `37c7d57` | Docs-only design note; `git diff --check` passed before commit | `docs/design-notes/live-side-by-side-model-arena.md` |
| 2026-05-12 | Wave 7 | Scalar live arena runtime prototype | `bb4aac1` | Red targeted protocol/frame-buffer/worker tests failed before implementation; targeted shared run passed with 5 files and 70 tests; targeted web run passed with 51 files and 348 tests; `pnpm test`, `pnpm lint`, `pnpm build`, repeated `pnpm test:perf`, and `git diff --check` passed before commit | `packages/shared/src/workerProtocol.ts`, `apps/web/src/worker/frameBuffer.ts`, `apps/web/src/worker/training.worker.ts`, `docs/worker-protocol.md` |
| 2026-05-12 | Wave 7 | Scalar live arena UI prototype | `0344bbe` | Red targeted web tests failed before implementation; targeted web run passed with 51 files and 350 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, Browser QA Mode B, and `git diff --check` passed before commit | `apps/web/src/components/controls/RunHistoryPanel.tsx`, `apps/web/src/hooks/useTraining.ts`, `apps/web/src/store/useTrainingStore.ts`, `docs/qa/browser-qa/wave-7-live-arena-ui.md` |
| 2026-05-12 | Wave 7 | Scalar live arena UI QA/state | `5dd79e7` | Docs-only evidence commit after Wave 7 live arena UI full verification and Browser QA passed; `git diff --check` passed before commit | `docs/qa/browser-qa/wave-7-live-arena-ui.md`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |
| 2026-05-12 | Wave 7 | Slow-motion backprop explanation mode design note | `4e08d29` | Docs-only design note; `git diff --check` passed before commit | `docs/design-notes/slow-motion-backprop-explanation-mode.md` |
| 2026-05-12 | Wave 7 | Dry-run backprop summary engine foundation | `afdc067` | Red targeted engine tests failed before implementation and after review-found gaps; targeted engine run passed with 13 files and 281 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, two subagent review passes, and `git diff --check` passed before commit | `packages/engine/src/network.ts`, `packages/engine/src/types.ts`, `packages/engine/src/__tests__/network.test.ts`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-12 | Wave 7 | Backprop explanation worker one-shot RPC | `10d59d1` | Red targeted worker tests failed before implementation; targeted web run passed with 51 files and 355 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, two approval-gate subagent passes, two implementation review passes, and `git diff --check` passed before commit | `apps/web/src/worker/training.worker.ts`, `apps/web/src/worker/training.worker.test.ts`, `docs/worker-protocol.md`, `docs/perf/PERFORMANCE_BASELINE.md` |
| 2026-05-13 | Wave 7 | Slow-motion backprop preview UI | `7dce9f9` | Targeted web run passed with 51 files and 359 tests; `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, Browser QA Mode B, two implementation review passes, and `git diff --check` passed before commit | `apps/web/src/components/controls/InspectionPanel.tsx`, `apps/web/src/components/controls/InspectionPanel.test.tsx`, `apps/web/src/styles/index.css`, `docs/qa/browser-qa/wave-7-backprop-preview.md` |
| 2026-05-13 | Wave 7 | Slow-motion backprop preview QA/state | `775d515` | Docs/evidence commit after the Wave 7 backprop preview UI full verification and Browser QA passed; `git diff --check` passed before commit | `docs/qa/browser-qa/wave-7-backprop-preview.md`, `docs/qa/browser-qa/wave-7-backprop-preview-desktop.png`, `docs/qa/browser-qa/wave-7-backprop-preview-compact.png`, `docs/perf/PERFORMANCE_BASELINE.md`, `docs/roadmap/ROADMAP_STATE.md` |

## Current Verification Status

- Tests: `pnpm test` passed on 2026-05-13 after the Wave 7 slow-motion backprop preview UI with engine 281 tests, shared 70 tests, and web 359 tests. Targeted `pnpm --filter @nn-playground/web test -- src/components/controls/InspectionPanel.test.tsx` passed with 51 files and 359 tests.
- Lint: `pnpm lint` passed on 2026-05-13 after the Wave 7 slow-motion backprop preview UI.
- Build: `pnpm build` passed on 2026-05-13 after the Wave 7 slow-motion backprop preview UI with the existing Vite chunk-size warning. Relevant chunks: `training.worker-S7ocudoV.js` 83.50 kB, `InspectionPanel-C0sjYXVR.js` 10.98 kB gzip 2.91 kB, `RunHistoryPanel-CTUmNQkd.js` 16.70 kB gzip 5.14 kB, `index-BtaMrfLu.js` 366.75 kB gzip 111.02 kB.
- Browser QA: Wave 0, Wave 1, Wave 2, Wave 4, Wave 6A, Wave 6B, Wave 6D, Wave 6E desktop, and Wave 7 Mode B checks passed. Slow-motion backprop preview Browser QA passed on 2026-05-13 after the dev server was restarted and the user manually restored the in-app browser tab from a generated connection-error page.
- Accessibility: Wave 1 component `jest-axe` coverage passed for the rendered action-card panel; Wave 4 histogram UI uses a native labelled select and `role="img"` text alternative covered by Testing Library assertions and Browser QA; Wave 6E checkpoint timeline uses a native labelled range and native restore button covered by component tests and Browser QA keyboard checks; Wave 7 saved-run/live arena uses native labelled selects, native buttons, labelled model regions, grouped scalar summaries, and keyboard activation covered by component tests and Browser QA. The slow-motion backprop preview uses a native button, a small `role="status"` live region, semantic layer-summary list, wrapping scalar metrics, and Testing Library plus Browser keyboard assertions.
- Performance: `pnpm test:perf` passed on 2026-05-13 after the Wave 7 slow-motion backprop preview UI with 2 benchmark files and 4 benchmark tests. Details are recorded in `docs/perf/PERFORMANCE_BASELINE.md`.

## Browser QA Evidence

- `docs/qa/browser-qa/wave-0-baseline.md`
- `docs/qa/browser-qa/wave-1-explanation-actions.md`
- `docs/qa/browser-qa/wave-2-lesson-depth.md`
- `docs/qa/browser-qa/wave-4-activation-histogram.md`
- `docs/qa/browser-qa/wave-6a-run-comparison.md`
- `docs/qa/browser-qa/wave-6b-saved-run-thumbnails.md`
- `docs/qa/browser-qa/wave-6d-dataset-lab.md`
- `docs/qa/browser-qa/wave-6e-checkpoint-timeline.md`
- `docs/qa/browser-qa/wave-7-side-by-side-arena.md`
- `docs/qa/browser-qa/wave-7-live-arena-ui.md`
- `docs/qa/browser-qa/wave-7-backprop-preview.md`
- Prior decision-boundary screenshot: `/private/tmp/nn-playground-decision-overlay-errors.png`
- Wave 0 compact screenshot: `/private/tmp/nn-playground-wave0-compact.png`
- Wave 0 final screenshot: `/private/tmp/nn-playground-wave0-final.png`
- Wave 1 action-card screenshot: `/private/tmp/nn-playground-wave1-action-card.png`
- Wave 1 compact action-card screenshot: `/private/tmp/nn-playground-wave1-action-card-compact.png`
- Wave 2 learning-rate lesson screenshot: `/private/tmp/nn-playground-wave2-learning-rate-lesson.png`
- Wave 4 activation histogram screenshot: `/private/tmp/nn-playground-wave4-activation-histogram.png`
- Wave 6A run comparison screenshot: `/private/tmp/nn-playground-wave6a-run-comparison.png`
- Wave 6B saved-run thumbnails screenshot: `/private/tmp/nn-playground-wave6b-run-thumbnails.png`
- Wave 6D dataset lab screenshot: `/private/tmp/nn-playground-wave6d-dataset-lab.png`
- Wave 6E checkpoint timeline desktop screenshot: `docs/qa/browser-qa/wave-6e-checkpoint-timeline-desktop.png`
- Wave 7 saved-run arena desktop screenshot: `docs/qa/browser-qa/wave-7-side-by-side-arena.png`
- Wave 7 saved-run arena desktop scrolled screenshot: `docs/qa/browser-qa/wave-7-side-by-side-arena-scrolled.png`
- Wave 7 saved-run arena compact screenshot: `docs/qa/browser-qa/wave-7-side-by-side-arena-compact.png`
- Wave 7 saved-run arena compact scrolled screenshot: `docs/qa/browser-qa/wave-7-side-by-side-arena-compact-scrolled.png`
- Wave 7 live arena UI screenshot: unavailable; Browser screenshot capture timed out. DOM snapshots and console checks are recorded in `docs/qa/browser-qa/wave-7-live-arena-ui.md`.
- Wave 7 slow-motion backprop preview desktop screenshot: `docs/qa/browser-qa/wave-7-backprop-preview-desktop.png`
- Wave 7 slow-motion backprop preview compact screenshot: `docs/qa/browser-qa/wave-7-backprop-preview-compact.png`

## Performance Evidence

- `docs/perf/PERFORMANCE_BASELINE.md`
- Wave 1 post-change `pnpm test:perf` passed. Observed values: `predictGrid` 1299.7205 ms, `predictGridInto` 1277.3120 ms, `predictGridWithNeurons` 820.7586 ms, `predictGridWithNeuronsInto` 719.5801 ms, Adam/L2/Clip applyGradients 4.6340 ms, SGD applyGradients 1.7477 ms.
- Wave 1 build main bundle changed from 351.56 kB to 354.89 kB, below the 10% build-size warning threshold.
- Wave 4 post-change `pnpm test:perf` passed. Observed values: `predictGrid` 1083.7991 ms, `predictGridInto` 1060.3090 ms, `predictGridWithNeurons` 673.5523 ms, `predictGridWithNeuronsInto` 576.1459 ms, Adam/L2/Clip applyGradients 3.7961 ms, SGD applyGradients 1.4921 ms.
- Wave 4 build main bundle changed from 351.56 kB at Wave 0 to 360.52 kB, about 2.5%, and worker changed from 68.66 kB to 71.53 kB, about 4.2%.
- Wave 5 worker lifecycle tests changed test files only. `pnpm test:perf` first run was noisy (`predictGrid` 1715.3990 ms, `predictGridInto` 1407.6730 ms, `predictGridWithNeurons` 1168.0452 ms, `predictGridWithNeuronsInto` 802.7233 ms, Adam 6.8347 ms, SGD 1.9094 ms). Repeat run passed with `predictGrid` 1161.8212 ms, `predictGridInto` 1112.8496 ms, `predictGridWithNeurons` 707.6835 ms, `predictGridWithNeuronsInto` 601.9497 ms, Adam 4.9256 ms, SGD 1.4850 ms.
- Wave 5 fallback tests changed test files only. `pnpm test:perf` passed with `predictGrid` 1178.1976 ms, `predictGridInto` 1105.2287 ms, `predictGridWithNeurons` 699.9329 ms, `predictGridWithNeuronsInto` 610.8358 ms, Adam/L2/Clip applyGradients 5.1271 ms, SGD applyGradients 1.5424 ms.
- Wave 5 final verification changed tests/docs only. `pnpm test:perf` passed with `predictGrid` 1148.0024 ms, `predictGridInto` 1148.9127 ms, `predictGridWithNeurons` 719.4125 ms, `predictGridWithNeuronsInto` 604.6402 ms, Adam/L2/Clip applyGradients 4.8508 ms, SGD applyGradients 1.6078 ms.
- Wave 6A comparison summaries changed the lazy run-history UI chunk only. `pnpm test:perf` passed with `predictGrid` 1137.5487 ms, `predictGridInto` 1108.1885 ms, `predictGridWithNeurons` 708.8439 ms, `predictGridWithNeuronsInto` 611.1152 ms, Adam/L2/Clip applyGradients 4.3936 ms, SGD applyGradients 1.4319 ms.
- Wave 6B saved-run thumbnails changed the lazy run-history UI chunk only. `pnpm test:perf` passed with `predictGrid` 1137.8122 ms, `predictGridInto` 1093.4358 ms, `predictGridWithNeurons` 697.4345 ms, `predictGridWithNeuronsInto` 592.1591 ms, Adam/L2/Clip applyGradients 4.7540 ms, SGD applyGradients 1.4204 ms.
- Wave 6C report export changed the lazy run-history UI chunk only. `pnpm test:perf` passed with `predictGrid` 1138.3189 ms, `predictGridInto` 1108.7949 ms, `predictGridWithNeurons` 690.4257 ms, `predictGridWithNeuronsInto` 589.8016 ms, Adam/L2/Clip applyGradients 4.6273 ms, SGD applyGradients 1.4548 ms.
- Wave 6D dataset lab changed the main web UI bundle only. `pnpm test:perf` passed with `predictGrid` 1160.4786 ms, `predictGridInto` 1112.4462 ms, `predictGridWithNeurons` 695.9391 ms, `predictGridWithNeuronsInto` 597.1600 ms, Adam/L2/Clip applyGradients 4.8577 ms, SGD applyGradients 1.5220 ms.
- Wave 6E engine checkpoint state changed the engine class imported by the worker. `pnpm test:perf` passed with `predictGrid` 1195.5230 ms, `predictGridInto` 1120.8068 ms, `predictGridWithNeurons` 712.0676 ms, `predictGridWithNeuronsInto` 588.5592 ms, Adam/L2/Clip applyGradients 4.8142 ms, SGD applyGradients 1.5587 ms.
- Wave 6E checkpoint protocol metadata changed shared runtime guards and protocol docs. `pnpm test:perf` passed with `predictGrid` 1185.8776 ms, `predictGridInto` 1139.6066 ms, `predictGridWithNeurons` 699.6503 ms, `predictGridWithNeuronsInto` 578.9710 ms, Adam/L2/Clip applyGradients 4.8995 ms, SGD applyGradients 1.4268 ms.
- Wave 6E worker checkpoint ring buffer changed worker runtime only. `pnpm test:perf` passed with `predictGrid` 1138.2182 ms, `predictGridInto` 1081.3733 ms, `predictGridWithNeurons` 693.0297 ms, `predictGridWithNeuronsInto` 589.9068 ms, Adam/L2/Clip applyGradients 4.5410 ms, SGD applyGradients 1.4516 ms.
- Wave 6E checkpoint timeline UI and protocol guard fix changed the main web UI bundle and shared runtime guard. `pnpm test:perf` passed with `predictGrid` 1141.9338 ms, `predictGridInto` 1122.8077 ms, `predictGridWithNeurons` 684.5251 ms, `predictGridWithNeuronsInto` 587.9544 ms, Adam/L2/Clip applyGradients 4.5803 ms, SGD applyGradients 1.4364 ms.
- Wave 7 saved-run arena changed the lazy run-history UI chunk only. `pnpm test:perf` passed with `predictGrid` 1286.9436 ms, `predictGridInto` 1278.7490 ms, `predictGridWithNeurons` 773.1364 ms, `predictGridWithNeuronsInto` 653.8458 ms, Adam/L2/Clip applyGradients 6.0251 ms, SGD applyGradients 1.5177 ms.
- Wave 7 scalar live arena runtime changed the worker bundle and scalar frame-buffer/protocol contracts. Initial `pnpm test:perf` was noisy; repeated `pnpm test:perf` passed with `predictGrid` 1138.0670 ms, `predictGridInto` 1093.3958 ms, `predictGridWithNeurons` 696.0030 ms, `predictGridWithNeuronsInto` 592.8361 ms, Adam/L2/Clip applyGradients 4.4625 ms, SGD applyGradients 1.4232 ms.
- Wave 7 live arena UI changed the lazy run-history UI chunk and App/MainArea callback wiring only. `pnpm test:perf` passed with `predictGrid` 1093.0898 ms, `predictGridInto` 1076.1186 ms, `predictGridWithNeurons` 681.7128 ms, `predictGridWithNeuronsInto` 579.4028 ms, Adam/L2/Clip applyGradients 3.9796 ms, SGD applyGradients 1.4334 ms.
- Wave 7 slow-motion backprop engine foundation changed the engine class imported by the worker. `pnpm test:perf` passed with `predictGrid` 1137.3643 ms, `predictGridInto` 1147.3317 ms, `predictGridWithNeurons` 725.1739 ms, `predictGridWithNeuronsInto` 606.6910 ms, Adam/L2/Clip applyGradients 4.1593 ms, SGD applyGradients 1.5099 ms. The worker bundle increased from 79.35 kB to 82.68 kB, about 4.2%, below the 10% threshold.
- Wave 7 slow-motion backprop worker RPC changed worker code and protocol docs only. `pnpm test:perf` passed with `predictGrid` 1080.3162 ms, `predictGridInto` 1072.9316 ms, `predictGridWithNeurons` 672.7654 ms, `predictGridWithNeuronsInto` 579.6610 ms, Adam/L2/Clip applyGradients 3.8635 ms, SGD applyGradients 1.5098 ms. The worker bundle increased from 82.68 kB to 83.50 kB, about 1.0%, below the 10% threshold.
- Wave 7 slow-motion backprop preview UI changed the lazy Inspection panel chunk and CSS only. `pnpm test:perf` passed on 2026-05-13 with `predictGrid` 1370.2497 ms, `predictGridInto` 1239.0260 ms, `predictGridWithNeurons` 750.3219 ms, `predictGridWithNeuronsInto` 646.0797 ms, Adam/L2/Clip applyGradients 6.1264 ms, SGD applyGradients 1.6279 ms. Earlier 2026-05-12 runs were noisy but passed; the best repeat was `predictGrid` 1237.1647 ms, `predictGridInto` 1223.3140 ms, `predictGridWithNeurons` 771.2188 ms, `predictGridWithNeuronsInto` 659.5370 ms, Adam/L2/Clip applyGradients 4.5729 ms, SGD applyGradients 1.5759 ms. The lazy Inspection chunk increased from 8.31 kB to 10.98 kB and CSS from 66.18 kB to 66.62 kB, below the 10% threshold.

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
| 2026-05-11 | Compare saved runs against the next older saved run | Reuses the existing newest-first run-history order and avoids persistence/schema changes or selectable-baseline state | Wave 6A implementation, `aa5fff1` |
| 2026-05-11 | Generate run thumbnails at render time from saved history | Avoids persistence schema changes, stored image data, and new runtime data collection while improving scanability | `docs/design-notes/saved-run-thumbnails.md`, `96ee7a4` |
| 2026-05-11 | Enrich markdown reports from saved record data only | Improves experiment handoff without adding export dependencies, stored data, or schema changes | Wave 6C implementation, `2d3ad9d` |
| 2026-05-11 | Expose sample count through bounded presets | Improves dataset experimentation while reusing existing `DataConfig.numSamples` and avoiding schema changes | Wave 6D implementation, `d6eeffa` |
| 2026-05-11 | Gate checkpoint/timeline implementation behind explicit approval | Wave 6E likely touches worker protocol, runtime restore behavior, checkpoint memory, and model-state determinism | `docs/design-notes/training-checkpoints-timeline.md`, `c2896a2` |
| 2026-05-12 | Store checkpoint payloads as engine runtime state, not persistence or URL state | Preserves public config, URL/config serialization, run-history schema, and large-array React-state boundaries | Wave 6E engine checkpoint implementation |
| 2026-05-12 | Send only checkpoint summaries through streamed snapshots | Keeps heavy checkpoint arrays in the worker and gives React bounded scalar timeline metadata | Wave 6E protocol metadata implementation |
| 2026-05-12 | Keep checkpoint payloads in a worker-local ring buffer | Avoids React large-array state, persistence, URL/config serialization, run-history schema changes, and new dependencies | Wave 6E worker checkpoint implementation |
| 2026-05-12 | Render the timeline from bounded metadata only | Keeps React state limited to checkpoint summaries while restore reads heavy payloads by worker-local checkpoint id | Wave 6E checkpoint timeline UI implementation |
| 2026-05-12 | Treat undefined optional histogram fields as omitted in the runtime guard | Worker snapshot assembly can include optional keys with `undefined` values when histogram demand is off; Browser QA proved the guard otherwise rejected valid snapshots | Wave 6E Browser QA and shared protocol regression test |
| 2026-05-12 | Implement Wave 7 Phase 1 from saved run history only | Delivers a usable side-by-side arena without live dual-model runtime, worker, protocol, persistence, URL/config, public config, or dependency changes | `docs/design-notes/side-by-side-model-arena.md`, `76c37ae` |
| 2026-05-12 | Prefer one-worker sequential live arena before multi-worker designs | Minimizes concurrency, lifecycle, and memory risk while preserving deterministic stepping and the existing single-model path | `docs/design-notes/live-side-by-side-model-arena.md`, `37c7d57` |
| 2026-05-12 | Ship the first live arena runtime slice as scalar-only Comlink APIs | Lets tests prove two isolated model slots and side-tagged bounded summaries before adding UI streaming, paired boundaries, persistence, URL/config state, or multiple workers | `docs/design-notes/live-side-by-side-model-arena.md`, `bb4aac1` |
| 2026-05-12 | Expose live arena UI through saved-run records and scalar summaries only | Gives learners a usable live comparison prototype while avoiding persistence, URL/config state, paired heavy arrays, public config, dependencies, and multiple-worker execution | `docs/design-notes/live-side-by-side-model-arena.md`, `0344bbe` |
| 2026-05-12 | Implement slow-motion backprop as an engine-only dry-run foundation first | Provides deterministic, bounded layer-level summaries while avoiding worker/protocol/frame-buffer/UI/URL/config/persistence/dependency changes in the first slice | `docs/design-notes/slow-motion-backprop-explanation-mode.md`, `afdc067` |
| 2026-05-12 | Expose backprop previews through a worker one-shot RPC only | Reuses the bounded engine summary without streamed messages, frame-buffer fields, React state, persistence, URL/config, or public config changes | `docs/design-notes/slow-motion-backprop-explanation-mode.md`, `10d59d1` |
| 2026-05-13 | Require a Loss Landscape Probe design note before implementation | Two approval-gate reviewers recommended revising before code because the feature is high-risk and lacked exact probe bounds, transport choices, and rollback constraints | `docs/design-notes/loss-landscape-probe.md` |

## Known Issues

- `main` is ahead of `origin/main` by local roadmap/pre-roadmap commits.
- `.claude/worktrees/` existed before Wave 0 as untracked local agent state and is intentionally ignored.
- Vite build has an existing large chunk warning; this is not a Wave 0 regression unless the warning changes materially.
- Browser QA exercised the live `test-metrics-stale` action card. Hyperparams/loss action targeting is covered by component and app integration tests.
- Wave 4 Browser QA on the fresh `5176` URL reported development-only perf warnings for slow interactions, but no console errors.
- Wave 6A compares each saved run with the next older saved run. Selectable comparison baselines are deferred.
- Wave 6B thumbnails are generated in render from existing bounded history; stored thumbnails and selectable thumbnail styles are deferred.
- Wave 6C report export does not include images, weights, or raw arrays; richer report media/export formats are deferred.
- Wave 6D sample controls are bounded presets only. Custom sample count editing and new dataset parameters are deferred.
- Wave 6E compact viewport visual Browser QA remains pending from that earlier slice; Wave 7 compact viewport Browser QA was executed with the Browser viewport capability.
- Wave 7 live arena UI Browser QA screenshot capture timed out in the in-app Browser backend; DOM snapshots and console checks passed and are recorded.
- Wave 7 slow-motion backprop preview Browser QA required the user to manually restore the local URL after the Browser tool could not navigate away from Chrome's generated connection-error `data:` page. After that manual recovery, Browser QA passed with no console errors and two development-mode perf warnings.

## Blocked Items

- Wave 7 paired heavy visualization, continuous arena streaming, URL/config serialization, persistence, public config shape changes, dependencies, engine math changes, and multiple-worker live arena designs remain blocked/deferred. Scalar-only runtime and UI prototypes are complete.

## Deferred Items

- Additional Wave 2 lesson topics beyond the completed tuning/failure-mode lesson batch in `e822187`.
- Additional Wave 3 QA infrastructure beyond `d9cc432` and the recorded Wave 0/Wave 1/Wave 2/Wave 4 browser evidence.
- Wave 4 follow-up visualization inspection improvements, including any gradient-flow overlay design.
- Wave 4 existing-data text alternative/accessibility polish beyond the completed activation histogram explorer is explicitly deferred. No additional Wave 4 runtime, protocol, or frame-buffer data path is approved in this run.
- Wave 6E persistence of checkpoints across reloads remains deferred and requires separate approval.
- Wave 7 paired live model arena visualizations, loss landscape probe, multiclass mode, advanced architecture comparison, and interactive gradient explanation mode remain deferred pending separate mandatory approvals.
- Live arena URL/config serialization, persistence, paired boundary rendering, paired histograms, checkpoint sharing, continuous streaming, and multiple-worker execution are explicitly deferred from the scalar runtime/UI prototype.

## Approval Gates Reached

- Wave 4 activation histogram explorer was approved by the user on 2026-05-11 and implemented in `590114d`.
- Wave 6E checkpoints and timeline scrubber design note was prepared in `c2896a2`; implementation was explicitly approved by the user on 2026-05-12 with "YES".
- Wave 7 larger product bets require separate proposal and explicit approval before each feature. Proposal prepared in `docs/roadmap/WAVE_7_PROPOSAL.md`.
- Wave 7 Side-by-Side Model Arena design note was approved to create on 2026-05-12 and committed as `64ed867`.
- Wave 7 Side-by-Side Model Arena Phase 1 saved-run implementation was approved by the user's `/goal complete everything` continuation and committed as `76c37ae`.
- Wave 7 live Side-by-Side Model Arena runtime design note was committed as `37c7d57`.
- Wave 7 scalar live arena runtime slice was approved by the user's continuation and committed as `bb4aac1`.
- Wave 7 scalar live arena UI slice was approved by the user's continuation and committed as `0344bbe`.
- Further Wave 7 work that adds paired heavy visualizations, URL/config state, persistence, public config changes, dependencies, engine math changes, or multiple-worker execution requires a separate approval gate.
- Wave 7 Slow-Motion Backprop Explanation Mode design note was committed as `4e08d29`. The user asked to continue and required two subagent review passes at approval gates; the first engine-only bounded dry-run foundation was implemented in `afdc067` after two subagent reviews rejected the initial draft, fixes were applied, and two re-review passes approved the corrected slice.
- Wave 7 Slow-Motion Backprop worker one-shot RPC was implemented in `10d59d1` after two approval-gate subagent passes and two implementation review passes approved the bounded worker-only scope.
- Wave 7 Slow-Motion Backprop preview UI was implemented in `7dce9f9` and QA/state evidence was recorded in `775d515`.
- Wave 7 Loss Landscape Probe reached an approval gate on 2026-05-13. Two subagents recommended design-note-first scope discipline. `docs/design-notes/loss-landscape-probe.md` asks for approval of an engine-only, deterministic, `7x7` maximum, 64-sample maximum, two-scalar-coordinate dry-run probe.

## Next Recommended Slice

Next recommended step: wait for explicit approval of the exact question in `docs/design-notes/loss-landscape-probe.md`. If approved, implement only the first engine-only Loss Landscape Probe slice with TDD. Do not touch worker RPCs, frame buffers, UI, URL/config serialization, persistence schema, public config shape, dependencies, or training behavior in that first slice.

## Handoff Notes

Required preflight was run on 2026-05-11:

- `pwd`: `/Users/kevincontreras/CascadeProjects/neural-network-playground`
- `git status --short`: `?? .claude/worktrees/`
- `git branch --show-current`: `main`
- `git remote -v`: `origin https://github.com/DenseDevKev/neural-network-playground.git`
- `git log --oneline -5`: `da480c8`, `97d9e9f`, `56a4565`, `0fe2196`, `dc7160a`

The requested branch `codex/wave-0-review-packaging` was created after preflight. The first branch creation attempt hit a sandboxed Git ref write issue and succeeded when Git branch creation was approved.
