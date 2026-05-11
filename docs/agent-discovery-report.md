# Temporary Agent Discovery Report

Date: 2026-05-11

This is a temporary handoff note for the autonomous Neural Network Playground
2.0 run. It is a planning aid, not a substitute for tests.

## Verification Commands and Current Status

Package scripts:

- Root: `pnpm dev`, `pnpm build`, `pnpm test`, `pnpm test:engine`, `pnpm test:perf`, `pnpm bench`, `pnpm lint`, `pnpm clean`.
- Web: `pnpm --filter @nn-playground/web dev`, `build`, `preview`, `test`.
- Engine: `pnpm --filter @nn-playground/engine test`, `test:watch`, `test:perf`, `bench`.
- Shared: `pnpm --filter @nn-playground/shared test`.

Baseline evidence gathered so far:

- `pnpm --filter @nn-playground/web test -- NetworkGraphCanvas.test.tsx`
  initially failed in `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
  because the stale architecture assertion expected `x, y` while the current
  engine feature labels render `X₁, X₂`; the queried summary container also
  includes the capacity badge and Gaussian topology hint.
- After updating the stale assertion and wrapping the store update that caused
  a React `act(...)` warning, the same command passed with 50 test files and
  304 tests in the web package run.
- Lesson-slice TDD: `pnpm --filter @nn-playground/web test -- src/lessons/lessonRegistry.test.ts`
  failed after adding the expanded lesson-id expectation, then passed with
  50 test files and 304 tests after adding the lesson definitions.
- Final verification run status is tracked in the final handoff. At the time
  this report was updated, `pnpm test`, `pnpm lint`, and `pnpm build` had all
  exited 0 in the coordinator session.
- Decision-boundary explainer TDD: `pnpm --filter @nn-playground/web test -- DecisionBoundary.test.tsx MainArea.test.tsx`
  first failed on the missing canvas `role="img"`/`aria-describedby` and
  missing visible overlay helper text. After implementation, the expanded
  targeted web run with `DecisionBoundary.test.tsx`, `MainArea.test.tsx`, and
  `appShell.integration.test.tsx` passed with 50 test files and 308 tests.

## Architecture Map

Monorepo boundaries:

- `packages/engine` is the DOM-free neural-network core. It owns typed domain
  types, deterministic dataset generation, feature transforms, losses,
  activations, schedules, optimizers, WebGPU grid parity helpers, and the
  packed-buffer `Network` implementation.
- `packages/shared` owns cross-package contracts: defaults, presets,
  serialization/import validation, worker protocol and guards, code export,
  experiment-memory validation, color scales, and structural equality.
- `apps/web` owns React UI, Zustand stores, worker bridge/runtime, frame buffer,
  guided lessons, explanations, layout, controls, and visualizations.

Runtime flow:

- `usePlaygroundStore` holds durable playground config, transient generated
  dataset state, visualization demand, and UI-only renderer toggles.
- `useTrainingStore` holds volatile runtime state: status, snapshot scalars,
  history version, frame-buffer version counters, train/test points, config
  sync state, worker errors, pause reason, and stale test-metric state.
- `useTraining` initializes the worker, syncs config/demand/WebGPU toggles,
  owns play/pause/step/reset, and applies streamed snapshots.
- `workerBridge.ts` owns the Comlink proxy plus MessageChannel stream, validates
  messages, installs SharedArrayBuffer views, rAF-gates snapshot application,
  writes heavy arrays into `frameBuffer.ts`, and sends `frameAck`.
- `frameBuffer.ts` keeps large typed arrays out of React/Zustand and exposes
  narrow version counters for output grid, neuron grids, params, layer stats,
  and confusion matrix.
- `training.worker.ts` owns data/network rebuilds, training loop, demand-gated
  expensive computations, optional WebGPU grid prediction, stop conditions, and
  snapshot publishing.

Visualization and education surfaces:

- `DecisionBoundary`, `NetworkGraphCanvas`, `NetworkGraphSVG`, `LossChart`,
  `ConfusionMatrix`, and `InspectionPanel` read snapshots/frame-buffer domains
  according to demand and version counters.
- `NetworkGraphCanvas` is the default graph renderer, with canvas painting for
  graph primitives and DOM overlays for controls, heatmaps, tooltips, lesson
  callouts, and accessible summaries.
- `lessonRegistry.ts` now defines six web-local lessons and the
  `GuidedLessonPanel` renders the registry automatically.
- `trainingExplanations.ts` and `TrainingExplanationPanel.tsx` provide
  deterministic "Why did this happen?" explanations from existing metrics.

## Protected Contracts and Risk Areas

Treat these as approval-gated unless a design note is written and accepted:

- Worker protocol and runtime guards in `packages/shared/src/workerProtocol.ts`.
- URL/config serialization and import validation in
  `packages/shared/src/serialization.ts`.
- Public config types/defaults/presets shared between engine, shared, and web.
- Run-history persistence shape in `packages/shared/src/experimentMemory.ts`
  and the web experiment-memory store.
- Worker stream semantics, `frameAck` back-pressure, SharedArrayBuffer seqlock
  reads, and frame-buffer/version-counter behavior.
- Engine math, optimizer/regularization semantics, deterministic datasets,
  typed-array layout, and WebGPU parity/fallback behavior.
- Deployment/base-path behavior in Vite and GitHub Pages workflows.

Lower-risk areas for this run:

- Web-local lesson content in `apps/web/src/lessons/lessonRegistry.ts`.
- Web-local tests around lessons and graph summaries.
- Small explanatory copy that reuses existing metrics and stores.

Additional read-only scout notes:

- UX/accessibility scout recommended lesson-outcome copy, explanation action
  chips, and decision-boundary overlay explainer as low-risk UI slices. It
  flagged canvas topology keyboard inspection parity and richer decision
  boundary text alternatives as future accessibility work.
- Engine/worker scout confirmed that protocol, URL/import shape,
  experiment-memory schema, and frame-buffer domains are the highest-risk
  contracts. It also flagged initial grid cadence, WebGPU adapter-limit
  fallback, and direct worker demand/cadence tests as worthwhile future
  hardening work, but not part of this low-risk lesson slice.

## Existing Uncommitted Work

`git status --short` showed:

- `?? .claude/worktrees/`

This appears unrelated to the current task and was left untouched.

Current changes from this run:

- `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
  updated for the current architecture summary labels and React `act(...)`
  hygiene.
- `apps/web/src/lessons/lessonRegistry.test.ts` updated to pin the expanded
  lesson library and target/tab alignment.
- `apps/web/src/lessons/lessonRegistry.ts` expanded with circle, feature
  engineering, and spiral lessons that reuse existing presets.
- `apps/web/src/lessons/types.ts` and `apps/web/src/App.tsx` updated so
  feature-focused lesson steps can highlight the existing Features panel.
- `apps/web/src/__tests__/appShell.integration.test.tsx` updated to cover the
  Features-panel lesson highlight path and keep the decision-boundary partial
  mock aligned with the new overlay-copy helper.
- `apps/web/src/components/visualization/DecisionBoundary.tsx` now exposes
  overlay copy and describes the selected overlay mode through the canvas'
  accessible description.
- `apps/web/src/components/layout/MainArea.tsx` now renders live helper text
  for the existing decision-boundary overlay controls.
- `docs/agent-discovery-report.md` added/updated as this temporary handoff
  artifact.

## Recommended First Feature Slice

### Slice A: Expand the Guided Lesson Library

Learning goal:

- Give learners more structured paths through existing presets without adding
  new runtime data, schema, protocol, or engine behavior.

User-facing behavior:

- The guided lesson selector offers additional lessons for feature engineering,
  circular boundaries, and complex spiral depth.
- Starting a new lesson applies the existing matching preset and steps through
  existing panels with concise educational copy.

Technical approach:

- Add lesson definitions only in `apps/web/src/lessons/lessonRegistry.ts`.
- Update lesson registry tests to pin the expanded explicit lesson IDs and
  preserve invariants: valid targets, non-empty text, existing presets, no
  embedded config snapshots/functions.
- Rely on the existing `GuidedLessonPanel` selector rendering and preset start
  flow tests.

Likely touched files:

- `apps/web/src/lessons/lessonRegistry.ts`
- `apps/web/src/lessons/lessonRegistry.test.ts`

Risk level:

- Low. Web-local lesson content only; no protected contract changes.

Required tests:

- Red/green targeted: `pnpm --filter @nn-playground/web test -- src/lessons/lessonRegistry.test.ts`
- Web package or full test after implementation.

Rollback plan:

- Revert the added lesson objects and the expanded expected lesson-id list.

### Slice B: Contextual Explanation Action Cards

Learning goal:

- Turn existing deterministic explanations into next-action prompts that help
  learners focus the relevant panel or control.

User-facing behavior:

- Explanation panel shows one or more concise action chips/cards, such as
  "Open Hyperparams" or "Inspect Network", using existing layout actions.

Technical approach:

- Extend web-local explanation UI mapping without changing metrics, worker
  payloads, stores, serialization, or shared contracts.
- Use existing `useLayoutStore` tab/phase actions for focus.

Likely touched files:

- `apps/web/src/explanations/trainingExplanations.ts`
- `apps/web/src/components/visualization/TrainingExplanationPanel.tsx`
- Related tests in the same directories.

Risk level:

- Low to medium. UI-only, but it introduces interactive focus behavior.

Required tests:

- Explanation ordering/action mapping tests.
- Component test proving action control moves the expected existing tab.
- Accessibility check for semantic buttons.

Rollback plan:

- Remove action metadata and card UI; keep existing explanation text behavior.

### Slice C: Decision Boundary Overlay Explainer

Learning goal:

- Make existing boundary overlay modes easier to understand without changing
  visualization data or worker demand.

User-facing behavior:

- Changing Output, Uncertain, Errors, or Split modes updates concise helper text
  and a richer accessible description for the canvas.

Technical approach:

- Reuse existing overlay state/props and add web-local copy only.
- Avoid new frame-buffer domains, worker messages, or dataset schema.

Likely touched files:

- `apps/web/src/components/layout/MainArea.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.test.tsx`

Risk level:

- Low. Existing UI state only, with accessibility upside.

Required tests:

- Component tests for overlay copy and `aria-describedby`/label behavior.
- Browser QA for visual fit if helper copy is rendered visibly.

Rollback plan:

- Remove helper text and description wiring; keep existing canvas behavior.

## Recommendation

Slice A was implemented in the initial run, and Slice C was implemented as the
follow-up low-risk continuation. The next feature slice should wait for
explicit approval and should prefer Slice B, unless the team chooses to harden
worker-demand/WebGPU test coverage first.
