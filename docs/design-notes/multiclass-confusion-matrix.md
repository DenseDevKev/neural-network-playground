# Design Note: Multiclass Confusion Matrix Readout

## Problem

Multiclass Classification Mode now has hidden 3-class engine, worker, URL/config opt-in, persistence eligibility, dataset, code-export, and decision-boundary foundations. The Confusion Matrix panel still intentionally renders only the binary `{ tn, fp, fn, tp }` shape and suppresses hidden multiclass state.

A real worker-authored 3x3 matrix would require a new metric/data contract. The current engine `Metrics.confusionMatrix`, shared worker snapshot field, frame buffer slot, and UI all describe binary matrices only.

## Proposed Change

Use a staged approach:

1. First implementation slice: add a UI-only, derived 3-class test-set readout for the currently loaded network using existing frame-buffer weights/biases, existing feature flags, and existing `testPoints`.
2. Later implementation slice: if the matrix must become an official worker metric, add a separate design note for protocol/frame-buffer/runtime guard changes before touching those protected surfaces.

The first slice must not add worker snapshot fields, frame-buffer fields, persistence fields, URL/config fields, public config fields, dependencies, public controls, or training behavior changes.

## User Value

Learners can see which classes are confused with each other instead of only seeing aggregate multiclass accuracy. This pairs naturally with the multiclass decision-boundary renderer: the boundary shows class regions, and the readout shows how test samples land across predicted/actual classes.

## Affected Files

First UI-only slice:

- `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.test.tsx`
- `apps/web/src/styles/index.css` if compact 3x3 layout needs minor styling.
- `docs/roadmap/ROADMAP_STATE.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/qa/browser-qa/*` for QA evidence if a visible path can be exercised.

Explicitly not affected in the first slice:

- `packages/engine/src/types.ts`
- `packages/engine/src/network.ts`
- `packages/shared/src/workerProtocol.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/worker/training.worker.ts`
- URL/config serialization
- run-history persistence schema
- public config shape

## Affected Runtime/Data Flow

The first slice derives a small readout locally:

1. Subscribe to existing `paramsVersion`, `testPoints`, current `network`, and current `features`.
2. Read existing frame-buffer `weights`, `biases`, and `weightLayout`.
3. Reconstruct a temporary `Network` from the bounded flattened params.
4. Transform each test point with the existing feature transforms.
5. Run `forward` and compute `actual x predicted` counts for exactly classes `0`, `1`, and `2`.
6. Render a 3x3 table plus totals and text summary.

The derived counts are small UI data. Heavy parameter arrays remain in the frame buffer. Raw prediction arrays are not stored in React state.

## Compatibility Risks

- Stale parameter cache: the UI must subscribe to `paramsVersion` and return an unavailable state when params are missing or shape-mismatched.
- Main-thread compute: the first slice should only derive from bounded test points and current frame params; it should not run during normal scalar binary workflows.
- Educational wording: label the first slice as a test-set readout derived from current network params, not as a worker-persisted metric.
- Bundle size: importing runtime engine helpers into the visible panel may affect the main app/engine chunks and must be measured.

## Accessibility Impact

- Matrix cells need row and column labels in accessible names.
- The summary must not rely on color alone.
- Compact viewport layout must preserve readable headers and totals.
- Missing-data state must explain what is needed without implying a worker failure.

## Performance Impact

Expected impact is small and bounded:

- Reconstruction uses existing frame-buffer params only when the panel renders under a hidden 3-class softmax config.
- The readout iterates current test points, usually bounded by existing dataset sample presets.
- No worker hot-path or training-loop work changes.

Record `pnpm test:perf` and build chunk sizes. If the main bundle or engine chunk grows above the roadmap threshold, split or reconsider.

## Test Plan

- Add red pure/helper coverage for deriving a 3x3 readout from flattened frame params.
- Add component tests for:
  - 3x3 row/column labels and totals.
  - accessible cell names.
  - missing params unavailable state.
  - preservation of existing binary matrix behavior.
  - stale binary matrix still hidden for multiclass state.
- Run targeted web tests.
- Run `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf`.

## Browser QA Plan

Use Browser Mode B if available. If Browser plugin tooling is unavailable, use the same Dia/Computer Use fallback and clearly mark any hidden multiclass-specific browser check as pending.

Minimum visible checks before marking public multiclass complete:

1. App loads.
2. Confusion panel still works for public scalar binary flow.
3. Hidden 3-class state can render the 3x3 readout through a safe test-only/manual path.
4. Keyboard and compact viewport behavior are checked.

## Rollback Plan

Revert the UI-only commit and its evidence docs. Because the first slice does not touch worker protocol, frame-buffer shape, URL/config, persistence, public config, dependencies, or engine math, rollback is limited to the visualization component/tests/styles and docs.

## Approval Required

The UI-only first slice is approved by the user's broad continuation as long as it stays local to the component and existing frame-buffer data.

Any worker-authored multiclass confusion metric, new snapshot field, frame-buffer field, persistence field, URL/config change, public config change, dependency, public control, or training behavior change requires a separate approval gate.

## Decision

Proceed with the UI-only derived readout slice. Do not implement a worker/protocol/frame-buffer multiclass matrix in this slice.
