# Design Note: Loss Landscape Probe

## Problem

The playground now supports guided lessons, activation histograms, run comparison, checkpoints, a scalar live arena, and bounded slow-motion backprop summaries. Learners can see that training changes weights, but they still cannot inspect the local loss surface that the optimizer is moving across.

A full neural-network loss landscape is high-dimensional and easy to misrepresent. A safe first slice should teach the idea of a local valley or slope without implying that a tiny 2D view is the whole model. It must also avoid mutating live model state, blocking the worker, or adding new persisted/configured state.

## Proposed Change

Implement Loss Landscape Probe as staged Wave 7 work.

First approved implementation slice:

1. Add an engine-only deterministic dry-run helper that evaluates a tiny 2D loss grid around the current checkpoint.
2. Probe exactly two deterministic trainable scalar coordinates in the first slice.
3. Evaluate a bounded grid with a maximum size of `7x7` and a maximum of 64 samples.
4. Return bounded scalar metadata and a tiny flattened loss grid.
5. Prove the live network checkpoint, weights, biases, optimizer state, and step count are unchanged after probing.

Later slices, only after the engine-only foundation is proven:

1. Add a one-shot worker RPC for the same bounded response.
2. Add an Inspection panel UI with one explicit `Probe loss surface` action.
3. Render a compact heatmap plus text summary.

No slice should continuously recompute the probe, stream probe data, save probes, alter URL/config state, or add dependencies.

## User Value

- Shows how small parameter changes can raise or lower loss.
- Connects optimizer motion, learning rate, clipping, and local minima/valleys to an observable surface.
- Gives teachers a bounded demonstration of why optimization is local and approximate.
- Builds toward richer landscape and optimizer explanations without destabilizing runtime contracts.

## Affected Files

First engine-only slice:

- `packages/engine/src/types.ts`
- `packages/engine/src/network.ts`
- `packages/engine/src/__tests__/network.test.ts`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

Later worker/UI slices, if separately approved:

- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/training.worker.test.ts`
- `apps/web/src/components/controls/InspectionPanel.tsx`
- `apps/web/src/components/controls/InspectionPanel.test.tsx`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/`

Avoid touching `packages/shared/src/workerProtocol.ts`, frame-buffer files, URL/config serialization, persistence/run-history schema, public config shape, deployment, or dependencies for the first slice.

## Affected Runtime/Data Flow

First engine-only slice:

1. Caller provides a small input/target sample set and training loss configuration.
2. Engine creates a dry-run copy from the current checkpoint.
3. Engine selects two deterministic scalar coordinates, initially the first two trainable weight coordinates in layer order.
4. For each grid cell, the dry-run copy restores the original checkpoint, perturbs those two scalar weights by bounded offsets, evaluates loss on the bounded sample set, and records one scalar loss value.
5. Engine returns a compact result:
   - `gridSize`
   - `sampleCount`
   - `axisA` / `axisB` labels and offsets
   - flattened `Float32Array` or readonly number array of at most 49 losses
   - `centerLoss`
   - `minLoss`
   - `maxLoss`
   - best cell coordinates
   - short textual summary
6. The live network is never mutated.

Later worker/UI flow:

1. User activates `Probe loss surface`.
2. Main thread sends one explicit Comlink request.
3. Worker evaluates the bounded engine probe on a bounded sample set.
4. UI stores only the compact one-shot response in local component state.

The first UI response may keep the tiny 49-value grid in local component state because it is bounded and user-triggered. Raw weights, gradients, activations, checkpoints, and large grids must not enter React state.

## Compatibility Risks

High-risk areas:

- Live model mutation if checkpoint restore is wrong.
- Optimizer-state drift if probe state is not isolated.
- Worker stalls if grid/sample caps are too high.
- Misleading product claims if the UI implies a 2D slice is the full landscape.

Explicitly forbidden in the first implementation:

- URL/config serialization changes.
- Persistence/run-history schema changes.
- Public config shape changes.
- Worker streaming or MessageChannel protocol changes.
- Frame-buffer/version-counter changes.
- New dependencies.
- Continuous recomputation.
- Raw weight, gradient, activation, or checkpoint transport to React.

## Accessibility Impact

The first UI slice must be understandable without color:

- Native `button` for `Probe loss surface`.
- Keyboard activation by Enter and Space.
- Small `role="status"` region for loading/error/summary.
- Text summary including center loss, min loss, max loss, and best direction.
- Heatmap cells must not be the only source of information.
- Compact viewport check at about `390x844`.
- No animation is required; reduced-motion risk is low for the first slice.

## Performance Impact

First slice budget:

- Max grid size: `7x7`.
- Max evaluation samples: 64.
- Max loss evaluations per probe: 49 cells x 64 samples.
- One explicit user action only.
- No live training-loop work.
- No per-frame probe work.

If a local/browser probe visibly stalls the UI or exceeds a practical one-shot budget, reduce the grid or sample cap before expanding scope. Compare `pnpm test:perf` and build chunks against `docs/perf/PERFORMANCE_BASELINE.md`.

## Test Plan

Engine-only first slice:

- Red test: probe method does not exist.
- Deterministic grid test with fixed seed and fixed samples.
- Immutability test comparing checkpoint before and after probe.
- Test that finite center/min/max losses are returned.
- Test invalid grid size, sample count, or empty sample set throws bounded errors.
- Test result length is capped by `gridSize * gridSize`.

Later worker/UI slices:

- Worker test rejects before initialization.
- Worker test returns current `runId`, `step`, bounded grid, and summary.
- Worker test proves no training step advances and next real step matches control behavior.
- Component test renders button/loading/success/error states.
- Component test verifies keyboard activation.
- Accessibility assertions for status text and non-color summary.

Required verification for code slices:

- Targeted affected tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

## Browser QA Plan

Browser QA is required for any UI slice:

1. App loads with no console errors.
2. Start or step training a few times.
3. Open Inspection.
4. Trigger `Probe loss surface` by mouse.
5. Trigger the same action by keyboard.
6. Verify heatmap and text summary appear.
7. Verify training play/pause/step/reset still work after probing.
8. Verify compact viewport has no text overlap.
9. Record screenshots under `docs/qa/browser-qa/`.

## Rollback Plan

Keep slices isolated:

1. Engine probe types/method/tests.
2. Worker one-shot RPC/tests.
3. Inspection UI/tests/styles.
4. QA/performance/roadmap evidence.

Rollback should revert only the current slice files, then run targeted tests and the full verification commands. If a later slice touches worker or UI, verify existing training and the slow-motion backprop preview still work after rollback.

## Approval Required

Yes. This is a Wave 7 feature and it can touch engine internals, worker runtime, visualization UI, accessibility, and performance.

The first implementation may proceed only with the bounded engine-only scope above. Any worker RPC, UI, frame-buffer, shared protocol, persistence, URL/config, public config, dependency, or continuous-probe expansion requires a separate approval gate.

## Decision

Design note prepared after two approval-gate subagent reviews on 2026-05-13. Both reviewers recommended revising into a design-note-first, tiny one-shot slice before implementation.

## Exact Approval Question

Do you approve implementing only the first engine-only Loss Landscape Probe slice: a deterministic dry-run `7x7` maximum, 64-sample maximum, two-scalar-coordinate probe that returns bounded loss-grid metadata, preserves live network state, adds engine tests, and does not touch worker protocol, frame buffers, UI, URL/config serialization, persistence schema, public config shape, dependencies, or training behavior?
