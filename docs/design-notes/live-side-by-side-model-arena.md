# Design Note: Live Side-by-Side Model Arena

## Problem

Wave 7 Phase 1 compares two saved runs in the History panel using existing stored summaries, loss thumbnails, and report data. That gives learners a useful static comparison, but it does not yet let them watch two models train from different configurations under a shared scenario.

A live arena would help learners compare learning dynamics directly: one model might learn faster, overfit sooner, produce a smoother boundary, or respond differently to regularization. That value requires runtime design work because the current app intentionally owns one live training model, one worker stream, one frame buffer, and one set of visualization-demand counters.

## Proposed Change

Implement the live arena as a separate approved initiative after this design note. The safest implementation direction is a staged live arena, not an immediate two-worker or two-hot-model rewrite.

Recommended implementation phases:

1. Arena runtime design and test harness:
   - Add tests that describe two isolated model owners without changing production behavior.
   - Define per-side lifecycle states, bounded memory, demand ownership, and frame-buffer domains.
   - Keep the existing single-model worker path untouched until tests prove the isolated path.

2. Sequential, one-worker prototype:
   - Prefer one worker that owns two model slots and advances them sequentially per tick.
   - Keep one MessageChannel stream, but identify each arena-side payload explicitly if protocol changes are approved.
   - Keep visualization demand explicit per side and cadence-gated.
   - Preserve the existing single-model mode as the default code path.

3. Minimal live UI:
   - Let users choose two configurations from existing presets/current config/saved runs.
   - Provide shared controls: start, pause, step, reset.
   - Render two labelled model panes with scalar metrics first.
   - Add paired boundary/loss visuals only after frame-buffer ownership is proven.

4. Later optimization only if necessary:
   - Consider multiple workers only after sequential execution proves too slow and a performance note justifies the complexity.
   - Consider shareable arena configs only after public config and URL/schema migration are separately approved.

## User Value

- Compare how two neural-network configurations learn over time.
- Make tradeoffs visible: speed versus generalization, smooth boundaries versus overfitting, feature choice versus model capacity.
- Give teachers a live demonstration surface without replacing the existing single-model lab.
- Build naturally on saved-run arena, run history, loss chart, decision boundary, checkpoint timeline, and explanation cards.

## Affected Files

Likely implementation files after approval:

- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/worker/frameBufferLayout.ts`
- `apps/web/src/store/useTrainingStore.ts`
- `apps/web/src/components/controls/TrainingControls.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/layout/MainArea.tsx`
- `apps/web/src/components/layout/deriveVisualizationDemand.ts`
- `apps/web/src/components/visualization/*`
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/__tests__/workerProtocol.test.ts`
- `packages/engine/src/network.ts`
- `packages/engine/src/__tests__/network.test.ts`
- `docs/worker-protocol.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/qa/browser-qa/*`

Files that must not change without separate approval:

- URL/config serialization files.
- Run-history persistence schema files.
- Public config shape contracts.
- Deployment configuration.
- Dependencies and lockfiles.

## Affected Runtime/Data Flow

Current runtime model:

- One `Network` instance in the training worker.
- Comlink RPC for setup, reset, stepping, checkpoint restore, demand updates, and WebGPU toggles.
- One MessageChannel for streaming snapshots.
- One back-pressure gate with `frameAck`.
- One set of cadence counters for grid, metrics, layer stats, activation histograms, and checkpoint metadata.
- One frame buffer with domain-specific version counters.
- React state receives bounded scalar data and version counters; heavy arrays stay outside React state.

Live arena runtime would need one of these explicit designs:

1. **One worker, two model slots, sequential stepping**:
   - Preferred first live design.
   - Each model slot owns its network, optimizer state, dataset split, checkpoint ring, demand counters, and pause reason.
   - One worker tick advances model A and model B sequentially with bounded steps per side.
   - Snapshot messages include side-specific bounded payloads.
   - Lower concurrency risk, easier deterministic testing, no extra worker startup cost.

2. **Two workers, one model each**:
   - Later option only if performance data proves sequential stepping is too slow.
   - Requires lifecycle coordination, duplicated frame-buffer domains, two streams, more memory, and more browser QA.

3. **One current model plus one replayed saved run**:
   - Lower-risk hybrid, but less educational than true live comparison.
   - Could be a fallback if live dual runtime proves too risky.

Any true live arena almost certainly requires worker-protocol and frame-buffer changes, so implementation is gated.

## Compatibility Risks

High-risk areas:

- Worker protocol: side-specific snapshot messages or a new arena stream contract.
- Frame buffer: per-side domains and version counters for boundary, neurons, parameters, stats, histograms, confusion matrix, and checkpoints.
- Runtime snapshot format: every streamed payload would need side ownership.
- Worker lifecycle: start/pause/reset/step/restore semantics become paired.
- Memory: duplicated networks, optimizer buffers, grids, histograms, and checkpoints.
- Determinism: sequential stepping order must be fixed and tested.
- Existing single-model flow: must remain default and must not regress.

Out of scope unless separately approved:

- Public config shape changes.
- URL/config serialization changes.
- Run-history persistence schema changes.
- Saved arena sessions across reloads.
- New dependencies.
- Engine math changes.

## Accessibility Impact

Live arena UI must not rely on two colorful canvases alone.

Requirements:

- Two semantic regions: `Model A live arena` and `Model B live arena`.
- Shared status region that announces whether both models are idle, running, paused, diverged, or mismatched.
- Native controls for start, pause, step, reset, model source selection, and speed.
- Keyboard order: shared controls, Model A setup, Model B setup, comparison summaries, visual panes.
- Text summaries for relative loss, accuracy, generalization gap, pause reason, and boundary confidence.
- No color-only winner/loser indicators.
- Reduced-motion-safe updates. If animated training indicators are added, they must respect reduced motion.
- Compact layout must stack panes without text overlap or canvas-only controls.

## Performance Impact

Expected costs:

- Training work roughly doubles in the sequential one-worker design for equal steps per side.
- Boundary/grid work may double only when both panes demand boundary visualization.
- Parameter and neuron-grid frame-buffer domains may need duplicated memory.
- Checkpoint memory may double if both models keep bounded timelines.
- Main-thread render cost can increase if both panes paint heavy visuals every frame.

Mitigation:

- Demand-gate every heavy visualization per side.
- Start with scalar and loss summaries before paired boundaries.
- Use cadence intervals per side.
- Keep large arrays in worker/frame-buffer domains, not React state.
- Cap checkpoints per model.
- Compare fixed scenarios against `docs/perf/PERFORMANCE_BASELINE.md`.
- Keep the single-model mode on the existing path to avoid default-regression risk.

Warning thresholds:

- Production build size increase greater than 10%.
- Fixed training scenario runtime increase greater than 20%.
- Decision-boundary update time increase greater than 20%.
- Worker message cadence degradation greater than 20%.
- Memory increase greater than 25% for the same scenario.

## Test Plan

Before implementation:

- Add protocol tests for any new arena message shape.
- Add frame-buffer tests for per-side version counters if duplicated domains are introduced.
- Add worker lifecycle tests for start, pause, step, reset, stale snapshot rejection, and error propagation per side.
- Add demand/cadence tests proving one side can request a visualization without forcing the other side to compute it.
- Add determinism tests proving sequential A/B stepping order is stable.

During implementation:

- TDD each runtime surface with failing tests first.
- Preserve all existing single-model worker and hook tests.
- Add UI tests for labelled model panes, shared controls, keyboard access, and comparison summaries.
- Add accessibility tests with semantic assertions and `jest-axe` where the existing setup applies.

Required verification:

- Targeted worker/protocol/frame-buffer/hook/component tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- Browser QA Mode B for desktop and compact viewport.

## Browser QA Plan

Live arena Browser QA must record:

1. App loads without console errors.
2. Existing single-model training still starts, pauses, steps, resets, and renders visuals.
3. Arena can select two model sources.
4. Arena start, pause, step, and reset affect both labelled model panes predictably.
5. Model A and Model B expose readable scalar summaries.
6. Paired visualizations render only when their panes demand them.
7. Keyboard traversal reaches all controls and updates focus predictably.
8. Compact viewport stacks or scrolls without text overlap.
9. Console errors remain empty.
10. Screenshots are stored under `docs/qa/browser-qa/`.

## Rollback Plan

Keep live arena implementation isolated behind one feature boundary.

Rollback steps:

1. Revert live arena UI files.
2. Revert arena-specific hook/store state.
3. Revert arena-specific worker protocol changes and protocol docs.
4. Revert arena-specific frame-buffer domains and tests.
5. Preserve the single-model worker path and saved-run arena Phase 1.
6. Run `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf`.
7. Run browser regression QA for the original single-model workflow and saved-run arena.

If rollback cannot cleanly separate arena work from existing user work, stop and write a block report instead of reverting broadly.

## Approval Required

Implementation requires explicit approval because live arena work is a Wave 7 feature and likely touches mandatory-gated runtime surfaces:

- Worker protocol.
- Frame-buffer semantics.
- Runtime snapshot format.
- Worker lifecycle.
- Possibly engine checkpoint/model-state behavior.
- Performance-sensitive training and visualization paths.

This note does not approve implementation.

## Decision

Do not implement live arena runtime code yet.

Recommended first approved implementation slice:

- Add protocol and frame-buffer tests for a side-tagged arena snapshot shape, then implement the smallest shared-contract change needed for two independent scalar-only model summaries.
- Do not add paired boundaries, paired histograms, checkpoint sharing, URL/config serialization, persistence, or multiple workers in the first live slice.

## Exact Approval Question

Do you approve implementing the first live Side-by-Side Model Arena runtime slice as a scalar-only, one-worker, two-model-slot prototype with side-tagged bounded summaries, including the required worker-protocol/frame-buffer design updates and tests, while still forbidding URL/config serialization, persistence schema, public config shape, dependency, and engine-math changes?
