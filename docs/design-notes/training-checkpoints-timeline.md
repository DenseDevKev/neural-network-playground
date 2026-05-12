# Design Note: Training Checkpoints and Timeline Scrubber

## Problem

Learners can currently train, pause, reset, inspect live snapshots, and compare saved runs, but they cannot move backward through a single training run to inspect how a model reached its current state. That limits explanations of overshooting, recovery, saturation, plateaus, and the relationship between earlier decisions and later behavior.

Checkpointing and a timeline scrubber could make training feel replayable, but this is a high-risk runtime feature because it may require model-state snapshots, bounded memory policy, worker coordination, and clear restore semantics.

## Proposed Change

Add an opt-in training timeline that captures bounded checkpoints during an active run and lets users scrub between them while training is paused.

The safest implementation should be split into separate approved slices:

1. Runtime checkpoint model: define a compact, bounded checkpoint payload that can restore network weights, biases, optimizer state if needed, training counters, and enough scalar metadata to explain the point in time.
2. Worker coordination: capture checkpoints inside the worker at a fixed cadence and expose restore behavior through an explicit command.
3. UI timeline: add an accessible scrubber in the Run or Inspection area that pauses training before restore, previews checkpoint metadata, and makes restore state obvious.
4. QA and performance: verify memory bounds, determinism, restore behavior, and browser interactions against the performance baseline.

The initial implementation should not persist checkpoints across reloads. Persistent timeline history should be treated as a separate approval-gated feature.

## User Value

- Learners can rewind to the moment a run began overfitting, diverging, recovering, or plateauing.
- Teachers can demonstrate cause and effect without asking students to reproduce fragile timing manually.
- Users can inspect how decision boundaries, loss, and activations changed across training milestones.

## Affected Files

Likely files if approved:

- `packages/engine/src/network.ts`
- `packages/engine/src/types.ts`
- `packages/shared/src/workerProtocol.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/store/*`
- `apps/web/src/components/layout/*`
- `apps/web/src/components/controls/*`
- `apps/web/src/components/visualization/*`
- `docs/worker-protocol.md`
- Tests in engine, shared protocol, worker bridge, worker runtime, store, and web component areas

## Affected Runtime/Data Flow

The feature would likely add a new runtime-only checkpoint path:

1. The worker periodically clones compact model state after training steps.
2. The worker stores a bounded ring buffer of checkpoints or publishes compact checkpoint metadata to the main thread.
3. The main thread keeps only lightweight metadata in React/store state.
4. Heavy checkpoint payloads stay in the worker or a dedicated frame-buffer-like transport.
5. Restoring a checkpoint pauses training, applies model state in the worker, invalidates stale visualization buffers, and emits a fresh snapshot.

Raw per-sample activations, prediction grids, and large visualization arrays should not be stored in checkpoint metadata. They should be recomputed or refreshed through the existing snapshot/frame-buffer paths after restore.

## Model-State Snapshot Strategy

Minimum restorable state likely includes:

- network layer sizes and activation/output activation assumptions for the current run,
- weights,
- biases,
- optimizer accumulators/moments if the optimizer requires them,
- training step, epoch, and loss history position,
- current run ID or a restore-safe generation counter,
- scalar summary at capture time.

Before implementation, the engine/runtime owner must confirm whether restoring weights and biases alone is sufficient for each optimizer. If optimizer state is required, checkpoint payload size and determinism tests must cover it explicitly.

## Checkpoint Cadence

Initial cadence should be fixed and conservative:

- capture only while training is running,
- skip capture while a previous checkpoint operation is in progress,
- capture every N training steps or every M emitted snapshots, whichever is simpler and more deterministic,
- do not capture on every frame,
- always capture an initial checkpoint after run initialization or first train step if memory allows.

Cadence should be runtime-local rather than URL/config/public-config state in the first implementation.

## Memory Limits

Initial implementation should use explicit bounds:

- fixed maximum checkpoint count,
- fixed maximum approximate bytes per checkpoint or total timeline memory,
- oldest-checkpoint eviction when over the bound,
- no persistent checkpoint storage,
- no raw activation arrays or decision-boundary grids in checkpoints.

The UI should disclose when older checkpoints have been evicted through concise timeline state, not through noisy warnings.

## Compatibility Risks

- Worker protocol changes are likely and require shared guards, worker tests, bridge tests, and protocol docs.
- Frame-buffer or transport changes may be needed if checkpoint payloads leave the worker.
- Optimizer-state restore could affect determinism if incomplete.
- Restored snapshots could confuse run history if restore semantics are not explicit.
- Persistence, URL/config serialization, and public config shape must remain unchanged unless a separate approval explicitly authorizes a schema migration.

## Accessibility Impact

The timeline must be usable without canvas-only interactions:

- native range input or equivalent keyboard-operable control,
- labelled restore and return-to-live buttons,
- visible focus,
- clear paused/live/restored status text,
- screen-reader announcement when a checkpoint is selected or restored,
- no reliance on color alone for checkpoint status,
- reduced-motion-safe timeline updates.

## Performance Impact

Performance risks:

- cloning large weight/optimizer buffers during hot training paths,
- memory growth from retained checkpoints,
- extra worker/main message traffic,
- visualization invalidation churn after restore.

Mitigations:

- bounded ring buffer,
- conservative cadence,
- clone only typed arrays needed for restore,
- keep heavy payloads outside React state,
- avoid persistent storage,
- compare `pnpm test:perf` and build output against `docs/perf/PERFORMANCE_BASELINE.md`,
- add targeted memory/cadence regression tests where feasible.

## Test Plan

- Engine tests for deterministic snapshot/restore of weights, biases, counters, and optimizer state if included.
- Shared protocol guard tests for any new checkpoint commands or messages.
- Worker lifecycle tests proving checkpoints pause/restore safely and do not continue training unexpectedly.
- Frame-buffer/version-counter tests if checkpoint restores invalidate heavy visualization domains.
- Store/hook tests for live/restored status and timeline selection.
- Component tests for keyboard operation, accessible names, empty states, and restore controls.
- Regression tests that URL/config serialization, persistence/run-history schema, and public config shape are unchanged.

## Browser QA Plan

Use Browser QA Mode B if available:

1. Load the app.
2. Start training on a common preset.
3. Wait until at least two checkpoints exist.
4. Pause training.
5. Scrub between checkpoints with mouse and keyboard.
6. Restore an earlier checkpoint.
7. Confirm loss, decision boundary, and inspection surfaces refresh.
8. Resume training from the restored point.
9. Check compact viewport.
10. Check current-URL console errors.

Record screenshots and any performance uncertainty under `docs/qa/browser-qa/`.

## Rollback Plan

- Revert UI timeline controls.
- Revert worker checkpoint commands/messages and shared protocol guards.
- Revert any checkpoint transport/frame-buffer additions.
- Revert engine checkpoint/restore helpers if added.
- Keep unrelated Wave 6A-6D experiment workflow improvements intact.
- Re-run `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` after rollback.

## Approval Required

Yes. Wave 6E is explicitly mandatory-approval work. Implementation is likely to touch worker protocol, runtime restore behavior, model-state snapshot semantics, frame-buffer invalidation, and possibly optimizer-state determinism.

## Decision

Design note prepared on 2026-05-11. Implementation is blocked until explicit user approval is granted for the checkpoint/timeline design and its protected-contract impacts.
