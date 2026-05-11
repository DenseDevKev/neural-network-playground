# Design Note: Activation Histogram Explorer

## Problem

The playground can show network topology, decision boundaries, loss curves, confusion matrices, and inspection details, but it does not yet give learners a compact way to see whether layers are saturated, inactive, or broadly distributed during training.

Activation histograms could teach concepts like dead neurons, saturation, hidden-layer capacity, and feature scaling. However, histograms require activation distribution data that is not currently exposed as a stable UI contract.

## Proposed Change

Add an activation histogram explorer in two stages:

1. Existing-data assessment: verify whether current snapshot/frame-buffer layer stats are sufficient for a coarse layer-level summary.
2. If not sufficient, design a demand-gated histogram data path that sends bounded bins, not raw per-sample activations.

The preferred implementation would publish compact histogram bins per layer only when the inspection panel requests them.

## User Value

- Learners can see when activations cluster near zero, saturate near activation limits, or spread across useful ranges.
- Teachers can explain why gradients and learning rate interact with activation functions.
- The feature would complement, not replace, the existing network graph and inspection panel.

## Affected Files

Likely files if approved:

- `packages/engine/src/network.ts`
- `packages/engine/src/types.ts`
- `packages/shared/src/workerProtocol.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/components/controls/InspectionPanel.tsx`
- `apps/web/src/components/visualization/*`
- Tests in engine, shared, worker, and web component areas

## Affected Runtime/Data Flow

Existing runtime snapshots and frame buffers prioritize scalars, grid predictions, neuron grids, parameters, layer stats, and confusion data. A reliable histogram explorer would likely need either:

- a new bounded histogram payload, or
- a new frame-buffer domain for histogram bins and version counters.

Raw activations should not be streamed into React state.

## Compatibility Risks

- Worker protocol changes would require shared guards, bridge tests, worker tests, and protocol docs.
- Frame-buffer additions would require version-counter tests and memory review.
- URL/config serialization and persistence should remain unchanged.
- Engine determinism must be preserved.

## Accessibility Impact

The histogram needs text alternatives, such as layer-level summaries:

- percentage near zero,
- percentage near activation limits,
- most populated bin range,
- whether the layer appears sparse, saturated, or broadly active.

Keyboard users must be able to select layers and read the summary without canvas-only interactions.

## Performance Impact

Potential risks:

- extra worker computation per snapshot,
- extra large-array transport,
- memory churn if raw activations or per-sample arrays are retained,
- render churn if histogram updates are not demand-gated.

Mitigations:

- compute fixed-width bins only,
- gate by inspection demand,
- update on cadence rather than every frame,
- keep arrays outside React state,
- compare against `docs/perf/PERFORMANCE_BASELINE.md`.

## Test Plan

- Engine deterministic histogram-bin tests with fixed data.
- Shared protocol guard tests if a protocol payload is added.
- Worker demand/cadence tests proving histograms are computed only when requested.
- Frame-buffer version-counter tests if a new buffer domain is added.
- Component tests for layer selection, empty states, and text alternatives.
- Accessibility tests with Testing Library and `jest-axe`.

## Browser QA Plan

- Load app without console errors.
- Start/pause training.
- Open inspection panel.
- Enable histogram explorer.
- Select at least two layers.
- Verify chart/text alternative updates.
- Verify keyboard selection.
- Check compact viewport.
- Compare against performance baseline if runtime data is added.

## Rollback Plan

- Remove histogram UI component and tests.
- Remove any demand flag or frame-buffer domain added for histograms.
- Revert worker/protocol guard changes as a single slice.
- Keep unrelated inspection panel behavior unchanged.

## Approval Required

Yes, if implementation requires new worker protocol fields, frame-buffer domains, runtime snapshot fields, or engine activation collection.

## Decision

Approved by the user on 2026-05-11 for a small high-risk slice using demand-gated, bounded layer-level histogram bins only.

Implementation constraints:

- Do not stream or store raw activation arrays in React state.
- Keep histogram bins bounded and layer-level.
- Use worker demand/cadence gating.
- Preserve URL/config serialization, persistence/run-history schema, public config shape, and dependencies.
- Do not combine this slice with gradient-flow overlay, checkpoints, timeline scrubber, or other Wave 4+ features.
