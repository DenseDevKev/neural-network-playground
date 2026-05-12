# Design Note: Slow-Motion Backprop Explanation Mode

## Problem

The playground now teaches model tuning, run comparison, checkpoints, activation distributions, and scalar live arena comparisons. It still does not let learners inspect how error signals move backward through the network or how those signals become weight updates.

Backpropagation is a core neural-network concept, but exposing it incorrectly can destabilize the custom engine, worker protocol, frame-buffer cadence, and accessibility model. A safe first slice must be bounded, demand-gated, and educational before it becomes a high-fidelity gradient visualizer.

## Proposed Change

Add a Wave 7 slow-motion backprop explanation mode as a staged feature:

1. Start with a design/education-only approval gate.
2. If approved, implement a first usable mode that pauses training and explains the current backprop step with bounded layer-level summaries.
3. Prefer existing scalar state and engine-internal deterministic computations.
4. Add any new gradient/weight-delta data only as bounded summaries, not raw per-sample arrays.
5. Gate computation through explicit user action or visualization demand.
6. Keep the existing single-model training path unchanged unless a failing test proves a correction is required.

The first approved implementation should not animate every edge or stream per-frame gradient arrays. It should help learners answer:

- Where is error largest?
- Which layer is receiving a strong or weak update?
- Are updates tiny, healthy, or unstable?
- How do learning rate, clipping, and activation saturation affect the next update?

## User Value

- Makes backpropagation visible without requiring users to read math notation first.
- Connects existing concepts: loss, activation saturation, learning rate, gradient clipping, and checkpoints.
- Gives teachers a deterministic step-through demonstration mode for classroom use.
- Builds toward future gradient-flow overlays without jumping directly into raw high-risk payloads.

## Affected Files

Likely design/implementation files if approved:

- `packages/engine/src/network.ts`
- `packages/engine/src/types.ts`
- `packages/engine/src/__tests__/network.test.ts`
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/__tests__/workerProtocol.test.ts`
- `packages/shared/src/index.ts`
- `docs/worker-protocol.md`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/training.worker.test.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/__tests__/frameBuffer.test.ts`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/hooks/useTraining.test.tsx`
- `apps/web/src/store/useTrainingStore.ts`
- `apps/web/src/components/controls/InspectionPanel.tsx`
- `apps/web/src/components/controls/InspectionPanel.test.tsx`
- `apps/web/src/components/visualization/NetworkGraph.tsx`
- `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
- `docs/qa/browser-qa/`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

Preferred first implementation flow if approved:

1. User opens Inspection or Network graph and activates a slow-motion backprop control.
2. Main thread sends a one-shot Comlink request or explicit demand-gated request to the worker.
3. Worker computes one deterministic training-step explanation from the current model/data state.
4. Worker returns bounded scalar summaries only:
   - layer index
   - activation regime summary if already available
   - mean/max absolute error signal
   - mean/max absolute gradient/update magnitude
   - clipped update count or ratio if clipping is active
   - short textual status bucket such as `tiny`, `healthy`, `large`, `clipped`
5. Large arrays remain in the worker or frame buffer. React stores only summary metadata and version counters.
6. UI renders a step explanation, layer summary list, and accessible text alternative.

Any per-neuron, per-edge, per-sample, animation-frame, or continuous streaming variant is out of scope for the first implementation and requires another design review.

## Compatibility Risks

- High if the implementation changes engine math, optimizer behavior, batch ordering, or determinism.
- High if new worker protocol messages or frame-buffer fields are needed.
- High if raw gradients or activations are transported to React.
- Medium if the UI adds new inspection state without persistence.
- Low for copy-only or existing-data explanations.

No implementation may change:

- URL/config serialization.
- Persistence/run-history schema.
- Public config shape.
- Existing training behavior.
- Deployment behavior.
- Dependencies.

## Accessibility Impact

The mode must be usable without animation or color-only encoding.

Requirements:

- Native button or segmented-control entry point.
- Keyboard-reachable step controls.
- Visible focus states.
- Reduced-motion-safe behavior.
- `aria-live` summary for the current explanation step.
- Text alternative for any graph/edge highlighting.
- Clear status buckets with text labels, not color alone.
- Compact viewport check for the dense lab shell.

## Performance Impact

Potentially high if gradient internals are computed repeatedly or transported as arrays.

Performance constraints:

- First implementation should be one-shot or explicitly demand-gated.
- No per-frame raw gradient streaming.
- No raw per-sample arrays in React state.
- Bounded summary sizes by layer count, not sample count or edge count.
- Compare `pnpm test:perf` and build chunks against `docs/perf/PERFORMANCE_BASELINE.md`.
- Add targeted tests proving demand gating or one-shot behavior.

## Test Plan

Before implementation:

- Add failing engine tests for deterministic summary computation if engine support is required.
- Add failing shared protocol guard tests if protocol shape changes.
- Add failing frame-buffer/version-counter tests if frame-buffer fields are added.
- Add failing worker tests for demand gating or one-shot RPC behavior.
- Add failing component tests for UI rendering, keyboard controls, and text alternatives.

Required verification:

- Targeted affected tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- Browser QA Mode B or Mode C pending-human evidence if Browser is unavailable.

## Browser QA Plan

Scenarios:

1. App loads with no console errors.
2. Open Inspection or Network graph.
3. Activate slow-motion backprop mode by mouse.
4. Activate the same control by keyboard.
5. Step once and verify explanation text updates.
6. Verify reduced-motion-safe behavior if animation is present.
7. Verify compact viewport layout.
8. Verify training play/pause/reset still work.
9. Verify no blank graph or framework overlay appears.

## Rollback Plan

Keep implementation slices isolated:

1. Engine summary support, if required.
2. Protocol/frame-buffer support, if required.
3. Worker demand/one-shot computation.
4. UI controls and accessible summaries.
5. QA/state evidence.

Rollback should revert only the current slice files. If protocol or frame-buffer changes are introduced, rollback must also remove docs and guard tests for that contract in the same slice.

## Approval Required

Yes. This is a Wave 7 feature and likely touches high-risk areas:

- Engine internals.
- Gradient/update summaries.
- Worker protocol or one-shot worker RPC.
- Frame-buffer/version counters if bounded summaries are stored outside React.
- Visualization and accessibility behavior.
- Performance-sensitive runtime paths.

## Decision

Pending human approval. Do not implement until the user explicitly approves the exact scope below.

## Exact Approval Question

Do you approve implementing the first slow-motion backprop explanation slice with bounded layer-level summaries only, using one-shot or demand-gated computation, with no URL/config serialization changes, no persistence schema changes, no public config shape changes, no dependencies, no raw per-sample gradient arrays in React state, and no continuous gradient streaming?
