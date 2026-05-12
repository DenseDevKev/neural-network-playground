# Wave 7 Proposal: Larger Product Bets

Date: 2026-05-12

Status: Approval required before implementation.

## Context

Waves 0 through 6E are implemented through verified, reviewable slices. Wave 7 is intentionally treated as a separate initiative because each candidate can affect engine math, worker/runtime data flow, visualization complexity, accessibility, performance, and product direction.

No Wave 7 feature is approved for implementation yet.

## Candidate 1: Side-by-Side Model Arena

### User Value

Learners can compare two models under the same dataset and see how architecture, learning rate, regularization, and feature choices change learning behavior.

### Likely Technical Design

- Run two bounded experiment configurations side by side.
- Prefer sequential or time-sliced worker execution before considering multiple live workers.
- Reuse existing decision boundary, loss, confusion, and run-summary components.
- Keep saved comparison metadata separate from persistence unless a later approved design extends run history.

### Engine Impact

No engine math changes expected for a first slice.

### Worker / Runtime Impact

Medium to high. Multiple model states or multiple workers need careful lifecycle, demand, cadence, and memory boundaries.

### Persistence / URL Impact

Potentially high if shareable arena configurations are required. Any public config or URL/config change needs separate approval.

### Visualization Impact

Medium. Requires paired layouts and synchronized summaries without overloading the dense lab shell.

### Accessibility Plan

Use labelled model regions, accessible comparison summaries, keyboard-reachable controls, and avoid color-only diff states.

### Test Plan

Worker lifecycle tests, component comparison tests, store tests for arena state, browser QA for start/pause/reset in both panes, and performance comparison.

### Browser QA Plan

Desktop and compact viewport checks, console checks, training start/pause/reset, preset application, and screenshot evidence.

### Performance Plan

Measure worker cadence, build size, memory, and fixed-step training time against the existing baseline.

### Rollback Plan

Keep arena state and UI isolated behind one route/panel or feature branch. Revert arena files without touching existing single-model flow.

## Candidate 2: Slow-Motion Backprop Explanation Mode

### User Value

Learners can step through how errors flow backward and how weights change, making gradients and credit assignment visible.

### Likely Technical Design

- Start with textual and layer-level explanations from existing data.
- Add bounded gradient snapshots only after a design note proves the data path is safe.
- Avoid raw per-sample gradient streaming.

### Engine Impact

High if exposing new gradient internals. Requires determinism and numerical correctness review.

### Worker / Runtime Impact

High if new runtime data is collected or transported. Any protocol/frame-buffer change needs approval.

### Persistence / URL Impact

None for a first educational overlay. Persistence of backprop traces is out of scope unless separately approved.

### Visualization Impact

High. Needs careful animation, reduced-motion behavior, and text alternatives.

### Accessibility Plan

Provide non-animated summaries, keyboard step controls, reduced-motion-safe transitions, and screen-reader-friendly layer summaries.

### Test Plan

Engine gradient tests, worker demand/protocol tests if data changes, component tests, accessibility tests, and browser QA.

### Browser QA Plan

Verify step controls, reduced-motion behavior where feasible, console health, and no overlap in compact layouts.

### Performance Plan

Compare fixed training/inspection scenarios against baseline; watch worker cadence and memory.

### Rollback Plan

Keep as a separate inspection mode. Remove mode and data request without changing core training.

## Candidate 3: Loss Landscape Probe

### User Value

Learners can see how small parameter changes affect loss, helping explain local minima, valleys, and optimizer behavior.

### Likely Technical Design

- Begin with a one-shot, demand-triggered probe over a tiny bounded parameter slice.
- Avoid continuous landscape recomputation.
- Keep probe arrays out of React state.

### Engine Impact

Medium to high. Requires deterministic temporary parameter perturbation without mutating live training state.

### Worker / Runtime Impact

High if probe computation runs in the worker and transports grid data.

### Persistence / URL Impact

None for first slice unless saved probes are later approved.

### Visualization Impact

Medium to high. Needs a readable 2D heatmap or contour-like surface plus accessible text summary.

### Accessibility Plan

Provide text summaries for lowest/highest loss regions and keyboard-reachable probe controls.

### Test Plan

Engine immutability/determinism tests, worker one-shot tests, frame-buffer tests if arrays are transported, component tests, and browser QA.

### Browser QA Plan

Verify probe generation, controls, console health, and compact layout.

### Performance Plan

Set strict bounds for probe grid size and compare worker responsiveness against baseline.

### Rollback Plan

Keep probe as an isolated inspection panel. Revert probe RPC/UI without touching training.

## Recommendation

Start Wave 7 with a design-only slice for the Side-by-Side Model Arena. It has the clearest product value and can begin with comparison UX and existing run data before any high-risk runtime changes.

Do not implement any Wave 7 feature until the specific feature and design are approved.

## Exact Approval Question

Do you approve starting Wave 7 with a design note for the Side-by-Side Model Arena, without implementation until that design note is reviewed?
