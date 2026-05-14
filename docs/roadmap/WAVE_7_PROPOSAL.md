# Wave 7 Proposal: Larger Product Bets

Date: 2026-05-12

Status: Historical proposal. Several bounded Wave 7 slices have since been approved and implemented. Use `docs/roadmap/ROADMAP_STATE.md` as the current source of truth.

## Context

Waves 0 through 6E are implemented through verified, reviewable slices. Wave 7 is intentionally treated as a separate initiative because each candidate can affect engine math, worker/runtime data flow, visualization complexity, accessibility, performance, and product direction.

This proposal originally established the Wave 7 approval model. Since then, bounded slices for Side-by-Side Model Arena, Slow-Motion Backprop, and Loss Landscape Probe have been approved and implemented. Any further Wave 7 feature still requires a fresh design note and explicit approval before implementation.

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

The original recommendation to start with Side-by-Side Model Arena has been superseded by completed bounded slices recorded in `docs/roadmap/ROADMAP_STATE.md`.

Next recommended Wave 7 target after the Loss Landscape Probe UI is Multiclass Classification Mode, documented in `docs/design-notes/multiclass-classification-mode.md`.

Do not implement any additional Wave 7 feature until the specific feature and design are approved.

## Exact Approval Question

Do you approve creating only the first implementation slice for Multiclass Classification Mode after reviewing `docs/design-notes/multiclass-classification-mode.md`, with the exact slice constrained to the approved first target and no worker protocol, frame-buffer, persistence/run-history schema, URL/config format, public config rollout, dependency, deployment, or UI implementation changes without separate approval?
