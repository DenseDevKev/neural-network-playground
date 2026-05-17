# Frontend Hierarchy Polish Design

Date: 2026-05-17
Status: Approved for implementation planning

## Goal

Improve NN.FORGE frontend scanability through a hierarchy cleanup pass. The app should keep its current workspace model and training workflow, but feel calmer, easier to read, and more intentional.

This is not a structural redesign. The existing layout variants, state wiring, responsive behavior, accessibility labels, URL state, worker protocol, serialization, persistence, and training behavior must remain intact.

## Direction

Use the "Hierarchy Cleanup" direction:

- Keep the current workspace recognizable.
- Reduce border, glow, shadow, and filled-surface competition.
- Make Network Topology read as the visual anchor.
- Make outputs feel like sibling views in one shared output system.
- Keep Start/Resume/Pause as the only primary actions.
- Let Guided Lesson become contextual instead of always visually loud.

## Implementation Guardrails

- Use CSS variables/tokens for muted surfaces, subtle borders, active states, and spacing.
- Prefer reducing contrast, shadow strength, glow, border density, and competing filled backgrounds before changing layout.
- Preserve all current component behavior, panel variants, responsive behavior, data/control wiring, accessibility labels, and training state flow.
- Preserve or improve visible focus states, disabled states, hover states, and readout legibility.
- Do not introduce new interaction patterns, animations, or visual metaphors unless they directly support the existing training workflow.
- Add markup/class hooks only when CSS cannot target the hierarchy cleanly.
- No new dependencies.

## Component Polish Scope

### Top Bar

Metrics should become secondary status indicators. Start/Resume/Pause remains the only true primary action.

Expected changes:

- Reduce metric tile contrast and shadow.
- Keep numeric readouts legible with tabular styling.
- Preserve updated-value feedback if it remains subtle and readable.
- Keep layout controls available but quieter than the training action.

### Left Configuration Panel

Make the cognitive flow clearer: Dataset/Data -> Network -> Training/Hyperparams -> Config. The panel should feel like a guided configuration surface, not a stack of equal-weight controls.

Expected changes:

- Improve active tab and section grouping rhythm.
- Normalize button heights, segmented controls, slider/readout spacing, and group gaps.
- Reduce visual weight of inactive options while preserving hover, focus, and disabled clarity.
- Avoid changing data, network, feature, hyperparameter, or config behavior.

### Network Topology Panel

Network Topology is the visual anchor. Graph controls should feel attached to the graph card as a compact toolbar, header, or footer treatment, not floating controls competing with the visualization.

Expected changes:

- Give the graph stage clearer internal spacing.
- Reduce overlapping or overly prominent control chrome.
- Make graph controls feel related to the topology panel rather than separate UI islands.
- Preserve graph rendering, graph interactions, and canvas/SVG behavior.

### Right Output Panel

Boundary, Loss, Confusion, and Inspection should feel like one shared output system. The tab strip and panel chrome should be quieter than the active visualization.

Expected changes:

- Make inactive output tabs lower contrast.
- Keep the active output state clear.
- Make loading and empty states calmer and consistent.
- Avoid changing output demand logic or lazy-loaded panel behavior.

### Training Controls And Guided Lesson

Training controls stay functional, prominent, and accessible. Guided Lesson drops back visually unless the user is actively using lesson mode or a lesson step requires attention.

Expected changes:

- Keep Start/Resume/Pause prominent.
- Make Step, Reset, speed buttons, timeline, and Restore visually secondary but easy to use.
- Reduce default Guided Lesson prominence.
- Allow stronger lesson highlight/accent only for active lesson states.
- Preserve keyboard shortcuts and training lifecycle behavior.

## Concrete Visual Rules

### Surface Hierarchy

The shell can stay dark and technical, but inactive panels should use muted surfaces, subtler borders, and lighter shadow treatment. Active or primary areas get the clearest contrast.

### Action Hierarchy

Only Start/Resume/Pause should read as primary actions. Step, Reset, Reshuffle, Restore, graph tools, and similar utilities should use a quieter secondary treatment.

### Panel Headers

Panel titles should be scannable and consistent. Phase labels like Build and Run should remain small metadata, not competing badges.

### Control Density

Keep the dense tool feeling, but avoid cramped clusters. Use consistent gaps, button heights, segmented-control spacing, and slider/readout alignment.

### Topology Controls

Move toward a cohesive graph-control treatment: compact toolbar or edge/footer controls visually attached to the topology panel. Avoid controls floating over graph content unless they are clearly overlaid tools.

### Right Output Tabs

The right panel tab strip should look like a shared view switcher. Active tab is clear; inactive tabs are lower contrast.

### Guided Lesson

Default state should be calm. Active lesson/highlight state can use stronger accent, but idle lesson UI should not compete with training controls or topology.

### Responsive Boundary

Mobile should preserve the same visual priority: header primary action, configuration flow, topology anchor, outputs, and accessible training controls. No new mobile-only behavior beyond layout polish.

## Files And Boundaries

Primary styling target:

- `apps/web/src/styles/forge.css`

Likely component hook targets, only if needed:

- `apps/web/src/App.tsx`
- `apps/web/src/components/layout/Header.tsx`
- `apps/web/src/components/layout/RegionShell.tsx`
- `apps/web/src/components/controls/TrainingControls.tsx`
- `apps/web/src/components/controls/GuidedLessonPanel.tsx`
- `apps/web/src/components/visualization/NetworkGraph.tsx`

Out of scope:

- Worker protocol changes.
- Serialization changes.
- URL/config format changes.
- Persistence or run-history data changes.
- Shared contract changes.
- New visualization features.
- New layout modes.
- New dependencies.

## Validation Plan

Run focused technical and rendered validation:

- `pnpm build`
- Relevant component/style tests for touched files.
- Browser QA on desktop and mobile.
- Verify page identity, nonblank render, no framework overlay, console health, screenshot evidence, and at least one training/control interaction.
- Check hover, focus, disabled states, and readout legibility after visual contrast reductions.

## Success Criteria

- The app remains behaviorally unchanged.
- Start/Resume/Pause is still the strongest action on the screen.
- Network Topology reads as the visual anchor in desktop dock layout.
- The left panel scans more clearly as a configuration flow.
- Output tabs feel like one shared output system.
- Guided Lesson is calmer by default and stronger only when context requires it.
- Desktop and mobile screenshots show no obvious clipping, overlap, unreadable text, or new horizontal overflow.
