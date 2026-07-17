# Precision Lab Production Integration Design

**Date:** 2026-07-16

**Status:** Approved design awaiting implementation plan

**Scope:** Replace the production Build and Run presentation together with the
approved Precision Lab workspace while preserving the existing neural-network
engine, runtime, scientific evidence, experiment document, and persistence
contracts.

## Intent

Make the neural network the central attraction in a calm, professional learning
workspace. Topology and decision boundary remain visible together, configuration
tools appear without displacing the experiment, evidence stays close to the
network, and training controls no longer dominate the viewport.

The approved prototype under `prototypes/precision-lab-complete` is the visual
reference, not a code or state source. Its hard-coded architecture, metrics,
datasets, checkpoints, and local component state must not enter production.
Production components continue to read canonical prepared recipes, live worker
frames, revision-scoped evidence, and saved-run artifacts through their existing
adapters.

## Success criteria

The integration is successful when:

- both Build and Run use the Precision Lab shell;
- the real network is the dominant center visualization at supported sizes;
- the real decision boundary is always mounted beside or immediately with the
  topology and is never hidden behind an evidence tab;
- Build controls, Run evidence, Presets, Lessons, History, audience modes,
  Advanced Tools, checkpoints, saved runs, and keyboard workflows retain their
  existing behavior;
- the topology, boundary, Loss, and Confusion surfaces resize without unwanted
  internal horizontal scrollbars;
- training does not cause layout shifts or center-page stutter;
- the production test, accessibility, bundle, performance, and browser gates
  pass without weakening assertions or expanding timeouts to hide regressions.

## Chosen integration approach

Create a new production `PrecisionLabShell` using the existing
`BuildRunShellProps` content composition contract, extended only with explicit
compact-boundary and shell-navigation inputs, then replace only the shell call in
`App.tsx`. The existing root orchestration remains responsible for compatibility
states, keyboard shortcuts, drawer state, visualization demand, worker ownership,
and error recovery.

This is preferred over rewriting `BuildRunShell` in place because the old and new
compositions can be tested independently until the new shell is proven. It is
preferred over porting the prototype application because the prototype simulates
the product with local mock state and cannot preserve production invariants.

The old shell is removed only after consumer searches, focused tests, full tests,
and browser smoke prove that `App.tsx` is the sole production consumer.

## Product architecture

### Preserved owners

- `usePlaygroundStore` owns the shareable V2 experiment document, async recipe
  preparation, import/URL compatibility, and visualization-demand delivery cache.
- `useTrainingStore` owns runtime status, generation/revision identity, evidence
  references, points, frame versions, configuration lifecycle, checkpoints,
  errors, and training speed.
- `useTraining` remains the only training and worker orchestration owner. The new
  shell receives its commands; it never creates a second hook instance.
- The worker remains authoritative for trained parameters and scientific
  artifacts.
- `frameBuffer` remains the main-thread typed-array and provenance cache. React
  components subscribe only to the narrow version counters they need and then
  read the accepted buffer snapshot.
- `metricHistoryBuffer` remains the bounded loss/accuracy-series owner.
- `useLayoutStore` owns only local workspace navigation, active Build/Evidence
  targets, audience mode, Advanced Tools disclosure, and code tab.
- `experimentMemoryStore` owns pending, accepted, incompatible, and rejected
  saved-run artifacts.

### Shell boundary

`PrecisionLabShell` receives prepared content nodes, display-safe values, and
commands from `App.tsx`. Focused feature adapters outside the shell read stores,
frame-buffer snapshots, recipe identity, and scientific selectors, then derive
the immutable display models used by the recipe strip, run status, selection
deck, and boundary rail. The shell never receives raw worker responses,
scientific evidence envelopes, or frame artifacts. It owns presentation-only
composition:

- header and recipe strip placement;
- Build/Run module rail;
- context panel/drawer chrome;
- topology, boundary rail, selection deck, and evidence-deck placement;
- responsive layout tracks;
- bottom transport placement;
- focus movement associated with opening and closing shell surfaces.

It does not mutate recipes, call worker APIs, interpret scientific evidence, or
copy frame arrays. Existing production controls remain command boundaries;
focused feature adapters and existing evidence components remain data and
scientific-interpretation boundaries. Adapters may expose formatted labels,
freshness states, bounded scalar summaries, typed presentation states, and
event commands, but no API that lets the shell reach back into a store or worker.

## Shared workspace composition

### Header

The header adopts the Precision Lab brand and compact visual hierarchy while
preserving the production Header behavior and accessible names:

- Build and Run remain a `Workspace view` control;
- live status identifies idle, running, paused, syncing, error, and restored
  states without depending on color alone;
- the audience selector remains a native `Audience mode` control;
- Presets, Lessons, History, and Advanced Tools remain available in every mode;
- `Advanced Tools` retains `aria-expanded`, `aria-controls`, focus restoration,
  and Escape behavior;
- the mobile training action retains the existing lifecycle blocking rules.

Header metrics are reduced to the information needed to orient the current run.
Detailed loss and accuracy claims remain in evidence surfaces where their basis,
revision, and freshness are visible.

Production keeps its current Inter and Space Grotesk type system. Chrome icons use
direct, tree-shaken imports from `@tabler/icons-react`; icons remain decorative
when adjacent text supplies the accessible name. No handcrafted SVG, emoji, or
prototype raster asset substitutes for functional interface icons.

### Recipe strip

A compact strip below the header summarizes the actual prepared document:

- dataset;
- architecture;
- hidden activation;
- output activation/objective;
- seed;
- evaluation freshness and trained-recipe drift;
- an Edit Recipe action that selects the relevant Build target.

The strip uses existing recipe identity and scientific evidence selectors. It
must distinguish the current prepared recipe from the trained recipe instead of
presenting pending settings as trained state.

### Module rail and context panel

A slim left rail exposes the visible tools for the current audience profile.
Build targets are Data, Network, Features, Hyperparameters, and Configuration.
Run targets are Boundary, Loss, Confusion, Inspect, and Code. Presets, Lessons,
History, and Advanced Tools remain header-level surfaces rather than duplicate
rail actions.

Selecting a Build target opens one context panel over the left portion of the
workspace. The panel mounts the existing production control component. It does
not replace or resize the topology/boundary tracks on compact desktop layouts.
Closing the panel restores focus to the rail trigger. Direct lesson and
explanation navigation opens any required disclosure and selects the correct
target without changing audience mode.

Audience-profile visibility continues to come from the existing pure profile and
visibility helpers. Hidden controls are unmounted but their recipe values remain
unchanged and discoverable through the recipe summary.

## Build workspace

Build keeps the network and boundary live while settings change. The selected
control panel overlays the left side of the experiment canvas on desktop and
becomes a full-width sheet on phone layouts.

### Data

`DataPanel` gains deterministic dataset previews produced from the production
dataset definitions and seeds. Previews are small canvases or source-backed image
surfaces, not CSS drawings. Dataset identifiers map directly to production IDs,
including `gauss`, `checkerboard`, and `three-class-clusters`. Selecting a preview
uses the existing recipe-edit transaction and lifecycle blocking.

### Network, Features, Hyperparameters, and Configuration

Existing controls retain validation, labels, pending-state behavior, and command
paths. The new shell changes grouping, spacing, and panel chrome only. Rapid
edits must continue to compose against the latest candidate document, and stale
preparation completions must remain fenced.

## Run workspace

Run makes the trained network primary. The topology and boundary remain visible
while the evidence deck changes between Boundary, Loss, Confusion, Inspect, and
Code. The Current Run and Recipe summaries become compact contextual summaries,
not permanent columns competing with the visualization.

The evidence tablist preserves its existing semantic names, one mounted
tabpanel, roving focus, Arrow key navigation, Home/End support, audience-profile
visibility, and Advanced Tools behavior. Selecting Boundary in the evidence deck
shows its detailed controls and explanation without mounting a second live
boundary canvas.

## Network visualization

The existing production graph renderer and painter remain the data source and
interaction engine. The visual treatment changes from small circular nodes to
responsive activation tiles:

- hidden and output neurons display real activation grids when available;
- empty or unavailable grids use an explicit neutral state rather than a fake
  heatmap;
- node size scales within bounded minimum and maximum values based on available
  graph height and layer density;
- layer labels and counts remain legible without forcing graph-level scrolling;
- zoom, pan, fit, Weights/Activations mode, signed-edge filters, tooltips, health
  states, and lesson callouts remain supported;
- keyboard and screen-reader summaries retain the `Architecture summary`
  contract.

Selecting a neuron persists the selection until the user selects another neuron,
clears it, changes generation, or changes architecture. The graph highlights a
small number of strongest incoming and outgoing paths using actual weights.
Highlight width and opacity encode magnitude; sign remains distinguishable by the
existing positive/negative palette and not color alone. Unselected connections
remain visible at reduced emphasis. The selection deck reads accepted frame data
and describes the selected neuron, activation distribution, bias, and strongest
influences without calling the worker directly.

Both the Canvas renderer and the supported fallback must preserve functional
parity for graph commands, weight and activation data accuracy, persistent neuron
selection, selected-path meaning, layer/node labels, status states, keyboard
operation, and accessible summaries. Visual geometry is not part of this parity
contract: the fallback may retain its existing node shapes and spacing while
presenting the same data and interactions. The fallback must not be silently
removed during this visual slice.

## Pinned decision-boundary rail

Topology and boundary are one workspace composition. A new compact boundary
presentation consumes the existing `useDecisionBoundaryModel` display model and
shares its accepted frame snapshot semantics. The pinned rail owns the only live
decision-boundary canvas in the product shell. It remains mounted while Build/Run
and evidence tabs change, and it does not mount `BoundaryContent` or create
another store adapter.

The rail contains:

- the real decision-boundary canvas;
- task-appropriate train/test points;
- freshness/drift state;
- compact accuracy or loss evidence when scientifically available;
- an expand action that opens the detailed Boundary evidence state without
  replacing the topology.

Selecting the Boundary evidence tab does not render another canvas. Its tabpanel
contains overlay, train/test, and discretization controls; provenance and
freshness; accessible scalar and class summaries; and the existing explanatory
content. Those controls update the single pinned canvas through the focused
boundary controller and existing document-view/local-overlay commands. The
expand action selects and focuses this Boundary tabpanel while the pinned canvas
stays in place.

Because the boundary is visible in both Build and Run, visualization demand
requests the required boundary artifact whenever a ready Precision Lab workspace
is mounted. Demand remains a fresh immutable object when changed. Scientific
evaluation cadence remains independent of visibility. Cached domains are retained
when a cadence-gated payload is omitted.

The extra always-visible request must pass the existing worker performance gates.
If it exceeds the approved medians, reduce only visual artifact cadence or paint
frequency; do not reduce scientific evaluation accuracy or hide the rail.

## Evidence deck

The deck occupies a bounded track beneath the topology on desktop. It changes
content without changing the outer workspace height.

- Boundary shows controls, provenance, accessible summaries, and explanatory
  context for the single pinned boundary canvas. It does not render a canvas of
  its own.
- Loss uses a responsive chart viewport with labels and legends outside the plot
  collision area. It must not use a horizontal scrollbar.
- Confusion uses a responsive matrix-and-metrics grid. Metrics wrap or stack at
  narrow widths; the selected responsive state must not create an internal
  horizontal scrollbar.
- Inspect retains its lazy controller/view architecture and worker request
  fencing.
- Code retains lazy loading, format tabs, copy/export behavior, and existing
  accessibility labels.

Evidence content may scroll vertically when its information genuinely exceeds
the bounded deck height. Horizontal overflow is a defect at every supported
viewport.

## Transport

The transport remains persistent but becomes visually secondary. It retains:

- Start/Pause;
- one-step execution;
- reset;
- save run;
- 1, 5, 10, 25, and 50 steps per frame;
- checkpoint timeline, selected checkpoint, restore, and restore guarantee;
- lifecycle blocking and exact accessible action names.

Desktop uses a compact single row. Compact desktop may wrap metadata into a
second line without changing the workspace track above it. Phone layouts use two
intentional rows with 44px minimum interactive targets. Running/paused label
changes must not change the primary button width or shift neighboring controls.

## Responsive model

Responsive behavior is container-led and uses explicit stable tracks.

### Wide: at least 1180px

- 48–56px module rail;
- flexible topology track;
- 250–280px boundary rail;
- 180–220px evidence track;
- compact single-row transport.

### Compact desktop/tablet: 680–1179px

- module rail remains visible;
- topology remains the largest track;
- boundary rail contracts to 190–230px;
- context panels overlay rather than permanently shrinking the topology;
- evidence becomes a compact full-width track;
- transport may use a planned second metadata row.

The 735×860 reference viewport must show topology and a usable boundary together
without page-level horizontal scrolling.

### Phone: below 680px

- header utilities collapse into an accessible menu while Build/Run and status
  stay visible;
- the rail becomes a compact horizontal tool strip;
- topology appears first and the compact boundary appears immediately beneath it
  in the same primary workspace, never behind an evidence tab;
- Build context panels become full-width sheets;
- evidence follows the primary workspace;
- transport uses two rows and keeps all controls reachable;
- the document has no horizontal overflow at 320px and all interactive controls
  keep a 44px target.

At 200% zoom, the layout follows the same compact/phone composition. Content is
not CSS-hidden solely to obtain containment.

## Stutter and rendering constraints

The center-page stutter is treated as a layout stability defect, not merely an
animation preference. The implementation must:

- keep header, recipe, topology, evidence, and transport tracks dimensionally
  stable while training;
- avoid mounting/unmounting structural workspace regions on each frame;
- paint activation grids, connections, and boundary updates in canvases without
  changing their CSS dimensions;
- subscribe to narrow frame-domain versions rather than broad runtime objects;
- keep typed arrays outside React state and avoid copying grids for display;
- batch ResizeObserver results and ignore sub-pixel/no-op size changes;
- prevent changing numeric labels from altering grid column widths;
- use CSS containment on visualization and evidence regions where it does not
  break overlays or accessibility;
- disable nonessential transitions under `prefers-reduced-motion: reduce`.

Layout stability is a pass/fail browser requirement. At 1437×742, 735×860, and
320×844, the test waits for the ready shell, records the bounding rectangles of
the header, recipe strip, topology, boundary rail, evidence deck, and transport,
starts training at 50 steps per frame, and samples those regions for five
seconds. With no resize, navigation, drawer, or panel interaction during the
sample, every region's `x`, `y`, `width`, and `height` must remain within 1 CSS
pixel of its recorded value. Chromium's cumulative layout shift over the same
post-start window, excluding entries with `hadRecentInput`, must equal zero. In
both Chromium and WebKit, `document.documentElement.scrollWidth` and the primary
workspace `scrollWidth` must remain no more than 1 CSS pixel greater than their
respective `clientWidth` before, during, and after the sample.

Performance instrumentation may measure render/layout activity in development,
but browser tooling entries must remain filtered so DevTools interaction does not
produce false slow-interaction warnings.

## State transitions and failure handling

- Recipe changes retain the existing prepare → pause/acknowledge → staged worker
  initialization → generation commit → URL/points/demand synchronization
  transaction.
- Pending configuration, worker failure, divergence, incompatible URL/import,
  empty evidence, stale trained recipe, and unavailable visualization states use
  the existing recovery commands and provenance-aware copy.
- Shell panels never call worker mutations directly.
- Checkpoint restore remains in the same generation, advances to the next
  revision, replaces evidence/history, and fences delayed pre-restore frames.
- Saved runs remain evidence artifacts, not resumable parameter snapshots.
- Failed saved-run persistence retries the exact pending artifact.
- Incompatible and rejected bytes are not silently discarded.
- Direct lesson/explanation navigation to a hidden tool opens Advanced Tools
  atomically without changing audience mode or experiment state.

## Accessibility contract

The replacement preserves these production semantics:

- main landmark named `Neural network playground workspace`;
- `Workspace view` with Build and Run controls;
- `Audience mode`;
- `Advanced Tools` disclosure;
- Presets, Lessons, and History dialogs and close actions;
- `Recipe summary`, `Current run`, `Timeline strip`, and `Status bar` regions;
- `Evidence views` tablist with Boundary, Loss, Confusion, Inspect, and Code;
- `Checkpoint timeline` and its controls;
- existing training action names and disabled-state explanations;
- generation and revision data attributes used by scientific workflow tests.

All icon-only actions receive explicit accessible names. Focus remains visible.
Drawers and sheets restore focus to their trigger. Escape closes the most local
surface before Advanced Tools. Representative Beginner, Explore, and Lab states
receive automated axe coverage in Build and Run.

## Styling and asset policy

Add an isolated Precision Lab style layer after existing base/component styles.
Use a `precision-*` namespace for new shell chrome. Reuse the production token
system, extending it only for the approved navy surface hierarchy, cyan/violet
accents, fixed track sizes, and graph tile geometry. Do not import the prototype
stylesheet wholesale or add global resets.

Dataset previews are generated from production data. Neuron and boundary images
are live canvases. Existing prototype screenshots are QA references only and are
not shipped as UI assets.

## Verification contract

Implementation follows focused TDD and keeps the current semantic browser
workflows green. Required acceptance includes:

1. Focused Vitest suites for `PrecisionLabShell`, Header, layout visibility,
   visualization demand, graph selection, compact boundary, Data previews,
   Loss/Confusion responsiveness, TrainingControls, and responsive CSS guards.
2. Existing App and shell integration tests, including the full audience/profile
   axe matrix, focus restoration, Escape ordering, and scientific-state
   invariants.
3. `pnpm lint`.
4. Full `pnpm test` across engine, shared, and web packages.
5. `pnpm build` and a pass/fail zlib-gzip comparison against the recorded release
   baseline. The main entry may grow by at most 10 KiB, from 142,005 to 152,245
   bytes. The Inspection chunk may grow by at most 1 KiB, from 6,349 to 7,373
   bytes. Total JavaScript may grow by at most 5%, from 223,010 to 234,161 bytes.
   A limit failure blocks completion and must be addressed through direct icon
   imports, dead-code removal, or an existing lazy boundary; the baseline or cap
   is not raised without a separate reviewed decision.
6. Repeated `pnpm test:perf` runs on an idle host, judged by medians. Forced
   evaluation remains at most 250ms and save capture at most 500ms.
7. Chromium and WebKit smoke with zero retries and no console/page errors,
   covering training, pause/step, presets, checkpoint restore, saved runs,
   audience/disclosure invariants, all evidence tabs, dataset preview selection,
   graph controls, and wide/735px/320px containment.
8. The five-second post-start layout-stability measurement defined above,
   including the 1 CSS pixel region-bounds tolerance, zero Chromium CLS, and
   cross-browser overflow threshold at all three reference viewports.
9. Keyboard-only, reduced-motion, 200% zoom, 44px touch-target, and horizontal
   overflow checks.
10. Side-by-side reference and production screenshots at identical states and
   viewports. A screenshot alone is not acceptance; interactions and scientific
   invariants must also pass.
11. `git diff --check` and production-consumer searches before deleting the old
    shell path.

No assertion, retry count, timeout, performance threshold, or scientific cadence
may be weakened simply to obtain a passing result.

## Delivery shape

The user requested one coordinated Build-and-Run replacement rather than a
Run-first release. The work therefore ships from one integration branch, but is
implemented through internal checkpoints:

1. shell contract and visual tokens;
2. header, recipe strip, rail, and context surfaces;
3. responsive network tiles and selection paths;
4. pinned compact boundary and demand integration;
5. Build panels and dataset previews;
6. evidence deck and compact transport;
7. responsive, accessibility, performance, and browser verification;
8. removal of the proven-unused old shell.

Each checkpoint must keep focused tests green. The branch is not presented as
complete until the full verification contract passes.

## Non-goals

- No engine-math, optimizer, objective, or dataset-generation rewrite.
- No worker-protocol, V2 schema, URL, checkpoint, export, or saved-run migration.
- No second shell or mode-specific state graph.
- No generic view-model framework or new state-management abstraction.
- No persistence of neuron selection, open panels, or evidence-only display
  preferences into the experiment document.
- No removal of legacy compatibility paths until production searches and tests
  prove they have no consumers.
- No unrelated scientific-copy or visualization-correctness fixes folded into
  this visual integration.
