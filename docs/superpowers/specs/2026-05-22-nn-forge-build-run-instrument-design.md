# NN.FORGE Build / Run Instrument Design

Date: 2026-05-22

## Goal

Redesign the NN.FORGE interface around two understandable global views, **Build** and **Run**, while keeping the existing feature set intact. The product should feel like a compact ML instrument panel: sharp, technical, modular, dense, and coherent.

This specification refines the earlier Lab Notebook + Cockpit direction. It preserves the product grammar of current recipe, trained snapshot, active run, evidence, and saved runs, but removes the current global layout complexity. Users should not need to understand internal layout modes such as Dock, Focus, Grid, or Split to configure, train, inspect, and compare experiments.

This is a design specification, not an implementation plan. It does not authorize model behavior changes, worker protocol changes, persistence changes, saved-run contract changes, URL/config format changes, or feature removal.

## Product Grammar

The interface must keep these objects distinct:

- **Current recipe:** the editable experiment setup, including dataset, features, network, hyperparameters, optimizer, learning rate, batch size, loss, and regularization.
- **Trained snapshot:** the model state produced by the last accepted run or step sequence.
- **Active run:** the current training operating state, including status, step, epoch, metrics, pause/failure state, freshness, and staleness.
- **Evidence:** the views that explain a run or snapshot, including Boundary, Loss, Confusion, Inspection, Code, and History.
- **Saved runs:** persisted records that can be restored, titled, compared, and used as references.

The main loop remains:

```text
Set recipe -> Run experiment -> Inspect evidence -> Compare -> Adjust
```

## Section 1: Top-Level Structure

NN.FORGE should have only two global views: **Build** and **Run**.

**Build** is the experiment-construction workspace. It uses a compact modular cockpit layout: Data/Recipe controls on the left, Network Topology as the center stage, and Features/Hyperparameters on the right. The topology stays visually dominant and always auto-centers/fits after layer or neuron changes.

**Run** is the experiment-operation and diagnosis workspace. It keeps topology central, moves transport controls into a persistent strip, and shows evidence views as diagnostic companions. History/comparison opens from a drawer or menu. Lessons open from their own menu and only surface as small contextual guidance when active.

`Dock`, `Focus`, `Grid`, and `Split` should no longer be global product modes. Their useful ideas may survive as internal responsive behaviors or layout states, but users should not have to choose between them.

## Section 2: Visual System Direction

NN.FORGE should move toward a sharper **technical instrument-panel** aesthetic instead of a softer stacked-card dashboard style.

Use:

- Compact modules with thin borders
- Dark layered surfaces with restrained contrast
- Slim top metrics
- Crisp uppercase module headers
- Dense but aligned controls
- Small status tags that feel like instrumentation, not badges everywhere
- One consistent control language for buttons, tabs, menus, sliders, selects, chips, and status readouts
- Dark, subtle scrollbars across the app

Avoid:

- Big rounded card stacks
- White/native scrollbars in normal app chrome
- Floating controls that feel detached from their panel
- Multiple competing active states
- Always-visible advanced panels
- Decorative symbols that do not explain the workflow
- Layout modes that sound like implementation details

Core rule: **The app should feel like a focused ML instrument, not a collection of dashboard cards.**

## Section 3: Feature Placement

No feature should be removed, but features should stop competing for permanent screen space.

**Always visible in Build:**

- Data
- Network Topology
- Features
- Hyperparameters
- Top metrics
- Phase/status
- Primary Play/Start control

**Build top bar / menu access:**

- Presets
- Lessons
- History
- More/Commands

**Presets:**

Presets should become a compact dropdown, menu, or popover. It should show the preset name, type/difficulty marker, and short explanation, with apply/select as the clear action. It may show the current preset or "custom recipe" state in the recipe/build area, but the full preset list should not take permanent panel space.

**Available in Build through menus or compact module controls:**

- Advanced topology display settings
- Graph fit/zoom/view controls
- Command palette
- Tweaks
- Less common visualization settings
- History and comparison when explicitly opened
- Lessons when explicitly opened

**Always visible in Run:**

- Topology
- Primary training controls
- Current run state
- Key metrics
- One active evidence view

**Available in Run through tabs, menus, or drawers:**

- Boundary
- Loss
- Confusion
- Inspection
- Code
- History
- Saved runs
- Comparison
- Restore
- Title/annotation
- Lessons
- Topology display settings
- Advanced diagnostics

**History and comparison:**

History is a drawer/menu surface, not a default right-column takeover. Comparison lives inside History as a workflow: current vs previous, best vs current, restore, annotate, and architecture/metric differences.

**Lessons:**

Lessons get their own menu. An active lesson can show a small contextual hint near the relevant area, but the app should not reserve a large persistent lesson panel by default.

## Section 4: Interaction Rules

**Global navigation**

Only `Build` and `Run` are global views. They should be visible in the top bar as the main switch. No `Dock`, `Focus`, `Grid`, or `Split` global mode controls.

**Menus**

Use top-bar menus for `Presets`, `Lessons`, `History`, and `More/Commands`.

`Presets` opens a compact recipe preset picker.

`Lessons` opens guided learning options.

`History` opens saved runs, restore, and comparison.

`More/Commands` contains less common settings such as tweaks, display density, chart style, command palette, and advanced view controls.

**Network Topology**

The topology module should focus on representing the architecture cleanly. It should not contain inline `+ layer`, `+ neuron`, `- neuron`, or `remove` controls inside the graph/stage area.

Architecture editing controls should live in the surrounding Build modules:

- Hidden layer count belongs in the Network module controls.
- Neurons per layer belong in per-layer rows or an adjacent Network inspector.
- Activation/output activation controls belong in Network or Hyperparameters as appropriate.
- Fit, zoom, display mode, weights/activations, and centering controls may remain attached to the topology header because they control visualization, not architecture editing.

The topology itself should auto-center and auto-fit after architecture changes, so users do not need to manually rescue the graph after adding a layer or changing neuron counts.

The graph/stage should remain visually smooth: architecture shown clearly, controls kept outside the canvas, no floating edit buttons over nodes or labels.

**Build editing**

Build should feel like editing a recipe. Data, Features, Network, and Hyperparameters are visible modules. Changes should update the current recipe and preserve existing recipe drift/snapshot rules.

**Run operation**

Run should feel like operating and diagnosing a trained/current experiment. Transport controls stay persistent. Evidence views change through tabs or segmented controls, with History available as a drawer.

**Lessons**

Lessons never own the layout by default. When active, they may show a small contextual hint or checklist near the relevant module, plus a menu/drawer state for full lesson details.

**Scrollbars**

No bright white/native scrollbars in normal app chrome. Use subtle dark scrollbars that match the instrument-panel style, while preserving accessibility and platform fallback behavior.

## Section 5: Implementation Boundaries

This redesign is a UI/UX restructuring, not a model or data rewrite.

Preserve:

- Existing datasets, features, network controls, hyperparameters, optimizer/loss/regularization behavior
- Existing training lifecycle
- Worker protocol
- Engine math
- URL/config format
- Persistence and saved-run contracts
- Recipe drift and trained snapshot semantics
- Output lazy-loading behavior
- Existing evidence types: Boundary, Loss, Confusion, Inspection, Code, History
- Keyboard shortcuts and accessibility labels
- Mobile support

Allowed changes:

- Replace `Dock / Focus / Grid / Split` global modes with `Build / Run`
- Move Presets, Lessons, History, and More/Commands into menus/drawers
- Reorganize existing panels into Build and Run layouts
- Restyle the app toward the sharper instrument-panel system
- Add small local UI primitives where repeated patterns need consistency
- Add topology auto-center/auto-fit behavior after layer/neuron changes
- Move architecture editing controls out of the topology stage/canvas
- Add dark scrollbar styling
- Add class/markup hooks where needed for clean styling

Not allowed without a separate plan:

- New ML concepts
- New datasets or model behavior
- New persistence schema or migrations
- New worker messages
- New saved-run contract changes
- New external UI framework just for aesthetics
- Removing existing functionality
- Turning History into a separate analytics product
- Making Lessons a permanent page that competes with Build/Run

## Section 6: Validation And Success Criteria

The redesign works if a user can understand, configure, run, inspect, and compare NN.FORGE experiments without learning internal layout concepts or hunting through hidden UI structure.

### Success Criteria

- A user sees only two global views: `Build` and `Run`.
- In `Build`, the user can identify and edit the current recipe through Data, Network, Features, and Hyperparameters without moving through multiple navigation layers.
- Presets are available from a clear dropdown/menu, not a permanent panel.
- The topology is visually smooth, centered, and dominant, with architecture editing controls outside the graph/stage area.
- Adding/removing layers or neurons automatically recenters/refits the topology.
- Topology remains legible across shallow, deep, narrow, and wide network configurations.
- In `Run`, the user can start, pause, resume, step, reset, read status, and inspect one active evidence view without switching global modes.
- Evidence always has clear ownership: current recipe, trained snapshot, active run, or saved historical run.
- Recipe drift remains visible after config changes so the user can tell when the current recipe no longer matches the latest trained snapshot.
- History and comparison are available through a drawer/menu and do not dominate the default screen.
- Lessons are available through their own menu and do not reserve permanent layout space unless actively guiding a step.
- The UI uses one consistent instrument-panel system for modules, controls, menus, metrics, tabs, statuses, focus states, disabled states, and scrollbars.
- Bright/native white scrollbars are not visible in normal app chrome.
- Existing behavior, data contracts, training lifecycle, worker protocol, URL format, persistence, saved runs, and accessibility are preserved.

### Manual Validation Checklist

Manual validation should cover:

- Build view at desktop width
- Run view at desktop width
- Mobile Build
- Mobile Run
- Presets menu
- Lessons menu
- History drawer/comparison flow
- Topology after adding/removing layers
- Topology after changing neurons per layer
- Topology with small, large, deep, and wide architectures
- Start -> Pause -> Resume
- Step and reset behavior
- Editing config after training and confirming recipe drift remains clear
- Evidence ownership after config changes
- Saved run/history comparison after recipe changes
- Empty, loading, invalid, disabled, and error states
- Worker/training failure or interruption states
- URL reload/deep-link behavior
- Persistence across refresh
- Keyboard/focus/hover/disabled/readout legibility
- Drawer/menu keyboard escape behavior, focus return, and focus trapping where modal behavior is used
- No clipping, overlap, white scrollbars, or horizontal overflow
