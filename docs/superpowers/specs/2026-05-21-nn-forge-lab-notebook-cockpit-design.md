# NN.FORGE Lab Notebook + Cockpit Design

Date: 2026-05-21

## Goal

Reinvent the NN.FORGE interface while keeping it grounded in the existing product. The direction is a Lab Notebook + Cockpit: a fast experiment workspace by default, with a diagnostic cockpit that becomes more prominent when the model is running, paused after a run, or inspected in focus mode.

This document is a design specification, not an implementation plan. It does not authorize a rewrite, dependency migration, model-behavior change, persistence change, or new feature expansion beyond the scope described here.

## Product Grammar

The design should make these objects legible and distinct:

- **Current recipe:** the editable experiment setup, including dataset, features, network, hyperparameters, optimizer, learning rate, batch size, and loss.
- **Trained snapshot:** the model state produced by the last run or step sequence. This may differ from the current recipe after the user edits configuration.
- **Active run:** the current operating state of training for the trained snapshot, including status, step, epoch, metrics, pause reason, freshness, and staleness.
- **Evidence:** the views that explain a run or snapshot, including boundary, loss, confusion, inspection, code, and history.
- **Saved runs:** persisted records that can be restored, titled, compared, and used as references against the current recipe, trained snapshot, or active run.

## Section 1: Product Spine

The reinvented NN.FORGE should become a **Lab Notebook + Cockpit**: a fast experiment workspace by default, with a diagnostic cockpit that becomes more prominent when the model is running, paused after a run, or inspected in focus mode.

The main loop should be:

```text
Set recipe -> Run experiment -> Inspect evidence -> Compare -> Adjust
```

"Recipe" means the current dataset, features, network, and hyperparameters as one coherent experiment setup, not scattered controls. "Cockpit" means topology, boundary, loss, confusion, activations, and inspection become one connected diagnostic system instead of separate panels competing for attention.

This keeps the app grounded because it does not invent a new product category. It reorganizes what already exists around the actual user job: trying a model, seeing what happened, and deciding what to change next.

## Section 2: Layout Model

This section defines where things go. Use one mental model across modes: **Recipe / Topology / Evidence / Timeline**.

On desktop, the strongest default should be a three-zone lab:

- **Left: Recipe rail** for dataset, features, network, hyperparameters, and presets, grouped as the experiment setup.
- **Center: Topology stage** as the visual anchor, with build controls attached to the graph and run overlays appearing only when useful.
- **Right: Evidence column** for current run summary, outputs, comparison, and history.
- **Bottom: Timeline strip** for Start/Pause/Resume, step/reset/speed, progress, restore, and current active run state.

Modes should keep the same product grammar while changing emphasis:

- **Dock:** balanced recipe, topology, evidence, and timeline.
- **Focus:** larger topology and one selected evidence view, with recipe reduced to compact reference/editing.
- **Grid:** grouped evidence families for scanning relationships.
- **Split:** clearer Build and Run phases that still share recipe, topology, evidence, and timeline language.

On mobile, keep the current "dock only" constraint, but turn it into a deliberate vertical sequence: **Recipe summary -> Topology -> Evidence -> Timeline**, with controls collapsed into compact sections rather than pretending desktop modes exist.

## Section 3: Mode Behavior

This section defines what each mode is for.

**Dock** is the default lab bench for ordinary configuration changes and quick experiments.

**Focus** is deep work on one experiment: topology stays large, one evidence surface gets priority, and recipe controls become compact reference/editing rather than a tall scroll fight.

**Grid** is the comparison wall. It should group evidence into readable families: topology, output behavior, learning curves, diagnostics, and history/comparison. It is for scanning relationships, not editing every setting or watching every diagnostic at full fidelity.

**Split** shows Build and Run as two sides of the same experiment, not two separate apps. Build emphasizes recipe and topology construction. Run emphasizes topology behavior, outputs, loss, confusion, and inspection. The phase banner should act like a calm mode label, not an alert.

**Mobile** stays dock-only, but should feel intentional: a vertical lab notebook where the user can review the recipe, inspect topology, check evidence, and control training without horizontal panel complexity.

## Section 4: Design System Direction

Use a **judicious hybrid design system** extracted from repeated NN.FORGE needs, not a generic component library built in advance.

The goal is to stabilize the app's visual and interaction language: surfaces, control rhythm, readouts, tabs, charts, diagnostics, comparison views, and mobile stacking. The system should live inside the app and serve the Lab Notebook + Cockpit model. External accessible primitives are allowed only where they solve meaningful interaction complexity, such as tabs, popovers, menus, tooltips, dialogs, or roving keyboard controls. No dependency should be added just to restyle buttons or panels.

The design system should define:

- **Tokens:** surfaces, text hierarchy, borders, focus, active state, warning/success/info, graph colors, chart colors, spacing, density, radius, elevation.
- **Primitives:** shell, panel, toolbar, tab strip, segmented control, action button, chip, slider row, metric readout, run status, empty state.
- **Patterns:** recipe section, evidence panel, diagnostic group, timeline strip, run comparison card, mobile section stack.

The design system governs the surrounding interface: panels, controls, tabs, metrics, timeline, evidence frames, and comparison surfaces. The topology stage should remain more bespoke because it is the product's primary visual anchor, with a custom internal visual contract where that helps the product feel distinct.

Core rule: **Systemize before redesigning, but do not abstract prematurely.** If a visual or interaction problem appears in multiple places, solve it through a token, primitive, or pattern. If it appears once, keep the fix local.

Tokens and primitives should be introduced at the point of repeated use and documented where the app consumes them. The design-system layer should remain app-local unless a later implementation plan justifies otherwise.

## Section 5: Concrete UX Improvements

First make the current recipe, trained snapshot, and active run legible and operable, then make evidence and comparison more powerful.

First wave:

**1. Recipe Summary**

Answers: "What experiment am I looking at?"

Show dataset, feature count, hidden layers, activation, optimizer, learning rate, batch size, loss, and whether the current recipe/config differs from the last trained snapshot. The summary should include Recipe Drift / Config Drift when the editable recipe no longer matches the trained snapshot or active run.

**2. Current Run Card**

Answers: "What is the state of this run?"

Show status, step, epoch, train/test loss, accuracy, generalization gap, pause reason, trained snapshot identity, freshness/staleness, and drift. The Run Card explains state; it does not own primary transport controls.

**3. Timeline Strip**

Answers: "How do I operate it?"

Unify primary action, step/reset/speed, progress, restore, and current active run status into one coherent experiment transport. The Timeline Strip operates state; the Current Run Card explains state.

Second wave:

**4. Evidence System**

Unify Boundary, Loss, Confusion, Inspection, Code, and History as sibling evidence views with clearer empty states and quieter chrome. Follow the Evidence View Taxonomy in Section 6 rather than forcing identical layouts.

**5. Diagnostic Cockpit**

Connect topology more clearly to outputs during run/focus contexts: weights/activations, boundary behavior, loss trend, confusion shifts, and layer inspection should feel like one model-state diagnostic layer. The cockpit should emerge through state-aware emphasis and already available model/run data unless a later implementation plan explicitly justifies new data hooks, worker protocol changes, or model-side changes.

**6. Comparison Loop**

After the current recipe, trained snapshot, and active run are legible, improve run history into current run/snapshot vs previous run/snapshot, best saved run vs current run/snapshot, restore, annotate/title, and architecture/metric comparison workflows. History is the record surface; Comparison is the workflow that uses history.

## Section 6: State, Interaction, And Edge-Case Rules

These rules make the product grammar hard to misread when current recipe, trained snapshot, active run, evidence, and saved runs diverge.

### 1. State Divergence / Recipe Drift

When the editable current recipe differs from the trained snapshot or active run, the UI must communicate Recipe Drift / Config Drift with a visible but non-alarming indicator.

The drift indicator should answer:

- What changed?
- Is the active/trained model using the old value or the new value?
- What action resolves the mismatch?

Recipe drift should appear in:

- Recipe Summary
- Current Run Card
- Relevant changed recipe fields

Existing evidence must not silently appear to belong to a newly edited recipe if it actually belongs to an older trained snapshot.

### 2. Topology State Contract

The topology stage has three possible state meanings:

- **Draft Blueprint:** editable architecture/config implied by the current recipe.
- **Trained Snapshot:** model state produced by the last completed run or step sequence.
- **Live Run:** model currently being trained.

When the current recipe diverges from the trained snapshot or live run, topology must visibly distinguish draft vs trained/live state. During active training, diagnostic overlays should privilege the live run. During editing, build affordances should privilege the draft recipe. In mixed states, the distinction must be labeled.

### 3. Cockpit Emergence Pattern

The cockpit should not appear as a separate dashboard. It should emerge through state-aware emphasis:

- Topology gains relevant overlays.
- Current Run Card becomes more prominent.
- Active evidence view reflects the current model state.
- Timeline Strip reflects live status.

State emphasis:

- **Idle/editing:** recipe and topology are primary.
- **Running:** topology, run status, and live evidence are primary.
- **Paused-after-training:** evidence and comparison become more prominent.
- **Focus:** topology and one selected evidence view become the paired diagnostic surface.

Cockpit connections should use already available model/run data unless a later implementation plan explicitly justifies new data hooks, worker protocol changes, or model-side changes.

### 4. Run Card / Timeline Relationship

The **Current Run Card explains state**. It should answer:

- What run/snapshot the user is looking at.
- What metrics are fresh or stale.
- Why the run is paused, stopped, failed, or stale.

The **Timeline Strip operates state**. It owns primary transport controls:

- Start
- Pause
- Resume
- Step
- Reset
- Speed
- Progress
- Status

Restore may appear in the Timeline Strip only as a secondary contextual action tied to the selected saved run or snapshot. It should not compete with primary transport controls.

### 5. Mobile Timeline Rule

On mobile, the Timeline Strip should become a sticky bottom transport during active or paused runs. It may collapse into a compact bar, but primary controls and run status must remain reachable without scrolling. Secondary controls can expand from the bar or move into compact sections.

### 6. Evidence View Taxonomy

Boundary, Loss, Confusion, Inspection, Code, and History are sibling evidence views, but should not be forced into identical layouts.

They should share:

- Title
- Run/snapshot context
- Freshness/staleness state
- Empty/loading/error state
- Compact explanation of what the view proves

Preserve their different interaction models:

- **Boundary:** plot-based
- **Loss:** time-series
- **Confusion:** matrix-based
- **Inspection:** hierarchical
- **Code:** textual
- **History:** record-based

Systemize framing and state language, not the internal interaction model of every evidence view.

### 7. History vs Comparison

History is the record surface. It contains:

- Saved runs
- Titles
- Timestamps
- Restored states
- Run metadata

Comparison is the workflow that uses history. It answers:

- What changed?
- Which performed better?
- What should I try next?

Keep comparison focused on current run/snapshot vs previous run/snapshot, best saved run vs current run/snapshot, restore, title/annotate, and architecture/metric differences. Do not let comparison become a separate analytics product.

### 8. Error And Failure States

The UI should cover these edge states:

- Invalid recipe
- Failed run
- Worker crash
- Unavailable evidence
- Stale metrics
- Incompatible saved run

Errors should appear where the user can act:

- Recipe errors in Recipe rail / Recipe Summary
- Run failures in Timeline Strip and Current Run Card
- Evidence failures inside the affected evidence view

Error states should explain the next useful action without adding new ML concepts or hiding the topology stage.

### 9. Grid And Split Clarifications

Grid is not a wall of equal panels. It groups evidence into families: topology, output behavior, learning curves, diagnostics, and history/comparison. Grid is for scanning relationships, not editing settings or watching every diagnostic at full fidelity. During active training, grid diagnostics may use simplified or throttled updates to preserve responsiveness.

Split should not become two separate apps. Build retains compact topology and recipe state. Run retains compact recipe summary and trained snapshot identity. The phase banner changes emphasis, not the underlying product grammar.

## Section 7: Implementation Boundaries And Non-Goals

The reinvention should **reorganize and systemize the existing product around the experiment loop**, not replace NN.FORGE with a new app.

This is not a full rewrite, a new product category, a generic design-system migration, a visual-only reskin, or a feature explosion. Every change should support the loop:

```text
Set recipe -> Run experiment -> Inspect evidence -> Compare -> Adjust
```

Boundaries:

- Do not rebuild the app from scratch.
- Do not introduce a large external UI framework just for aesthetics.
- Do not bury the topology stage; it remains the product's visual anchor.
- Do not turn the timeline strip into a dashboard.
- Do not add new ML concepts unless they clarify the existing experiment loop.
- Do not prioritize mobile polish before the desktop experiment loop is stable.
- Do not treat comparison/history as a separate analytics product.
- Do not optimize for screenshot appeal at the expense of experiment clarity.
- Do not change model behavior, training lifecycle, graph rendering, or worker protocol unless a later implementation plan explicitly justifies it.
- Do not change persistence, URL/config format, or saved run contracts unless migration and compatibility risks are explicitly handled.
- Do not remove educational clarity; make it contextual, not absent.

Positive boundary:

Changes should primarily clarify the relationship between **current recipe**, **trained snapshot**, **active run**, **evidence**, and **saved runs**. That is the new product grammar.

## Section 8: Validation And Success Criteria

The reinvention succeeds only if the experiment loop becomes easier to understand and operate. Do not judge it by whether it looks more modern in screenshots.

Success criteria:

- **Experiment legibility:** A user can tell what recipe is active, whether it differs from the last trained snapshot, and what active run state they are looking at.
- **Run operability:** A user can start, pause, resume, step, reset, restore, adjust speed, and understand run status from the timeline strip without hunting across the UI.
- **Evidence clarity:** A user can identify Boundary, Loss, Confusion, Inspection, Code, and History as evidence views, understand what each explains, and know what action produces empty-state content.
- **Diagnostic connection:** During training, paused-after-training, focus, and split-run contexts, topology, metrics, boundary, loss, confusion, and inspection read as connected views of the same model state.
- **Comparison usefulness:** A user can compare current run/snapshot vs previous run/snapshot or best saved run vs current run/snapshot, restore or title a run, and understand architecture/metric differences.
- **Decision support:** After inspecting evidence or comparison, a user can identify at least one plausible next adjustment to the recipe.
- **Scope discipline:** The implementation improves `set recipe -> run experiment -> inspect evidence -> compare -> adjust` without changing model behavior, training lifecycle, graph interactions, worker protocol, persistence contracts, URL/config compatibility, or saved run compatibility.
- **Responsive continuity:** Mobile keeps the same product grammar in a simpler stack; it should not introduce separate concepts or hide the core run controls.
- **Accessibility continuity:** Focus states, labels, keyboard shortcuts, disabled states, readout legibility, and chart/graph interpretability are preserved or improved.

Release-blocking validation:

The first implementation wave is not successful unless users can:

- Distinguish current recipe from trained snapshot.
- Operate a run from the Timeline Strip.
- Identify stale/config-drift state.
- Understand which evidence belongs to which run or snapshot.

Stress tests:

1. **Mid-run tweak:** User changes learning rate during active training. UI must show the active run is still using the prior value while the current recipe has pending drift.
2. **Config change after training:** User trains a model, edits hidden layers, then views evidence. UI must show that evidence belongs to the trained snapshot, not the edited recipe.
3. **Paused or failed run:** User pauses or hits a failure state. Timeline Strip and Current Run Card must explain what happened and what action is available.
4. **Saved-run reference:** User opens a saved run. UI must show whether they are viewing a reference, restoring it, or comparing it to the current recipe/active run context.
5. **Mobile active run:** User can pause/resume without scrolling away from topology or evidence.

Practical validation should cover: desktop dock, focus, grid, and split layouts; split build/run behavior; mobile compact dock behavior; Start -> Pause -> Resume; config changes after training; empty evidence states; and at least one current run/snapshot-vs-saved comparison path.
