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
- **Saved runs:** persisted records that can be restored, titled, compared, and used as references against the current experiment.

## Section 1: Product Spine

The reinvented NN.FORGE should become a **Lab Notebook + Cockpit**: a fast experiment workspace by default, with a diagnostic cockpit that becomes more prominent when the model is running, paused after a run, or inspected in focus mode.

The main loop should be:

```text
Set recipe -> Run experiment -> Watch topology/outcomes -> Compare result -> Adjust recipe
```

“Recipe” means the current dataset, features, network, and hyperparameters as one coherent experiment setup, not scattered controls. “Cockpit” means topology, boundary, loss, confusion, activations, and inspection become one connected diagnostic system instead of separate panels competing for attention.

This keeps the app grounded because it does not invent a new product category. It reorganizes what already exists around the actual user job: trying a model, seeing what happened, and deciding what to change next.

## Section 2: Layout Model

Use one mental model across modes: **Recipe / Topology / Evidence / Timeline**.

On desktop, the strongest default should be a three-zone lab:

- **Left: Recipe rail** for dataset, features, network, hyperparameters, and presets, grouped as the experiment setup.
- **Center: Topology stage** as the visual anchor, with build controls attached to the graph and run overlays appearing only when useful.
- **Right: Evidence column** for current run summary, outputs, comparison, and history.
- **Bottom: Timeline strip** for Start/Pause/Resume, step/reset/speed, progress, restore, and current active run state.

Modes should become different emphases of the same model:

- **Dock:** everyday lab bench, balanced configuration/topology/evidence.
- **Focus:** topology and one evidence view get more space; recipe becomes reference rather than a scrolling wall.
- **Grid:** comparison wall for multiple evidence surfaces, but with fewer visible boxes and stronger grouping.
- **Split:** Build and Run become clearer work phases, but still share the same recipe/topology/evidence/timeline language.

On mobile, keep the current “dock only” constraint, but turn it into a deliberate vertical sequence: **Recipe summary -> Topology -> Evidence -> Timeline**, with controls collapsed into compact sections rather than pretending desktop modes exist.

## Section 3: Mode Behavior

Each mode should support the same experiment loop, but with a different emphasis.

**Dock** should be the default lab bench: recipe on the left, topology in the center, evidence on the right, timeline below. It should feel like the place to make ordinary changes and run quick experiments.

**Focus** should become “deep work on one experiment.” Topology stays large, one evidence surface gets priority, and recipe controls become compact reference/editing rather than a tall scroll fight.

**Grid** should become the comparison wall. Instead of many boxed panels shouting at once, it should group evidence into readable families: topology, output behavior, learning curves, diagnostics, history. It is for scanning relationships, not editing every setting.

**Split** should make Build and Run feel like two sides of the same experiment, not two separate apps. Build emphasizes recipe and topology construction. Run emphasizes topology behavior, outputs, loss, confusion, and inspection. The phase banner should act like a calm mode label, not an alert.

**Mobile** should stay dock-only, but should feel intentional: a vertical lab notebook where the user can review the recipe, inspect topology, check evidence, and control training without horizontal panel complexity.

## Section 4: Design System Direction

Use a **judicious hybrid design system** extracted from repeated NN.FORGE needs, not a generic component library built in advance.

The goal is to stabilize the app’s visual and interaction language: surfaces, control rhythm, readouts, tabs, charts, diagnostics, comparison views, and mobile stacking. The system should live inside the app and serve the Lab Notebook + Cockpit model. External accessible primitives are allowed only where they solve meaningful interaction complexity, such as tabs, popovers, menus, tooltips, dialogs, or roving keyboard controls. No dependency should be added just to restyle buttons or panels.

The design system should define:

- **Tokens:** surfaces, text hierarchy, borders, focus, active state, warning/success/info, graph colors, chart colors, spacing, density, radius, elevation.
- **Primitives:** shell, panel, toolbar, tab strip, segmented control, action button, chip, slider row, metric readout, run status, empty state.
- **Patterns:** recipe section, evidence panel, diagnostic group, timeline strip, run comparison card, mobile section stack.

The topology stage should remain more bespoke because it is the product’s primary visual anchor. Surrounding panels, controls, diagnostics, metrics, tabs, charts, and comparison views should be systemized more aggressively, but the graph/cockpit center can keep custom treatment where that helps the product feel distinct.

Core rule: **Systemize before redesigning, but do not abstract prematurely.** If a visual or interaction problem appears in multiple places, solve it through a token, primitive, or pattern. If it appears once, keep the fix local.

## Section 5: Concrete UX Improvements

First wave:

**1. Recipe Summary**

Answers: “What experiment am I looking at?”

Show dataset, feature count, hidden layers, activation, optimizer, learning rate, batch size, loss, and whether the current recipe/config differs from the last trained snapshot.

**2. Current Run Card**

Answers: “What is the state of this run?”

Show status, step, epoch, train/test loss, accuracy, generalization gap, pause reason, freshness/staleness, and trained snapshot identity.

**3. Timeline Strip**

Answers: “How do I operate it?”

Unify primary action, step/reset/speed, progress, restore, and current run status into one coherent experiment transport.

Second wave:

**4. Evidence System**

Unify Boundary, Loss, Confusion, Inspection, Code, and History as sibling evidence views with clearer empty states and quieter chrome.

**5. Diagnostic Cockpit**

Connect topology more clearly to outputs during run/focus contexts: weights/activations, boundary behavior, loss trend, confusion shifts, and layer inspection should feel like one model-state diagnostic layer.

**6. Comparison Loop**

After the current experiment and trained snapshot are legible, improve run history into current vs previous, best vs current, restore, annotate/title, and architecture/metric comparison workflows.

## Section 6: Implementation Boundaries And Non-Goals

The reinvention should **reorganize and systemize the existing product around the experiment loop**, not replace NN.FORGE with a new app.

This is not a full rewrite, a new product category, a generic design-system migration, a visual-only reskin, or a feature explosion. Every change should support the loop:

```text
Set recipe -> Run experiment -> Inspect evidence -> Compare -> Adjust
```

Boundaries:

- Do not rebuild the app from scratch.
- Do not introduce a large external UI framework just for aesthetics.
- Do not bury the topology stage; it remains the product’s visual anchor.
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

## Section 7: Validation And Success Criteria

The reinvention succeeds only if the experiment loop becomes easier to understand and operate. Do not judge it by whether it looks more modern in screenshots.

Success criteria:

- **Experiment legibility:** A user can tell what recipe is active, whether it differs from the last trained snapshot, and what active run state they are looking at.
- **Run operability:** A user can start, pause, resume, step, reset, restore, adjust speed, and understand run status from the timeline strip without hunting across the UI.
- **Evidence clarity:** A user can identify Boundary, Loss, Confusion, Inspection, Code, and History as evidence views, understand what each explains, and know what action produces empty-state content.
- **Diagnostic connection:** During training, paused-after-training, focus, and split-run contexts, topology, metrics, boundary, loss, confusion, and inspection read as connected views of the same model state.
- **Comparison usefulness:** A user can compare current vs previous or best vs current, restore or title a run, and understand architecture/metric differences.
- **Decision support:** After inspecting evidence or comparison, a user can identify at least one plausible next adjustment to the recipe.
- **Scope discipline:** The implementation improves `set recipe -> run experiment -> inspect evidence -> compare -> adjust` without changing model behavior, training lifecycle, graph interactions, worker protocol, persistence contracts, URL/config compatibility, or saved run compatibility.
- **Responsive continuity:** Mobile keeps the same product grammar in a simpler stack; it should not introduce separate concepts or hide the core run controls.
- **Accessibility continuity:** Focus states, labels, keyboard shortcuts, disabled states, readout legibility, and chart/graph interpretability are preserved or improved.

Practical validation should cover: desktop dock, focus, grid, and split layouts; split build/run behavior; mobile compact dock behavior; Start -> Pause -> Resume; config changes after training; empty evidence states; and at least one current-vs-saved comparison path.
