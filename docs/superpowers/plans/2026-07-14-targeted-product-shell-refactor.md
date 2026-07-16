# Targeted Product-Shell Refactor — Full Staged Roadmap

> **Historical roadmap:** This document records the original staged contract.
> Current release execution is tracked in
> [`2026-07-16-release-ready-product-shell.md`](2026-07-16-release-ready-product-shell.md),
> and current runtime boundaries are documented in
> [`../../architecture/product-shell.md`](../../architecture/product-shell.md).
> References below to an App/InspectionPanel dual demand writer and live
> `RegionShell` consumers describe the pre-shell baseline, not current
> production ownership.

## Summary

Refactor the UI through two behavior-preserving pilots, then build progressive disclosure, a bounded terminology catalog, and shared Beginner/Explore/Lab profiles.

Keep one shell, one state graph, and the existing engine, V2 experiment document, worker protocol, checkpoints, saved runs, and URL behavior. Do not introduce a generic view-model framework or three separate mode-specific interfaces.

## Guardrails and ownership

- Preserve the existing uncommitted Scientific Trust plan edit.
- No engine-math, worker-protocol, V2 schema, URL, checkpoint, or saved-run migrations.
- Additive local layout persistence is allowed through the existing sanitizer.
- Refactor branches preserve behavior; shell and mode branches change only their declared visibility and navigation behavior.
- Log—but do not combine into these branches—the existing multiclass-overlay, regression-copy, split/test-visibility, and inspection disabled-state defects.
- Do not remove deprecated aliases, `dataset/regenerateData`, or legacy render paths until searches and tests prove they have no production consumers.

Document these boundaries:

- `usePlaygroundStore`: shareable V2 document, including recipe and shareable `document.view`; preparation/import/URL compatibility; temporary visualization-demand delivery cache.
- `useTrainingStore`: runtime status, evidence references, train/test points, frame versions, configuration synchronization, trained-recipe identity, checkpoints, errors, and session speed.
- `metricHistoryBuffer`: bounded metric-series storage.
- Worker: authoritative producer of scientific artifacts.
- `frameBuffer`: accepted main-thread typed-array/provenance cache; invalidation versions remain in `useTrainingStore`.
- `useLayoutStore`: persisted workspace navigation, disclosure, code tab, and audience mode.
- `experimentMemoryStore`: saved, rejected, and pending run artifacts.
- Feature hooks/components: request state, selections, focus, hover, and ephemeral drawer state.
- Visualization demand: visible shell state is authoritative; `useTraining` remains the only sender. The current App/InspectionPanel dual-writer arrangement is a documented temporary exception until the shell stage.

## Staged implementation

### Task 1: Baseline and ownership contract

Branch: `codex/refactor-state-ownership`

- Save this plan at `docs/superpowers/plans/2026-07-14-targeted-product-shell-refactor.md`.
- Add `docs/architecture/state-ownership.md` with the ownership rules and known exceptions above.
- Record the current commit/status, focused and full test results, production chunk sizes, total JavaScript gzip size, and idle-machine performance medians.
- Record dead-path and compatibility candidates without deleting them.
- Use direct Vitest invocation for focused files; do not use the package `test -- <file>` form because it still runs the entire web suite.

### Task 2: InspectionPanel pilot

Branch: `codex/refactor-inspection-panel`

Keep the existing `InspectionPanel.tsx` import path as a thin public wrapper. Add a feature-local controller, pure model, and presentational view:

```ts
type InspectionTraceSource = 'train' | 'test';

interface InspectionPanelController {
    model: InspectionPanelDisplayModel;
    commands: {
        selectHistogramLayer(index: number): void;
        selectTraceSource(source: InspectionTraceSource): void;
        selectSampleIndex(index: number): void;
        requestTrace(): Promise<void>;
        requestBackprop(): Promise<void>;
        requestLandscape(): Promise<void>;
    };
}
```

- Put all store, frame-buffer, scientific-selector, and worker-bridge access in `useInspectionPanelController`.
- Keep request IDs, loading/errors, results, selections, and cancellation logic local to the controller.
- Make `inspectionPanelModel.ts` React-free and return display-safe primitives rather than raw worker responses or frame artifacts.
- Make `InspectionPanelView` consume only the model and commands.
- Subscribe to `layerStatsVersion` instead of broad `frameVersion`; read point snapshots imperatively when commands run.
- Preserve current raw-versus-effective sample-index clamping and current button-disabled behavior.
- Retain the panel’s demand lifecycle temporarily for direct/legacy rendering compatibility.
- Redistribute the existing tests instead of duplicating them:
  - Controller: stale success/failure for all three requests, model/source changes, response mismatch, clamping, and newer-request-wins.
  - Model: labels, maxima, histograms, provenance, summaries, zero-spread data, and effective indices.
  - View: accessibility, loading, disabled/empty states, and event forwarding without stores or worker mocks.
  - Wrapper integration: demand lifecycle and one worker-to-display flow.

### Task 3: DecisionBoundary visualization adapter

Branch: `codex/refactor-decision-boundary`

- Export `DecisionBoundaryProps` while preserving the runtime component API.
- Preserve and re-export `getDecisionOverlayCopy` and `classifyPointFromGrid`.
- Add a React-free `deriveDecisionBoundaryModel` and a focused `useDecisionBoundaryModel`.
- Use a discriminated display model with `empty`, `scalar`, `multiclass`, and `unavailable` states.
- Preserve typed-array identity; do not copy grids.
- Subscribe only to `outputGridVersion` and `multiclassBoundaryVersion`.
- Read `frameBuffer` once for each derived model so painting, summaries, and accessibility consume the same snapshot.
- Keep ResizeObserver, device-pixel-ratio sizing, and canvas painting inside `DecisionBoundary`.
- Test scalar smooth/discrete rendering, overlays, split visibility, binary/multiclass/regression selection, malformed multiclass artifacts, accessible summaries, typed-array identity, exact version refresh, and no refresh for unrelated frame changes.

Mandatory stop/go after stages 2–3:

- Store/worker/frame dependencies are confined to the two adapters.
- Presentational tests require no global stores or worker setup.
- No new layer merely forwards every prop unchanged.
- Existing component APIs and workflows remain compatible.
- Inspection lazy gzip grows by no more than 2 KiB, entry gzip by no more than 1 KiB, and total JavaScript gzip by no more than 2%.
- Idle performance medians remain within 120% of baseline.
- All tests, builds, and Chromium/WebKit smoke scenarios pass.

If any condition fails, stop before shell work and simplify or abandon the extraction pattern.

### Task 4: Shared shell and Advanced Tools

Branch: `codex/refactor-shell-disclosure`

- Replace the Header’s “More” drawer with one persisted `advancedToolsOpen` disclosure.
- Closed Explore shell:
  - Build: Current Recipe, Data, Topology, Network, Features, Hyperparameters.
  - Run: Current Run, Recipe, Topology, Boundary, Loss, Confusion.
- Open disclosure additionally shows Configuration in Build and Inspect/Code in Run.
- Remove the duplicate Code/Configuration “More” path only after consumer searches pass.
- Add one shared `resolveVisibleEvidenceView` used by both rendering and visualization-demand derivation; legacy `history` resolves to Boundary.
- Remove the unused `historyDrawerOpen` demand argument.
- After proving direct/legacy consumers are non-production, remove InspectionPanel’s demand writer and make App shell visibility the sole demand writer.
- Selecting Inspect or Code opens Advanced Tools atomically.
- Explicit collapse while an advanced view is active selects Boundary before unmounting.
- Hydration with a persisted advanced target opens disclosure without stealing focus.
- Put the disclosure outside the evidence tablist with `aria-expanded` and `aria-controls`.
- Implement roving evidence-tab focus with Arrow keys, Home, and End.
- Closing by button or Escape returns focus to the disclosure trigger before advanced content unmounts.
- Opening disclosure alone must not request diagnostic artifacts.

### Task 5: Bounded terminology catalog

Branch: `codex/refactor-concept-copy`

Add a React-free typed catalog for:

```ts
type ConceptId =
    | 'data-loss'
    | 'training-objective'
    | 'decision-boundary'
    | 'activation'
    | 'gradient'
    | 'checkpoint';

interface ConceptEntry {
    inlineLabel: string;
    titleLabel: string;
}
```

- Migrate only genuinely repeated strings from CurrentRunCard, LossChart, InspectionPanelView, ExperimentStateContext, and Header.
- Preserve existing rendered wording exactly.
- Leave one-off copy local.
- Do not add an `ExplanationDensity` abstraction, provider, i18n system, or copy CMS yet.
- Test catalog completeness and unchanged rendered DOM copy.

### Task 6: Shared audience profiles

Branch: `codex/refactor-audience-modes`

Add app-local types and a pure profile table:

```ts
type AudienceMode = 'beginner' | 'explore' | 'lab';

interface AudienceProfile {
    label: string;
    coreBuildModules: readonly BuildModuleId[];
    coreEvidenceViews: readonly EvidenceViewId[];
    advancedDefaultOpen: boolean;
}
```

Add `audienceMode`, `advancedToolsOpen`, `setAudienceMode`, and `setAdvancedToolsOpen` to `useLayoutStore`. Add a compact Header `<select aria-label="Audience mode">`.

Profile matrix:

| Mode | Core Build | Core Run | Advanced default |
|---|---|---|---|
| Beginner | Data, Network | Boundary, Loss | Closed |
| Explore | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Closed |
| Lab | Same as Explore | Same as Explore | Open |

Current Recipe/Run, Topology, transport, Presets, Lessons, and History remain available in every mode. Advanced Tools always exposes the complete union: Features, Hyperparameters, Configuration, Confusion, Inspect, and Code as applicable. Hidden content is unmounted, not CSS-hidden.

Transition rules:

- Explore is the default for new, missing, or invalid persisted state.
- An explicit mode change applies that profile’s disclosure default.
- If closing disclosure hides the active Build/Run target, fall back atomically to Data/Boundary and synchronize legacy aliases.
- Direct navigation or a lesson action targeting a hidden tool opens Advanced Tools without changing mode.
- Hydration preserves a persisted hidden target by opening disclosure instead of changing the target.
- A valid manually collapsed Lab state survives reload.
- Mode changes never alter Build/Run view, recipe, runtime, checkpoints, saved runs, URL, exports, or ephemeral drawers.
- Persist mode/disclosure only in `nn-playground-layout`; never add them to V2 experiment state.
- New copy is limited to “Mode,” “Beginner,” “Explore,” “Lab,” and a short note that mode changes visible tools only. Mode-specific explanatory rewriting remains a later evidence-backed copy project.

## Verification and acceptance

For each branch:

```bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run <focused-files> --pool=forks --reporter=dot
pnpm lint
pnpm test
pnpm build
pnpm test:perf
pnpm test:e2e
git diff --check
```

Add coverage for:

- Layout sanitization, backward-compatible hydration, manual Lab collapse, hidden-target hydration, fallbacks, and legacy aliases.
- Exact modules/tabs for every profile and the full Advanced union.
- Guided lesson and explanation actions opening hidden Beginner tools.
- Rendering and demand using the same resolved evidence view.
- Advanced-open alone producing no inspection/confusion demand.
- Focus restoration, keyboard tab navigation, and axe checks in all profiles.
- A cross-browser paused-run scenario that records URL/hash, step/model identity, checkpoint timeline, and saved-run count; cycles all modes and disclosure states; and proves those invariants remain unchanged.
- Existing training, pause/step, presets, checkpoint restore, and saved-run workflows.

After implementation, run five usability sessions—three neural-network newcomers and two experienced users—covering preset selection, training/pause, boundary interpretation, train/test evidence, and locating Advanced Tools. Treat findings as input to a separate copy/profile-tuning slice, not as a gate for shipping the three shared profiles.

## Assumptions

- Modes are flexible profiles, never locked product tiers.
- Explore is the default.
- Terminology work is infrastructure-only and preserves current wording.
- Known visualization and inspection UX defects remain separate bug-fix branches.
- Performance tests are regression gates, not proof that this UI refactor improves interaction speed.
