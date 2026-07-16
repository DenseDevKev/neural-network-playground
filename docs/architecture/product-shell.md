# Product-Shell Architecture

NN·FORGE uses one product shell, one experiment document, and one training
runtime. Beginner, Explore, and Lab are visibility and guidance profiles over
that shared system. They are not separate applications, permission tiers, or
alternate business-logic paths.

The state authority behind the shell is defined in
[`state-ownership.md`](state-ownership.md). The approved design is recorded in
[`2026-07-16-release-ready-product-shell-design.md`](../superpowers/specs/2026-07-16-release-ready-product-shell-design.md).

## Profile capability model

`apps/web/src/productShell/audienceProfiles.ts` is the source of truth for
profile labels, descriptions, core modules, core evidence, disclosure defaults,
and guidance density. `apps/web/src/productShell/visibleShell.ts` resolves the
actual visible target. Components must consume those tables and resolvers rather
than add scattered profile conditionals.

| Profile | Core Build modules | Core Run evidence | Advanced default | Guidance |
|---|---|---|---|---|
| Beginner | Data, Network | Boundary, Loss | Closed | High |
| Explore | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Closed | Standard |
| Lab | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Open | Compact |

Current Recipe/Run, Topology, transport, Presets, Lessons, and History remain
available in every profile. Opening Advanced Tools exposes the complete
applicable union: Features, Hyperparameters, Configuration, Confusion,
Inspection, and Code.

Hidden modules are unmounted, not CSS-hidden. Their recipe values are preserved
and must remain discoverable through Current Recipe and an Advanced Tools
recovery path; changing visibility never resets or silently replaces them.

## Layout persistence and transitions

`useLayoutStore` persists local workspace state under `nn-playground-layout`:

- Build/Run view and selected recipe/evidence target
- selected code-export tab
- `audienceMode`
- `advancedToolsOpen`

These fields never enter the V2 experiment document, URL, checkpoints, exports,
or saved runs. Missing or invalid profiles sanitize to Explore. Missing or
invalid disclosure state uses the selected profile's default. A valid manually
collapsed Lab state survives reload. If persisted navigation targets a hidden
tool, hydration opens Advanced Tools instead of replacing the target or moving
focus.

An explicit profile change applies that profile's disclosure default. If the
new visibility would hide the active target, the store falls back atomically to
Data in Build or Boundary in Run and keeps deprecated aliases synchronized.
Direct lesson or explanation navigation to a hidden tool opens Advanced Tools
without changing profile.

Profile changes must preserve the recipe, URL/hash, model identity and training
step, checkpoints, saved runs, export selection, and ephemeral drawers. Tests
and browser release evidence treat those values as invariants.

## Advanced Tools interaction contract

Advanced Tools is one inline disclosure, not a miscellaneous drawer and not a
duplicate control path.

- The trigger is outside the evidence tablist and exposes `aria-expanded` and
  `aria-controls`.
- The controlled region explains what opening it reveals.
- Opening the disclosure alone does not request diagnostic worker artifacts.
- The evidence tablist contains only visible tabs and uses roving focus with
  Arrow keys, Home, and End.
- Selecting Inspection or Code opens the disclosure atomically.
- Collapsing while an advanced target is active resolves to Data or Boundary
  before the hidden content unmounts.
- Button- and Escape-driven closure restore focus to the disclosure trigger.
  A nested dialog or concept disclosure handles its own Escape first.
- Hydration never steals focus.
- Narrow layouts retain reachable controls and at least 44px interactive
  targets. Nonessential motion is disabled under `prefers-reduced-motion`.

Presets, Lessons, and History remain distinct transient drawers. Their open
state stays local to `App` and is not persisted with the shell.

## Visualization demand flow

Rendered shell visibility is authoritative:

```text
audience profile + Advanced Tools + selected evidence
    -> resolveVisibleEvidenceView
    -> App derives and caches VisualizationDemand
    -> useTraining sends demand
    -> worker produces requested scientific artifacts
```

The same resolved evidence target drives rendering, accessibility context, and
demand. Legacy `history` resolves to Boundary. `App` is the sole production
demand writer; `InspectionPanel` does not mutate demand on mount.

## Terminology catalog

`apps/web/src/concepts/conceptCatalog.ts` is the React-free source of truth for
the bounded domain vocabulary. Each entry has a stable ID, canonical term,
plain definition, aliases, related concepts, applicable profiles, and
difficulty. Extended explanations, examples, UI targets, and documentation
links are optional. `ConceptHelp` renders accessible on-demand help while
keeping essential labels and instructions visible without hover.

To add or change a concept:

1. Add or update its stable ID in `CONCEPT_IDS`.
2. Add one frozen `CONCEPTS` entry with an accurate plain definition, unique
   canonical/alias terms, valid related IDs, profiles, and difficulty.
3. Reuse that entry through `ConceptHelp`; do not duplicate its definition in a
   component, tooltip, empty state, or onboarding path.
4. Use the active profile's guidance level only to control optional detail. Do
   not rewrite the scientific meaning by profile.
5. Extend catalog tests for completeness, collisions, relationships, lookup,
   filtering, and stable order, then add an accessible rendering test at the
   consuming surface.

Do not add a provider, copy CMS, i18n layer, or generic view-model framework
solely to extend this catalog.

## Compatibility and loading

- The engine, worker protocol, frame-buffer semantics, V2 schema, URL format,
  checkpoints, and saved-run formats are unchanged.
- Deprecated layout aliases remain synchronized while older persisted layout
  state and compatibility tests still depend on them.
- Configuration, Inspection, Code Export, and Run History retain meaningful
  lazy boundaries. Hidden advanced panels are not mounted; final release
  evidence records the resulting chunks and initial JavaScript cost.
- The catalog and profile tables are small static modules shared by all modes;
  there are no duplicated mode-specific application trees.

## Known boundaries

- Profiles change visible tools and explanation density only. They do not lock
  capabilities or weaken validation and safety rules.
- Profile and disclosure preferences are device-local and intentionally do not
  travel with a shared experiment URL.
- The initial catalog contains six high-value concepts. Search, broader
  onboarding, localization, and copy/profile tuning are separate evidence-led
  extensions.
- Legacy layout aliases and fallback render files remain compatibility
  candidates even though current production consumer searches are empty.
