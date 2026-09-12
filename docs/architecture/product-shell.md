# Signal Atelier product shell

NN·FORGE has one experiment document and one training runtime. `AtelierShell`
presents Playground, Saved runs, and Lessons. New visitors arrive at
Playground → Network with a paused default experiment and an optional lesson
invitation. See [state ownership](state-ownership.md) and the
[design contract](signal-atelier-design.md).

## Navigation and guidance

| Destination | Views |
|---|---|
| Playground | Setup, Network, Results, Inspect |
| Setup | Dataset, Network, Training |
| Results | Boundary, Learning progress, Errors |
| Inspect | Trace, Activations, Gradients |
| Saved runs | Local records and explicit two-record comparison |
| Lessons | Ten-lesson library and the active guided journey |

The utility menu opens Export/import, Session checkpoints, Guidance, or
Shortcuts and help. One utility dialog owns focus at a time. Escape closes the
innermost dismissible surface; closing restores focus to its trigger. The
worker-recovery overlay remains a separate blocking recovery state.

Guidance More, Standard, and Compact map to the retained Beginner, Explore, and
Lab explanation densities. Guidance never hides a feature. Native finite-choice
selects, keyboard tablists, persistent inline errors, and visible focus are shared
across destinations. Navigation never starts, pauses, or resets the experiment.

## Local preferences and compatibility

`useLayoutStore` stores navigation, code format, guidance, and lesson invitation
preferences under `nn-playground-layout-v2`. It reads and migrates the old
`nn-playground-layout` key without deleting it. Old Build selections map into
Setup; Run evidence maps into Results, Inspect, or Saved runs. Old configuration and code
selections create a session-only export request consumed by the shell.

Deprecated layout names are accepted only by the migration reader. Live layout
state and explanation actions use the current destination and tab interfaces. The V2 experiment fragment remains exclusively a shareable experiment
document; shell navigation and theme never enter it. Modal state, comparison
selection, draft values, export requests, and active lesson execution are not
persisted in experiment data. Saved-list position and comparison selection remain
stable while navigating within the current mounted session.

System, Light, and Dark are handled separately by the theme store and a
before-paint initializer. System follows live device changes. Explicit choices
survive reload when storage is available; unavailable storage leaves the current
session usable. Canvas and SVG use the same theme tokens without resetting the
worker or changing evidence identity.

## One recipe draft and one configuration transaction

`App` owns `useRecipeDraft`. Dataset, Network, and Training edit one candidate,
including raw incomplete numeric text. Whole-candidate validation uses canonical
schema limits. Invalid combinations have field and cross-field errors; values
are never silently clamped. Apply is unavailable while unchanged, invalid, or
submitting. Cancel discards all unsubmitted edits.

Apply submits one complete recipe through the existing worker synchronization
path with source `setup` and an expected-base recipe identity. Success requires
the matching accepted worker configuration; training remains paused. An old
draft cannot overwrite a newer imported, lesson, or saved recipe. Preparation
failures retain edits; submitted failures retain the synchronization retry path.
An import whose dialog unmounts still releases its own failed preparation
transaction, without clearing a newer operation.

Leaving dirty Setup opens Apply / Discard / Stay and page unload is guarded.
Dataset, activation, layer, and preset shortcuts stage changes in this same
draft. Imported JSON is separately staged, validated, summarized, and applied
explicitly; it uses the same publication and worker-acknowledgement boundary.

## Controllers and visualization demand

`App` owns the single production training, save, boundary, and network-selection
controllers. The shell receives display content and commands. Expensive plots
mount only while visible; navigating away does not dispose of the runtime.

```text
active destination + workspace tab + task + compact network region
    → App derives VisualizationDemand
    → usePlaygroundStore delivery cache
    → useTraining sends demand
    → worker produces requested scientific artifacts
```

App is the sole production demand writer. Inspection never writes demand on
mount. On a compact viewport, Data, Network, and Prediction region tabs mount
only the chosen region; hidden neurons do not request activation grids. A hidden
inspector cannot keep those requests alive. Each plotted artifact retains its
own recipe, generation, revision, and step provenance.

Both graph renderers share geometry, selection, filters, and scientific colors.
Graph-local pan/zoom and scrolling preserve target size for the largest networks.
Structural summaries and keyboard neuron buttons provide access beyond Canvas.
Selecting or closing an inspector restores the relevant selection focus.

## Scientific utilities and local evidence

Saved runs retain their existing format and 20-record limit. Save, rename, and
delete retain cross-tab locking and refresh. A failed save retains the exact
artifact; Retry and Download pending evidence use it without recapture. A new
capture is blocked until it is resolved. Applying a saved recipe creates a fresh
model, while Download evidence preserves the stored observation.

Comparison requires exactly two explicit records. Loss ranking requires matching
dataset and objective identities. Stored EMA, evaluation, and objective histories
retain their separate meanings and consistent chart legends.

Trace, activation statistics, backpropagation, and parameter probes use the
existing worker requests with duplicate/stale response guards. One-off requests
require a paused model and expose Pause while running. Checkpoint selection only
previews metadata; explicit paused Restore restores parameters and optimizer
state. Checkpoints are session-only; future shuffles may differ.

## Lessons and terminology

The library retains all ten canonical lessons and their presets. The active
lesson controller survives target navigation, and completion observes its actual
model session. Instructions point to Setup and Apply changes. Desktop lessons
use a side panel; compact lessons leave the target control reachable.

`concepts/conceptCatalog.ts` is the React-free vocabulary authority. The nine
concepts are available through Shortcuts and help; contextual help and training
explanations reuse them. Optional extended definitions/examples follow Guidance.
Do not duplicate scientific definitions in new tooltips or onboarding copy.

## Loading and maintained boundaries

Education, workspace utilities, code exports, inspection, and results plots have
meaningful lazy boundaries. Sharing a lazy chunk does not mount hidden panels.
Engine, worker protocol, frame-buffer semantics, V2 schema, URL format, saved-run
format, and checkpoint format remain unchanged. The replaced Build/Run shell,
unused panels, and old shell styles are retired after consumer checks; the SVG
network renderer remains a live fallback.

Runtime tokens live in `styles/atelier.css`. `styles/components.css` contains
shared feature styling. Avoid reintroducing old shell overrides or hardcoded
Canvas/SVG theme colors. The full feature and release gates live in the
[Atelier acceptance checklist](../qa/signal-atelier-acceptance.md).
