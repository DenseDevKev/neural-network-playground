# Signal Atelier feature acceptance

This is the complete feature map for the approved 20-screen vision. Implementation,
automated coverage, actual browser inspection, and public release are distinct
claims. Candidate receipts must name the exact revision and commands; this map
alone is not a release receipt.

## Screen and interaction coverage

| Screen | Surface and required behavior | Primary automated coverage |
|---|---|---|
|01|Data → square activation maps → prediction; theme, transport, pan/zoom, selection, filters|NetworkGraph renderer/painter tests; precision-acceptance, precision-lab-layout|
|02|Dataset draft: 11 generators, sample presets, seed/noise/split/counts, deterministic preview|Setup editor/draft tests; atelier-critical-flows unified draft|
|03|Network draft: 9 features, 0–6 hidden layers, 1–16 neurons, 8 activations, 4 initializers|Recipe catalog/draft tests; Setup editor|
|04|Training draft: 3 optimizers and their parameters, schedules, regularization, clipping, regression loss|Canonical schema/worker tests; draft combination and validation cases|
|05|Neuron inspector: accepted map/statistics, strongest signed links, independently labeled steps, focus return|Canvas/SVG selection and inspection tests; precision-acceptance|
|06|Real EMA/full train/test evaluation/objective histories and evidence age|LossChart/model tests; playground-smoke; layout stability|
|07|Boundary overlays and full held-out binary/multiclass confusion data|Boundary/ConfusionMatrix tests; atelier-critical-flows task transitions|
|08|Saved records: 20 limit, 120-code-point names, apply/download/delete/rename|RunHistory/experiment memory tests; saved-run-concurrency; smoke|
|09|Exactly two explicit records; differences filter; matching dataset/objective ranking; stored history plots|Saved comparison tests; actual visual comparison capture|
|10|Ten-lesson library with canonical presets and reset consequence|Lesson registry/controller tests; all ten critical browser journeys|
|11|Active guided panel, real-session completion, Previous/Continue/Restart/Exit and target navigation|Lesson controller races; all ten actual browser completions|
|12|Trace coordinates/target/full output/loss/step and activation statistics/histograms|Inspection controller/model tests; paused actual worker traces|
|13|Forward/backward gradients and parameter probe with true axes/objective/sample basis|Inspection worker/model races; actual browser backprop/probe|
|14|Pseudocode/NumPy/TF.js, guarded learned parameters, JSON/link, staged import, clipboard recovery|Config/Code tests; critical browser import/export downloads|
|15|Three outputs, full softmax tuple, class labels/regions, task-aware diagnostics|Multiclass scientific tests; critical browser full tuple|
|16|Continuous regression legends, linear output/losses, no classification-only controls|Regression/boundary/inspection tests; critical browser task switch|
|17|Exact failed artifact retained; Retry/Download/Discard; reject overlapping capture|Save controller/persistence tests; precision-acceptance quota recovery|
|18|Checkpoint selection previews only; paused explicit Restore, optimizer/revision/timeline validation|Worker checkpoint tests; browser preview/restore and pause|
|19|Mobile Results: plots, legends, controls, local overflow, safe-area access|Responsive/browser touch tests and mobile visual evidence|
|20|Mobile Setup: full draft, visible errors, Apply/Cancel access and keyboard-aware forms|Zoom/short viewport tests; unified draft and mobile visual evidence|

Each reference state requires both Light and Dark actual-app captures. Generated
reference values are visual intent only; screenshot fixtures use deterministic
recipes and real accepted engine results. Review captures before accepting image
regression baselines.

## Cross-cutting state gates

- Theme: initial System, live device changes, explicit choice/reload, unavailable
  storage; preserve recipe, generation, revision, step, checkpoints, saved records.
- Layout migration: retain old key, code preference and lesson invitation; preserve
  the experiment URL, including skip-link and local navigation.
- Draft: across all three tabs, raw incomplete input, no silent clamp, entire
  candidate validation, unchanged/invalid/busy Apply, Cancel, stale base, exact
  worker acknowledgement, preparation and synchronization failures, guarded exit.
- Visualization: both renderers, zero/multiclass/maximum networks, missing grids,
  zoom/pan/resized coordinates, keyboard selection and focus restoration. App is
  the sole demand writer and requests only visible artifacts.
- Diagnostics: paused one-off requests, reachable Pause, duplicate guards, retry,
  model/sample/request invalidation, no delayed response replacing new evidence.
- Persistence: capacity, Unicode titles, valid/rejected/legacy/incompatible records,
  exact pending-artifact retry/download, explicit delete, cross-tab concurrency.
- Lessons: ten starts and completions in real worker sessions, explicit reset,
  new targets and Apply instructions, stale training evidence rejected.
- Utilities: canonical JSON/code validators, invalid import keeps experiment,
  clipboard denial has selectable text, checkpoint selection does not restore.
- Recovery: persistent worker/artifact/storage/compatibility errors, bounded copy,
  no false success or hidden destructive state changes.

## Visual and accessibility matrix

Use 1440, 1280, 1024, 768, 390, and 360px; also test 760/1200 breakpoint boundaries,
landscape, short available height, 200% zoom, reduced motion and forced colors.
Check no page-level horizontal overflow, readable axes/legends, 44px controls,
keyboard/focus/menus/dialogs, long names, loading and failure states, stable metric
layout, and reachable mobile comparison/diagnostics/import/export/checkpoints.

`atelier-visual-evidence.spec.ts` is the opt-in deterministic capture suite; its
reviewed artifacts and instructions are recorded with the visual receipt. The
normal functional suite includes accessibility, mobile touch, resize, document
zoom, and layout stability. Automated axe results do not substitute for actual
keyboard and visual inspection.

## Candidate and release gates

Run unchanged correctness, browser, recovery, bundle, and performance contracts:

```text
node --test scripts/*.test.mjs
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm test:bundle
pnpm test:e2e
pnpm test:e2e:recovery
pnpm test:perf
```

Qualify Chromium and WebKit on normal preview and the actual project subpath
without isolation headers. After fault tests, rebuild normally and check that
fault requests are ignored. Retain gzip caps 152245 entry, 7373 inspection,
234161 all JavaScript including worker. Initial shared chunks must not escape entry
accounting. Preserve the paired same-runner performance gate and 1.20 maximum factor.

Before merge, qualify the exact integration revision with the release workflow.
If main changes, integrate and rerun affected gates. After merging the single
integration PR, require main CI, exact-revision Pages deployment, and both-browser
public checks of theme/training/saves/lessons/exports/layout/assets. Record PR,
merge SHA, deployment URL, checks, visual evidence, and rollback by reverting the
integration PR. Saved records and old layout storage remain intact.
