# State Ownership Contract

This contract fixes the state boundaries for the staged product-shell refactor. The refactor must preserve the V2 experiment document, URL behavior, engine and worker protocols, checkpoints, and saved runs.

## Ownership

| Owner | Authoritative responsibility |
|---|---|
| `usePlaygroundStore` | The shareable V2 document (`recipe` plus `document.view`), preparation/import/URL compatibility, and the temporary visualization-demand delivery cache. |
| `useTrainingStore` | Volatile runtime status; worker-authored live-evidence references; train/test points; React invalidation versions; configuration synchronization and errors; trained-recipe identity; checkpoint metadata; pause/worker errors; and session speed. |
| `metricHistoryBuffer` | Bounded metric-series storage. `useTrainingStore` publishes its independent invalidation versions. |
| Worker | Authoritative model state and scientific artifacts, including checkpoint payloads. |
| `frameBuffer` | Accepted main-thread typed-array and provenance cache. Its React invalidation versions live in `useTrainingStore`; it is not a second scientific authority. |
| `useLayoutStore` | Persisted workspace navigation, disclosure, code-tab, and audience preferences. Disclosure and audience fields are added in later roadmap stages. |
| `experimentMemoryStore` | Saved, rejected, and pending run artifacts, plus their persistence/compatibility bookkeeping. |
| Feature hooks/components | Request state, selections, focus/hover state, and ephemeral drawers. These do not belong in the shareable document or persisted layout state. |

Rendered visibility is authoritative for visualization demand. `usePlaygroundStore.demand` is only the delivery cache, and `useTraining` is the sender to the worker.

## Current exceptions

- `App` derives demand from shell visibility while `InspectionPanel` also toggles layer-stat and histogram demand on mount. This temporary dual-writer arrangement remains through the two pilots; Task 4 makes the shell the sole writer after direct/legacy consumers are proven non-production.
- `InspectionPanel` and `DecisionBoundary` still read stores, `frameBuffer`, selectors, and worker bridges directly. Tasks 2 and 3 confine those dependencies to their feature adapters without changing ownership.
- `usePlaygroundStore.featuresUI` remains an existing renderer/capability-switch location; this roadmap does not relocate it.
- Ephemeral `App` drawers remain local. Request lifecycle and selection state remain local to feature hooks.

## Baseline evidence

- Commit: `5e9ea79b1cd9c14592c1eca30f5fdf8a8e097126`.
- Status: the isolated branch was clean before `docs/superpowers/plans/2026-07-14-targeted-product-shell-refactor.md` was added.
- Focused direct Vitest: `InspectionPanel` plus `DecisionBoundary`; 2 files, 25 tests passed.
- Full suite: engine 480, shared 325, web 747; 1,552 tests passed.
- Lint and production build exited 0. Browser smoke passed all 6 Chromium/WebKit scenarios.

Production gzip, measured with default `gzip -c` after `pnpm build`:

| Artifact | Bytes |
|---|---:|
| Main entry | 140,981 |
| `InspectionPanel` lazy chunk | 4,301 |
| Non-worker JavaScript | 162,936 |
| All JavaScript including worker | 218,638 |

Idle-machine performance medians:

| Measurement | Baseline |
|---|---:|
| Web forced paired evaluation | 5.3901 ms |
| Web save capture | 7.7354 ms |
| Engine `predictGrid` | 951.3629 ms / 100 |
| Engine `predictGridInto` | 971.4793 ms / 100 |
| Engine `predictGridWithNeurons` | 601.7652 ms / 50 |
| Engine `predictGridWithNeuronsInto` | 583.2278 ms / 50 |
| Adam `applyGradients` | 3.9853 ms |
| SGD `applyGradients` | 1.1428 ms |

## Stop/go budgets after Tasks 2 and 3

- `InspectionPanel` gzip: at most 6,349 bytes (baseline + 2 KiB).
- Main-entry gzip: at most 142,005 bytes (baseline + 1 KiB).
- All-JavaScript gzip, including the worker: at most 223,010 bytes (2% growth). The non-worker subtotal remains a diagnostic, not a separate gate.
- Idle medians: at most 120% of baseline—6.4681 ms, 9.2825 ms, 1,141.6355 ms/100, 1,165.7752 ms/100, 722.1182 ms/50, 699.8734 ms/50, 4.7824 ms, and 1.3714 ms respectively, in the table order above.
- All focused/full tests, lint, build, and Chromium/WebKit smoke scenarios must pass.

## Deferred candidates

These are candidates only; do not delete or relocate them in this slice.

- `usePlaygroundStore.dataset` and `regenerateData`: baseline searches find only the store implementation and its unit test, but removal requires a dedicated consumer search and compatibility proof.
- Deprecated layout aliases (`layout`, `phase`, `activeTabLeft`, `activeTabRight`, and their setters): persisted-state sanitization accepts them, and `RegionShell` still has production consumers. Retain until consumers and compatibility tests migrate.
- The default legacy `MainArea` and `Sidebar` render paths: retained for direct-render tests and fallback contexts until production-consumer searches and replacement tests prove removal safe.
- `NetworkGraphSVG`: a live runtime fallback selected by `featuresUI.canvasNetworkGraph`, not a dead path.
- `deriveVisualizationDemand.historyDrawerOpen`: currently accepted but does not affect demand; Task 4 removes it with its tests.

## Exact verification commands

Run from the repository root. Focused files must use direct Vitest invocation; `pnpm test -- <file>` still runs the full web suite.

```bash
git rev-parse HEAD
git status --short
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/InspectionPanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx --pool=forks --reporter=dot
pnpm lint
pnpm test
pnpm build
pnpm test:perf
pnpm test:e2e
git diff --check
git diff --cached --check
```

After the build, reproduce the bundle measurements with default gzip compression:

```bash
for file in apps/web/dist/assets/*.js; do bytes=$(gzip -c "$file" | wc -c | tr -d ' '); printf '%s %s\n' "$(basename "$file")" "$bytes"; done
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' ! -name 'training.worker-*.js' -exec gzip -c {} \; | wc -c
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' -exec gzip -c {} \; | wc -c
```
