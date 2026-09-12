# State Ownership Contract

This contract records the state boundaries retained by the Precision Lab branch and its behavior-preserving maintenance pass. The refactor must preserve the V2 experiment document, URL behavior, engine and worker protocols, checkpoints, and saved runs.

The release continuation is specified in `docs/superpowers/specs/2026-07-16-release-ready-product-shell-design.md`. Its measured pre-change evidence and browser diagnosis are recorded in `docs/qa/2026-07-16-product-shell-baseline.md`. The implemented shell rules and extension points are documented in `docs/architecture/product-shell.md`.

## Ownership

| Owner | Authoritative responsibility |
|---|---|
| `usePlaygroundStore` | The shareable V2 document (`recipe` plus `document.view`), preparation/import/URL compatibility, and the temporary visualization-demand delivery cache. |
| `useTrainingStore` | Volatile runtime status; worker-authored live-evidence references; train/test points; React invalidation versions; configuration synchronization and errors; trained-recipe identity; checkpoint metadata; pause/worker errors; and session speed. |
| `metricHistoryBuffer` | Bounded metric-series storage. `useTrainingStore` publishes its independent invalidation versions. |
| Worker | Authoritative model state and scientific artifacts, including checkpoint payloads. |
| `frameBuffer` | Accepted main-thread typed-array and provenance cache. Its React invalidation versions live in `useTrainingStore`; it is not a second scientific authority. |
| `useLayoutStore` | Persisted local workspace navigation, disclosure, code-tab, and audience preferences. These values never enter the V2 document, URL, checkpoints, exports, or saved runs. |
| `experimentMemoryStore` | Saved, rejected, and pending run artifacts, plus their persistence/compatibility bookkeeping. |
| Feature hooks/components | Request state, selections, focus/hover state, and ephemeral drawers. These do not belong in the shareable document or persisted layout state. |

Rendered visibility is authoritative for visualization demand. `usePlaygroundStore.demand` is only the delivery cache, and `useTraining` is the sender to the worker.

## Current adapter and compatibility boundaries

- `App` is the sole production writer of visualization demand. It derives demand from resolved shell visibility, writes the delivery cache, and `useTraining` remains the only sender to the worker. `InspectionPanel` no longer mutates demand on mount.
- Inspection store, frame-buffer, selector, and worker-bridge access is confined to `useInspectionPanelController`; `inspectionPanelModel` is React-free and `InspectionPanelView` consumes display-safe model data and commands.
- Decision-boundary store and frame-buffer access is confined to `useDecisionBoundaryModel`; `deriveDecisionBoundaryModel` is React-free and the canvas component owns only painting and responsive sizing.
- `usePlaygroundStore.featuresUI` remains an existing renderer/capability-switch location; this roadmap does not relocate it.
- Ephemeral `App` drawers remain local. Request lifecycle and selection state remain local to feature hooks.

## Precision Lab production composition and maintenance guards

`App`/`CompatiblePlayground` owns the production `useTraining`, `useSaveCurrentRun`, `useNetworkSelectionController`, and `useDecisionBoundaryController` instances. `PrecisionLabShell` receives display content and commands. The canonical live boundary stays mounted; shell navigation never becomes an experiment URL fragment.

Transport and History share the App-owned save controller. Capture obtains one artifact; persistence retry reuses the store's pending artifact. Exact-artifact retry is not a solution to independent-tab lost updates; that separate finding remains in `BUGS-TO-REVIEW.md`.

`nn-forge/dependency-boundaries` in `scripts/eslint-architecture.mjs` rejects engine/shared imports of UI, engine imports of shared, production imports of tests/prototypes, and newly introduced controller-owner modules. It checks relative/package-alias static imports, re-exports, type imports, literal `import()` and literal `require()`; nonliteral computed imports are outside its resolution scope. Runtime controller counts still require integration tests.

Type-only imports of controller interfaces are permitted. The existing `RunHistoryPanel` save wrapper and `NetworkGraph`/Canvas/SVG selection wrappers are exact, documented exceptions, not permission to add more owners. Remove exceptions only with consumer-proven retirement. `NetworkGraphSVG` remains a live fallback.

Production explicit `any` is an error. Test/spec/benchmark entries, `__tests__`, `__benchmarks__`, and `apps/web/src/test` retain intentionally malformed-input exceptions. Runtime `apps/web/src/testing/e2eFaults.ts` is production and is not exempt.

Maintenance tooling is not shipped in the app. Start with `docs/maintenance/README.md` for commands, source receipts and connector-friendly evidence.

## Resolved transition searches

- The retired `BuildRunShell`, `RegionShell`, `MainArea`, and `Sidebar` presentation paths have no production or test consumers. `App` composes the live display exports from `PrecisionLabContent` through `PrecisionLabShell` and retains controller ownership.
- The unused `deriveVisualizationDemand.historyDrawerOpen` argument has been removed.

## Historical July product-shell baseline evidence

These measurements are historical receipts, not current qualification or current budget values. Current executable bundle caps are in `scripts/check-web-bundle-gzip.mjs`; use `pnpm test:bundle`. Current candidate status belongs in the root living execution plan.

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

## Historical stop/go budgets after July Tasks 2 and 3

- `InspectionPanel` gzip: at most 6,349 bytes (baseline + 2 KiB).
- Main-entry gzip: at most 142,005 bytes (baseline + 1 KiB).
- All-JavaScript gzip, including the worker: at most 223,010 bytes (2% growth). The non-worker subtotal remains a diagnostic, not a separate gate.
- Idle medians: at most 120% of baseline—6.4681 ms, 9.2825 ms, 1,141.6355 ms/100, 1,165.7752 ms/100, 722.1182 ms/50, 699.8734 ms/50, 4.7824 ms, and 1.3714 ms respectively, in the table order above.
- All focused/full tests, lint, build, and Chromium/WebKit smoke scenarios must pass.

## Deferred candidates

These are candidates only; do not delete or relocate them in this slice.

- `usePlaygroundStore.dataset` and `regenerateData`: baseline searches find only the store implementation and its unit test, but removal requires a dedicated consumer search and compatibility proof.
- Deprecated layout aliases (`layout`, `phase`, `activeTabLeft`, `activeTabRight`, and their setters): persisted-state sanitization and compatibility tests still accept them. Retain until a dedicated compatibility migration proves older local layout state remains safe without them.
- `NetworkGraphSVG`: a live runtime fallback selected by `featuresUI.canvasNetworkGraph`, not a dead path.

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

For historical reproduction only, the July measurements used default command-line gzip. These commands do not define the current automated gate:

```bash
for file in apps/web/dist/assets/*.js; do bytes=$(gzip -c "$file" | wc -c | tr -d ' '); printf '%s %s\n' "$(basename "$file")" "$bytes"; done
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' ! -name 'training.worker-*.js' -exec gzip -c {} \; | wc -c
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' -exec gzip -c {} \; | wc -c
```
