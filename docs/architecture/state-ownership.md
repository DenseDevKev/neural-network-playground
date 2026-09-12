# State ownership contract

Signal Atelier retains the V2 document, experiment URL, engine and worker
protocols, checkpoints, and saved-run formats. Its current presentation contract
is [product-shell.md](product-shell.md); historical receipts below remain dated.

## Ownership

| Owner | Authoritative responsibility |
|---|---|
| `usePlaygroundStore` | Shareable V2 recipe and document view; preparation/import/URL compatibility; temporary demand delivery cache. |
| `useTrainingStore` | Volatile runtime, worker-authored evidence references, data points and invalidation versions, configuration synchronization/errors, trained recipe identity, checkpoint metadata, pause reasons, and speed. |
| `metricHistoryBuffer` | Bounded metric series, independently invalidated by the training store. |
| Worker | Model, optimizer, scientific artifacts, and checkpoint payloads. |
| `frameBuffer` | Accepted main-thread typed arrays and provenance; not a second scientific authority. |
| `useLayoutStore` | Versioned local navigation, guidance, code format, and lesson invitation preferences; legacy preference migration and session export requests. |
| Theme store | System/Light/Dark preference and resolved theme; no experiment or runtime state. |
| `useRecipeDraft` | App-owned session candidate, base identity, raw numeric text, dirty/validation/submission state, and atomic Apply/Cancel. |
| `useAtelierViewport` | Effective available viewport, including document zoom and virtual keyboard; no scientific state. |
| `experimentMemoryStore` | Saved, rejected, legacy/incompatible, and exact pending artifacts plus persistence bookkeeping. |
| Feature hooks/components | Request lifecycle, focus/hover/selection, dialog state, comparison selection, and active lesson execution. |

No layout, theme, modal, draft, comparison, or lesson-execution state enters the
persisted experiment document. Old layout storage is read without deletion.

## Production composition and adapters

`App`/`CompatiblePlayground` owns the production `useTraining`,
`useSaveCurrentRun`, `useNetworkSelectionController`, and
`useDecisionBoundaryController` instances. `AtelierShell` composes their display
models and commands. Hidden visualizations unmount while controllers survive.
App alone derives demand from actual visibility, including the compact focused
network region, and `useTraining` alone sends the cached demand to the worker.

Inspection store/frame-buffer/worker access stays in
`useInspectionPanelController`; its React-free display model feeds the view.
Boundary state access stays in `useDecisionBoundaryModel`; its pure derivation
feeds a canvas responsible only for painting and size. Canvas and SVG network
renderers remain supported and share geometry, tokens, and selection semantics.

Transport and Saved runs share one save controller. Capture obtains one artifact;
persistence Retry and Download pending evidence reuse it exactly. Native locking
and digest checks preserve independent-tab saves, renames, and deletions.

A recipe commit has one preparation/publication/configuration acknowledgement
lifecycle. Draft Apply uses expected-base identity and `setup` source. Submitted
failures use synchronization recovery; unsubmitted preparation errors retain the
draft. Staged imports validate before preview and prepare on explicit Apply.
An unmounted import releases only its own failed pending transaction.

## Maintenance guards

`nn-forge/dependency-boundaries` in `scripts/eslint-architecture.mjs` rejects
engine/shared imports of UI, engine imports of shared, production imports of
tests/prototypes, and new controller-owner modules. It checks static imports,
re-exports, type imports, literal `import()` and literal `require()`; nonliteral
computed imports are outside resolution. Integration tests verify runtime owner
counts and demand behavior.

Type-only controller interfaces are permitted. Existing RunHistoryPanel save and
NetworkGraph renderer selection wrappers are exact documented exceptions, not
permission to add owners. `NetworkGraphSVG` is a live fallback. Production explicit
`any` is an error; malformed-input test exceptions do not exempt runtime fault
injection code. Maintenance tooling is not shipped; see
[maintenance commands](../maintenance/README.md).

Replaced shell components and styles were retired only after import-closure and
compatibility checks. Deprecated layout setters and duplicate live fields were removed after consumer
checks. The migration reader still accepts old persisted names, and explanation
actions now navigate and focus the current destination and tab controls.

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
- `NetworkGraphSVG`: a live runtime fallback selected by `featuresUI.canvasNetworkGraph`, not a dead path.

## Exact verification commands

Run from the repository root. Focused files must use direct Vitest invocation; `pnpm test -- <file>` still runs the full web suite.

```bash
git rev-parse HEAD
git status --short
node --test scripts/*.test.mjs
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/InspectionPanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx --pool=forks --reporter=dot
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm test:bundle
pnpm test:perf
pnpm test:e2e
pnpm test:e2e:recovery
git diff --check
git diff --cached --check
```

For historical reproduction only, the July measurements used default command-line gzip. These commands do not define the current automated gate:

```bash
for file in apps/web/dist/assets/*.js; do bytes=$(gzip -c "$file" | wc -c | tr -d ' '); printf '%s %s\n' "$(basename "$file")" "$bytes"; done
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' ! -name 'training.worker-*.js' -exec gzip -c {} \; | wc -c
find apps/web/dist/assets -maxdepth 1 -type f -name '*.js' -exec gzip -c {} \; | wc -c
```
