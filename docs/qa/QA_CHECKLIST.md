# NN.FORGE QA checklist

This checklist separates local developer checks from release qualification. Do not mark a release green from only one layer.

## 1. Fast local correctness

```bash
pnpm install --frozen-lockfile
node --test scripts/*.test.mjs
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm test:bundle
```

Historical package counts at the September 6 release baseline (current counts belong in the candidate receipt):

- engine: 497 tests;
- shared: 334 tests;
- web: 1,043 tests;
- total package tests: 1,874;
- helper tests after performance comparator: 88.

The build may still print Vite's generic configured 200 kB chunk warning. The reviewed JavaScript release contract is `pnpm test:bundle`, not the generic warning.

## 2. JavaScript bundle contract

`pnpm test:bundle` must pass all fixed limits:

| Dimension | Maximum gzip bytes |
|---|---:|
| application entry | 152,245 |
| InspectionPanel | 7,373 |
| all JavaScript including worker | 234,161 |

The checker fails closed for missing/ambiguous entry or Inspection chunks, symlinks/traversal, invalid measurements, and one-byte overruns. CSS/fonts are not part of these JavaScript dimensions.

## 3. Normal browser suite

```bash
pnpm test:e2e
```

This launches the normal local isolated preview and runs Chromium + WebKit with zero retries.

Historical September 6 release-baseline expectation (Atelier adds new cases):

- 42 expected/passed;
- 6 intentional skips;
- 0 unexpected;
- 0 flaky.

The six skips are four external-hosting-only contracts plus two fault-enabled cases excluded from a normal build.

## 4. Real static project-subpath suite

Build normally, start the checked-in fixture, then point Playwright to its project base:

```bash
pnpm build
node scripts/serve-release-fixture.mjs apps/web/dist 4174
PLAYWRIGHT_BASE_URL=http://127.0.0.1:4174/neural-network-playground/ pnpm test:e2e
```

Historical September 6 release-baseline result (Atelier adds new cases):

- 46 expected/passed;
- 2 intentional fault-enabled skips;
- 0 unexpected;
- 0 flaky.

Required hosting evidence includes:

- actual worker request below the project path;
- non-isolated fallback operation;
- lazy Inspection/Code/History/Configuration chunks;
- paired evaluation after a manual step;
- canonical V2 shared recipe URL;
- fresh-context reload with same recipe/architecture and fresh runtime;
- no collapse from project path to origin root;
- same-origin local font resources with no Google-font requests.

## 5. Fault recovery

```bash
pnpm test:e2e:recovery
```

The fault-enabled build must pass the recovery scenario in Chromium and WebKit. After that test, rebuild without `VITE_E2E_FAULTS` and verify the normal build ignores fault requests:

```bash
env -u VITE_E2E_FAULTS pnpm build
pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --grep '@fault-disabled'
```

Do not suppress console/resource failures or fake successful font responses to make recovery green.

## 6. Performance — local calibration vs release qualification

### Local historical absolute gates

```bash
pnpm test:perf
```

This still executes the historical engine constants plus the worker scientific-trust tests. The engine constants were calibrated from five medians on an isolated development machine and remain unchanged. A different host can fail those frozen engine values even when no regression exists.

Do **not** raise those constants merely because another machine is slower.

### Release performance gate

The release workflow uses the historical policy itself, not a foreign machine's absolute result:

1. exact intake baseline and candidate run on the same `macos-15` Apple Silicon runner;
2. five complete runs per revision;
3. execution order alternates baseline/candidate;
4. engine candidate median must be <= 120% of same-runner baseline median for:
   - predictGrid;
   - predictGridInto;
   - predictGridWithNeurons;
   - predictGridWithNeuronsInto;
   - Adam + L2 + clipping;
   - SGD zero-gradient adapter;
5. forced paired evaluation must remain <= 250 ms;
6. save capture must remain <= 500 ms;
7. exact SHAs, environment, exits, raw logs, and comparison output are retained.

The comparator is `scripts/compare-performance-reference.mjs`. It requires exactly five baseline and five candidate logs and fails closed for incomplete/ambiguous evidence.

## 7. State-integrity checks

Before accepting a release, explicitly verify:

- skip-link keyboard and pointer activation do not change the V2 experiment URL;
- skip-link activation does not change generation, revision, or training step;
- destination, workspace, guidance, and theme changes do not mutate the experiment;
- unified Setup commits once, Cancel leaves the active experiment untouched, and dirty navigation is guarded;
- saved-run records do not imply trained-parameter persistence;
- Apply Saved Recipe creates the recipe state rather than pretending to restore a model;
- paired train/test evaluation remains distinct from batch/EMA evidence;
- evaluation age/drift language remains present where required.

## 8. Accessibility

For normal release checks:

- keyboard-only navigation works;
- skip link is usable and preserves the experiment hash;
- focus remains visible;
- major controls have appropriate accessible names;
- evidence tabs/panels expose correct semantics;
- axe checks remain green;
- reduced-motion behavior does not remove essential state feedback;
- 200% zoom remains usable.

Signal Atelier acceptance additionally requires 44px touch targets, all20reference states in both themes, and the viewport/layout-stability matrix in [signal-atelier-acceptance.md](signal-atelier-acceptance.md).

## 9. Release-verification workflow

Before merge, the exact candidate SHA must have a green `Release verification` run with all six jobs:

- source-evidence;
- correctness;
- browsers (preview);
- browsers (subpath);
- browsers (recovery);
- performance.

Do not treat a previous SHA's green run as proof for a later code/config change.

## 10. After merge

Require normal `main` CI to pass. Record the resulting `main` SHA.

The existing deployment workflow consumes the successful `main` CI SHA; do not manually deploy an untested different revision.

## 11. Live deployment verification

Once GitHub Pages reports an actual `page_url`, run the existing browser suite against it:

```bash
PLAYWRIGHT_BASE_URL=<actual-page-url> pnpm test:e2e
```

At minimum verify in Chromium and WebKit:

- initial load;
- no page/console errors;
- same-origin fonts;
- training worker / non-isolated fallback;
- step/evaluation flow;
- Inspection, Code, History, Configuration;
- shared V2 recipe URL and fresh-context reload;
- skip-link state integrity;
- project subpath preservation.

A successful static fixture is not a substitute for this live check, and an expected GitHub Pages URL is not a deployment receipt.
