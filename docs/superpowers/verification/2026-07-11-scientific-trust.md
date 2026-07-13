# Scientific Trust Verification Record

## Baseline

- Captured: `2026-07-11T02:14:55-04:00`
- Branch: `codex/scientific-trust-v2`
- Base snapshot: `6b83026925b68a5be42ed9e2d600cb5f210aed59`
- Environment: same isolated worktree and development machine required for the post-change comparison.

### Repository gates

| Gate | Result | Evidence |
|---|---|---|
| `pnpm test` | PASS | 74 files, 958 tests: engine 320, shared 125, web 513 |
| `pnpm lint` | PASS | ESLint exit 0, no findings |
| `pnpm build` | PASS | TypeScript and Vite exit 0; existing `>200 kB` chunk warning recorded |
| `pnpm test:perf` | PASS | 2 benchmark files, 4 tests on each of five runs |
| `git diff --check` | PASS | exit 0, no output |

### Five-run raw performance measurements

Times are milliseconds per operation. Grid totals printed by the benchmark were divided by their declared iteration counts: 100 for standard grids and 50 for neuron grids.

| Run | predictGrid | predictGridInto | Adam + L2 + clip | SGD | predictGridWithNeurons | predictGridWithNeuronsInto |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.375342 | 9.288689 | 3.9737 | 1.3456 | 11.924420 | 10.273716 |
| 2 | 9.322705 | 9.454270 | 4.0115 | 1.3694 | 11.977600 | 10.199000 |
| 3 | 9.504869 | 9.277257 | 4.0267 | 1.3895 | 11.986086 | 10.190458 |
| 4 | 9.327635 | 9.280918 | 3.9383 | 1.3745 | 11.963984 | 10.205682 |
| 5 | 9.535484 | 9.405236 | 3.9302 | 1.4301 | 12.004992 | 10.180192 |
| **Median** | **9.375342** | **9.288689** | **3.9737** | **1.3745** | **11.977600** | **10.199000** |

The post-change gate is at most 120% of each median above, plus the new absolute gates of at most 250 ms for a maximum-recipe paired evaluation and at most 500 ms for a maximum-recipe save capture.

The historical `SGD` subject calls `applyGradients` with zero accumulators. Its
immutable value and threshold are retained below, but the test and post-change
record name it **SGD zero-gradient adapter** so it is not mistaken for seeded
nonzero-update throughput.

## Implementation slices

| Task | Committed implementation slices |
|---:|---|
| 1 | `e8b0a4a`, `7c974a9` |
| 2 | `598e7d3`, `1fb545d` |
| 3 | `1772a13`, `9645e3f`, `49070e7`, `f486def`, `a774caa`, `475f220`, `6604e69`, `b3aa627`, `ea98e17` |
| 4 | `0bbca62` |
| 5 | `30edb7b`, `cbfc2ab`, `0e24c8d`, `7aa6667`, `1bec4cf`, `c05da39` |
| 6 | `129a725`, `d88f466`, `e3ad181`, `8c2bfc6`, `6dba50e`, `90f1610`, `14f5de1`, `2549791` |
| 7 | `beaaf7d` |
| 8 | `a5c0fed`, `b8d5fb4`, `66ffcf0`, `4e66c15` |
| 9 | `53665e0`, `41c632b` |
| 10 | `5d67ae5`, `23db2e5`, `6dd0412` |
| 11 | `d52d04d`, `afbb68e`, `1db9722`, `d64c93a`, `32be12c` |
| 12 | `5893cfc`, `e6c3d7a` |

## Post-change repository gates

### Focused performance and correctness gates

| Gate | Result | Evidence |
|---|---|---|
| Worker performance TDD RED | PASS | Web `test:perf` was absent before the dedicated script/config; exit 1 with `ERR_PNPM_RECURSIVE_RUN_NO_SCRIPT`. |
| Existing engine threshold TDD RED | PASS | Immutable assertions rejected Adam `48.8066 ms > 4.76844 ms` and SGD `29.8710 ms > 1.6494 ms`. |
| Engine performance-fix focus | PASS | 4 files, 120 tests, exit 0; includes complete-gradient clipping, exact-zero evidence, and late non-finite atomicity. |
| Final provenance/cadence focus | PASS | `pnpm --filter @nn-playground/engine exec vitest run src/__tests__/networkObjective.test.ts src/__tests__/numericalError.test.ts --reporter=dot`: 2 files, 31 tests, exit 0. |
| Final shared audit focus | PASS | `pnpm --filter @nn-playground/shared exec vitest run src/__tests__/experimentMemory.test.ts src/__tests__/workerProtocol.test.ts --reporter=dot`: 2 files, 29 tests, exit 0; hostile accessor/proxy guard then passed 16/16 independently. |
| Final web audit focus | PASS | Eight worker/hook/UI/persistence/protocol files, 207 tests, exit 0. |
| Full engine correctness | PASS | 20 files, 480 tests, exit 0. |
| Web TypeScript | PASS | `pnpm --filter @nn-playground/web exec tsc --noEmit`, exit 0. |
| Scoped ESLint | PASS | All touched engine/web performance source, tests, and configs; exit 0. |
| Normal/performance discovery separation | PASS | Normal web Vitest excludes `scientificTrust.performance.test.ts`; dedicated config includes only it. |
| Performance review | PASS | Independent scoped review found no unresolved critical, important, or minor issue. |
| Full-range review and re-audit | PASS | Broad local review found five Important issues; all were reproduced and fixed. Re-audit found one remaining exception-safety issue; hostile accessor/proxy RED-GREEN closed it. Final verdict: no Critical, Important, or Minor findings and ready to commit. |
| Final root `pnpm test` | PASS | Exact command at `2026-07-12T20:22` AST; 93 files, 1,533 tests: engine 480, shared 325, web 728; exit 0. |
| Final root `pnpm lint` | PASS | Exact command at `2026-07-12T20:22` AST; ESLint exit 0, no findings. |
| Final root `pnpm build` | PASS | Exact command at `2026-07-12T20:22` AST; TypeScript and Vite exit 0; existing `>200 kB` warning recorded, largest application chunk `485.83 kB` (`140.37 kB` gzip). |
| Final root `pnpm test:perf` | PASS | Exact command at `2026-07-12T20:23` AST; engine 4/4 and worker 2/2, exit 0. Controller values: grid `9.135023`, grid-into `9.167500`, neurons `12.546784`, neurons-into `11.488300`, Adam `3.7793`, SGD zero-gradient adapter `1.1876`, forced pair `5.0035`, save capture `6.7509` ms. |
| Final root `git diff --check` | PASS | Exact command at `2026-07-12T20:23` AST; exit 0, no output. |

### Final five-pass performance collection

- Command for every pass: `pnpm test:perf`
- Final-tree collection window: `2026-07-12T20:18:01-04:00` through
  `2026-07-12T20:19:58-04:00`.
- Engine benchmark files ran serially (`fileParallelism: false`, one worker) so
  post-timing validation in one file could not contaminate another file's timed
  region. Iterations, subjects, baseline values, and thresholds were unchanged.
- The web gate imported the worker-owned strict V2 API directly. Only
  `Comlink.expose` was isolated at module load; evaluation, provenance,
  validation, bounded-history serialization, and async save validation all ran
  through shipped worker code.

Engine times are milliseconds per operation. Printed grid totals were divided
by 100 for standard grids and 50 for neuron grids.

| Run (timestamp AST) | predictGrid | predictGridInto | Adam + L2 + clip | SGD zero-gradient adapter | predictGridWithNeurons | predictGridWithNeuronsInto | Forced pair median | Save capture median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (`20:18:01–20:18:21`) | 9.686058 | 9.188860 | 3.8966 | 1.4827 | 11.766350 | 11.772256 | 5.0475 | 6.5996 |
| 2 (`20:18:26–20:18:45`) | 9.774131 | 9.342484 | 3.7911 | 1.4530 | 11.760222 | 11.476332 | 5.2404 | 7.0109 |
| 3 (`20:18:52–20:19:11`) | 9.162944 | 9.209807 | 3.7111 | 1.2105 | 12.330712 | 11.783522 | 5.1399 | 6.6978 |
| 4 (`20:19:16–20:19:35`) | 9.176247 | 9.220630 | 3.8302 | 1.4015 | 11.958630 | 11.441054 | 4.9950 | 6.4109 |
| 5 (`20:19:39–20:19:58`) | 9.106741 | 9.246962 | 3.7562 | 1.4560 | 11.922596 | 11.732152 | 5.0382 | 6.5068 |
| **Median** | **9.176247** | **9.220630** | **3.7911** | **1.4530** | **11.922596** | **11.732152** | **5.0475** | **6.5996** |

| Benchmark | Baseline median | Post median | Change | Fixed maximum | Verdict |
|---|---:|---:|---:|---:|---|
| predictGrid | 9.375342 | 9.176247 | -2.1236% | 11.2504104 | PASS |
| predictGridInto | 9.288689 | 9.220630 | -0.7327% | 11.1464268 | PASS |
| Adam + L2 + clip | 3.9737 | 3.7911 | -4.5952% | 4.76844 | PASS |
| SGD zero-gradient adapter | 1.3745 | 1.4530 | +5.7112% | 1.6494 | PASS |
| predictGridWithNeurons | 11.977600 | 11.922596 | -0.4592% | 14.37312 | PASS |
| predictGridWithNeuronsInto | 10.199000 | 11.732152 | +15.0324% | 12.2388 | PASS |
| Maximum-recipe forced pair | n/a | 5.0475 | n/a | 250 | PASS (2.02% of budget) |
| Maximum-recipe save capture | n/a | 6.5996 | n/a | 500 | PASS (1.32% of budget) |

Every one of the five forced-pair and save-capture run medians independently
passed its absolute threshold. Their 20-sample measured sets were:

<details>
<summary>Worker raw samples (milliseconds)</summary>

```text
Run 1 forced: 5.1157, 4.9612, 5.1497, 4.8757, 4.8660, 4.9248, 5.1205, 4.7838, 4.8217, 4.9071, 4.8623, 5.3460, 5.2327, 5.1223, 4.9032, 5.1924, 5.2484, 5.3892, 5.0561, 5.0389
Run 1 save:   6.8836, 5.8436, 5.8127, 6.0507, 6.4182, 6.0078, 6.4930, 6.2535, 6.5809, 6.4778, 6.6184, 6.7542, 7.3140, 6.9354, 6.8570, 7.0076, 7.7927, 6.7167, 7.1566, 6.5150
Run 2 forced: 4.9754, 5.4027, 4.9597, 5.2357, 5.2908, 5.2459, 5.0346, 5.0766, 5.2522, 5.3612, 5.0471, 5.2413, 5.2001, 5.3301, 5.2396, 5.3359, 6.4899, 5.0759, 5.1142, 5.3799
Run 2 save:   7.7316, 5.8141, 5.8339, 5.9121, 6.3891, 5.9960, 6.2591, 6.3005, 7.0205, 7.0012, 6.3559, 7.3994, 7.6152, 7.3085, 7.2752, 7.3128, 8.0061, 6.9405, 7.3991, 7.2412
Run 3 forced: 5.0700, 5.1027, 5.5489, 6.2400, 16.7887, 7.3310, 6.0780, 5.0547, 5.0824, 5.0607, 5.0468, 5.0388, 5.1199, 5.1965, 5.2048, 5.1159, 4.9344, 5.3199, 5.1598, 5.2983
Run 3 save:   7.2639, 5.7768, 5.7590, 5.9905, 6.1657, 6.1604, 6.0366, 5.9991, 6.2662, 6.3802, 6.5643, 6.9548, 6.9388, 6.9402, 6.8768, 6.9182, 7.5513, 6.8320, 7.0238, 6.8314
Run 4 forced: 4.8472, 5.0895, 4.9164, 5.1421, 4.8112, 4.8859, 4.8147, 4.8430, 4.9033, 9.0723, 11.3050, 7.1361, 5.8095, 5.0378, 5.4046, 4.9522, 5.1011, 4.9165, 5.1315, 4.8757
Run 4 save:   6.9192, 5.8661, 5.8669, 5.8974, 6.1608, 6.0169, 6.0304, 6.2882, 6.2798, 6.0486, 6.2995, 6.9194, 7.1807, 6.8380, 7.1209, 6.7695, 7.9743, 6.5222, 7.1029, 6.9044
Run 5 forced: 4.8275, 5.0654, 4.9599, 5.0384, 4.9455, 5.0243, 4.9585, 5.1273, 5.0401, 5.1367, 5.0415, 4.9373, 5.0871, 5.1191, 5.2212, 5.0168, 5.0380, 5.0188, 5.0578, 4.9299
Run 5 save:   7.0467, 5.8229, 5.8952, 5.8117, 6.0173, 5.9326, 6.1149, 6.3922, 6.2648, 6.2450, 6.2680, 6.9992, 6.8683, 7.0105, 6.7299, 7.1666, 7.9740, 6.6214, 6.8128, 6.9756
```

</details>

### Rejected and superseded timing attempts

No threshold or baseline was changed in response to noise:

- Initial collection was deferred at load average `18.29` with TradingView
  renderer `83.3%`, GPU `51.6%`, and WindowServer `61.4%`.
- The first engine threshold run revealed a reproducible product regression,
  not noise: Adam `54.9266 ms` and SGD `30.7933 ms`; isolated runs reproduced
  it. Repeated robust gradient scans and per-weight `Math.hypot` were fixed with
  a finite-checked fused preparation path.
- Pre-serialization runs that failed only neuron-grid-into at `12.7215 ms` and
  `12.5465 ms` were rejected after isolated grid evidence passed. Serial engine
  files removed measured-region cross-interference without changing subjects.
- A serialized SGD `1.9184 ms` outlier and a later load-`8.36` run with grids
  around 2.5x were rejected as contaminated.
- Four otherwise complete passes were superseded after a later checkpoint
  correctness fix changed the source tree. A later five-pass set from
  `19:46:53–19:48:49` was also superseded by final audit fixes for post-update
  batch loss, 50-step checkpoint cadence, and persistence/protocol/UI
  hardening. The table above is the only final set: all five runs completed
  without rejection after the final browser-quiet handoff.

## Browser scenarios

Target: `http://127.0.0.1:5173/` in the Codex in-app browser on
`2026-07-12` AST. All interaction assertions below came from fresh accessible
DOM snapshots; screenshots are under `docs/superpowers/verification/browser/`.

| # | Scenario | Result | Evidence |
|---:|---|---|---|
| 1 | Empty/default V2 URL and reload | PASS | Empty URL prepared Circle (`2 -> 4 x 4 -> 1`, SGD `0.03`, BCE, `x,y`); reload retained the identical recipe. `01-default-v2.jpg`. |
| 2 | Regression to XOR | PASS | Regression and XOR reached their complete canonical fingerprints; XOR matched pinned `r2.1.s7KkpuX9x5Ct1FqICqBkuk3veDCjyFcslR9HVKOwIv4`. `02-regression-to-xor.jpg`. |
| 3 | Every public preset transition | PASS | Full 7 x 7 UI transition matrix, 49/49 source/destination fingerprint checks, zero failures. `03-all-public-presets.jpg`. |
| 4 | Live step 51 vs evaluation 50 | PASS | Exact DOM state: batch EMA step 51 (`0.6428`), train/test full-split evaluation step 50 (`0.5986` / `0.5957`), timeline current 51/checkpoint 50. Running-page screenshot illustrates the same separate cadence at live 507/evaluation 505 because taking a screenshot does not freeze the render loop. `04-live-vs-full-evaluation.jpg`. |
| 5 | Pause and manual-step forced pairs | PASS | Pause aligned EMA/train/test at step 1153; one manual step aligned all three at 1154 (`0.0122` / `0.0120` / `0.0123`). `05-pause-manual-forced-pairs.jpg`. |
| 6 | Three-class matrix and copy | PASS | Evaluation 256/model step 1154 used all 150 test samples: diagonal `49,52,49`, zero off-diagonal, 100% accuracy. Copy URL synchronized the canonical address whose fingerprint matched pinned three-class `r2.1.4EVeEMIUv4zdaWIk0GQRVg9UKkktYAdCix5PPECMozU`. `06-three-class-matrix.jpg`. |
| 7 | Regression terminology | PASS | Regression displayed `mean squared error`, `Train data loss (full split)`, and `Training objective`; no banner accuracy metric. `07-regression-objective.jpg`. |
| 8 | V1/future URL and JSON incompatibility | PASS with browser API limitation | V1 `#v=1&r=legacy` stayed byte-for-byte preserved and rendered `legacy-state` at `v`; future `#v=3&r=future` rendered `unsupported-version` at `v`; neither exposed the workspace. `08-v1-url-incompatibility.jpg`, `08-future-url-incompatibility.jpg`. Exact unversioned/future JSON import cases passed focused `ConfigPanel`/`CompatibilityState` tests; the in-app browser rejects file-chooser upload commands, so those two file selections could not be replayed manually without prohibited DOM mutation. |
| 9 | Save pair and Apply saved recipe | PASS | Saved Circle at steps 0 and 1 with full 150/150 evaluations; comparison named the lower test-data-loss record by `0.0005`. Applying the older record retained the recipe fingerprint and correctly started a fresh step-0 model rather than pretending to restore weights. `09-save-pair-apply-recipe.jpg`. |
| 10 | Untouched legacy localStorage | PASS | Existing version-1 notice survived reload, disappeared on explicit notice dismissal, then reappeared after the next reload/hydration, proving dismissal did not delete the legacy key. `10-legacy-storage-notice.jpg`. |
| 11 | Incompatible saved comparison | PASS | Regression atop Circle rendered `Not directly comparable` and the dataset/objective identity requirement, with no numeric winner claim. `11-incomparable-saved-runs.jpg`. |
| 12 | Checkpoint restore and limited guarantee | PASS after two browser-found fixes | First replay exposed an evicted UI checkpoint (`V2 checkpoint not found`, `12-checkpoint-restore-failure.jpg`); pause now synchronizes the authoritative worker ring. Second replay exposed stale live-revision preference (`checkpoint restore must advance...`, `12-checkpoint-restore-revision-failure.jpg`); restore now validates against the newest accepted scientific revision. Final replay restored owned Step 1900, remained paused, advanced evaluation 420 to 421 at the same model step, cleared stale live trend, and retained `Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only.` `12-checkpoint-restore-limited-guarantee.jpg`. |

The two checkpoint failures were reproduced before their fixes, encoded as
focused regressions, and replayed in the same high-speed browser flow. Final
focused verification was web `111/111`, shared protocol `14/14`, web TypeScript,
scoped ESLint, and `git diff --check`, all exit `0`.

## Limitations

- Worker budgets measure worker-owned computation and strict boundary validation,
  not browser scheduling, Comlink transfer, or structured-clone latency. Those
  transport surfaces are covered by browser scenarios rather than this CPU gate.
- Local timing remains sensitive to interactive host load; explicit threshold
  assertions, serial benchmark files, host observation, and five complete runs
  reduce but cannot eliminate machine noise.
- The in-app browser does not support file chooser uploads. Unversioned and
  future-version JSON import compatibility is therefore covered by exact
  source-preserving component tests, while URL incompatibility is covered in a
  real browser.
- The running-page screenshot for scenario 4 visually demonstrates separate
  live/evaluation cadence but cannot freeze the exact 51/50 frame; the exact
  values were captured from the accessible DOM before the page advanced.
- Production build retains the pre-existing Vite chunk-size warning; it is
  recorded rather than hidden and remains a future code-splitting opportunity.
