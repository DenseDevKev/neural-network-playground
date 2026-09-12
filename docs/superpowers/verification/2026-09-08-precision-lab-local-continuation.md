# Precision Lab — September 8 local continuation verification

## Disposition

**Locally implemented and tested; not committed/pushed, not browser-qualified, not a release.**

The available GitHub connector exposes reads but no file/commit/ref writes or workflow dispatch. The source and CI artifacts below came from that connector; the patch was developed in an isolated Linux workspace. No remote repository state, settings, visibility or deployment was changed. A GitHub write-capable session must apply/review the patch and run the exact committed candidate's full workflow.

The canonical root `NN-FORGE-LIVING-EXECUTION-PLAN.md` was recovered from the original Library file, not replaced by a new roadmap. Its historical baseline and resolved performance sections are reconciled with current refs. New implementation tasks remain unchecked until their committed/test acceptance conditions are met.

## Source authority and recovery

| Item | Exact evidence |
|---|---|
| Fresh main ref | `98f29b86e469a2a545be75928ae6f32309fd1582` |
| Main merge parents | `ae09b9863ae90f8fb2f62545834fcc138755ba9a`, `2cd5b896a1e11e95e04c1122b100421cefe9ab77` |
| Main CI | `34002500733`, success |
| Active branch | `codex/nn-forge-precision-lab` |
| Fresh branch head | `3e7da9a4dabd2d7bc275c9aa4c614a04d9b26732` |
| Exact recovered Git tree | `80d2b9b4bf1a6a5a55a32752b844f7a239967477` |
| Branch qualification | `34254380062`; correctness/build/bundle passed, browser jobs failed |
| Source archive artifact | `10067245424`; ZIP SHA-256 `39985fa48953e0502afcd0c0bc047d85e50b1b9f8f09a361849f882eb93a0ad1` |
| Dependency workspace | Artifact `10035307936`, run `34169725927`; ZIP SHA-256 `298782d0c43764af7875601efb8cad63c3b3d49ec9b7edbabd685f5eb6d15262` |

The source archive's internal checksum was verified and its reconstructed Git tree matched the remote tree exactly. All five relevant package/lockfile manifests matched the dependency archive before it was reused. No pending previous-Mac source was assumed. The source at `3e7da9a` already integrates the shell, selection deck and one live boundary; it supersedes the supplied older `6e06b678` head.

The local repository has a synthetic source-import root only to permit diffs/worktrees. It is **not upstream history** and must not be pushed. The deliverable is a normal unified patch against the exact upstream tree, not a root-commit patch or branch replacement. External `provenance.json`, patch checksum and source/dist manifests identify the final local candidate without inventing an upstream SHA.

Read references: July 16 design, July 17 implementation plan, consolidated release roadmap, committed progress ledger and accepted-release verification. The historical prototype remains unavailable; no pixel-parity claim is made.

## Recovered defects versus tooling failures

The prior unreadable-tool-results failure is not attributed to the application. Current upstream browser logs identify separate issues:

- Three failures use a Features helper expecting `x` / `y`, whereas the retained accessible names are `X₁` / `X₂`. The locators were corrected; product names were not changed to satisfy tests.
- Precision Lab paint containment traps fixed help within the workspace. A help region extended below the viewport. Help is now portaled beside the workspace, with bounded viewport styling, focus on open, normal Tab access and Escape returning focus without leaking to outer navigation.
- An obsolete two-column transport grid reserved an empty column, squeezing controls; short compact viewports also clipped keyboard shortcuts against the status bar. The local patch removes the empty column, provides vertical overflow space, wraps controls and retains touch-target sizing.

These fixes have focused component/static regression evidence. Their **original real-browser reproductions have not been rerun successfully** in this sandbox, so browser bug closure is not claimed.

The integration tests already mocked visualization to isolate training, but the newly integrated boundary canvas was not in that mock list. The exact unchanged baseline also emitted JSDOM ResizeObserver warnings while all six integration tests passed. The test-only canvas mock was updated; this is not represented as an application crash fix.

## Implementation boundaries

### Single ownership

App retains its single `useTraining`, selection and boundary controllers. No second live canvas or worker-command owner is introduced. `useSaveCurrentRun` adds one App-owned save controller shared by transport and History. Legacy standalone presentation adapters remain for compatibility and tests until real-browser acceptance permits removal.

### Exact-artifact save/retry

The shared synchronous guard prevents transport/History events racing before React rerenders. The worker still owns capture of recipe, snapshot and evidence; the UI supplies only ID/title/timestamps. Default title enrichment occurs once after capture.

A failed persistence attempt retains the store's exact pending artifact. Retry invokes only `retryPersistence`: no worker recapture, new UUID, timestamp or current recipe read. Tests cover changed recipes after failure, repeated failed/successful retries, concurrent calls, StrictMode, drawer unmount/reopen while a capture is pending, rejected capture and hydration/access/title guards. Applying a saved recipe and trained-parameter limitations retain their existing semantics.

### Responsive evidence/transport

LossChart uses rounded, coalesced ResizeObserver measurements with bounded height, skips equivalent dimensions and cancels pending resize frames on cleanup. Evidence buffers/provenance remain unchanged. Confusion metrics now use semantic `dl`/`dt`/`dd` structures and a compact container layout. No confusion mathematics, evaluation cadence or scientific labels were changed.

Transport adds Save and explicit pending-artifact retry/discard controls, shares disabled reasons, and preserves Run/Pause, Step, reset, checkpoint controls and shortcuts. Required 44 px control sizing and no-overflow layout are implemented but still require actual browser acceptance.

### Targeted bundle reduction

The first expanded build measured entry **152,821** and total JavaScript **234,451** bytes and failed the unchanged checker. The final patch groups lessons and training explanations in one on-demand `EducationContent` chunk, marks safe unused memo wrappers pure for tree-shaking, removes unused App nodes and uses two ordinary Terser compression passes. A separate Hyperparameter lazy chunk was tested and rejected because it increased total compressed size; it is not in the final patch.

No feature, engine algorithm, protocol/schema/persistence format, scientific cadence or size limit was changed. The Vite generic large-chunk warning remains visible; it is not the executable gzip contract.

## Local final verification

Environment: Linux x86_64, Node **22.16.0**, pnpm **9.15.9**, exact recovered lockfile. CI uses its existing Node 20 setup; local success does not replace that rebuild. Final results are recorded in the delivered evidence files, with exit codes and full Vitest JSON rather than inferred from progress dots.

| Check | Result | Raw evidence |
|---|---|---|
| Engine unit suite | 497 passed, 0 failed | `engine-final.json`, exit 0 |
| Shared unit suite | 334 passed, 0 failed | `shared-final.json`, exit 0 |
| Web unit/integration suite | 1,170 passed, 0 failed | `web-final.json`, exit 0 |
| Total package tests | **2,001 passed** | Three complete package runs above |
| Infrastructure helpers | 91 passed, 0 failed | `helpers-all-final.log`, exit 0 |
| Lint | Pass | `lint-final.log`, exit 0 |
| Complete typechecks | Pass | `typecheck-final.log`, exit 0 |
| Production build | Pass | `build-final.log`, exit 0 |
| Executable gzip guard | Pass, limits unchanged | `bundle-final.log`, exit 0 |
| Second normal rebuild | All 87 files byte-identical; gzip guard passed again | `build-repeat.log`, `rebuild-equivalence.json`, `dist.sha256`, `bundle-repeat.log` |
| Playwright discovery | 66 scenarios in seven files | `browser-discovery.log`, exit 0; **not execution** |
| Browser execution attempts | Both blocked before app launch | `browser-attempt.log` and `browser-webkit-attempt.log`, each exit 1; missing Chromium/WebKit executables |
| Local performance diagnostic | Engine historical absolute FAIL; worker PASS | `performance-local.log`, exit 1 overall |

The first aggregate `pnpm test` invocation was interrupted while web tests were running. It is **not** counted as a passing aggregate run. Each final package suite above completed separately with machine-readable results. Earlier RED/diagnostic failures are retained where useful and are not relabeled as passes.

### Final gzip measurement

| Dimension | Actual bytes | Fixed maximum | Headroom |
|---|---:|---:|---:|
| Entry | 151,131 | 152,245 | 1,114 |
| InspectionPanel | 5,431 | 7,373 | 1,942 |
| Total JavaScript including worker | 234,068 | 234,161 | **93** |

Total headroom is narrow. Run the unchanged checker against the exact CI build; do not raise limits if a rebuild exceeds them.

### Performance interpretation

The accepted policy is already resolved and the baseline merged. The local diagnostic does not reopen that investigation. Local forced-pair median was **8.0115 ms** (fixed limit 250 ms); save-capture median **9.6054 ms** (fixed limit 500 ms). Historical absolute engine benchmarks reported two failing tests and one passing test. Those raw results are retained.

No paired baseline comparison on the accepted macOS/ARM64 runner was performed here, so no candidate-specific engine regression or accepted performance pass is asserted. The workflow patch reuses the original exact baseline and five alternating paired collections with the same 120% median comparator and independent worker budgets.

## Qualification workflow and remaining acceptance

The local workflow patch adds all infrastructure helper tests, exact source/dist build evidence, the existing recovery/normal-rebuild/fault-disabled sequence, and the accepted paired performance job. It remains read-only, preserves original assertions/zero retries, and has no deployment or settings writes. Three workflow contract tests are included in the 91-helper result. The workflow has **not** been pushed, dispatched or observed passing.

The layout specifications now cover Build and Run at 1437×742, 735×860 and 320×844, one connected boundary, URL preservation, major region bounds, overflow, 5.5 seconds at 50 steps/frame, <=1 CSS px unexpected movement and Chromium CLS = 0. Axe coverage adds all three required viewports. These are authored/discovered acceptance scenarios, not measured layout or accessibility results. Chromium launch requires the absent `chromium_headless_shell-1228` executable; WebKit launch requires the absent `webkit-2311/pw_run.sh`. Both attempts failed before the application ran.

Still required before acceptance: real Chromium/WebKit preview and non-isolated subpath execution; browser save/retry and recovery; 200% zoom, keyboard/focus/touch/reduced-motion checks; measured layout stability; exact committed-candidate reference performance; safe legacy-consumer removal and requalification; reviewed merge and main CI.

Pages remains a separate repository-settings blocker. Existing deployment run `34002943565` failed. No visibility change, successful deployment or live URL is claimed.
