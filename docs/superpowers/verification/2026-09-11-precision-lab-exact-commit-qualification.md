# NN.FORGE Precision Lab exact-commit qualification

Date: 2026-09-11

## Qualified product candidate and documentation boundary

- Repository: `DenseDevKev/neural-network-playground` (public before the successful rerun; visibility was changed by the owner).
- Branch: `codex/nn-forge-precision-lab`.
- Commit: `fcc382658396defa1a19c3a5543478ebe6b0986f`.
- Git tree: `e74b3561f2287d3f921851bb877686dbd0cca83a`.
- Latest product-changing commit: `6f478eae8a390e5a0c5d0ea80263198b10c66ddd`.
- Main remains `98f29b86e469a2a545be75928ae6f32309fd1582`.
- Qualification workflow: `.github/workflows/precision-lab-verification.yml`, workflow ID `351256324`.
- Exact-head run: [34543410273](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273), attempt 2.

Attempt 1 failed before runner allocation. Its six annotations said that recent account payments had failed or the spending limit needed to be increased. After the owner made the repository public, one bounded rerun allocated standard GitHub-hosted runners, reached checkout in every job and completed successfully. No repository code, workflow, test, retry, timeout, skip, scientific contract or acceptance limit was changed to obtain the green run.

## Required job results

| Job | Job ID | Result |
| --- | ---: | --- |
| `source-evidence` | [103381716547](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716547) | Passed |
| `focused` | [103381716376](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716376) | Passed |
| `browsers (preview)` | [103381716657](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716657) | Passed |
| `browsers (subpath)` | [103381716630](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716630) | Passed |
| `browsers (recovery)` | [103381716667](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716667) | Passed |
| `performance` | [103381716566](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34543410273/job/103381716566) | Passed |

All six successful job records identify run attempt 2 and head SHA `fcc38265`. The bounded workflow rerun is documented above; Playwright retries remained `0`, and no failed step, flaky result or unexplained failure is present in attempt 2.

## Correctness, build and bundle evidence

- Infrastructure contracts: 155 passed, 0 failed, 0 skipped.
- Engine: 497 passed across 21 files.
- Shared: 334 passed across 10 files.
- Web: 1,191 passed across 106 files.
- Package total: 2,022 passed.
- Changed tests, lint, complete typechecks, production build, tracked-source cleanliness and build provenance: passed.
- Workflow environment: Node `v20.20.2`, pnpm `9.15.9`.

| Gzip measurement | Actual | Maximum | Result |
| --- | ---: | ---: | --- |
| Main entry | 151,321 bytes | 152,245 bytes | Passed |
| InspectionPanel | 5,432 bytes | 7,373 bytes | Passed |
| Total JavaScript (9 files) | 233,766 bytes | 234,161 bytes | Passed |

The build-evidence artifact records source SHA `fcc38265` and deterministic SHA-256 values for the emitted assets.

## Browser and recovery evidence

- Normal preview: 82 passed, 6 intentional mode skips, 0 unexpected, 0 flaky.
- Project-subpath hosting: 86 passed, 2 intentional fault-mode skips, 0 unexpected, 0 flaky.
- The preview-only skips are the project-hosting-specific journeys plus the opt-in fault journey; those contracts execute in the subpath and recovery jobs respectively.
- Both Chromium and WebKit passed the full 200% document-zoom build/run journeys and the short/tall/zoom/reset transport regression.
- The suites retained real built assets, normal and `/neural-network-playground/` hosting, real worker/font/lazy-import paths, URL and navigation integrity, keyboard/touch/focus/accessibility coverage, layout stability, selection and exact-artifact save retry.
- Recovery fault-enabled report: 2 passed (Chromium and WebKit), 0 skipped/flaky/unexpected.
- After the clean rebuild, fault-disabled report: 2 passed (Chromium and WebKit), 0 skipped/flaky/unexpected.

## Accepted performance comparison

Reference `ae09b9863ae90f8fb2f62545834fcc138755ba9a` and candidate `fcc38265` were measured in five alternating pairs on the same isolated macOS 15.7.9, Apple M1 ARM64 runner. The accepted comparator passed every contract.

| Measurement | Baseline median | Candidate median | Contract |
| --- | ---: | ---: | --- |
| `predictGrid` | 15.7632 ms | 15.4138 ms | ≤120% baseline; passed |
| `predictGridInto` | 15.2790 ms | 15.3420 ms | ≤120% baseline; passed |
| `predictGridWithNeurons` | 18.7426 ms | 17.5569 ms | ≤120% baseline; passed |
| `predictGridWithNeuronsInto` | 16.9050 ms | 16.0847 ms | ≤120% baseline; passed |
| `adam` | 4.7510 ms | 4.7748 ms | ≤120% baseline; passed |
| `sgd` | 1.0936 ms | 1.1667 ms | ≤120% baseline; passed |
| Forced paired evaluation | 8.3788 ms | 9.8853 ms | ≤250 ms; passed |
| Saved-run capture | 10.9122 ms | 11.8374 ms | ≤500 ms; passed |

The raw `pnpm test:perf` collection process exited `1` for both the baseline and candidate because both exceeded preserved historical absolute engine constants on this hosted runner. The workflow intentionally retains those raw results and then applies the accepted symmetric same-runner policy above. Their identical exit classification is not a candidate-only regression and is not reinterpreted as a universal-host release failure.

## Provenance and closure boundary

- Every compact summary records `sourceSha: fcc38265`, `sourceTree: e74b3561...`, `trackedChanges: false`, `runId: 34543410273`, `attempt: 2` and `status: passed`.
- The source artifact records SHA `fcc38265`, tree `e74b3561...` and source archive SHA-256 `0f2482b2a538c5b7160e40a09ad197c5cad346b68a53d28b604495b137697536`.
- Raw browser reports/traces, fault-on and fault-off recovery reports, command receipts, build hashes and all ten performance sample logs are retained as run artifacts.
- This establishes full qualification for exact candidate `fcc38265`; it is not a claim that finite tests prevent every future failure.
- This documentation-only closure commit must receive its own exact-head six-job qualification before it becomes the final branch authority. Its result should be recorded externally to avoid a self-referential documentation-commit loop.

Main, Pages settings and billing settings were not changed by the qualification agent. Repository visibility is now public following the owner's action. Pages remains a separate settings/deployment question, and no live deployment is claimed. Legacy-shell retirement and cross-tab saved-run overwrite/concurrency remain separate work.
