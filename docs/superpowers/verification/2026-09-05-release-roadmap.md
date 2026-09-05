# NN.FORGE release-roadmap verification record

**Recorded:** September 5, 2026. This is an immutable branch-qualification event,
not a claim of a deployed release or that every roadmap task is complete.

| Identity | Value |
|---|---|
| Repository | DenseDevKev/neural-network-playground (private) |
| Authoritative intake main | `ae09b9863ae90f8fb2f62545834fcc138755ba9a` |
| Previous execution checkpoint | `96fdcfffe405bfa52e9fe0fec9a18afb70f7fc48` |
| Code verified in this record | `fba63307cd31a7646af15dfb4fff95825dff02f9` |
| Execution branch | `codex/nn-forge-release-roadmap` |
| Combined qualification | [Actions 33974892532](https://github.com/DenseDevKev/neural-network-playground/actions/runs/33974892532), attempt 1 |
| Publication | Not performed; owner audience/settings authorization remains open |

Main was not advanced. No pull request, historical merge, branch deletion,
visibility change, Pages setting change, or publication was performed. The
GitHub connector supplied source and Actions evidence; an isolated analysis
container ran the dependency-free helper tests. The previous Mac and local-only
prototypes were not accessed. No independent subagent review is claimed.

## Outcome

The WebKit recovery/resource blocker is resolved by same-origin font delivery.
The current correctness, preview, non-isolated subpath, recovery, clean-rebuild,
and fixed JavaScript bundle checks passed. **Engine timing qualification remains
failed.** Both baseline and candidate fail on the measured Linux runner and in
all five complete paired collections on a fresh virtual Apple M1 runner. A
failing baseline is not a passed candidate gate.

Read the [roadmap](../plans/2026-09-05-nn-forge-release-roadmap.md) and
[historical reconciliation](../README.md) before starting further work.
Precision Lab's replacement UI is not implemented; only its P1 gzip guard was
pulled forward without changing the production shell.

## Executed combined results at fba6330

Runner: Ubuntu 24.04.4, Node `20.20.2`, pnpm `9.15.9`; frozen lockfile installation.
The correctness log and browser report metadata identify the exact source SHA.

| Command / job | Observed result |
|---|---|
| `node --test scripts/*.test.mjs` | 77 passed, 0 failed, 0 skipped (includes 24 new bundle-helper tests) |
| `pnpm lint` | Exit 0 |
| `pnpm typecheck` | Exit 0, engine/shared/web and web test TypeScript projects |
| `pnpm test` | 1,874 passed: engine 497, shared 334, web 1,043 |
| Normal `pnpm build` | Exit 0; existing configured 200 kB warning remains visible |
| `pnpm test:bundle` | Exit 0; measurements below |
| `pnpm test:e2e`, isolated preview | 42 passed, 6 intentional skips, 0 failures, 0 flaky cases |
| Same suite at non-isolated project subpath | 46 passed, 2 intentional skips, 0 failures, 0 flaky cases |
| `pnpm test:e2e:recovery` | 2 passed, 0 skipped; Chromium and WebKit |
| Clean rebuild, then `worker-recovery.spec.ts --grep '@fault-disabled'` | 2 passed, 0 skipped |
| `pnpm test:perf` | Engine failed; worker budgets passed; aggregate failure retained |
| Same runner's pinned intake `pnpm test:perf` | Engine failed; failure retained, not a waiver |

Preview's six skips are four external-only hosting cases plus the two existing
fault-enabled cases in a normal bundle. On the non-isolated fixture all four
hosting cases execute, leaving only the two expected fault-enabled skips.
Browser report metadata shows one worker, zero flaky cases, and the actual target:
`http://127.0.0.1:4173/` for preview and
`http://127.0.0.1:4174/neural-network-playground/` for the external fixture.
These are not a live Pages URL.

Correctness job `101329794150`; preview `101329794298`; subpath `101329794313`;
recovery `101329794291`; performance `101329794235`.

### JavaScript gzip measurements

Measured by the new checked-in CLI in the Node 20 correctness job:

| Dimension | Actual gzip bytes | Fixed maximum bytes | Result |
|---|---:|---:|---|
| Application entry | 146,940 | 152,245 | Pass |
| InspectionPanel | 5,414 | 7,373 | Pass |
| All JavaScript, including worker | 226,417 | 234,161 | Pass |

These are the limits written in the July 17 Precision Lab plan, now executable
in both CI and release verification. No cap, minification setting, or chunking
strategy was relaxed. This is a JavaScript-only guard: it excludes CSS and fonts
and does not certify cold-load latency or total transferred bytes.

Self-hosted font assets increase the static build inventory; CSS in this build
is approximately 127.25 kB before gzip and 20.93 kB gzip. The main application JS
is approximately 521.39 kB before gzip. No startup-speed improvement is claimed.

## Q2 — Actual WebKit font reload correction

The prior fault-recovery trace loaded external Google CSS/fonts initially, then
reported a cross-origin resource-policy failure during conditional font reload.
The worker itself recovered and stepped; the test correctly rejected console
errors. The earlier anonymous-CORS stylesheet trial did not solve the failure
and was reverted before this continuation.

A new browser contract was committed before the replacement. At
`221e110e4f1692ebb3d0b6ade87a2a95e6bad5a5`,
[run 33973441436](https://github.com/DenseDevKev/neural-network-playground/actions/runs/33973441436)
failed in both browsers at the intended assertion: actual Google-hosted font
requests were present when the expected list was empty. This was not a missing
browser or broken test setup.

Commit `cf02bf51c8763c11ba481304c9c4ad39c4c79816` replaced external font links with
exactly pinned `@fontsource/inter@5.3.0` and
`@fontsource/space-grotesk@5.3.0`, imported from `styles/fonts.css` before the
application CSS. Inter 400/500/600/700 and Space Grotesk 400/500/600 remain the
requested families/weights. Vite emits their assets below the application base.
The stock dependency resolver's unrelated churn was discarded; the committed
lockfile adds only the two importer entries, package integrities, and snapshots.
The reviewed lockfile blob is `0f95025a6cb614998f597db18e9eeafad57800d5`.

Font tests exercise the real font faces, accented Latin text, and two native
reloads with no request interception. Both browsers require same-origin/project
font loads, no Google font requests, and no console/page/font-request errors.
The original recovery assertion is unchanged. Isolation, worker behavior,
evaluation cadence, and scientific state are unchanged. The upstream author
copyrights and full OFL text ship in `apps/web/public/font-licenses.txt`.
No pixel-identical historical remote-font revision is claimed.

The initial green
[run 33973942312](https://github.com/DenseDevKev/neural-network-playground/actions/runs/33973942312)
passed correctness and all three browser jobs; the later fba6330 run reproduced
that result alongside the new bundle guard. Normal output was rebuilt after
injected-fault testing, and normal-build fault-query inertness passed again.

A temporary dependency-intake workflow was used to resolve package metadata and
register one reviewed Git lockfile blob. It did not create commits, update refs,
or publish. It was removed in the font fix; the retained verification workflows
use read-only repository permissions.

## P1 — Executable bundle guard

Files: `scripts/check-web-bundle-gzip.mjs`, its `.test.mjs`, root `package.json`,
`.github/workflows/ci.yml`, and `.github/workflows/release-verification.yml`.

The helper measures the actual module entry, exactly one InspectionPanel chunk,
and recursively every physical `.js` asset. It rejects missing/ambiguous inputs,
external entry URLs, traversal, symlinks, invalid numeric measurements, and every
one-byte overrun; one failure reports every failed dimension with actual/allowed
bytes. Fonts and CSS are intentionally outside these reviewed JS dimensions.

The test file was run before the helper existed (expected missing-module RED),
then passed 24/24 after implementation in the isolated Node 22 analysis runtime.
All Node helpers then passed 77/77 there. GitHub's Node 20 run independently
executed the same 77 helper tests, lint, full typechecks, unit tests, production
build, and the real gzip CLI successfully. No product shell was activated.

## Q1 — Five paired reference collections: still failed

[Performance reference run 33974596143](https://github.com/DenseDevKev/neural-network-playground/actions/runs/33974596143)
compared candidate `ebdb7b6c0006d5d38b72e20e49f061310a3aabd6` with the exact intake
`ae09b9863ae90f8fb2f62545834fcc138755ba9a` in one job. This candidate includes the
font correction and reference workflow, before the later JS-only budget helper.

Observed host: Apple M1 (Virtual), arm64, 3 logical CPUs, 7,516,192,768 bytes RAM,
macOS 15.7.9 build 24G830, Node 20.20.2, pnpm 9.15.9. This is a newly measured VM,
not the original physical development Mac and not a replacement for its missing
reproducible host calibration.

Five complete collections per revision ran serially, with order alternating
between baseline-first and candidate-first. Every exit and raw log was retained;
all ten aggregate runs exited 1. All four grid limits failed on both revisions
in every collection. Worker evaluation/save checks passed in all collections.

Medians below summarize the five recorded measurements; they do not erase a
failed individual run or establish a statistically significant speed change.

| Measurement (ms) | Baseline median | Candidate median | Existing absolute maximum |
|---|---:|---:|---:|
| predictGrid | 14.5525 | 14.6987 | 11.2504104 |
| predictGridInto | 14.8714 | 14.6643 | 11.1464268 |
| predictGridWithNeurons | 17.3578 | 17.2455 | 14.37312 |
| predictGridWithNeuronsInto | 15.8438 | 15.8801 | 12.2388 |
| Adam + L2 + clipping | 4.5260 | 4.4715 | 4.76844 |
| SGD zero-gradient adapter | 1.1316 | 1.1034 | 1.6494 |
| Forced full paired evaluation | 8.4089 | 8.7056 | 250 |
| Worker-owned save capture | 10.3354 | 10.5895 | 500 |

An Adam outlier of 9.0914 ms in candidate collection 2 is preserved; baseline
collection 2 also exceeded that cap at 4.7928 ms. Do not use the medians to call
all Adam runs passing. No engine, worker performance implementation, benchmark
fixture, diagnostic, or threshold changed in this continuation.

### Raw per-collection summaries (milliseconds)

| Pass / revision | Grid | GridInto | Neurons | NeuronsInto | Adam | SGD adapter | Forced pair | Save |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / baseline | 14.5525 | 15.2412 | 17.7335 | 16.3101 | 4.5260 | 1.1316 | 8.3431 | 10.3354 |
| 1 / candidate | 14.6987 | 14.7876 | 17.3276 | 15.8137 | 4.4715 | 1.1018 | 8.7157 | 10.2270 |
| 2 / baseline | 14.5746 | 15.3991 | 16.8319 | 15.6305 | 4.7928 | 1.0862 | 8.4089 | 10.0641 |
| 2 / candidate | 14.7310 | 14.6939 | 17.2455 | 15.8801 | 9.0914 | 1.0985 | 8.8683 | 10.6311 |
| 3 / baseline | 14.6308 | 14.7953 | 16.8892 | 15.9391 | 4.5336 | 1.2917 | 8.4648 | 10.3444 |
| 3 / candidate | 14.5622 | 14.5828 | 17.4720 | 16.0528 | 4.4682 | 1.1526 | 8.7056 | 10.5895 |
| 4 / baseline | 14.3592 | 14.8714 | 17.3578 | 15.8438 | 4.3422 | 1.0941 | 8.3218 | 10.1632 |
| 4 / candidate | 14.5847 | 14.6643 | 17.0797 | 15.5999 | 4.3383 | 1.1034 | 8.3449 | 10.5821 |
| 5 / baseline | 14.5194 | 14.7330 | 18.0778 | 15.8028 | 4.3429 | 1.1398 | 8.5537 | 10.4002 |
| 5 / candidate | 14.7109 | 14.5733 | 17.2345 | 16.0666 | 4.6395 | 1.1146 | 8.4082 | 10.6704 |

The runtime failure is not resolved by moving runners, by a passing web budget,
or by the small baseline/candidate differences. Qualification remains blocked
until a documented reference environment meets the unchanged contract or a
separately reviewed policy decision explicitly changes the contract. This record
makes no such policy change and does not authorize publication.

## Artifact register

All listed artifacts were inspected as metadata or, for browser reports and
reference measurements, downloaded and parsed. Build metadata records provenance;
no byte-identical dist comparison to the original product is claimed after the
skip-link and font changes. Retention is seven days, so future readers must check
expiry rather than assume the artifacts remain available.

| Run | Artifact ID | Name / use | SHA-256 digest |
|---|---:|---|---|
| 33974892532 | 9972110088 | Preview report | `1031c297473e490c9acf00feb491f75208927d210fbfe8f94468fc15230eedc2` |
| 33974892532 | 9972105096 | Subpath report | `146b2bc1077df45ae8d8c6b10732bb7f89da0d0dc995f7ac60b77f54c172d3ca` |
| 33974892532 | 9972052733 | Recovery report plus clean-build report | `6f6159cc549901e5b03ba999a916e87d7a461cd399779595c34b36525b9147c6` |
| 33974892532 | 9972078111 | Normal build and source/dist-manifest evidence | `311e66aa39d10b6a21bd0b2030510110a68f36659770a998918dd3b9b604d943` |
| 33974892532 | 9972042208 | Linux baseline/candidate performance | `697972611c636b2817d7b92b79c2beb04164f0a667dc842762c91712327f0c27` |
| 33974892532 | 9972019222 | Exact tracked source archive | `5f4146960e6d1c6f7d19d2e127d249346e22d792a51887e6eed50ee23c5527aa` |
| 33974596143 | 9972000036 | All five baseline/candidate reference collections | `839cef3aad2d63da3166610588d11b61d34cdb409c2081b3a8bf11f768fe2740` |

## Historical and scope disposition

The initial successful main CI `33944120268` and failed Pages deployment
`33944571511` are historical intake evidence, not runs of this branch. The old
local test/performance claims remain reported, not newly executed here. The
current fresh evidence above supersedes the earlier branch's open WebKit font
blocker but does not erase its original failing logs.

The production product is still Build/Run. Scientific-trust requirements already
have committed implementation slices and an older verification record. Their
unchecked plan boxes do not reopen the project. Precision Lab's UI integration
and visual-reference checks remain planned, not implemented. The central index
reconciles the five historical documents without modifying their original text.

No physical Quest, iOS device, production Pages site, or 200% manual browser-zoom
session was verified in this event. Browser fixtures are real built-application
tests but not a certification of all environments. The configured Vite warning
and GitHub action-runtime maintenance annotation remain visible. No external-font
resource error is suppressed to obtain the successful recovery result.
