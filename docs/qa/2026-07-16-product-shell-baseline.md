# Product-Shell Release Baseline

> **Historical pre-change evidence:** Preserve the commands, measurements, and
> known gaps below as the comparison baseline. They are not current release
> results. Current behavior is documented in
> [`../architecture/product-shell.md`](../architecture/product-shell.md); final
> verified results belong in `docs/qa/2026-07-16-product-shell-release.md` after
> the complete release matrix runs.

**Recorded:** 2026-07-16  
**Commit:** `97ce506` (`codex/refactor-decision-boundary`)  
**Worktree:** `/private/tmp/neural-network-playground-targeted-product-shell-refactor`

## Repository and CI inspection

- The monorepo uses pnpm recursive scripts for lint, Vitest suites, Vite production build, performance gates, and Playwright.
- `.github/workflows/ci.yml` installs dependencies, runs lint, the complete test suite, a production build, installs Chromium/WebKit, and runs `pnpm test:e2e`; it uploads Playwright traces, screenshots, and reports on failure.
- `playwright.config.ts` defines Desktop Chrome and Desktop Safari projects. CI uses one worker; local runs use Playwright's normal worker selection.
- Existing smoke coverage includes training/pause/step/checkpoint restore, preset transitions and multiclass evidence, and saved-run persistence in both browsers.

## Pre-change commands and results

```bash
pnpm lint
pnpm test
pnpm build
pnpm test:e2e
pnpm test:perf
git diff --check
```

| Gate | Result |
|---|---|
| Lint | Pass |
| Engine tests | 480 pass |
| Shared tests | 325 pass |
| Web tests | 782 pass across 71 files |
| Total tests | 1,587 pass |
| Production build | Pass; TypeScript and Vite (133 modules) |
| Browser smoke | Pass; 6/6 in 12.1 seconds |
| Diff whitespace | Pass |
| First performance sample | One contaminated 1.36% threshold miss; rerun required on idle host |

Exact zlib gzip measurements after the production build:

| Artifact | Bytes |
|---|---:|
| Main entry | 141,153 |
| `InspectionPanel` lazy chunk | 5,496 |
| Code Export lazy chunk | 3,599 |
| Run History lazy chunk | 3,985 |
| Engine chunk | 5,770 |
| React chunk | 4,206 |
| Worker | 55,674 |
| Non-worker JavaScript | 164,209 |
| Total JavaScript | 219,883 |

All pilot bundle budgets pass: Inspection at most 6,349 bytes, entry at most 142,005 bytes, and total JavaScript at most 223,010 bytes.

## Chromium/WebKit failure diagnosis

Earlier smoke attempts failed under severe host contention rather than at one deterministic product assertion:

- WebKit exhausted the 45-second test budget at `page.goto('/')` in all three scenarios, while the preview server returned HTTP 200 in 0.023396 seconds.
- Chromium advanced through the UI but exhausted the same wall-clock budget at different later workflow stages.
- The host showed load averages above 20 with WindowServer, GeForceNOW, Vitest/Node workers, and other services consuming CPU.
- A sequential diagnostic reduced concurrency but did not remove starvation, so merely changing worker count was not a valid fix.
- With GeForceNOW absent and host contention reduced, the unchanged default two-worker command passed all six scenarios: Chromium in 1.6–3.2 seconds and WebKit in 2.9–3.6 seconds; total 12.1 seconds.
- No pilot commit changed `tests/e2e`, `playwright.config.ts`, package scripts, or CI configuration.

**Root cause:** environment-contaminated wall-clock timeouts caused by host scheduling starvation, amplified by concurrent browser work. No browser-specific product defect or invalid assertion was reproduced. The release implementation must keep the current semantic assertions and must not add retries, skips, sleeps, or timeout inflation to mask host load.

## Performance caveat

The first fresh `pnpm test:perf` sample ran while `syspolicyd` (~85% CPU), `trustd` (~42%), Codex, Dia, and WindowServer were active. `predictGridWithNeuronsInto` measured 12.40575 ms/iteration against a 12.2388 ms cap, a 1.36% miss, while the other gates passed. This sample is classified as contaminated. Final verification will run the identical command repeatedly on an idle host and report medians without loosening thresholds.

## Known pre-change shell gaps

- `history` can be stored as an evidence target, rendered as Boundary, yet derive no Boundary demand.
- More is a miscellaneous Configuration/Code drawer and duplicates Code.
- Advanced disclosure, profile state, profile visibility, and profile switching do not exist.
- Evidence tabs lack roving keyboard focus and disclosure focus restoration.
- User-facing domain definitions are scattered; no typed catalog exists.
- Current Recipe does not yet flag every active advanced setting that may be hidden in Beginner.

These gaps are addressed by `docs/superpowers/plans/2026-07-16-release-ready-product-shell.md`.
