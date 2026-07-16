# Product-Shell Release Record

This record captures the final verification of the targeted product-shell
refactor. The product code under test is commit `29f23ce` on
`codex/refactor-shell-disclosure`; this evidence file is a documentation-only
follow-up. The pre-change comparison is
[`2026-07-16-product-shell-baseline.md`](2026-07-16-product-shell-baseline.md).

## Release metadata

| Field | Value |
|---|---|
| Recorded | 2026-07-16 |
| Baseline commit | `97ce506817a570ee1c1bfc0bad7d6f53bba746ae` |
| Verified product commit | `29f23ce9c701d23acf961c0591f82399d57b4933` |
| Branch | `codex/refactor-shell-disclosure` |
| Host | macOS 27.0 (`26A5378n`) |
| Node | `v25.9.0` |
| pnpm | `10.31.0` |
| Time zone | `America/Puerto_Rico` |

The final design and ownership rules are documented in
[`../architecture/product-shell.md`](../architecture/product-shell.md) and
[`../architecture/state-ownership.md`](../architecture/state-ownership.md).
The implementation sequence and acceptance criteria remain in
[`../superpowers/plans/2026-07-16-release-ready-product-shell.md`](../superpowers/plans/2026-07-16-release-ready-product-shell.md).

## Automated release gates

Commands were run from the repository root with no skipped tests and no
Playwright retries:

```bash
pnpm --filter @nn-playground/web exec tsc --noEmit
pnpm lint
pnpm test
pnpm build
pnpm test:perf
pnpm test:perf
pnpm test:perf
pnpm --filter @nn-playground/web test:perf
pnpm test:perf
PLAYWRIGHT_PORT=4174 pnpm test:e2e -- --reporter=line
git diff --check
git diff --check 97ce506..HEAD
```

Port 4174 was used because an unrelated user project already owned 4173. The
override changes only the local preview port; it does not change scenarios,
timeouts, worker count, or assertions.

| Gate | Result |
|---|---|
| Web TypeScript | Pass |
| ESLint | Pass |
| Engine Vitest | 480 pass |
| Shared Vitest | 325 pass |
| Web Vitest | 891 pass across 78 files |
| Total Vitest | 1,696 pass |
| Product-shell integration | 26 pass, including all 12 axe states |
| Production build | Pass; TypeScript and Vite, 140 modules, 5.02 seconds |
| Chromium/WebKit smoke | 14/14 pass in 30.4 seconds, two workers, zero retries |
| Diff whitespace | Pass |

The browser suite covers training, pause/step, presets, checkpoint restore,
saved runs, responsive behavior, and a paused scientific-invariant scenario.
That scenario cycles every profile and disclosure state while proving the URL
hash, model identity and evaluation step, checkpoint timeline, saved-run count,
and selected code-export format do not change.

## Bundle measurements

Sizes are exact zlib gzip bytes from the production build.

| Artifact | Baseline | Current | Delta | Required cap | Result |
|---|---:|---:|---:|---:|---|
| Main entry | 141,153 | 140,001 | -1,152 | 142,005 | Pass |
| `InspectionPanel` lazy chunk | 5,496 | 5,414 | -82 | 6,349 | Pass |
| Worker | 55,674 | 54,194 | -1,480 | n/a | Recorded |
| Non-worker JavaScript | 164,209 | 164,793 | +584 | n/a | Recorded |
| Total JavaScript | 219,883 | 218,987 | -896 | 223,010 | Pass |

Retained lazy chunks include Code Export (3,900 bytes), Configuration (1,872
bytes), Inspection (5,414 bytes), and Run History (3,897 bytes). The engine
chunk is 5,536 bytes and React is 3,973 bytes. Production uses Terser
minification to remain inside the established entry and total-JavaScript caps.

## Performance evidence

Four engine samples and four web samples were collected. The host was not
idle: FL Studio, `syspolicyd`, `trustd`, WindowServer, and Codex all showed
material CPU use. The third engine run produced one SGD outlier and exited
non-zero before the root script could run the web gate, so the third web sample
was collected directly with the unchanged web performance command. The table
reports four-run medians rather than hiding the contaminated sample. The fourth
complete root run was executed after the final tab-panel accessibility fix and
passed every engine and web gate at the verified product commit.

| Measurement | Four-run median | Gate | Result |
|---|---:|---:|---|
| `predictGrid` | 9.902477 ms/iteration | 11.2504104 ms | Pass |
| `predictGridInto` | 10.291331 ms/iteration | 11.1464268 ms | Pass |
| `predictGridWithNeurons` | 12.476520 ms/iteration | 14.37312 ms | Pass |
| `predictGridWithNeuronsInto` | 11.824080 ms/iteration | 12.2388 ms | Pass |
| Adam + L2 + clipping | 4.0986 ms | 4.76844 ms | Pass |
| SGD adapter | 1.42655 ms | 1.6494 ms | Pass by median |
| Worker forced-evaluation pair | 5.2895 ms | 250 ms | Pass |
| Worker-owned save capture | 7.00565 ms | 500 ms | Pass |

The contaminated third SGD sample was 2.2039 ms against its 1.6494 ms cap.
The first two and fourth complete root runs passed, all four-run medians pass
without a threshold change, and the outlier remains disclosed as host-sensitive timing
evidence rather than a code regression.

## Manual browser and accessibility QA

Visible QA used the local development build at 1280x720, 390x844, and 320x844.
Automated semantic checks used jest-axe in all three profiles, both Build and
Run, and both disclosure states (12 combinations). Chromium and WebKit ran the
same checked-in smoke suite.

| Check | Evidence | Result |
|---|---|---|
| Profile visibility | Beginner, Explore, and Lab showed their documented core tools; Advanced Tools exposed the applicable full union | Pass |
| Disclosure safety | Opening Advanced Tools left hash, runtime status, and model step unchanged; collapsing restored focus to its trigger | Pass |
| Keyboard evidence tabs | Arrow keys, Home, End, Enter/Space behavior, and global focus order are covered by component/browser tests | Pass |
| Accessible structure | Browser accessibility snapshot exposed skip link, banner, main, named regions, tablists, tabs, tabpanels, status, and live regions | Pass |
| Axe | No serious or critical violations in 12 profile/view/disclosure states | Pass |
| Help without hover | Concept help opened by button, closed with Escape, restored focus, and remained inside the viewport | Pass |
| Focus visibility | Disclosure, tabs, profile select, and concept controls retained visible focus during keyboard QA | Pass |
| Reduced motion | Browser test verified the reduced-motion media rule disables nonessential transitions | Pass |
| Narrow layout | No document or shell horizontal overflow at 320x844 or 390x844 | Pass |
| Touch targets | Mode, speed, evidence, Code format, and workspace controls measured at least 44 CSS pixels high at 320px | Pass |
| Color and contrast | Changed controls remained readable in visual inspection and critical states retained text labels rather than color-only meaning; jsdom axe does not provide a conclusive computed-color audit | Pass (visual only; no numeric audit) |
| Error association | No changed validation contract introduced an unassociated error; existing error tests remained green | Pass (regression coverage) |
| Console | No warnings or errors during visible profile/disclosure/help QA | Pass |
| 200% zoom/text resize | The in-app browser runtime could not change real browser zoom; 320px and 390px layout stress is not equivalent proof | Not executed |

At 320px, measured Code format controls were 84x44, 59x44, and 47x44 CSS
pixels; speed buttons measured 44x44. The visible Data-loss and Checkpoint help
popovers remained entirely within the desktop viewport.

## Consumer and hygiene searches

- `InspectionPanel` no longer writes visualization demand; `App` is the only
  production writer, while the store interface remains the delivery boundary.
- `historyDrawerOpen` has no production consumer and remains only in historical
  documentation.
- The legacy `RegionShell` implementation has no production consumer and was
  retained rather than deleted because compatibility removal was outside this
  release slice.
- Searches of changed production and test files found no focused/skipped tests,
  `debugger` statements, temporary logging, or implementation TODO/FIXME
  markers. Matches that remain are plan/documentation text.

## Browser failure diagnosis

The earlier Chromium/WebKit failures were environment-contaminated wall-clock
timeouts, not a reproducible application defect. WebKit timed out while loading
a localhost page that returned HTTP 200 in about 23 ms; Chromium timed out at
different later steps. The host had load averages above 20 and heavy browser,
streaming, test, and WindowServer activity. The unchanged baseline suite passed
when contention fell, and the final expanded suite passes both engines with
zero retries. No timeout inflation, sleep, skip, or weakened assertion was used.

## Independent review

Independent slice reviews covered profile/copy behavior, release tests,
documentation, and deployment. A final read-only whole-branch audit compared
`97ce506` with the release working tree. It found one Important accessibility
issue: inactive evidence tabs referenced unmounted per-tab panel IDs. Commit
`29f23ce` fixes that issue test-first by pointing every tab to the one stable,
mounted tabpanel while retaining dynamic labelling and unmounting inactive
content. The focused 37-test shell/axe run, complete 1,696-test suite, build,
performance gate, and both browser engines passed afterward. The reviewer found
no remaining Critical or Important production, architecture, accessibility, or
deployment issue.

## Remaining limitations and follow-up

- A real 200% browser zoom or OS text-resize pass remains pending human QA. The
  narrow-viewport evidence is useful but is not represented as an equivalent.
- Performance is host-sensitive. One disclosed SGD sample failed while the
  three-run median and every other median passed; a quiet-host release run would
  produce cleaner timing evidence.
- Terser keeps the bundle inside the hard caps but increased the final measured
  production-build time to 5.02 seconds (an earlier Terser sample was 3.20
  seconds) from the roughly 1.3-second pre-Terser local build.
- The optional external CodeRabbit review could not run because its CLI was not
  authenticated. Independent local slice reviews and a whole-branch audit are
  used instead.
- The planned three-newcomer/two-experienced-user usability study requires
  external participants and was not performed in this coding session. It is a
  post-release input to a separate copy/profile-tuning slice, not a shipping
  gate for the shared profiles.
- Existing multiclass-overlay, regression-copy, split/test-visibility, and
  inspection disabled-state defects remain logged as separate bug-fix work; no
  engine, worker, schema, URL, checkpoint, or saved-run contract was changed to
  address them here.
