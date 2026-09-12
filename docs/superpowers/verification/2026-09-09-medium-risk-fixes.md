# NN.FORGE — Medium-risk bug fixes verified

Date: September 9, 2026. Scope: fix the two medium-risk findings, verify regressions, and leave the high-risk saved-run overwrite for a separate task.

## Source and qualification

- Repository: `DenseDevKev/neural-network-playground`.
- Branch: `codex/nn-forge-precision-lab`.
- Intake code: `f186a027ed151088f666074052c9cbc9b5b0362a`.
- Verified code and tests: `88e107a954cae016f4da3a71ffb29225d92028b7`.
- Verified tree: `c0d6eea9a99a2eb3c661f2311bdf7064861cd858`.
- Final Actions run: [34410864082](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34410864082), all six jobs successful, first attempt.
- Main remains `98f29b86e469a2a545be75928ae6f32309fd1582`. No merge or deployment.

The exact CI source archive was checked against all 461 locally tracked files: zero content differences. This verification record and the checklist closure are documentation-only additions after the successful code/test run; they do not imply an additional product build.

## Bug fixes — intentional presentation corrections

### Drawer touch interception

At 320px, measured close-button geometry was y=163, height=44, beneath a header extending to y=248.5. Hit-testing confirmed the header received the touch. The old fixed 156px mobile top offset was the cause, not a broken close callback.

`PrecisionLabShell.tsx` now portals drawers into the existing outer shell. CSS places them in the real workspace grid row, below the actual header and above the footer, outside the workspace scroller. IDs, non-modal semantics, close command and trigger-focus return remain. Initial focus does not scroll the page.

The added browser regression covers Presets, Lessons and History at 320px and 390px with header details expanded and collapsed. All 48 measured close hits across two browsers and two hosting modes reached the correct control, closed the drawer and preserved URL, checkpoint and disclosure state. The original touch journeys pass as well. Three new portal tests were observed failing before the fix, then passing after it.

### Training-induced layout shifts

The observer now records shifting source nodes and their rectangles without weakening its original assertions. The captured sources were intrinsic-width metric/transport rows, run metadata wrapping at larger step counts, and phone Epoch/status rows changing line layout.

Scoped CSS stabilizes neighboring metric tracks and reserves the demonstrated wrapping space. Intermediate headers size to their actual content. Phone Epoch text consistently occupies two lines; the complete footer credit occupies its own row. No numerical update is skipped, no scientific text is removed, and no timing, bundle or layout limit is raised.

Chromium reports zero non-input-associated layout shifts at 1437x742, 735x860 and 320x844, in both Build and Run, on both normal preview and non-isolated project-subpath hosting. Each scenario trains for 5.5 seconds at 50 steps/frame. Both browsers pass one-pixel geometry/overflow limits, preserve the same connected boundary canvas, and retain the experiment URL. WebKit does not provide the layout-shift API used here, so its result is geometry/interaction verification, not a measured zero CLS.

## Test-only correction

The full local regression exposed a race in the 70-case lesson catalog test: when source and destination recipes already matched, the recipe-only wait could finish before asynchronous lesson start called reset. The existing reset-count assertion now sits inside the completion wait. All 70 cases and their assertions remain; no lesson implementation changed.

## Executed verification

| Command or gate | Evidence/result |
|---|---|
| `pnpm test` | Local exit 0: engine 497, shared 334, web 1,179; 2,010 total. Exact-code CI full package regression also passed. |
| `node --test scripts/*.test.mjs` | 91 passed, zero failed; local and CI successful. |
| `pnpm lint` | Local and CI exit 0. |
| `pnpm typecheck` | Local and CI exit 0, including web source/test configurations. |
| `pnpm build` | Local and fresh CI build successful. |
| `pnpm test:bundle` | Local and CI passed unchanged caps. |
| `pnpm exec playwright test --max-failures=5` | CI preview: 62 passed, six intentional skips, zero failed/flaky/retries. |
| `PLAYWRIGHT_BASE_URL=http://127.0.0.1:4174/neural-network-playground/ pnpm exec playwright test --max-failures=5` | CI non-isolated subpath: 66 passed, two intentional skips, zero failed/flaky/retries. |
| `pnpm test:e2e:recovery` | CI: two fault-enabled recovery tests passed, one per browser. |
| `env -u VITE_E2E_FAULTS pnpm build` then `pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --grep '@fault-disabled'` | CI: normal rebuild and both fault-disabled checks passed. |
| Existing five-pair reference-performance gate | CI passed all engine comparisons and independent worker budgets; no engine speed gain claimed. |
| `git diff --check` and scoped source review | Passed; protected implementation/configuration paths unchanged. |

Local bundle output, in bytes:

```text
entry gzip 151201 / 152245
InspectionPanel gzip 5432 / 7373
total JavaScript gzip 234147 / 234161
```

Only 14 bytes of total JavaScript headroom remain in the local build. The fresh CI build independently passed the original guard. No cap was increased.

Local environment: Node v22.16.0, pnpm 9.15.9, exact frozen offline dependencies. CI uses its fresh frozen install; reference measurements ran on an ARM64 Apple M1 virtual runner with Node v20.20.2 and pnpm 9.15.9. The accepted reference baseline is `ae09b9863ae90f8fb2f62545834fcc138755ba9a`. The comparison result and raw samples are preserved separately.

## Changes and boundaries

Commits: `70fde76` diagnostics; `e6008ee` drawer; `2655584` metric tracks; `76ff339` intermediate wrapping; `8eac28d` lesson test wait; `88e107a` phone Epoch/status correction. Seven code/test files changed from intake: 158 insertions, 10 deletions, before documentation closure.

No changes to App training ownership, engine/shared implementation, worker messages, stores, hooks, persistence formats, dependencies, manifests, lockfile, CI workflow or thresholds. No Sentry frequency or production-latency claim is made. Linear showed no visible overlapping issue. Local application browser navigation was administrator-blocked; full application proof comes from CI. Inline diff review is not an independent security audit or proof that no undiscovered bug exists.

## Outstanding high-risk item

`BUGS-TO-REVIEW.md` item 1 remains open and unchanged: independent saved-run stores can both report success while one whole-envelope write overwrites the other. The medium-risk patch does not repair cross-tab write coordination. Tackle this data-loss risk next, with explicit conflict semantics and controlled two-tab concurrency tests.
