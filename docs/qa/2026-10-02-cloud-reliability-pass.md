# Cloud reliability and browser evidence — October 2, 2026

Executed the [cloud reliability plan](../superpowers/plans/2026-10-02-cloud-reliability-pass.md) against application commit `a48ff98eb261688cab3a17975aa650df24195a45`, freshly confirmed against `origin/main`. Application source, tests, manifests, and lockfile are unchanged. This pass adds evidence and fix scopes; the two reproduced issues below remain unfixed.

## Results

| Check | Observed result |
| --- | --- |
| Normal production build | Passed |
| Chromium visual gallery | 2 passed, 0 skipped; 20 states per theme |
| WebKit visual gallery | 2 passed, 0 skipped; 20 states per theme |
| Focused Share/mobile probes | Completed in both engines and both themes; 24 Results/Setup captures at 320, 360, and 390px; zero console/page errors |
| Injected worker startup failure and recovery | 2 passed, 0 skipped; both engines |
| Normal rebuild and fault-disabled verification | Build passed; 2 browser tests passed, 0 skipped |
| Full non-isolated project-subpath suite | 156 passed, 6 mode skips, 0 failures/flaky tests; all four hosting contracts executed |
| Review supplements | Recovery focus: both engines passed; running status: all 12 engine/theme/width cases measured; final normal rebuild and 2 repeated fault-disabled tests passed |
| Restored normal-build gzip budgets | Entry 150601 / 152245; InspectionPanel 6590 / 7373; total JavaScript 233101 / 234161 bytes |

The previous 1,838 unit/integration tests, 156 helper tests, lint, type checks, and 152 standard browser passes remain baseline evidence. They were not repeated for this documentation-only pass. Browser mode reports are archived separately; each listed run uses zero retries.

The gallery exercised 80 application states across engines and themes, including network inspection, actual training, lessons, saved comparisons, exports, a quota-failure screen, and checkpoints. Screenshot capture is evidence of the exercised state, not a pixel-golden comparison or user study.

## Screenshots

- [Desktop Network, Chromium, light, 1440×1024 viewport](cloud-reliability-2026-10-02/desktop-network-chromium.png).
- [Phone Results, Chromium, dark, 390×844 viewport](cloud-reliability-2026-10-02/mobile-results-chromium.png).
- [Overlapping trained phone metrics, Chromium, light, 320×844 viewport](cloud-reliability-2026-10-02/mobile-overlap-320-chromium.png).
- [Running phone status and metric overlap, WebKit, dark, 320×844 viewport](cloud-reliability-2026-10-02/mobile-running-overlap-320-webkit.png).
- [Share setup reopening Code, WebKit, light, 1440×1024 viewport](cloud-reliability-2026-10-02/share-entry-webkit.png).

Images are full-page captures, so image height can exceed viewport height. [Raw focused probe results](cloud-reliability-2026-10-02/probe-results.json) retain model identity, selected tab, element/text rectangles, touch-control dimensions, and console errors.

## Confirmed findings and focused fix scopes

### P2 — Trained Step/Epoch values overlap on phone layouts

**Impact:** The training position becomes hard to read during ordinary use. This is a presentation failure; the model, evaluation results, and stored data were not shown to be corrupted.

**Reproduction:** Open the normal app, set speed to 50×, train beyond 1,200 steps, and pause. View Results or Setup at widths 320, 360, and 390px. The measured paused Chromium run reached step 2,550 / epoch 170. Step text overlaps Epoch text by 15 CSS pixels; WebKit overlap is approximately 14.11px. Epoch also reaches about 5px into the status text's horizontal space. This reproduces in both themes and both browsers, across all 24 captured Results/Setup cases. Screenshots confirm actual glyph collisions. The page itself has no horizontal overflow, and measured transport buttons/selects retain at least 44×44px dimensions. A review-requested [running-status probe](cloud-reliability-2026-10-02/running-status.json) additionally covers all 12 engine/theme/width combinations while training: status text fits its 36px height without horizontal or vertical overflow, but Epoch glyph rectangles intersect its text rectangles in every case. Thus the long status does not introduce a separate overflow defect; it exposes the same fixed-column collision. Running model values can advance between measurement and screenshot capture.

**Root cause:** `apps/web/src/styles/atelier.css:198–210` assigns compact transport columns `44px 52px 44px minmax(60px,1fr) 44px`. Step and Epoch occupy the first two fixed button-width columns, while their non-wrapping label/value text grows beyond those columns. Status begins in column 3. These grid tracks suit the icon controls but cannot accommodate longer metric text. `AtelierTransport.tsx` correctly displays the actual step and epoch; the formatting must remain intact.

**Proposed fix scope:** Change compact metric/status placement in `apps/web/src/styles/atelier.css` so complete metric text has independently sufficient space; retain touch control sizes and full scientific values. Do not shrink labels into unreadability, truncate values, or suppress updates.

**Regression scope:** Add a focused case to `tests/e2e/precision-lab-layout.spec.ts` using real training until four-digit steps and three-digit epochs. At 320/360/390px in both engines/themes, require disjoint Step, Epoch, and status text rectangles as well as unchanged model identity across Results/Setup, existing no-overflow checks, and 44px controls. The current baseline should fail the overlap assertion. After a fix, rerun that case plus the existing layout/zoom checks and bundle guard.

### P3 — Share setup reopens Code instead of the sharing controls

**Impact:** The specific Share action takes users to unrelated export content and requires an extra tab selection. Sharing remains available; this is not a lost-recipe or blocked-export issue.

**Reproduction:** At desktop width, use Utilities → Export / import → Code → TF.js, close the dialog, then click Share setup. In all four engine/theme combinations, the selected export tab is Code and Copy setup link is not visible. Selecting Setup & sharing reveals the link control. Model generation, revision, training step, and URL remain identical before and after the sequence.

**Root cause:** `apps/web/src/App.tsx:97` keeps `exportTab` in state. The Share setup handler at line 176 calls the generic `openUtility('exports')`, whose action at lines 121–124 opens the surface without selecting the setup tab. The existing explicit export-request effect at lines 109–113 does select its requested mode. General Utilities access can reasonably retain the last selection, but the specifically labeled Share action should target sharing.

**Proposed fix scope:** Give the Share entry point an explicit setup-tab action inside the existing draft guard. Preserve the generic Utilities tab selection and Code language selection, and preserve Stay/Discard/Apply draft behavior.

**Regression scope:** In `tests/e2e/atelier-critical-flows.spec.ts`, reproduce the sequence and require Setup & sharing to be selected with Copy setup link visible and model identity unchanged. Cover a pending setup draft: Stay must retain the draft; Discard/Apply must follow the existing guard and then reveal sharing. The current baseline should fail the entry-tab assertion. Keep any state-level coverage in `apps/web/src/App.test.tsx` consistent with the browser behavior.

## Recovery and hosting evidence

Fault-enabled verification exercised the accessible error dialog, focus trap, refresh, return to step zero, and a successful real training step in both engines. The separate normal rebuild then passed both fault-disabled tests. The final build is normal, not a fault-injection build. The independent reviewer requested explicit focus evidence before helper clicks: [the added observation](cloud-reliability-2026-10-02/recovery-focus.json) shows BODY focus immediately after reload, outside hidden/inert content, followed by Tab reaching Skip to main content in both engines. A real training step then succeeds. This is the normal focus sequence after a full-page refresh, not restoration to the now-removed error dialog. After this extra fault-enabled probe, the app was rebuilt normally again; both fault-disabled checks and the unchanged bundle limits passed again.

The hosting fixture returned HTTP 200 at `/neural-network-playground/` with no COOP/COEP headers. It serves the actual production assets without cross-origin isolation, exercising the transferable worker fallback. All four hosting contracts passed, including successful worker training, lazy assets, and shared-recipe reloads. The full suite also passed exact-artifact save/retry and two-tab concurrency checks. Its six skips are the four opt-in gallery cases and two fault-enabled cases, all executed successfully in their separate runs above. This is a cloud hosting simulation, not proof of a live Pages deployment.

Full logs, browser reports, and source/command receipts are under `/workspace/outputs/cloud-reliability/`. `probe.mjs` there reproduces the focused measurements using the repository's real Playwright helpers. Every report is archived before the next mode overwrites Playwright's default paths. The [archived result summary](cloud-reliability-2026-10-02/summary.json) records per-mode statistics and every test outcome.

## Independent review

An Astra reviewer checked the report against raw screenshots, Playwright results, command receipts, CSS/handler source, fixture headers, and save/retry coverage. Two coverage gaps were identified: immediate post-recovery focus and long running-status geometry at all phone widths. Both were closed by the supplemental observations above. There were no deferred minor review findings. Supplemental commands ran with staged documentation/evidence additions; application source remained unchanged.

## Limits and execution decisions

- The deployed site and physical phones were not tested. Chromium/WebKit phone layouts are cloud emulation.
- No performance qualification is claimed. The existing five-pair same-host reference policy remains the requirement for such a claim.
- The historical favicon defect is already fixed in the tested main. Historical bundle headroom and old unexecuted checkboxes were not treated as current findings.
- Skill helper resources were unavailable and the checklist tool rejected this session mode, so the execution ledger uses explicit task records and the repository's command-receipt wrapper. This reduces bookkeeping automation, not browser assertions.
- Scope remains the approved verification/reporting plan: preserve raw evidence and produce fix scopes. The two defects need a subsequent implementation pass; no source changes or new publication were included.
