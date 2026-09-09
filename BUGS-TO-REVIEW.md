# Bugs to review — September 9, 2026

Audit baseline: `cad28a4e6e076245cba7191cea2e78c711335003` on `codex/nn-forge-precision-lab`. Medium-risk closure source: `88e107a954cae016f4da3a71ffb29225d92028b7`, verified by Actions run `34410864082`. The high-risk persistence finding and its implementation remain unchanged.

## Evidence and ordering

Sentry discovery and plugin lookup exposed no callable Sentry integration in this session. Error frequencies, affected-user counts, severity distributions and slowest transactions are **unknown, not zero**. Linear was checked before edits: the connected account returned no visible non-archived issues and no matching project. There are therefore no verified Sentry or Linear IDs to attach.

Until telemetry is available, ordering below uses demonstrated potential impact: silent saved-run loss first, then blocked touch navigation, then visual instability. This is not a production-frequency ranking. Browser evidence comes from exact-baseline GitHub Actions run [34273895548](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34273895548), preview job `102222301967`. That job stopped after five failures; unexecuted cases are not passing cases.

## Deferred bugs

### [ ] 1. Independent saved-run stores can overwrite another successful save

**Risk: high — possible silent saved-run data loss.**

**Locations:** `apps/web/src/store/experimentMemoryStore.ts:72-78`, `:95-120`, `:203-233`; `apps/web/src/hooks/useExperimentMemoryStorageSync.ts:22-25`. Existing same-instance overlap coverage: `apps/web/src/store/experimentMemoryStore.test.ts:97`.

**Symptom and evidence:** A deterministic local test creates two independent stores, hydrates both from the same empty localStorage envelope, and concurrently saves two different valid records. Both `saveRecord` calls resolve `true`, but the persisted envelope contains only the second record. The assertion expecting both IDs fails. This proves the independent-store lost-update case; a full two-browser-tab reproduction has not yet been executed, and production prevalence is unknown.

**Suspected root cause:** The promise queue belongs to each `createExperimentMemoryStore()` instance. Each instance assembles its candidate from its own `get().records`, then performs asynchronous serialization/verification followed by an unconditional whole-envelope `localStorage.setItem`. Storage-event hydration synchronizes views after writes but does not serialize writers or make the read/modify/write atomic.

**Reproduction:** Reuse the existing `makeRecord` fixture in `experimentMemoryStore.test.ts`; run the following as a temporary test alongside it. Do not permanently add a failing assertion to the normal regression gate before a fix is agreed.

```ts
window.localStorage.clear();
const first = createExperimentMemoryStore();
const second = createExperimentMemoryStore();
await Promise.all([first.getState().hydrate(), second.getState().hydrate()]);
const a = await makeRecord(0);
const b = await makeRecord(1);
expect(await Promise.all([
    first.getState().saveRecord(a),
    second.getState().saveRecord(b),
])).toEqual([true, true]);
const stored = JSON.parse(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)!);
expect(stored.records.map((record: { id: string }) => record.id).sort())
    .toEqual([a.id, b.id].sort()); // Fails: only one ID remains.
```

**Why deferred:** This is a concurrency and persisted-data problem, not a safe local cleanup. Save/delete/rename/eviction/retry need one coherent conflict policy. An ad hoc merge immediately before writing is still racy, and may mishandle rejected records or resurrect deletions.

**Suggested approach:** First add a genuine two-tab reproduction with controlled interleavings. Agree cross-context write ownership and conflict semantics, then implement coordinated or transactional read/validate/modify/write. Preserve rejected/incompatible bytes, capacity rules, and the exact pending-save retry artifact. Test simultaneous save/save, save/delete, rename/save, quota failure, stale hydration, and an incompatible envelope appearing during a write. No storage schema change should be bundled into this audit.

**References:** Sentry unavailable; no visible Linear match. Local independent-store regression probe; not a Sentry-reported incident.

## Resolved medium-risk bugs

### [x] 2. History drawer does not close after a touch hit at 320px and 390px

**Original risk: medium. Resolved and regression-tested.**

**Locations:** `apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx:108-116,282,344`; `apps/web/src/styles/precisionLab.css:11-13,159`. Regression: `tests/e2e/precision-drawer.spec.ts` and the existing touch-shell journeys.

**Confirmed root cause:** Real-browser hit diagnostics at 320px placed the close button at y=163 while the wrapping header extended to y=248.5. The header, not the close control, received the touch. The old fixed `top: 156px` did not track actual header height.

**Fix:** Render drawers into the existing outer shell and position them in its actual workspace grid row, outside the scrolling/paint-contained workspace. Keep the existing non-modal semantics, IDs, close command, and trigger-focus return. Initial focus uses `preventScroll`; no guessed header offset or global z-index escalation remains.

**Verification:** The final Chromium and WebKit runs each exercise all three drawers at 320px and 390px with header details both closed and expanded: 24 successful measured close hits per hosting mode, 48 across normal preview and project subpath. Checks include 44px targets, the actual hit recipient, close/return focus, unchanged URL, checkpoint, status and disclosure state. The original touch-shell journeys also pass. Three focused portal tests failed before implementation and pass after it.

**References:** Diagnostic run `34406669613`; successful final run [34410864082](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34410864082), code `88e107a9`. No verified Sentry or Linear reference was available.

### [x] 3. Training causes layout shifts in desktop, tablet and phone Build/Run

**Original risk: medium. Resolved and regression-tested for the required scenarios.**

**Locations:** `apps/web/src/styles/precisionLab.css:205-259`; `apps/web/src/components/controls/TrainingControls.tsx:271`. Regression and source attribution: `tests/e2e/precision-lab-layout.spec.ts:65-112`.

**Confirmed root cause:** Intrinsic-width flex rows let changing step/epoch digits and evaluation text reposition neighboring controls. Source-node diagnostics additionally identified tablet run metadata wrapping at five-digit step counts, a phone Epoch value changing lines at three digits, and the footer credit moving as STEP grew.

**Fix:** Use stable metric tracks and reserve the observed wrapping space. Keep complete scientific text and values. Let the intermediate header size to its actual content. The phone Epoch label/value always use two lines, and the phone status credit occupies its own full-width grid row. Only a decorative spacer is hidden; no metric, update, error or evidence is suppressed.

**Verification:** Final preview and subpath each pass Build and Run at 1437x742, 735x860 and 320x844 in both browsers. Chromium records exactly zero non-input-associated layout shifts in all 12 measured scenarios across the two hosting modes, during 5.5 seconds at 50 steps/frame. Both browsers retain the one-pixel outer-region movement/overflow limits, single connected live canvas and unchanged URL. WebKit passes geometry checks; its absent layout-shift API is not treated as a CLS measurement. No threshold or scientific cadence changed.

**References:** Source-attribution runs `34406669613`, `34408145015`, `34409044281`; successful final run [34410864082](https://github.com/DenseDevKev/neural-network-playground/actions/runs/34410864082), code `88e107a9`. No verified Sentry or Linear reference was available.

## Coverage follow-ups — not confirmed vulnerabilities

- [ ] Restore Sentry access, select the relevant project/environment and time window, retrieve frequent errors and slow transactions, and correlate stack/source maps and releases. Reorder the checklist from actual production evidence. No transaction name was available to designate as a post-deploy performance monitor; locate the real code-export/evidence interaction rather than inventing one.
- [ ] **[VERIFY WITH SCANNER]** Run `pnpm audit --audit-level high` in a network-enabled environment. This audit attempted `pnpm audit --json` with bounded retry/timeout settings; the registry audit request failed with `EAI_AGAIN`. This is no vulnerability verdict, and no CVE is asserted.
- [x] Complete browser qualification on code `88e107a9`: preview 62 passed / 6 intentional skips; subpath 66 passed / 2 intentional skips; zero failures, retries or flaky results. Fault-enabled recovery and normal rebuilt fault-disabled checks also pass in both browsers. Local application navigation remains administrator-blocked, so full application evidence is from GitHub Actions, not isolated local stylesheet fixtures. This is not a live deployment receipt.

## Scoped review notes

URL payloads and imported files were traced through size checks, decoding/JSON parsing, schema validation and experiment preparation before state publication. Displayed code and labels are React text, not executable HTML. The reviewed application has no server-side user/account API or authorization layer; local browser storage is not an authorization boundary, and authentication would not substitute for authorization in any future shared backend.

A targeted pattern scan of 419 tracked UTF-8 files found no matches for the selected private-key/token patterns. It was not a dedicated secret scanner, a Git-history scan, or evidence that every possible credential is absent. No secret value was printed. Production dependencies inspected have concrete import/tooling uses; no dependency or lockfile changes were made. Manifest ranges are constrained by the committed lockfile and CI's frozen install. Legacy migration components were retained intentionally.

This checklist does not authorize changing the engine, scientific identities, worker protocol, stored schema, release budgets, branch protections, `main`, or deployment settings.
