# Bugs to review — September 9, 2026

Audit baseline: `cad28a4e6e076245cba7191cea2e78c711335003` on `codex/nn-forge-precision-lab`. Source locations below refer to this audit's source; the scoped fixes do not modify the deferred implementation files.

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

### [ ] 2. History drawer does not close after a touch hit at 320px and 390px

**Risk: medium — touch navigation remains obstructed.**

**Locations:** `apps/web/src/styles/precisionLab.css:123-156`; `apps/web/src/styles/forge.css:500-509`, `:2791-2864`; `apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx:106-108`, `:247-266`; `apps/web/src/App.tsx:140-145`. Reproduction: `tests/e2e/playground-smoke.spec.ts:839-846`, `:886`.

**Symptom and evidence:** Exact-baseline Chromium CI opens History, measures the close target, performs the coordinate touch interaction, and still finds `#forge-surface-history.precision-drawer` visible. This occurs at both 320px and 390px in the preview job. It is not established that the close callback is reached. Local app navigation was blocked by administrator browser policy, so no local full-page reproduction is claimed.

**Suspected root cause:** The mobile drawer is fixed at `top: 156px` with z-index 30, while the wrapping header uses z-index 100 and variable height. A higher header may intercept the close hit; focus-induced scrolling or touch timing could also contribute. These are hypotheses, not confirmed attribution. This is separate from the confirmed desktop topology-overlay stacking bug corrected in this audit.

**Why deferred:** The available CI failure does not identify the event recipient. Raising z-index, moving the drawer by a guessed constant, or changing focus timing would change layout/interaction behavior without confirming the root cause.

**Suggested approach:** Capture the close-button bounding box, `document.elementFromPoint` at its center, header bounds, visual viewport, active element and click recipient immediately before the touch. Reproduce with genuine touch input in Chromium and WebKit, at both widths, zoomed and scrolled, with evaluation disclosures open and closed. Fix the demonstrated overlap or event ownership while keeping the non-modal drawer and focus-return contract intact.

**References:** GitHub Actions run `34273895548`, preview job `102222301967`, touch-shell failures at 320px and 390px. Sentry unavailable; no visible Linear match.

### [ ] 3. Training still produces unattributed layout shifts in desktop Build and Run

**Risk: medium — visual instability and a failing release-qualification contract.**

**Locations:** `tests/e2e/precision-lab-layout.spec.ts:65-94` (measurement and failing assertion). Candidate presentation areas to investigate, not confirmed causes: `apps/web/src/components/layout/Header.tsx:273-313`, `apps/web/src/components/controls/CurrentRunCard.tsx`, `apps/web/src/styles/precisionLab.css:185-200`.

**Symptom and evidence:** At 1437x742, the baseline preview CI reports non-input-associated layout-shift sums of `0.0010301131812217451` in Build and `0.0009349299441882596` in Run. The existing contract requires exactly zero. The preceding outer-region geometry checks pass, so stable outer boxes do not establish stable content. These are baseline CI measurements, not before/after optimization results.

**Suspected root cause:** Live metric/caption updates or internal content wrapping may move descendants without moving the tracked outer regions. The current observer records values but not the shift source nodes, so exact responsibility is unknown.

**Why deferred:** There is no attributed moving element yet. Broad fixed-height/min-height changes could clip evidence or change responsive behavior. No layout threshold or scientific update cadence should be relaxed to make the test green.

**Suggested approach:** Extend diagnostic evidence to include each shift source node and its previous/current rectangles, then identify the specific changing presentation box. Reserve or constrain only the proven source's footprint, preserving complete metric/evidence text. Verify Build and Run with loaded fonts, normal preview and subpath hosting, both engines, and compact layouts. Keep the zero-shift and overflow gates unchanged.

**References:** GitHub Actions run `34273895548`, preview job `102222301967`, `precision-lab-layout.spec.ts:94`. Sentry unavailable; no visible Linear match.

## Coverage follow-ups — not confirmed vulnerabilities

- [ ] Restore Sentry access, select the relevant project/environment and time window, retrieve frequent errors and slow transactions, and correlate stack/source maps and releases. Reorder the checklist from actual production evidence. No transaction name was available to designate as a post-deploy performance monitor; locate the real code-export/evidence interaction rather than inventing one.
- [ ] **[VERIFY WITH SCANNER]** Run `pnpm audit --audit-level high` in a network-enabled environment. This audit attempted `pnpm audit --json` with bounded retry/timeout settings; the registry audit request failed with `EAI_AGAIN`. This is no vulnerability verdict, and no CVE is asserted.
- [ ] Complete browser qualification on the newly published source. The baseline preview job reported 20 passed, 5 failed, 2 skipped and 39 not run. Local real-site browser attempts were blocked by `ERR_BLOCKED_BY_ADMINISTRATOR`; isolated stylesheet hit-tests do not replace the full application suite.

## Scoped review notes

URL payloads and imported files were traced through size checks, decoding/JSON parsing, schema validation and experiment preparation before state publication. Displayed code and labels are React text, not executable HTML. The reviewed application has no server-side user/account API or authorization layer; local browser storage is not an authorization boundary, and authentication would not substitute for authorization in any future shared backend.

A targeted pattern scan of 419 tracked UTF-8 files found no matches for the selected private-key/token patterns. It was not a dedicated secret scanner, a Git-history scan, or evidence that every possible credential is absent. No secret value was printed. Production dependencies inspected have concrete import/tooling uses; no dependency or lockfile changes were made. Manifest ranges are constrained by the committed lockfile and CI's frozen install. Legacy migration components were retained intentionally.

This checklist does not authorize changing the engine, scientific identities, worker protocol, stored schema, release budgets, branch protections, `main`, or deployment settings.
