# NN.FORGE — Living Execution Plan

**Role of this file:** This is the checklist I will actually execute against. It is deliberately more detailed than a normal roadmap so you can audit what I am doing.

## Status notation
- `[x]` completed with evidence.
- `[ ]` pending.
- `[!]` blocked / requires an explicit decision.
- `[-]` deferred or out of scope.
- I will update **this same file** as work progresses rather than creating a new plan every time.
- A box is not checked merely because code was written; the listed exit condition must be satisfied.

---

# 0. Control state

**Reconciled September 10, 2026 against GitHub refs and exact-source CI. Sections explicitly marked historical are receipts, not current acceptance.**

- Repository: `DenseDevKev/neural-network-playground` (private).
- Product authority: `main` at `98f29b86e469a2a545be75928ae6f32309fd1582`.
- Release baseline: **merged and accepted**; merge parents are `ae09b986...` and `2cd5b896...`. Main CI `34002500733` succeeded.
- Active branch: `codex/nn-forge-precision-lab`.
- Original continuation base: `3e7da9a4dabd2d7bc275c9aa4c614a04d9b26732`, not the supplied older `6e06b678...`.
- Maintenance intake: `6de861dd4e00f3223ec8985a0b6c00bdbfa7abe9`, tree `12a57ef8e463e7386c687b602f2501de088d798c`.
- Latest maintenance implementation: `7ec856a39079141a040624e9fe2927218e0db8f8`, tree `4447aed98747a411e9fef2211cb2785de1c2bde6`. This file's documentation closure is a descendant, not a new application feature.
- Exact-source maintenance CI receipt: `5092ee69abc8714ff906d5d63bbd5b5263f62368`, run `34529946854`; correctness, types, lint, build, recovery and accepted reference performance pass. The total-JavaScript guard still fails. `7ec856a3` only tightens reporting source-tree validation; its documentation-head CI is inspected separately.
- Main composition: accepted Build/Run presentation. Active-branch composition: **PrecisionLabShell already integrated**, not a future shell.
- Precision Lab merged to main: **no**. Successful Pages deployment: **no**.
- Continuation implementation is **committed on the active GitHub branch**: 32 non-workflow files at `20831874d436aac55c3c100ffe84949e967e8a37`, qualification workflow at `3b567950d3c725e5dc2856b4331813db26edba39`, and targeted browser corrections at `950142880d11e78d58eaa271904c6c8a4d2facb9`.
- Historical browser-fix qualification: code `88e107a9` passed run `34410864082`. Later selection/zoom additions and graph-control correction reached intake `6de861dd`; run `34522259192` failed preview/subpath and bundle while correctness, recovery and the accepted performance comparison passed. Do not carry the older green browser receipt onto this newer candidate.
- Maintenance scope is closed by section 27 and `docs/maintenance/2026-09-10-review.md`; Precision Lab release acceptance remains open. No inherited acceptance failure is waived.
- Source recovery: verified GitHub Actions source archive and matching-lockfile dependency workspace; no previous Mac was assumed.
- Canonical checklist: this same file was recovered from the user's Library and restored at repository root in the September 8 continuation. It is now maintained in GitHub; numbered milestones and historical receipts remain intact.

**Evidence boundary:** `[x]` below requires identified committed implementation and test evidence, or a directly verified historical/ref event. The continuation is now committed; acceptance boxes remain open until the exact candidate satisfies their required gates. Passing component/unit tests do not substitute for browser geometry, accessibility, recovery, performance, or release gates.

## Non-negotiable product invariants

- Current recipe → Trained snapshot → Active run → Evidence → Saved runs.
- Navigation/profile/disclosure changes must not mutate the experiment or V2 URL fragment.
- App owns training. Presentation receives typed display models and commands, never raw worker envelopes.
- Exactly one production selection controller and one canonical live boundary.
- Batch/EMA signals remain distinct from paired full-split evaluations; evaluation age and recipe drift stay explicit.
- Saved runs store recipe/evidence, not parameters. Apply Saved Recipe does not restore a model.
- Retry persists the exact pending artifact; it must not recapture the current model/recipe.
- Checkpoint restore does not promise an identical future stochastic trajectory.
- Visibility optimization cannot reduce scientific evaluation cadence.
- No engine/schema/protocol/persistence format changes, threshold increases, historical branch import, or visibility change without separately justified work.
- “Pinned boundary” is the always-mounted live boundary rail, **not** a saved pin/replace/clear feature.
- Tool-output/read failures are infrastructure failures, not evidence of an application defect.

---

# 1. Milestone R — Qualified release baseline

**Historical milestone completed and merged.** R tasks below describe the accepted release baseline, not work that must block Precision Lab again. Evidence: release record `docs/superpowers/verification/2026-09-05-release-roadmap.md`, qualified code `fbae4b98...`, run `34001183202`, merge `98f29b86...`, main CI `34002500733`.


**Goal achieved:** establish a stable release before Precision Lab presentation integration.

## R0 — Source authority

### R0.1 Pin the baseline
- [x] Confirm authoritative `main`.
- [x] Record exact intake SHA.
- [x] Record exact execution branch.
- [x] Avoid treating old local state as authority.

**Exit:** one unambiguous source baseline exists.

### R0.2 Isolate implementation
- [x] Keep release-readiness work off `main`.
- [x] Avoid wholesale merge of historical branches.
- [x] Preserve `main` until qualification is accepted.

**Exit:** unqualified work cannot silently become the product authority.

---

# 2. Browser / deployment infrastructure

**Historical accepted-baseline evidence.** These checks are not carried forward as a pass for the new Precision Lab candidate; its acceptance remains in sections 17–19.


## R1 — Explicit Playwright targets

- [x] Support explicit `PLAYWRIGHT_BASE_URL`.
- [x] Preserve project subpath.
- [x] Normalize trailing slash.
- [x] Reject credentials.
- [x] Reject malformed target URLs.
- [x] Reject base URL query/fragment.
- [x] Reject conflicting explicit port/base settings.
- [x] Preserve localhost preview defaults.
- [x] Preserve Chromium and WebKit projects.

**Exit:** local and external E2E targets are deterministic and fail closed.

## R2 — Static-host project-subpath fixture

- [x] Serve actual `apps/web/dist`.
- [x] Serve below `/neural-network-playground/`.
- [x] Deliberately omit COOP/COEP.
- [x] Serve correct JS MIME.
- [x] Preserve canonical trailing-slash redirect.
- [x] Return true 404s for missing assets.
- [x] Reject traversal.
- [x] Reject symlink escape.
- [x] Restrict to GET/HEAD.

**Exit:** the fixture behaves like a real static project-site host, not a fake app server.

## R2.1 Project-subpath contracts

- [x] Worker loads beneath project path.
- [x] Non-isolated worker fallback works.
- [x] Manual step reaches paired evaluation evidence.
- [x] Inspection lazy module loads.
- [x] Code lazy module loads.
- [x] History lazy module loads.
- [x] Configuration lazy module loads.
- [x] Canonical V2 shared URL survives reload.
- [x] Shared URL preserves recipe + architecture.
- [x] Reopened URL gets a fresh runtime model.
- [x] Project path remains intact.

**Exit:** production bytes work under realistic non-root static hosting in both browsers.

---

# 3. Reproduced product defects

**Historical accepted-baseline evidence.** These checks are not carried forward as a pass for the new Precision Lab candidate; its acceptance remains in sections 17–19.


## R2.2 Skip-link experiment-state defect

### Reproduction
- [x] Create a valid shared recipe URL.
- [x] Take a training step.
- [x] Activate skip link with pointer.
- [x] Activate skip link with keyboard.
- [x] Prove old behavior replaced experiment hash with `#main-content`.

### Fix
- [x] Prevent default fragment navigation.
- [x] Focus the existing main landmark programmatically.
- [x] Avoid serialization/worker/training changes.

### Verification
- [x] URL unchanged.
- [x] generation unchanged.
- [x] revision unchanged.
- [x] step unchanged.
- [x] main receives focus.
- [x] Chromium passes.
- [x] WebKit passes.

**Exit:** accessibility navigation cannot mutate experiment state.

## R2.3 WebKit font/recovery defect

### Diagnosis
- [x] Preserve strict console-error assertion.
- [x] Confirm worker recovery itself succeeds.
- [x] Identify failures as external font resource-policy errors.
- [x] Test anonymous-CORS hypothesis.
- [x] Revert failed CORS hypothesis.

### Correction
- [x] Pin Inter.
- [x] Pin Space Grotesk.
- [x] Serve font assets from application origin.
- [x] Preserve requested font weights.
- [x] Include font license notices.
- [x] Use real browser loads, no request interception.
- [x] Test native reloads.
- [x] Require no Google-font requests.
- [x] Require no font/page/console errors.
- [x] Preserve original recovery assertion unchanged.

### Verification
- [x] Chromium recovery.
- [x] WebKit recovery.
- [x] Normal preview.
- [x] Project-subpath fixture.
- [x] Clean rebuild.
- [x] Fault-disabled check.

**Exit:** font delivery no longer causes conditional-reload failures.

---

# 4. Repeatable release qualification

**Historical accepted-baseline evidence.** These checks are not carried forward as a pass for the new Precision Lab candidate; its acceptance remains in sections 17–19.


## R3.1 Workflow structure

- [x] source-evidence job.
- [x] correctness job.
- [x] preview-browser job.
- [x] project-subpath-browser job.
- [x] recovery job.
- [x] performance job.
- [x] preserve failure artifacts.
- [x] preserve exact source SHA.
- [x] preserve dist SHA-256 manifest.
- [x] use read-only repository permission.
- [x] no deployment side effect.

## R3.2 Correctness baseline

- [x] infrastructure helpers pass.
- [x] lint passes.
- [x] typecheck passes.
- [x] engine tests: **497 passed**.
- [x] shared tests: **334 passed**.
- [x] web tests: **1,043 passed**.
- [x] total package tests: **1,874 passed**.
- [x] production build passes.
- [x] source remains clean after build.

**Historical helper/build receipts:** accepted baseline had 88 helpers (run `34001183202`). The September 8 local continuation had 91 helpers and 87 byte-identical rebuilt files with a passing bundle guard; that was not browser recovery. Current maintenance counts and the failing current bundle gate are recorded in section 27.

---

# 5. P1 pulled forward — JavaScript bundle contract

**Caps unchanged. Current maintenance CI `5092ee69` measures entry 151,342 / 152,245, InspectionPanel 5,431 / 7,373 and total JavaScript 234,282 / 234,161 bytes: total is 121 bytes over the cap, so this gate FAILS.** Local intake and maintenance both measure 234,285 total (124 bytes over); their 13 emitted files are byte-identical. Local recovered dependencies omit font binaries, so complete asset fidelity comes from CI. September 8 values of 234,068 total and 93 bytes of headroom are historical and superseded. Evidence: run `34529946854`, compact focused summary and preserved build manifest.


## P1.1 Implement executable checker

- [x] Add `scripts/check-web-bundle-gzip.mjs`.
- [x] Add Node tests.
- [x] Resolve real app entry instead of guessing hashed filename.
- [x] Require exactly one InspectionPanel chunk.
- [x] Measure all JavaScript recursively.
- [x] Include worker in total JS.
- [x] Reject missing/ambiguous inputs.
- [x] Reject traversal/symlinks.
- [x] Reject invalid numeric limits.
- [x] Fail on a one-byte overrun.
- [x] Report all failed dimensions.
- [x] Add `pnpm test:bundle`.
- [x] Integrate with CI.
- [x] Integrate with release verification.

## P1.2 Fixed reviewed caps

- [x] Entry gzip ≤ **152,245 bytes**.
- [x] InspectionPanel gzip ≤ **7,373 bytes**.
- [x] Total JS gzip ≤ **234,161 bytes**.

## P1.3 Accepted baseline measurements (`fbae4b98...`, run `34001183202`)

- [x] Entry: **146,940 bytes**.
- [x] InspectionPanel: **5,414 bytes**.
- [x] Total JavaScript: **226,417 bytes**.

**Exit:** fixed documented limits are executable and green without raising them.

---

# 6. RESOLVED — Engine performance policy and accepted baseline

## Q1 disposition — do not reopen the completed investigation

- [x] Establish historical benchmark provenance: same-isolated-machine regression semantics from the July 11 Scientific Trust baseline.
- [x] Implement fail-closed comparison of five baseline and five candidate collections, alternating collection order.
- [x] Retain exact baseline `ae09b9863ae90f8fb2f62545834fcc138755ba9a` and candidate/host metadata.
- [x] Require candidate engine medians <= 120% of the same-runner baseline medians.
- [x] Preserve independent fixed worker limits: forced paired evaluation <= 250 ms and save capture <= 500 ms.
- [x] Keep historical absolute engine constants unchanged; retain their failures in raw logs instead of treating them as universal-host release limits.
- [x] Qualify accepted release code `fbae4b98...` with all six release jobs green in `34001183202`.
- [x] Merge release baseline to main (`98f29b86...`) and verify main CI (`34002500733`).

**Provenance:** comparator RED `f6aa6937...`; implementation `27e0b244...`; corrected fixture `5d637eb1...` (88 helper tests); release workflow `11facd97...`; dedicated reference workflow `78465112...`; final code `fbae4b98...`. Dedicated reference run `34001077012` and release run `34001183202` passed. The committed release verification document preserves the raw historical investigation and all ten-run medians. No engine optimization was needed for policy resolution.

## Current candidate performance — separate from resolved policy

- [x] Run the unchanged accepted five-pair macOS/ARM64 reference gate on the maintenance implementation `5092ee69`: Actions `34529946854`.
- [x] Preserve exact candidate/host identity, raw samples, comparator result and independent worker budgets in `precision-performance-5092ee69...` and its compact summary.

The accepted comparison passed all six engine comparisons plus independent forced-pair and save-capture budgets. Candidate worker medians in that run were 9.7946 ms and 11.6225 ms; no engine speed gain is claimed. Reporting-only changes and the documentation closure are requalified at their exact head separately.

The September 8 Linux diagnostic (historical absolute engine tests failed; worker medians 8.0115/9.6054 ms passed) was not a reference qualification. The workflow is now committed and has run repeatedly; the old claim that it was an unpushed local patch is obsolete. The policy investigation remains resolved and the release baseline remains merged.

---

# 7. R4 — Consolidated repository handoff (historical baseline)

- [x] Commit consolidated roadmap, verification record, README, deployment and QA guidance, reconciliation and font notices.
- [x] Review baseline branch boundaries: no engine/shared implementation change, no prototype import, no hidden threshold change.
- [x] Limit product changes to the reproduced skip-link correction and same-origin font delivery; retain regression tests and qualification infrastructure.
- [x] Keep release qualification read-only and distinct from deployment.

Evidence: committed release verification record, qualified code `fbae4b98...`, run `34001183202`, accepted merge `98f29b86...`. “Precision Lab is future work” was accurate for that release only; active branch `3e7da9a...` now uses PrecisionLabShell.

New continuation audit is recorded separately in sections 17–18 and the September 8 verification document.

---

# 8. R5 — Exact accepted release qualification (historical)

- [x] Freeze qualified code `fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`.
- [x] Helpers 88; engine 497; shared 334; web 1,043; lint/typecheck/build/bundle/source cleanliness.
- [x] Isolated Chromium/WebKit: 42 passes, six intentional skips, no failures/flakes.
- [x] Project-subpath Chromium/WebKit: 46 passes, two intentional skips, no failures/flakes.
- [x] Fault-enabled recovery, normal rebuild and fault-disabled verification.
- [x] Accepted five-pair engine reference and independent worker budgets.
- [x] Preserve source SHA, dist manifest, raw logs and artifacts.

Evidence: `docs/superpowers/verification/2026-09-05-release-roadmap.md`, Actions `34001183202`. These are **baseline results**, not new-candidate browser/performance results. Subsequent accepted baseline merge is `98f29b86...`; current candidate qualification remains open.

---

# 9. R6 — Acceptance into main (complete)

- [x] Merge accepted release baseline into main.
- [x] Record main `98f29b86e469a2a545be75928ae6f32309fd1582` and parents `ae09b986...` / `2cd5b896...`.
- [x] Confirm post-merge CI `34002500733` succeeded.
- [x] Re-read the current main ref on September 8; it is unchanged.
- [x] Keep unqualified Precision Lab work off main.

Evidence: current GitHub ref, merge commit and main Actions run read through the GitHub connector. The original baseline/performance blocker is resolved; this is not a reason to delay implementation again.

---

# 10. R7 — Live deployment

**[!] Repository-settings blocker remains.** Deployment run `34002943565` at accepted main failed. There is no successful Pages receipt or verified live URL. Do not change repository visibility, claim a deployment, or interpret settings failure as a product regression. This continuation makes no deployment changes.


**Owner-gated because this changes external release state.**

## Deployment decision
- [ ] Confirm desired audience/visibility.
- [!] Resolve the existing Pages repository-settings / plan blocker through owner-authorized settings.
- [ ] Confirm deployment source.
- [ ] Confirm project base path.

## Deploy
- [ ] Deploy exact accepted `main`.
- [ ] Record deployment run.
- [ ] Record deployed SHA.
- [ ] Record artifact/digest.
- [ ] Record actual `page_url`.

## Live validation
- [ ] Chromium live load.
- [ ] WebKit live load.
- [ ] no console/page errors.
- [ ] local font assets load.
- [ ] worker asset loads.
- [ ] non-isolated fallback behaves correctly.
- [ ] step/evaluation flow.
- [ ] Inspection.
- [ ] Code.
- [ ] History.
- [ ] Configuration.
- [ ] shared V2 URL.
- [ ] fresh-context shared URL.
- [ ] same recipe/architecture.
- [ ] fresh runtime model.
- [ ] skip-link integrity.
- [ ] project path preserved.

**Exit:** real deployment behaves like the qualified static-host fixture.

---

# 11. Release checkpoint

- [x] Accepted baseline: `98f29b86e469a2a545be75928ae6f32309fd1582`.
- [x] Historical qualification: `fbae4b98...`, Actions `34001183202`; main CI `34002500733`.
- [x] Continue Precision Lab on `codex/nn-forge-precision-lab` from the accepted baseline.
- [!] Deployment SHA / live URL remain unavailable because Pages settings are blocked.
- [ ] Complete exact Precision Lab candidate qualification before merging it.

Ref recovery September 8 found newer committed integration at `3e7da9a...`; do not rewrite already integrated shell/boundary work from the older `6e06b678...` handoff.

---

# 12. Precision Lab — historical September 8 intake

- [x] Re-read committed consolidated release roadmap.
- [x] Read July 16 design and July 17 implementation plan against current source.
- [x] Recover exact current source through the GitHub Actions source artifact and verify tree `80d2b9b4...`.
- [x] Recover a matching-lockfile dependency archive instead of assuming access to the prior Mac.
- [x] Re-read the original living execution plan from the user's Library; preserve its milestone structure in this continuation.
- [x] Identify already committed renderer integration (`6e06b678...`) and shell/boundary integration (`3e7da9a...`).
- [x] Record that the historical prototype is unavailable; make no unseen-reference pixel-parity claim.

Source documents: `docs/superpowers/plans/2026-09-05-nn-forge-release-roadmap.md`, `docs/superpowers/specs/2026-07-16-precision-lab-production-integration-design.md`, `docs/superpowers/plans/2026-07-17-precision-lab-production-integration.md`.

No separate pending shell files were found beyond available committed/source artifacts. The older tool-output failure is not an application defect. The current CI browser failures are tracked separately with their actual logs.

---

# 13. P2 — Display-safe Precision Lab shell

**Committed integration exists:** `3e7da9a...`, Actions `34254380062` focused correctness/build/bundle job passed; browser jobs failed, so the milestone is not accepted. App uses PrecisionLabShell, one `useTraining`, one selection controller and one live boundary controller. The committed continuation preserves this ownership and adds the shared save controller. Whole-product browser/keyboard/viewport acceptance remains pending for the current candidate.


## Architecture
- [x] exactly one `useTraining` owner. — committed `3e7da9a...`; focused/full package CI passed.
- [x] no raw worker envelopes in presentation props. — committed `3e7da9a...`; focused/full package CI passed.
- [x] typed display adapters. — committed `3e7da9a...`; focused/full package CI passed.
- [x] no duplicate runtime state system. — committed `3e7da9a...`; focused/full package CI passed.
- [ ] preserve experiment URL.
- [ ] preserve checkpoints.
- [ ] preserve saved runs.
- [ ] preserve code-export selection.
- [ ] preserve workspace profiles / Advanced Tools.
- [ ] preserve evaluation provenance.

## Shell models
- [ ] shell view model.
- [ ] workspace-profile visibility model.
- [ ] disclosure model.
- [ ] focus model.
- [ ] compact-layout model.
- [ ] unit tests for each.

## Composition
- [x] Precision Lab layout components. — committed `3e7da9a...`; focused/full package CI passed.
- [x] production state only, no mock experiment state. — committed `3e7da9a...`; focused/full package CI passed.
- [ ] Build/Run semantics preserved.
- [ ] status/evaluation age/drift preserved.
- [ ] keyboard navigation preserved.
- [ ] skip link preserved.

**Exit:** new presentation exists without changing experiment semantics.

---

# 14. P3 — Real network selection

**Committed renderer/controller evidence:** `6e06b678...` and `3e7da9a...`; retained Canvas/SVG/controller/model tests and upstream full package CI pass. Pointer/keyboard geometry in real browsers is not yet accepted. Selection is a neuron identity with layer context and ranked incoming/outgoing influences; the plan does not claim separately implemented layer/edge selection modes.


## Selection model
- [x] selectable element identities. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] selected neuron identity, layer context and ranked edge influences — committed controller/model tests; not independent layer/edge selection modes.
- [x] architecture invalidation. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] model-generation invalidation. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] deterministic strongest-path ranking. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] ranking tests. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] invalidation tests. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.

## Visualization
- [x] actual production numerical data. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] Canvas integration. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] accessible/SVG fallback where required. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [x] no large live-grid copies into React state. — committed renderer/controller tests at `6e06b678...` / `3e7da9a...`.
- [ ] accessible selection semantics.

## Interaction
- [ ] pointer.
- [ ] keyboard.
- [ ] clear.
- [ ] architecture changes.
- [ ] model generation changes.
- [ ] profile/disclosure transitions.
- [ ] mobile fallback.

**Exit:** actual network inspection works without a second model-state architecture.

---

# 15. P4 — One canonical live decision boundary

## Committed ownership

- [x] One live boundary controller and one canvas composed in App/PrecisionLabShell at `3e7da9a...`.
- [x] Boundary detail evidence does not create a duplicate live renderer.
- [x] Training remains in App; engine/scientific evaluation cadence unchanged.
- [x] Typed boundary display model retains current snapshot/evaluation provenance and recipe-drift semantics.

Evidence: committed App, DecisionBoundaryController, PrecisionLabShell and their focused tests; full package/build/bundle CI job in `34254380062` passed. This proves component ownership, not final browser geometry or performance.

## Pinned rail (live, not a saved snapshot)

- [ ] Qualify the same connected canvas across Build/Run, profiles, disclosures and evidence tabs in Chromium/WebKit.
- [ ] Prove layout remains readable and without unintended overflow at all three required viewports.
- [ ] Verify visibility/demand transitions do not lower required scientific evaluation cadence.

Browser specifications retain connected-canvas identity and URL assertions, both Build/Run and all major region bounds, and 5.5 seconds at 50 steps/frame with <=1 CSS px movement and Chromium CLS = 0. These ran successfully on historical code `88e107a9` in `34410864082`; the later whole-product candidate still fails separate acceptance gates. Do not describe the current tests as merely discovered or carry an older full acceptance result forward.

Do not implement pin/replace/clear or persistent pinned snapshots: that was a stale misreading of the design, not a missing feature.

---

# 16. P5 — Production-backed previews / evidence

**Committed implementation; final browser acceptance pending.** Shared App-owned `useSaveCurrentRun` serializes transport and History saves, keeps capture in the worker, preserves pending artifact identity across recipe changes and drawer unmount, and retries only the persistence store. Run `34271728441` passed the changed tests, infrastructure tests, full package regression, lint, typecheck, build and bundle guard on `3b567950...`. LossChart uses coalesced responsive canvas sizing; confusion metrics use semantic definition lists. Targeted browser fixes in `9501428...` do not change save/evidence ownership; exact candidate browser qualification remains open.


## Dataset previews
- [ ] enumerate all production dataset IDs.
- [ ] deterministic production generators.
- [ ] no mock point clouds.
- [ ] tests for every dataset.
- [ ] correct labels/classes/axes.

## Loss/confusion
- [x] compact LossChart — committed responsive/coalesced implementation + component/full-suite evidence in run `34271728441`.
- [x] compact ConfusionMatrix — committed semantic compact implementation + component/full-suite evidence in run `34271728441`.
- [x] provenance labels preserved — full regression suite green in run `34271728441`.
- [x] train/test distinction preserved — full regression suite green in run `34271728441`.
- [ ] no horizontal overflow.
- [ ] mobile operation.

## Training controls
- [x] Run/Pause — committed control tests + full regression suite green in run `34271728441`.
- [x] Step — committed control tests + full regression suite green in run `34271728441`.
- [x] reset — committed control tests + full regression suite green in run `34271728441`.
- [x] speed / steps-per-frame — committed control tests + full regression suite green in run `34271728441`.
- [x] disabled reasons — committed control tests + full regression suite green in run `34271728441`.
- [x] keyboard — committed control tests + full regression suite green in run `34271728441`.
- [ ] touch targets.

## Save/history
- [x] save exact intended artifact — App-owned save controller + committed retry/capture tests green in run `34271728441`.
- [x] preserve worker-owned capture — capture remains worker-owned; committed integration tests green in run `34271728441`.
- [x] failed save retries exact pending artifact — retry persistence reuses the captured pending artifact; dedicated tests green in run `34271728441`.
- [x] no silent recapture on retry — dedicated recipe-change/reopen/retry tests green in run `34271728441`.
- [x] saved-run explanation remains explicit — full regression suite green in run `34271728441`.
- [x] Apply Saved Recipe semantics remain explicit — full regression suite green in run `34271728441`.

**Exit:** compact evidence UI uses real production data and preserves artifact identity.

---

# 17. P6 — Production acceptance

**Current result: NOT YET RELEASE-QUALIFIED.** The September 9 browser-fix source `88e107a9` passed run `34410864082`, but later tests/corrections reached `6de861dd` and exposed new acceptance failures. Intake run `34522259192` failed preview/subpath and total JavaScript size; it passed correctness, recovery and accepted reference performance. Maintenance run `34529946854` separately verifies its changes while retaining the failing bundle gate and existing browser assertions. Maintenance completion does not check off this end-state release matrix. See section 27 and the maintenance review for current scoped evidence.

## Functional
- [ ] full unit suite.
- [ ] full E2E.
- [ ] build.
- [ ] bundle guard.
- [ ] recovery.
- [ ] shared URL.
- [ ] saved-run semantics.
- [ ] checkpoints.
- [ ] code export.
- [ ] evaluation provenance.

## Browsers
- [ ] Chromium.
- [ ] WebKit.
- [ ] zero retries.
- [ ] no accepted flaky cases.

## Required viewports
- [ ] `1437x742`.
- [ ] `735x860`.
- [ ] `320x844`.

At each:
- [ ] Build usable.
- [ ] Run usable.
- [ ] Advanced Tools usable.
- [ ] no unintended horizontal overflow.
- [ ] no inaccessible hidden controls.
- [ ] evidence readable.

## Layout stability
At 50 steps/frame for at least 5 seconds:
- [ ] record major bounds at start.
- [ ] record major bounds after interval.
- [ ] stable regions move ≤ 1 CSS px unexpectedly.
- [ ] diagnose intentional motion separately.
- [ ] Chromium post-start CLS = 0 for acceptance scenario.

## Accessibility
- [ ] skip link.
- [ ] keyboard only.
- [ ] visible focus.
- [ ] 44px required touch targets.
- [ ] reduced motion.
- [ ] 200% zoom.
- [ ] semantic regions.
- [ ] accessible network selection.
- [ ] accessible evidence tabs.
- [ ] no axe regressions.
- [ ] understandable status updates.

## Performance
- [ ] JS bundle caps green.
- [ ] forced paired evaluation ≤ 250 ms contract.
- [ ] save capture ≤ 500 ms contract.
- [ ] accepted engine reference gate green.
- [ ] no duplicate boundary work.
- [ ] no major live numerical-grid copy in React state.

**Exit:** Precision Lab is scientifically and operationally equivalent or better, not merely prettier.

---

# 18. P7 — One final production shell

**Migration state:** App already composes PrecisionLabShell, with later graph-control corrections through `6de861dd`. Maintenance removes only proven unused private primitives, not the legacy shell or styles. BuildRunShell/MainArea compatibility/test consumers remain until a dedicated migration and exact-candidate acceptance prove removal safe. P7 is not claimed complete; no browser, bundle or performance gate is waived.


## Migration
- [x] identify current production App composition and retained legacy/test consumers/styles — committed through product tree `bf7115da...`.
- [ ] migrate each consumer.
- [ ] verify each migration.
- [ ] keep old shell until all consumers are proven migrated.

## Removal
- [ ] remove obsolete shell.
- [ ] remove dead styles.
- [ ] remove obsolete presentation adapters.
- [ ] preserve engine/schema/protocol unless separately justified.
- [ ] confirm only one production shell remains.

## Final qualification
- [ ] correctness.
- [ ] browser matrix.
- [ ] responsive acceptance.
- [ ] accessibility.
- [ ] bundle.
- [ ] accepted performance.
- [ ] build provenance.
- [ ] verification record.
- [ ] diff review.

**Exit:** Precision Lab is the sole production presentation layer.

---

# 19. Precision Lab release

**Blocked by candidate qualification first, then the separate owner-controlled Pages settings issue. No merge, deployment or visibility change was made in this continuation.**


- [ ] Freeze qualified candidate.
- [ ] Merge to `main`.
- [ ] Re-run `main` CI.
- [ ] Deploy accepted `main`.
- [ ] Chromium live validation.
- [ ] WebKit live validation.
- [ ] shared URLs.
- [ ] worker.
- [ ] fonts.
- [ ] project base path.
- [ ] responsive layouts.
- [ ] record release evidence.

---

# 20. Post-release evaluation

Do **not** automatically invent another giant roadmap.

- [ ] Use NN.FORGE as a real learner/experimenter.
- [ ] Evaluate experiment creation.
- [ ] Evaluate architecture construction.
- [ ] Evaluate training controls.
- [ ] Evaluate evidence comprehension.
- [ ] Evaluate train/test distinction.
- [ ] Evaluate evaluation-age/drift clarity.
- [ ] Evaluate network inspection.
- [ ] Evaluate save/compare workflow.
- [ ] Evaluate code export.
- [ ] Evaluate sharing.
- [ ] Evaluate mobile use.
- [ ] Gather actual usage/feedback evidence.
- [ ] Rank problems by frequency/severity.
- [ ] Choose exactly one next milestone.

---

# 21. Explicitly deferred unless separately approved

- [-] accounts.
- [-] backend/cloud persistence.
- [-] collaboration/multiplayer.
- [-] generic AI features.
- [-] new dataset program.
- [-] new neural-network math.
- [-] broad engine rewrite.
- [-] framework migration.
- [-] historical branch merge.
- [-] prototype-state import.
- [-] unrelated design-system rebuild.
- [-] arbitrary benchmark-limit increases.
- [-] persisted trained-parameter restore.
- [-] deterministic checkpoint-future claim.
- [-] making repo public merely to simplify deployment.
- [-] deleting old branches just for cleanliness.

---

# 22. Decision log

### D001 — `main` is product authority
Status: active.

### D002 — Do not infer engine regression from a failing baseline
Because baseline and candidate fail the same current grid limits, engine changes require stronger evidence.
Status: active.

### D003 — Same-origin pinned fonts
Chosen as the narrow correction for the reproduced WebKit reload/resource-policy failure.
Status: implemented and verified.

### D004 — Executable bundle caps
Documented JS caps became real CI gates rather than treating the generic Vite chunk warning as the contract.
Status: implemented and verified.

### D005 — Do not build Precision Lab on an ambiguous release base
Finish/disposition release qualification first.
Status: satisfied for the accepted baseline; do not re-open.

### D006 — Single save owner and artifact identity
App owns one save controller shared by transport and History. Retry uses the store's exact pending artifact and never contacts the worker to recapture.
Status: committed in the September 8 continuation (`20831874` and `3b567950`); additional priority, invalidation and late-completion tests in `acf04179` pass in maintenance CI. Final whole-product browser acceptance remains separate.

### D007 — Honest evidence boundaries
GitHub connector reads and downloaded source artifacts are authority. Local patch/test results are not a remote commit, CI run or live deployment. Missing browser binaries and unreadable tool responses are tooling limits.
Status: active.

### D008 — No bundle-limit expansion
Educational content is grouped into one on-demand chunk; safe unused memo factories are tree-shaken; Terser uses two normal compression passes. No cap, scientific cadence, schema or engine algorithm was changed.
Status: caps preserved; current CI total is 234,282 / 234,161 bytes and fails. The 93-byte headroom was a September 8 historical receipt, not the current state. Maintenance leaves runtime output unchanged.

---

# 23. Immediate execution queue (reconciled September 10)

## NEXT-01 — Preserve and commit recovered integration (complete)
- [x] Recover newer shell/boundary instead of overwriting it from the older checkpoint: `3e7da9a`.
- [x] Preserve exact source/tree and distinguish prior tool failures from application evidence.
- [x] Commit shared save ownership/retries, responsive/evidence changes and lazy loading: `20831874`, with workflow `3b567950`.
- [x] Restore/reconcile this canonical plan: `3c8e48a7`; qualification record `cad28a4e`.

No local-patch handoff remains for these committed changes. The September 10 maintenance pass is separately tracked in section 27. Do not push synthetic local source-import history or reapply the old patch to a newer branch.

## NEXT-02 — Exact committed-candidate qualification
- [ ] Freeze and record the new upstream SHA.
- [ ] Run all helpers, lint, typecheck, engine/shared/web tests, build and fixed gzip guard on that SHA.
- [ ] Run preview and project-subpath Chromium/WebKit with zero retries and no accepted flaky cases.
- [ ] Verify capture/retry flows in browsers, shared URLs, checkpoint recovery and code export.
- [ ] Run fault-enabled recovery, rebuild normally and run fault-disabled verification.
- [ ] Qualify Build and Run at 1437x742, 735x860 and 320x844; Advanced Tools, focus, keyboard, touch targets, 200% zoom and reduced motion.
- [ ] Verify >=5 seconds at 50 steps/frame, <=1 CSS px major-region movement and Chromium post-start CLS = 0.
- [ ] Run the already accepted five-pair reference policy unchanged; preserve exact host and raw samples.

## NEXT-03 — Migrate/retire legacy presentation only after proof
- [ ] Remove remaining obsolete shell consumers/styles after successful acceptance.
- [ ] Repeat affected correctness/browser/bundle/performance gates and final diff review.

## NEXT-04 — Accept Precision Lab into main
- [ ] Merge only a fully qualified exact candidate.
- [ ] Record new main and passing main CI.

## NEXT-05 — Pages (separate settings blocker)
- [!] Resolve owner-controlled repository Pages settings without changing visibility implicitly.
- [ ] Deploy only accepted main and retain a real deployment receipt/live URL.
- [ ] Run live Chromium/WebKit validation before claiming publication.

Do not redo benchmark provenance work, rewrite the recovered shell, or confuse settings/tooling failures with application regressions.

---

# 24. Rules for checking a task off

## Code
Requires a committed implementation + focused tests + affected broader verification. Local passing tests are recorded as local evidence, not a completed upstream milestone.

## Bug fix
Requires reproduction → failing regression → fix → original reproduction passes → relevant regression suite passes.

## Performance
Requires exact environment + exact SHA + raw measurements + explicit benchmark policy + pass under that policy.

## Browser compatibility
Requires actual browser result; mocked/intercepted success does not count unless interception itself is the behavior under test.

## Release
Requires exact source SHA + required gates green + provenance + accepted `main`; live deployment claims additionally require live external checks.

## Documentation
Requires committed repository file and claims that match actual evidence.

---

# 25. Evidence ledger

| Date | Event | SHA | Evidence | Result |
|---|---|---|---|---|
| 2026-09-05 | Intake authority | `ae09b986...` | repository ref | baseline |
| 2026-09-05 | Skip-link fix | `a5f9172...` | focused browser run | green |
| 2026-09-05 | Initial combined qualification | `96fdcff...` | Actions `33949410527` | fonts + perf red |
| 2026-09-05 | Same-origin font fix | `cf02bf5...` | release verification | browser/recovery green |
| 2026-09-05 | 5-run Apple Silicon reference | `ebdb7b6...` | Actions `33974596143` | baseline + candidate grid red |
| 2026-09-05 | JS bundle guard | `fba6330...` | Actions `33974892532` | correctness/browser/bundle green; perf red |
| 2026-09-05 | Consolidated docs | `64ffa817...` | Actions `33975800331` | correctness/browser/recovery green; perf red |

| 2026-09-06 | Policy resolved / baseline qualified | `fbae4b98...` | Actions `34001183202`; committed release record | all release jobs green |
| 2026-09-06 | Accepted baseline merge | `98f29b86...` | merge parents; main CI `34002500733` | accepted main; Pages run `34002943565` failed |
| 2026-09-08 | Ref/source recovery | `3e7da9a...` | GitHub refs; source artifact `10067245424`; tree `80d2b9b4...` | newer shell/boundary already committed |
| 2026-09-08 | Upstream integration CI | `3e7da9a...` | Actions `34254380062` | correctness/build/bundle pass; browser jobs fail |
| 2026-09-08 | Local continuation | no upstream commit | engine 497/shared 334/web 1170; helpers 91; lint/types/build/bundle | local checks pass, not release qualification |
| 2026-09-08 | Local performance diagnostic | local patch | raw `performance-local.log` | historical absolute engine FAIL; worker PASS; accepted reference not run |
| 2026-09-08 | Browser launch attempt | local patch | `browser-attempt.log`, `browser-webkit-attempt.log`; 66 discovered scenarios | missing Chromium/WebKit executables before app launch; not an app defect |

| 2026-09-09 | Medium-risk browser fixes | `88e107a9` | Actions `34410864082` | historical exact-code qualification green; no deployment |
| 2026-09-10 | Expanded selection/zoom candidate | `6de861dd` | Actions `34522259192` | correctness/recovery/reference green; browser and bundle red |
| 2026-09-10 | Behavior-preserving maintenance | `5092ee69` | Actions `34529946854`; compact receipts | 2,019 package tests, 153 helpers, lint/types/build/recovery/reference pass; total JS cap fails |
| 2026-09-10 | Reporting source-tree review | `7ec856a3` | 154 local helper tests, source-tree RED/GREEN and lint | rejects wrong-tree completion receipts; no application change |

---

# 26. Living-plan changelog

## v1 — 2026-09-05
- Created living execution ledger.
- Backfilled completed release-readiness tasks.
- Marked engine benchmark-policy work as current blocker.
- Added exact merge/deploy/Precision Lab order.
- Added explicit completion rules to prevent casual checkbox completion.

---

## v2 — 2026-09-08 (historical local continuation checkpoint; subsequently committed)
- Recovered this canonical file instead of inventing a replacement checklist.
- Reconciled resolved performance policy, merged main, accepted baseline CI and blocked Pages status.
- Recovered newer committed shell/boundary integration and preserved App ownership.
- Corrected pinned-boundary semantics; no saved snapshot workflow was invented.
- Recorded local shared save/retry, responsive evidence/help/transport and bundle changes with raw test evidence.
- Kept real-browser, accessibility, zoom, recovery, layout, accepted reference performance, legacy removal, merge and deployment unchecked.

# Historical September 8 one-line status (superseded)

> **Accepted baseline is merged; Precision Lab shell/controller/boundary are committed upstream at 3e7da9a. The local continuation passes 2,001 package tests, 91 helpers, lint, typechecks, build and unchanged bundle limits, but it is not pushed or release-qualified: browser executables and write/dispatch access are unavailable, reference performance is unrun, and Pages settings remain blocked.**


## September 9, 2026 — scoped optimization and bug-hunt pass

This dated entry supersedes the historical one-line status above for this audit only. Baseline: `cad28a4e6e076245cba7191cea2e78c711335003` on `codex/nn-forge-precision-lab`; `main` was read at `98f29b86e469a2a545be75928ae6f32309fd1582`. No merge, deployment, scientific-state, schema, protocol, dependency or budget changes are part of this pass.

- [x] Check Linear before edits: no visible non-archived issues or matching project. Attempt Sentry discovery and lookup: no exposed Sentry tools; frequency and transaction rankings remain unknown.
- [x] Recover exact source, verify its Git tree, and reuse CI dependencies only after all manifests and the lockfile match. Report the repository map, commands and prioritized plan before source edits.
- [x] Reproduce Build-help interception with actual CSS and an isolated Chromium hit-test. Add a failing stacking regression, then contain topology descendants with `isolation: isolate`; the hit target changes from the toolbar to the intended help control. This is an intentional bug fix.
- [x] Separate close-button and Escape focus tests into independent mounts and await each focus return. Focus behavior is unchanged; the focused suite passes 20 tests.
- [x] Characterize code-export invalidation, then reuse the parameter snapshot across scalar-only evidence updates. Both parameter-unpack operations drop from six calls to one across five updates; displayed code stays unchanged. Parameter/provenance and generation changes remain covered. This is behavior-preserving work reduction, not a measured latency or Sentry improvement.
- [x] Inspect input/error boundaries, dependency uses and targeted credential patterns. No speculative dependency or legacy-component removal. Registry vulnerability scanning is blocked by DNS, not reported clean.
- [x] Write root `BUGS-TO-REVIEW.md`: reproduce independent-store lost updates without changing persistence, and defer unconfirmed mobile drawer and layout-shift roots. Include risk, locations, evidence, reproduction/fix approach and telemetry/scanner limitations.
- [x] Run local regression gates: engine 497, shared 334 and web 1,176 tests pass (2,007 total); 91 infrastructure tests pass; lint, source/test typechecks and build pass. The unchanged JavaScript gzip guard passes: entry 151,125/152,245 bytes, inspection 5,430/7,373, total JS 234,063/234,161. Recovered dependencies omit font binaries, so local build/font fidelity is limited and no complete-asset/browser claim follows.
- [x] Keep each concern in a small local commit for publication through the GitHub connector; preserve the verified baseline tree and active branch ancestry.
- [ ] Close the hard-bug checklist only after a safe fix and its relevant tests prove each root cause. No racing persistence or guessed responsive-layout fix was attempted.
- [ ] Obtain a passing exact-head browser qualification and post-change reference-performance result. Baseline run `34273895548` passed focused/performance/recovery but failed preview/subpath browsers. Local application navigation is administrator-blocked; isolated hit-tests are narrower evidence. Inspect the newly published run separately rather than treating the baseline as a post-change result.
- [ ] Merge or deploy: still outside this audit. Verify remote source/ref identity after publication and report the exact commits and CI status to the user.


---

# 27. Behavior-preserving maintenance — September 10 closure

**Approved scope:** improve the codebase without new product features or continuing the separate browser-fix project. Plan: `docs/maintenance/2026-09-10-plan.md` (`38179180`). Detailed decisions and evidence: `docs/maintenance/2026-09-10-review.md`. Contributor commands: `docs/maintenance/README.md`. These completed maintenance tasks do not waive section 17–19 acceptance.

- [x] **M1 — Production lint scoping.** Production explicit-any errors, narrow test/benchmark exemptions and equivalent structural WebGPU typing (`ee615300`); nine policy tests, lint/typechecks and full CI regression pass.
- [x] **M2 — Architecture guards.** Resolved static/literal import, re-export and type-boundary checks with exact legacy exceptions (`a2a68762`); 24 positive/negative ESLint tests and repository lint pass. Not a runtime controller-count proof.
- [x] **M3 — Small, honest CI evidence.** Streaming command receipts, bounded Playwright/focused/performance summaries, independent recovery JSON and failure-preserving build manifests (`d212779a`, `7ec856a3`). Nine subprocess tests, 17 summary tests and workflow contracts pass locally; real CI receipts preserve the bundle failure and report successful recovery/performance separately.
- [x] **M4 — Typed test fixtures.** Consolidate duplicate saved-artifact builders with deterministic metadata and isolated nested values (`acf04179`); three fixture tests, affected suites and complete 2,019-test CI run pass. Clone-removal mutation is detected.
- [x] **M5 — Proven unused-code cleanup.** Remove private Row/Segmented/Stepper and four unnecessary test-interface exports (`5092ee69`), with whole-tree reference/entry-point review. Retain CSS, legacy shells, package exports and the live SVG fallback. Local emitted output remains identical.
- [x] **M6 — Pure decision review.** Add save-disabled priority and access/hydration transition characterization (`acf04179`); retain the existing helper and App ownership instead of adding abstraction. Removing a required subscription makes its regression fail.
- [x] **M7 — Resource ownership review.** Audit specified listeners, observers, RAF/timers, worker subscription and accepted-save lifetime; repeated StrictMode cycles and late completion tests pass (`acf04179`). Unsubscribe-removal mutation fails. No claim of exhaustive heap/leak qualification.
- [x] **M8 — Reproducibility and documentation.** Document declared versus observed runtime versions, unchanged frozen lockfile, ownership, compact artifact retrieval, audit limits and this reconciled ledger. No runtime/dependency upgrade or false release-acceptance claim.

## Scoped verification

- Committed code `5092ee69`, Actions `34529946854`: changed tests, 153 helpers at that revision, 497 engine + 334 shared + 1,188 web tests, lint, source/test types, build, recovery, normal fault-disabled rebuild and accepted five-pair performance pass.
- Reporting-only review `7ec856a3`: source-tree mismatch regression observed failing before correction; final local helpers 154 and lint pass. Exact documentation-head CI is inspected after publication, not predicted here.
- CI bundle: entry 151342 / 152245; inspection 5431 / 7373; total JS 234282 / 234161 bytes. **The existing total cap remains failed, by 121 bytes.** Local intake/candidate both have a 124-byte overrun; no cap changed.
- All 13 locally emitted outputs are byte-identical before/after maintenance. Recovered dependencies omit font binaries, so complete asset delivery must use fresh CI, not this local equality claim.
- Three deliberate local mutations (fixture clone, storage unsubscribe, save access subscription) were detected and fully restored. Raw logs and source/build checksums accompany the handoff evidence.

## Remaining work is not maintenance completion

Precision Lab browser/zoom qualification, the inherited bundle overrun, consumer-proven legacy-shell retirement, and the independent-tab saved-run overwrite remain open. Main stays `98f29b86...`; no merge, deployment, Pages/settings or visibility change. The performance-policy investigation remains resolved.

## v3 — September 10 reconciliation

Replaced stale current-SHA, unpushed-patch, unrun-performance and 93-byte-headroom claims with dated source/CI receipts. Preserved historical milestone boxes and the September 8/9 records. Added maintenance-only completion evidence without marking blocked Precision Lab release tasks accepted.
