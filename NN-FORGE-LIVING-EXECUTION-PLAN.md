# NN.FORGE — Living Execution Plan

**Role of this file:** This is the checklist I will actually execute against. It is deliberately more detailed than a normal roadmap so you can audit what I am doing.

## September 12 completion authority

The owner has authorized finishing the outstanding branch work and merging a working release into `main`. [Completion PR #36](https://github.com/DenseDevKev/neural-network-playground/pull/36) is the authoritative record for final candidate SHA, reviews, qualification, merge, main CI and deployment receipts. Those events occur after this document is committed; their statuses are recorded on the PR and GitHub Actions instead of creating a new documentation commit for every completed check. The completion source/branch audit is in [the September 12 record](docs/superpowers/verification/2026-09-12-completion.md).

The completion candidate fixes saved-run concurrency and destructive-cleanup identities without changing the storage schema, retires the unused presentation paths, patches critical/high development dependency advisories, and extends the full Precision Lab qualification to pushed `main` commits. Existing scientific contracts, browser assertions and release budgets remain in force. Dated no-merge/no-deploy restrictions below describe their historical audit scopes; they do not override the owner's current completion request.

## Status notation
- `[x]` completed with evidence.
- `[ ]` pending.
- `[!]` blocked / requires an explicit decision.
- `[-]` deferred or out of scope.
- I will update **this same file** as work progresses rather than creating a new plan every time.
- A box is not checked merely because code was written; the listed exit condition must be satisfied.

---

# 0. Control state

**Historical qualification state — September 11, 2026:** exact tested product head `fcc382658396defa1a19c3a5543478ebe6b0986f`, tree `e74b3561f2287d3f921851bb877686dbd0cca83a`, passed all six required jobs on attempt 2 of exact-head run `34543410273`: source evidence, focused correctness/build/bundle, preview browsers, project-subpath browsers, recovery and accepted five-pair performance. The owner changed the repository from private to public before the bounded rerun; standard GitHub-hosted runners then started normally, removing the earlier account billing/spend-limit startup block. Preview passed 82 tests with 6 intentional mode skips; subpath passed 86 with 2 intentional mode skips; both reported zero unexpected and zero flaky tests. Recovery passed both fault-enabled tests and both clean-build fault-disabled tests. No additional product correction was needed after `6f478eae8a390e5a0c5d0ea80263198b10c66ddd`. Sections 28–29 preserve the historical failures and current evidence. A documentation-only closure commit created from this record must receive its own exact-head run before it becomes the final qualified branch head.

**Reconciled September 10, 2026 against GitHub refs and exact-source CI. Sections explicitly marked historical are receipts, not current acceptance.**

- Repository: `DenseDevKev/neural-network-playground` (public as of the September 11 bounded rerun; visibility was changed by the owner, not by the qualification agent).
- Product authority: `main` at `98f29b86e469a2a545be75928ae6f32309fd1582`.
- Release baseline: **merged and accepted**; merge parents are `ae09b986...` and `2cd5b896...`. Main CI `34002500733` succeeded.
- Active branch: `codex/nn-forge-precision-lab`.
- Original continuation base: `3e7da9a4dabd2d7bc275c9aa4c614a04d9b26732`, not the supplied older `6e06b678...`.
- Maintenance intake: `6de861dd4e00f3223ec8985a0b6c00bdbfa7abe9`, tree `12a57ef8e463e7386c687b602f2501de088d798c`.
- Historical maintenance implementation: `7ec856a39079141a040624e9fe2927218e0db8f8`, tree `4447aed98747a411e9fef2211cb2785de1c2bde6`. Maintenance documentation closure was `679789d5`; later qualification candidates are recorded in section 28.
- Exact-source maintenance CI receipt: `5092ee69abc8714ff906d5d63bbd5b5263f62368`, run `34529946854`; correctness, types, lint, build, recovery and accepted reference performance pass. The total-JavaScript guard failed in that historical maintenance run; it passes on the later `6bd20da6` intake run recorded in section 28. `7ec856a3` only tightens reporting source-tree validation; its documentation-head CI is inspected separately.
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

**Historical helper/build receipts:** accepted baseline had 88 helpers (run `34001183202`). The September 8 local continuation had 91 helpers and 87 byte-identical rebuilt files with a passing bundle guard; that was not browser recovery. Historical maintenance counts and its then-failing bundle gate are recorded in section 27; the later continuation is in section 28.

---

# 5. P1 pulled forward — JavaScript bundle contract

**Caps unchanged. Historical maintenance CI `5092ee69` measured entry 151,342 / 152,245, InspectionPanel 5,431 / 7,373 and total JavaScript 234,282 / 234,161 bytes: total was 121 bytes over the cap.** The later `6bd20da6` intake passes its exact CI bundle guard in run `34538909554`. Current product `6f478eae` passes locally at entry 151,332, inspection 5,430 and total 233,770 bytes, but its new CI jobs do not start; see section 28. Local recovered dependencies omit font binaries, so complete asset fidelity still requires CI. No cap was changed.


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

## Historical maintenance-candidate performance — separate from resolved policy

- [x] Run the unchanged accepted five-pair macOS/ARM64 reference gate on the maintenance implementation `5092ee69`: Actions `34529946854`.
- [x] Preserve exact candidate/host identity, raw samples, comparator result and independent worker budgets in `precision-performance-5092ee69...` and its compact summary.

The accepted comparison passed all six engine comparisons plus independent forced-pair and save-capture budgets. Candidate worker medians in that run were 9.7946 ms and 11.6225 ms; no engine speed gain is claimed. The later `6bd20da6` intake also passes the accepted reference gate in `34538909554`. The new `6f478eae` candidate has no executed CI performance result because its runners do not start; see section 28. No policy or engine change is inferred from that infrastructure failure.

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

**Current result: PRODUCT CANDIDATE RELEASE-QUALIFIED; FINAL DOCUMENTATION HEAD PENDING.** Exact tested product head `fcc382658396defa1a19c3a5543478ebe6b0986f` passed every item in this P6 matrix in attempt 2 of run `34543410273`. The product contents remain unchanged in the later documentation-only closure commits. Section 29 records the exact evidence and keeps branch closure open until the final documentation head completes the same six-job workflow.

## Functional
- [x] full unit suite.
- [x] full E2E.
- [x] build.
- [x] bundle guard.
- [x] recovery.
- [x] shared URL.
- [x] saved-run semantics.
- [x] checkpoints.
- [x] code export.
- [x] evaluation provenance.

## Browsers
- [x] Chromium.
- [x] WebKit.
- [x] zero retries.
- [x] no accepted flaky cases.

## Required viewports
- [x] `1437x742`.
- [x] `735x860`.
- [x] `320x844`.

At each:
- [x] Build usable.
- [x] Run usable.
- [x] Advanced Tools usable.
- [x] no unintended horizontal overflow.
- [x] no inaccessible hidden controls.
- [x] evidence readable.

## Layout stability
At 50 steps/frame for at least 5 seconds:
- [x] record major bounds at start.
- [x] record major bounds after interval.
- [x] stable regions move ≤ 1 CSS px unexpectedly.
- [x] diagnose intentional motion separately.
- [x] Chromium post-start CLS = 0 for acceptance scenario.

## Accessibility
- [x] skip link.
- [x] keyboard only.
- [x] visible focus.
- [x] 44px required touch targets.
- [x] reduced motion.
- [x] 200% zoom.
- [x] semantic regions.
- [x] accessible network selection.
- [x] accessible evidence tabs.
- [x] no axe regressions.
- [x] understandable status updates.

## Performance
- [x] JS bundle caps green.
- [x] forced paired evaluation ≤ 250 ms contract.
- [x] save capture ≤ 500 ms contract.
- [x] accepted engine reference gate green.
- [x] no duplicate boundary work.
- [x] no major live numerical-grid copy in React state.

**Exit:** Precision Lab is scientifically and operationally equivalent or better, not merely prettier.

---

# 18. P7 — One final production shell

**September 12 completion:** `App` composes `PrecisionLabShell` and the live display exports in `PrecisionLabContent`. Dead `BuildRunShell`, `RegionShell`, `Sidebar`, `MainArea` adapters and their unused private components are retired. Relevant configuration, boundary, profile and preset tests are migrated to current production consumers. App retains training/save/selection/boundary ownership. Dead selectors are removed while live topology-stage and lesson-cue styles remain.

The complete package suite, typechecks, production build and unchanged bundle budgets pass in the completion checkout. Final cross-browser, recovery, performance, review and exact-commit provenance receipts belong to [PR #36](https://github.com/DenseDevKev/neural-network-playground/pull/36); no historical result substitutes for the final candidate.

**Exit contract:** one production presentation layer, migrated behavior coverage and passing final qualification.

---

# 19. Precision Lab release acceptance

The owner authorized final integration and merge on September 12. Acceptance is recorded by post-commit events rather than a self-referential checklist:

| Required event | Authoritative record |
|---|---|
| Exact qualified candidate and review | [Completion PR #36](https://github.com/DenseDevKev/neural-network-playground/pull/36) and its head SHA |
| Merge into main | The PR's merge event and merge commit |
| Main correctness and full qualification | [Actions](https://github.com/DenseDevKev/neural-network-playground/actions) runs for that exact main SHA |
| Deployment and actual page URL | Successful `Deploy to GitHub Pages` run and its environment URL |
| Live Chromium/WebKit, URLs, worker, fonts, base path and responsive behavior | Post-deployment browser receipt recorded on PR #36 |

Qualification keeps the existing fixed bundle caps, zero Playwright retries, fault-on/fault-off recovery and five-pair performance comparison. Deployment must use the tested main SHA. Repository visibility and billing settings are outside this completion change; source is already public. No publication is claimed without the real deployment and live-browser receipts.

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
Status: caps preserved. Maintenance CI total 234,282 / 234,161 bytes was a historical failure; later `6bd20da6` passes its exact CI guard, and `6f478eae` measures 233,770 / 234,161 locally with new CI startup blocked. Section 28 is current; no older headroom or failure is carried forward.

---

# 23. Historical execution queue (September 10; superseded by September 12 completion authority)

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


---

# 28. All-jobs-green continuation — runner-start blocker

**Goal remains open.** Resume from the published product candidate; do not repeat or discard already committed fixes. This section supersedes older current-state assertions without changing historical receipts or accepted performance policy.

## Source and completed investigation

- [x] Refresh both remote refs before editing: main `98f29b86e469a2a545be75928ae6f32309fd1582`; intake branch `6bd20da6de276337ae62f6eba92c6c9847d4379e`, tree `b3c10470d417182bce294a5b29428cb567c666fa`.
- [x] Recover the exact Actions source archive and verify the reconstructed Git tree; reuse dependencies only after lockfile equality. Work in an isolated linked worktree; never publish synthetic local recovery history.
- [x] Inspect completed intake run `34538909554`: focused, source-evidence, recovery and accepted five-pair performance jobs succeed. Preview and subpath fail only the two WebKit 200% document-zoom journeys. Preview JSON records 78 passed, 6 intentional skips, 2 failures and zero flaky tests.
- [x] Trace the remaining zoom failure to the expected static transport still being sticky. The served width condition is a literal `48.5em`, while the short-height condition remains `calc(600em / 14)`. Preserve the complete application assertions; do not force clicks, skip WebKit or raise thresholds.
- [x] Add a failing static regression, then replace only the short-height condition with literal `42.8571428571em`. Add isolated literal-versus-calc evidence and a tall/short/zoom/reset browser regression. These are implemented in `6f478eae`, tree `b6c5927de660746587ac7d92ab8a26204e684dca`; cross-engine browser acceptance is still pending.
- [x] Run local verification for the published product contents: 497 engine, 334 shared and 1,191 web tests (2,022 total), 155 infrastructure tests, responsive RED/GREEN, lint, complete source/test typechecks, production build, bundle guard and whitespace check pass.
- [x] Compare staged tree with the tree published through Git data writes, re-read the active ref, and update it non-forced. No main, visibility, Pages, ownership, scientific, schema, protocol or persistence change.

## Bundle and browser evidence boundaries

Local gzip bytes on `6f478eae`: entry 151332 / 152245, InspectionPanel 5430 / 7373, total JavaScript 233770 / 234161. All limits are unchanged. The earlier 121-byte maintenance overrun is historical, not the current local result; intake `6bd20da6` also passes its exact CI bundle guard.

Local reconstructed dependencies omit font binaries and the WebKit executable cannot be installed because the browser download host does not resolve in this runtime. These limitations are not product defects and local unit/build results do not qualify Chromium/WebKit, font delivery, geometry or accessibility. No local full-browser pass is claimed.

## Current external blocker

- [!] Run `34542233759` for exact code `6f478eae` ends before any checkout, install or test step. All six jobs report failure, no steps, and no assigned runner. A source-evidence job metadata read reports `runner_id: 0` and an empty runner name; its log download has no blob.
- [x] Make one bounded infrastructure retry request for the source-evidence job, not an application-test retry. Attempt 2 again ends without executing steps. Preserve both failures; do not repeatedly rerun or alter the workflow to hide them.
- [!] The connector's available check-run outputs contain no explanatory title, summary or annotations. The exact account/platform reason is not established. A quota/billing restriction is a possibility, not a confirmed diagnosis; no billing or account-setting change has been made.
- [ ] Restore GitHub Actions runner startup and inspect the exact candidate's complete six-job qualification, including the new isolated and full-application zoom regressions. A failure before runner allocation cannot be counted as a product pass or fail.
- [ ] Fix any remaining reproduced application failure, then require all six jobs to succeed on one exact candidate. Preserve zero retries, existing failure caps, performance policy, scientific cadence and bundle/layout budgets.
- [ ] Reconcile final acceptance boxes only after those results, and inspect the documentation-head run separately. No future CI status or self-referential commit SHA is invented in this record.

## Unchanged separate boundaries

Legacy-shell retirement and cross-tab saved-run overwrite remain separate work. Accepted baseline/performance-policy investigation remains resolved. Main stays `98f29b86...`; Pages remains a repository-settings blocker. Nothing was merged, deployed or made public.

## v4 — September 10 qualification resume

Replaced the historical maintenance guard failure as current authority with the later measured evidence, recorded the precise WebKit correction and local regression results, and distinguished a new runner-start failure from the prior tool-output failures. All-jobs-green and release acceptance remain unchecked.


---

# 29. Exact-head qualification resume — September 11, 2026

**Execution order:** source identity → runner-start diagnosis → current browser correction → local non-browser gates → reviewed publication if justified → exact-head six-job CI → documentation closure. A step stays open until its stated exit condition and evidence are both present.

## 29.1 Source authority and inherited failures

- [x] Refresh `main` and the working branch before editing. Main remains `98f29b86e469a2a545be75928ae6f32309fd1582`; the working branch remains `fcc382658396defa1a19c3a5543478ebe6b0986f`, tree `e74b3561f2287d3f921851bb877686dbd0cca83a`.
- [x] Inspect the latest exact-head push run. Run `34543410273`, workflow `351256324`, is the only workflow run ID for `fcc38265`; attempt 1 materialized all six jobs but did not start them, and the bounded attempt 2 executed all six.
- [x] Recover exact source without treating synthetic ancestry as publishable. The verified `6bd20da6` source artifact and the five authoritative later file versions reconstruct Git tree `e74b3561...` exactly. Local `HEAD` is a synthetic recovery commit and must never be pushed.
- [x] Read the living plan, governing design/implementation/release/verification/maintenance/ownership records, workflow, manifests, lockfile importers, browser configuration, bundle/performance/evidence scripts, and implicated product/browser tests before considering a source correction.
- [x] Keep inherited failures distinct. Run `34538909554` at `6bd20da6` executed and failed only two WebKit document-zoom journeys. Run `34542233759` and attempt 1 of `34543410273` did not execute application code; attempt 2 of `34543410273` executed and passed the full qualification.

**Exit:** the current remote SHA/tree, source contents, and inherited execution boundary are unambiguous. **Evidence:** refreshed GitHub refs/commit/run/job metadata; verified source archive checksum/tree; local `git write-tree` equals `e74b3561...`.

## 29.2 Runner-start diagnosis

- [x] Inspect full exact-head jobs, check runs, attempts, timing, steps and log downloads. In attempt 1, five `ubuntu-latest` jobs and one `macos-15` job had `steps: []`, `runner_id: 0`, empty runner names and 0 billable milliseconds; log downloads returned `BlobNotFound`. Attempt 2 allocated normal GitHub-hosted runners and completed every step.
- [x] Verify this is not invalid workflow syntax. GitHub expanded all six current jobs, including the browser matrix, from the exact committed workflow.
- [x] Check broad service evidence without converting it into an account diagnosis. GitHub's public status reports Actions operational and no September 10–11 incident; this does not exclude a brief or account-scoped restriction.
- [x] Read the exact attempt-1 check annotation in the authenticated GitHub run UI. All six jobs said: “The job was not started because recent account payments have failed or your spending limit needs to be increased. Please check the 'Billing & plans' section in your settings.” This resolves the earlier connector visibility gap; attempt 2 supersedes those failed check runs with successful ones.
- [x] Apply only a directly supported remedy. The owner changed the repository from private to public; no billing setting, workflow, product code, test or limit was changed by the qualification agent.
- [x] Perform one bounded current-head infrastructure check after the supported remedy. Attempt 2 of run `34543410273` allocated standard hosted runners and reached checkout on all six jobs, proving the startup blocker was removed.

**Exit:** satisfied. **Evidence:** attempt 1's identical six-job billing/spend-limit annotations and zero-step metadata; attempt 2 of the same run/head allocated runners and completed all six jobs successfully after the repository became public.

## 29.3 Published WebKit correction

- [x] Re-read the current literal-height CSS, responsive source contract, isolated query diagnostic, full application zoom journeys, layout/accessibility suites, shell, transport and App ownership. The tests use CSS `documentElement.style.zoom`, not browser UI zoom or pinch scaling.
- [x] Run the current responsive source contract: 8/8 tests pass. The built CSS contains `42.8571428571em`; the full application WebKit journeys and the tall/short/zoom/reset journey remain present.
- [x] Record the local browser limitation separately. The frozen dependency install and build succeeded locally, but the workspace could not download either Playwright browser; no local page assertion was claimed as acceptance.
- [x] Run all current zoom journeys in Chromium and WebKit against normal preview and project-subpath hosting. Attempt 2 passed both full `precision-zoom.spec.ts` build/run journeys plus the short/tall/zoom/reset regression in both jobs, with intact application coverage and no unexpected or flaky result.
- [x] Inspect downstream results now that the zoom journeys pass. Preview passed 82 with 6 intentional mode skips; subpath passed 86 with 2 intentional mode skips. The committed `--max-failures=5` policy remained unchanged, and no remaining product assertion required another correction.

**Exit:** satisfied. **Evidence:** `precision-summary-preview-fcc38265...` and `precision-summary-subpath-fcc38265...`, with raw browser artifacts retained by run `34543410273`; zero unexpected, zero flaky and no omitted failures/errors.

## 29.4 Local exact-tree gates

- [x] Install the frozen lockfile with pnpm 9.15.9. The dependency tree includes the committed font packages; no manifest or lockfile changed.
- [x] Run all infrastructure tests: 155 passed, 0 failed.
- [x] Run all packages: engine 497, shared 334, web 1,191 — 2,022 passed, 0 failed.
- [x] Run lint after moving browser-generated reports outside the repository scan: pass. The first lint invocation only found generated Playwright report assets; it did not identify a tracked-source defect.
- [x] Run complete source/test typechecks: pass.
- [x] Run a normal production build with fault injection unset: pass, including emitted same-origin font assets.
- [x] Run the unchanged gzip guard: entry 151,321 / 152,245; InspectionPanel 5,432 / 7,373; total JavaScript 233,766 / 234,161 bytes — all pass.
- [x] Run `git diff --check` before this documentation edit: pass.

**Evidence boundary:** these commands ran against exact tree contents `e74b3561...`, but locally under Node 24 rather than the workflow's Node 20, and the recovered local commit ID is synthetic. They are strong diagnostic evidence, not GitHub exact-SHA acceptance and not browser/performance qualification.

## 29.5 Publication, exact candidate and closure

- [x] Obtain independent review of the closure diff for scope, test strength, provenance and scientific/ownership invariants. The first review identified stale attempt-1/current-state wording and an ambiguous retry statement; a follow-up review identified four remaining product-versus-documentation and historical-visibility scope ambiguities. All were corrected before final closure publication. No reviewer files changed.
- [ ] If a repository-local correction is demonstrated, publish one coherent non-forced Git-data commit based on the freshly re-read remote parent/tree. Never publish the synthetic recovery commit.
- [x] Require one exact product candidate to pass `source-evidence`, `focused`, `browsers (preview)`, `browsers (subpath)`, `browsers (recovery)` and `performance` with preserved raw and compact evidence. Attempt 2 of run `34543410273` passed all six on exact product head `fcc38265`, tree `e74b3561...`.
- [x] Inspect current required checks, exact source/build provenance, browser counts/skips/flakes, recovery fault-on/fault-off reports, gzip measurements and five-pair performance medians. All summaries report the exact SHA/tree, no tracked changes, status passed and attempt 2; source archive SHA-256 is `0f2482b2a538c5b7160e40a09ad197c5cad346b68a53d28b604495b137697536`. Raw `pnpm test:perf` collection exits remain `1` for both baseline and candidate because this hosted runner exceeds the preserved historical absolute engine constants; the accepted symmetric same-runner comparator and independent forced-pair/save-capture limits pass.
- [ ] If documentation creates the final commit, qualify that final documentation head; do not start a commit loop merely to write its own run ID.

**Exit:** all six jobs are green on one exact published head and the record accurately identifies that head/run, or the final report names the exact external error and smallest user action while leaving acceptance open.

## 29.6 Unchanged separate work

[-] Legacy-shell retirement, cross-tab saved-run overwrite/concurrency, Pages repository settings and future roadmap work remain separate. Main, Pages and billing settings are unchanged. Repository visibility is now public following the owner's action that enabled standard hosted runner use. The accepted release baseline/performance-policy investigation remains resolved.

## v5 — September 11 exact-head recovery

Refreshed the actual documentation head, corrected the earlier statement that no annotations existed, recorded the zero-billable-millisecond startup signature, captured current exact-tree local gates, and kept browser/CI acceptance open. No scientific contract, test retry/timeout/skip, bundle/performance/layout limit, workflow, product source, account setting, deployment or release state was changed.

## v6 — September 11 authenticated runner-start diagnosis

Authenticated GitHub UI inspection recovered the annotation hidden from the connected API and confirmed that every exact-head job was blocked by failed recent account payments or an insufficient spending limit. No workflow rerun, repository-local correction, billing change or publication was attempted; the remote head remains `fcc38265`, and exact-head CI acceptance remains open pending account-owner action.

## v7 — September 11 exact-head all-jobs-green receipt

After the owner made the repository public, one bounded rerun of `34543410273` started normally and passed all six required jobs on `fcc38265`. Focused qualification passed 155 infrastructure tests and 2,022 package tests, lint, types, build and the unchanged gzip guard (151,321 entry; 5,432 InspectionPanel; 233,766 total JavaScript). Preview passed 82 with 6 intentional hosting/fault-mode skips; subpath passed 86 with 2 intentional fault-mode skips; recovery passed 2 fault-enabled plus 2 fault-disabled tests; every browser summary reports zero unexpected and zero flaky tests. The accepted five-pair macOS/ARM64 comparator passed all engine medians plus forced-pair 9.8853 ms and save-capture 11.8374 ms. The preserved raw baseline and candidate performance collections both exited `1` on the historical absolute engine constants, symmetrically confirming why those constants are not universal-host release thresholds. Product acceptance for `fcc38265` is complete; only the documentation-only closure head remains to receive its own exact-head qualification.


# 30. September 12 branch completion

- Storage fix: `06ade9f` coordinates origin-wide writes, applies mutations to freshly validated persisted state, preserves exact retries and rejected bytes, and guards selected file deletion. Regression-first unit and real-browser reproductions are retained.
- Development dependency patches: `b1fe8e4`; critical/high findings are cleared. Two moderate entries for the same unconfigured Vitest mocker-server path remain disclosed in `BUGS-TO-REVIEW.md`; no scanner rule is suppressed.
- P7 presentation retirement: `7e29402`, preserving live content, ownership and migrated behavior tests.
- `codex/nn-forge-release-roadmap` is already merged. The `precision-apply` branch transports patches already committed in `precision-lab`; importing transport artifacts would add no product work. Old local cockpit/integration snapshots are superseded and remain archival. Other local branch heads are already ancestors of main.
- Final commit, qualification, merge and deployment receipts: [PR #36](https://github.com/DenseDevKev/neural-network-playground/pull/36). Source/check status remains per-commit, never inferred from a prior run.
