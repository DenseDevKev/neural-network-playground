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

**Reconciled September 8, 2026 against fresh GitHub refs, not a prior Mac.**

- Repository: `DenseDevKev/neural-network-playground` (private).
- Product authority: `main` at `98f29b86e469a2a545be75928ae6f32309fd1582`.
- Release baseline: **merged and accepted**; merge parents are `ae09b986...` and `2cd5b896...`. Main CI `34002500733` succeeded.
- Active branch: `codex/nn-forge-precision-lab`.
- Latest freshly read upstream head: `3e7da9a4dabd2d7bc275c9aa4c614a04d9b26732`, not the supplied older `6e06b678...`.
- Exact upstream tree: `80d2b9b4bf1a6a5a55a32752b844f7a239967477`.
- Main composition: accepted Build/Run presentation. Active-branch composition: **PrecisionLabShell already integrated**, not a future shell.
- Precision Lab merged to main: **no**. Successful Pages deployment: **no**.
- This continuation: **local patch on the exact upstream tree; NOT committed or pushed through GitHub**. The available GitHub actions are read-only and cannot dispatch qualification.
- Source recovery: verified GitHub Actions source archive and matching-lockfile dependency workspace; no previous Mac was assumed.
- Canonical checklist: this same file recovered from the user's Library. It was absent from the recovered branch tree; this patch restores it at repository root, preserving its numbered milestones/history.

**Evidence boundary:** `[x]` below requires identified committed implementation and test evidence, or a directly verified historical/ref event. New local implementation is described with test results but remains `[ ]` until committed and the item's required gates pass. Passing component tests do not qualify browser geometry, accessibility, recovery, performance, or release.

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

**Accepted baseline helper count:** 88 passed (run `34001183202`). Local continuation helper count: 91 passed; not yet an upstream run. Second normal rebuild: all 87 dist files byte-identical and fixed bundle guard green; this does not stand in for browser recovery.

---

# 5. P1 pulled forward — JavaScript bundle contract

**Caps unchanged in this continuation.** Current local final build measures entry **151,131 / 152,245**, InspectionPanel **5,431 / 7,373**, total JavaScript **234,068 / 234,161** bytes. Total headroom is only **93 bytes**, so exact CI rebuild verification is mandatory. The earlier baseline numbers below are historical, not the current bundle size. Evidence: `bundle-final.log` in the continuation evidence package.


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

- [ ] Run the accepted five-pair macOS/ARM64 reference gate for the new committed Precision Lab candidate.
- [ ] Record exact candidate SHA, host metadata, all raw samples, comparator result and independent worker budgets.

The local Linux / Node 22.16.0 diagnostic `pnpm test:perf` exited **1**: two historical absolute engine benchmark tests failed and one passed; the two worker scientific-trust tests passed. Forced-pair median **8.0115 ms**, save-capture median **9.6054 ms**. This is neither an accepted reference qualification nor proof of a candidate-specific regression. Engine/shared implementation and all timing constants are unchanged. Do not weaken thresholds or restart the policy investigation because of this diagnostic.

The local workflow patch reuses the accepted paired comparator on `macos-15` with an ARM64 assertion. It has not been pushed or executed.

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

# 12. Precision Lab — reconciled intake

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

**Committed integration exists:** `3e7da9a...`, Actions `34254380062` focused correctness/build/bundle job passed; browser jobs failed, so the milestone is not accepted. App uses PrecisionLabShell, one `useTraining`, one selection controller and one live boundary controller. Local continuation preserves this ownership and adds the shared save controller. Browser/keyboard/viewport equivalence remains pending.


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

Local browser specifications retain connected-canvas identity and URL assertions, add both Build/Run and all major region bounds, and preserve 5.5 seconds at 50 steps/frame with <=1 CSS px movement and Chromium CLS = 0. They have been discovered, **not executed successfully**.

Do not implement pin/replace/clear or persistent pinned snapshots: that was a stale misreading of the design, not a missing feature.

---

# 16. P5 — Production-backed previews / evidence

**Local implementation, not yet committed/accepted.** Shared App-owned `useSaveCurrentRun` serializes transport and History saves, keeps capture in the worker, preserves pending artifact identity across recipe changes and drawer unmount, and retries only the persistence store. Focused save regressions and all 1,170 web tests pass. LossChart uses coalesced responsive canvas sizing; confusion metrics use semantic definition lists. Browser geometry/a11y/recovery remains unqualified. Evidence: September 8 verification document and raw final test JSON files.


## Dataset previews
- [ ] enumerate all production dataset IDs.
- [ ] deterministic production generators.
- [ ] no mock point clouds.
- [ ] tests for every dataset.
- [ ] correct labels/classes/axes.

## Loss/confusion
- [ ] compact LossChart.
- [ ] compact ConfusionMatrix.
- [ ] provenance labels preserved.
- [ ] train/test distinction preserved.
- [ ] no horizontal overflow.
- [ ] mobile operation.

## Training controls
- [ ] Run/Pause.
- [ ] Step.
- [ ] reset.
- [ ] speed / steps-per-frame.
- [ ] disabled reasons.
- [ ] keyboard.
- [ ] touch targets.

## Save/history
- [ ] save exact intended artifact.
- [ ] preserve worker-owned capture.
- [ ] failed save retries exact pending artifact.
- [ ] no silent recapture on retry.
- [ ] saved-run explanation remains explicit.
- [ ] Apply Saved Recipe semantics remain explicit.

**Exit:** compact evidence UI uses real production data and preserves artifact identity.

---

# 17. P6 — Production acceptance

**Current result: NOT RELEASE-QUALIFIED.** Local package suites: engine **497**, shared **334**, web **1,170** = **2,001** passing tests; helpers **91**. Lint, complete typechecks, production build and unchanged gzip caps pass locally. Chromium and WebKit launch attempts both fail before application startup because their executables are absent; Playwright discovery lists 66 tests but is not execution. Required browser/recovery/a11y/layout/zoom and accepted macOS five-pair performance gates remain open. No local implementation box below is checked merely because code was written or a mock test passed.


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

**Migration state:** App already composes PrecisionLabShell at committed `3e7da9a...`. Legacy BuildRunShell/MainArea remain as compatibility code/test consumers until real-browser acceptance proves removal safe. This continuation marks removable memo factories pure for tree-shaking but does not delete the legacy shell or claim P7 complete. Browser/performance gates cannot be waived to force removal.


## Migration
- [x] identify current production App composition and retained legacy/test consumers — current tree `3e7da9a...`.
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
Status: locally implemented and unit-verified; awaiting commit and browser acceptance.

### D007 — Honest evidence boundaries
GitHub connector reads and downloaded source artifacts are authority. Local patch/test results are not a remote commit, CI run or live deployment. Missing browser binaries and unreadable tool responses are tooling limits.
Status: active.

### D008 — No bundle-limit expansion
Educational content is grouped into one on-demand chunk; safe unused memo factories are tree-shaken; Terser uses two normal compression passes. No cap, scientific cadence, schema or engine algorithm was changed.
Status: local final build passes fixed caps; total headroom only 93 bytes, so CI rebuild remains mandatory.

---

# 23. Immediate execution queue (reconciled September 8)

## NEXT-01 — Preserve and commit this exact patch
- [x] Re-read main and active branch; recover the newer committed shell/boundary instead of replacing it.
- [x] Preserve source archive/tree and CI failure evidence separately from the prior tooling failure.
- [ ] Review/apply the local patch on exact `3e7da9a4dabd2d7bc275c9aa4c614a04d9b26732` with a write-capable session.
- [ ] Commit shared save ownership/retries, responsive/evidence fixes, targeted lazy loading and regressions.
- [ ] Commit this restored canonical living plan and verification record.

The local evidence package contains the exact patch, source/diff/build digests and raw logs. No remote commit exists for these changes. Do not push the synthetic local source-import history.

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

---

# 26. Living-plan changelog

## v1 — 2026-09-05
- Created living execution ledger.
- Backfilled completed release-readiness tasks.
- Marked engine benchmark-policy work as current blocker.
- Added exact merge/deploy/Precision Lab order.
- Added explicit completion rules to prevent casual checkbox completion.

---

## v2 — 2026-09-08 (local continuation, awaiting commit)
- Recovered this canonical file instead of inventing a replacement checklist.
- Reconciled resolved performance policy, merged main, accepted baseline CI and blocked Pages status.
- Recovered newer committed shell/boundary integration and preserved App ownership.
- Corrected pinned-boundary semantics; no saved snapshot workflow was invented.
- Recorded local shared save/retry, responsive evidence/help/transport and bundle changes with raw test evidence.
- Kept real-browser, accessibility, zoom, recovery, layout, accepted reference performance, legacy removal, merge and deployment unchecked.

# Current one-line status

> **Accepted baseline is merged; Precision Lab shell/controller/boundary are committed upstream at 3e7da9a. The local continuation passes 2,001 package tests, 91 helpers, lint, typechecks, build and unchanged bundle limits, but it is not pushed or release-qualified: browser executables and write/dispatch access are unavailable, reference performance is unrun, and Pages settings remain blocked.**
