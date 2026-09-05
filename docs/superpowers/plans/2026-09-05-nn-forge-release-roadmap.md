# NN.FORGE consolidated release and product roadmap

> Execute reviewed tasks in dependency order with Superpowers executing-plans,
> test-driven-development, and verification-before-completion. A failed or
> unperformed gate must remain explicit.

**Goal:** Finish one reliable release of the existing product, then integrate one
coherent Precision Lab presentation—not a collection of competing product modes.

**Product authority:** `main`, intake commit
`ae09b9863ae90f8fb2f62545834fcc138755ba9a`.
**Execution branch:** `codex/nn-forge-release-roadmap`.
**Recorded code qualification:** `fba63307cd31a7646af15dfb4fff95825dff02f9`.
See the [verification record](../verification/2026-09-05-release-roadmap.md) for
actual run IDs, measurements, and limitations. Later documentation is not a new
claim that this code has been deployed.

## Architecture and authorization

Keep the existing prepared V2 experiment, engine, training hook, worker,
evaluation provenance, and saved-record contracts. BuildRunShell remains the
production composition until a fully reviewed replacement is accepted. Branch
code/docs execution was requested; publication, visibility changes, historical
merges, branch deletion, and unverified advancement of main were not authorized.

Preserve Current recipe -> Trained snapshot -> Active run -> Evidence -> Saved
runs. Navigation/profile/disclosure must preserve the recipe, model generation
and revision, step, checkpoint timeline, saved records, export selection, and
experiment URL. Batch/EMA and full-split train/test evaluation remain separate.

No math, schema, worker protocol, persistence format, evaluation cadence, blanket
code splitting, or relaxed thresholds are part of release qualification.
**Narrow dependency amendment:** Q2 adds only exactly pinned Inter and Space
Grotesk Fontsource 5.3.0 packages to remove a reproduced external-font reload
failure. This replaces the earlier absolute no-new-dependencies rule for that
bounded correction only. No external UI framework or backend is introduced.

## Current finite sequence

| Stage | Status at the recorded code qualification | Exit condition |
|---|---|---|
| R0 — Source authority | Completed | Private access, exact SHA, one isolated branch, immutable source evidence |
| R1 — Browser targets | Implemented and verified | Explicit targets fail closed and retain project paths; local defaults remain |
| R2 — Hosting contracts | Implemented and verified against a static fixture | Real worker, lazy panels, canonical recipe reload, and non-isolated training pass in both browsers |
| R3 — Release qualification | Executed, not fully qualified | Correctness, browser, recovery, and bundle checks pass; engine timing qualification remains open |
| R4 — Consolidated handoff | Documentation delivered with this record | One roadmap and verification event, accurate QA/deployment commands, historical reconciliation |
| R5 — Live publication | Owner-gated; not performed | Approved main SHA, settings, deployment provenance, and live browser verification |
| P1 — Executable gzip limits | Implemented and verified, pulled forward independently | Reviewed fixed JavaScript caps enforced in CI; no shell activation |
| P2–P7 — Precision Lab integration | Planned, not implemented | Reviewed presentation integration and all acceptance gates below |

R0 -> R1 -> R2 -> R3 -> R4 -> R5 remains the release order. R4 records failed gates
rather than hiding them. P1 was a safe guardrail pulled forward; it does not make
Precision Lab the current product or make R3/R5 complete.

## Release task contracts

### R0 — Source authority

Files: this plan and `.github/workflows/release-verification.yml`.
Capture only tracked source using `git archive`, record its full SHA and artifact
digest, and retain the intake CI/deployment as historical evidence. Main is not
moved. No previous Mac, dependencies, cache, or unuploaded prototype is required.

### R1 — Fail-closed Playwright target

Files: `scripts/playwright-target.mjs`, `scripts/playwright-target.d.mts`,
`scripts/playwright-target.test.mjs`, `playwright.config.ts`.
`resolvePlaywrightTarget(env)` preserves local port validation. An explicit
`PLAYWRIGHT_BASE_URL` accepts HTTPS or HTTP loopback, rejects credentials,
query/fragment, malformed input, and simultaneous explicit port. Preserve the
full project path; external mode never starts or falls back to preview. The
report identifies the normalized base URL and mode. Acceptance is the real Node
helper suite plus unchanged Chromium/WebKit/timeouts/retries.

### R2 — Production hosting and state integrity

Files: the three existing `tests/e2e/` specifications, new
`deployment-contract.spec.ts` and `navigation-integrity.spec.ts`, and
`scripts/serve-release-fixture.mjs` with its Node tests.
Use relative application navigation. Serve the actual normal build under
`/neural-network-playground/` with no isolation headers. Require secure-context,
non-isolated worker loading; step 0 -> step 1 paired evidence; successful lazy
Inspection/Code/History/Configuration resources; canonical V2 sharing into a
fresh context with identical recipe/architecture. Preserve original assertions.
The skip-link correction in `apps/web/src/App.tsx` focuses main without changing
the experiment fragment; both keyboard and pointer tests preserve model identity
and step. Do not spoof the worker or substitute successful asset responses.

### Q2 — Native font reload correction (closed by fresh browser evidence)

Files: `apps/web/index.html`, `apps/web/src/main.tsx`,
`apps/web/src/styles/fonts.css`, `apps/web/package.json`, `pnpm-lock.yaml`,
`apps/web/public/font-licenses.txt`, `tests/e2e/font-delivery.spec.ts`.
Retain Inter 400/500/600/700 and Space Grotesk 400/500/600 from exact package
versions, loaded through Vite from the application origin. Keep isolation headers
and the original recovery console assertion. Require actual font-face loading,
zero remote Google font requests, same-origin/project-path font assets, and two
native reloads in both browsers. The full normal/subpath/recovery suites must
pass. Font/CSS transfer size is outside the JavaScript-only budget; no cold-start
speedup or pixel-identical historical font revision is claimed.

### R3 / Q1 — Qualification and the remaining performance gate

Files: `.github/workflows/release-verification.yml`,
`.github/workflows/performance-reference.yml`, and the verification record.
Run infrastructure helpers, lint, complete typechecks/unit suite, production
build, bundle caps, preview, non-isolated fixture, and fault-enabled recovery.
Preserve the recovery report before a clean rebuild and normal-build inertness
check. Independent jobs retain failures and artifact digests.

The candidate and intake baseline both failed the unchanged engine timing caps
on the same Linux runner and across five complete paired collections on a fresh
virtual Apple M1 runner. All worker evaluation/save budgets passed. No engine or
benchmark source was changed. Do not call this a proven regression, a performance
improvement, or a passed gate. The virtual runner is not the prior physical Mac.

To close Q1, use a documented reference environment and run baseline and candidate
with the same Node/pnpm/dependencies/hardware, five complete collections with raw
results retained. Require all existing absolute gates and the agreed relative
regression criteria. If the reference baseline still fails, retain the failure
and resolve the reference-host/budget policy explicitly before any policy change.
Do not retry until one lucky pass, drop outliers, or silently recalibrate caps.
No source optimization is prescribed without a demonstrated new code regression.

### R4 — Documentation closure

Modify `README.md`, `docs/deployment.md`, `docs/qa/QA_CHECKLIST.md`, and this plan.
Create `docs/superpowers/README.md` and
`docs/superpowers/verification/2026-09-05-release-roadmap.md`.
Use a central dated reconciliation index rather than rewriting the original five
historical documents. Link current commands and executed evidence; distinguish
branch code from main, fixtures from live hosting, and a successful subset from
a successful release. Check relative links and `git diff --check`. Do not invent
an immutable-build comparison after a genuine product fix changed the build.

### R5 — Owner-authorized live release

No application change is prescribed for the original Pages 404. The owner must
approve public delivery and confirm private-repository Pages eligibility without
changing source visibility. Only then enable Pages with Source = GitHub Actions.
Accept reviewed branch work into main only when authorized and qualified. Verify
the actual CI-tested SHA selected by deployment, artifact digest, run/attempt,
`page_url`, and timestamp. Execute the external browser suite against that URL.
A fixture pass is not live verification; a manual dispatch is not proof of prior
CI. Until publication is approved and verified, R5 remains unperformed.

## One later product milestone: Precision Lab

Use the [July 16 design](../specs/2026-07-16-precision-lab-production-integration-design.md)
and [July 17 detailed plan](2026-07-17-precision-lab-production-integration.md) as
design evidence. Reconcile older labels such as `Audience mode` with the current
`Workspace profile`. A missing visual reference is a visual-comparison gate, not
permission to import mock prototype state or a dependency for release testing.

The original detailed file map and test steps remain in the July 17 plan. The
following table groups them into independently reviewable slices. Basenames below
are under `apps/web/src` unless a repository-relative path is specified.

| Slice | Files and change boundary | Required tests and acceptance |
|---|---|---|
| P1 — Gzip limits (completed) | `scripts/check-web-bundle-gzip.mjs`, its Node tests, root `package.json`, `.github/workflows/ci.yml` and release workflow | Parse actual entry, require one InspectionPanel chunk, recurse through all JS including worker; reject absent/ambiguous inputs and every one-byte cap overrun. Fixed limits: entry 152245, Inspection 7373, aggregate 234161 bytes. |
| P2 — Display-safe recipe and shell | `components/layout/precisionLab/` recipe models/hooks/views, `PrecisionLabShell.tsx`, `styles/precisionLab.css`, Header and existing layout actions; July tasks 2–4 | Test null/pending/drift/aged/fresh model precedence; adapt only canonical stores; no raw worker envelopes in shell props. One training-hook owner, correct profile visibility, focus return, and unchanged paused recipe/model/checkpoint/hash/export state. |
| P3 — Real network selection | `networkSelectionModel.ts`, `useNetworkSelectionController.ts`, `NetworkSelectionDeck.tsx`, Canvas/SVG renderers and painter; July tasks 6–8 | Deterministic strongest-path ranking from real typed arrays, functional Canvas/SVG parity, no copied live grids in React state, selection invalidates on incompatible generation/architecture. Selection must not change the recipe or model. |
| P4 — One pinned boundary | `useDecisionBoundaryController.ts`, `DecisionBoundaryCanvas.tsx`, `PinnedBoundaryRail.tsx`, `BoundaryEvidencePanel.tsx`, `components/layout/deriveVisualizationDemand.ts`, App; July tasks 9–10 | Exactly one live boundary canvas survives Build/Run and evidence changes; detail view does not duplicate it. Test demand transitions and scientific-state invariants; never lower full-evaluation cadence to satisfy visual performance. |
| P5 — Production previews and compact operation | `datasetPreviewModel.ts`, `DatasetPreviewCanvas.tsx`, DataPanel, LossChart, ConfusionMatrix, TrainingControls, `useSaveCurrentRun.ts`, RunHistoryPanel; July tasks 5, 11–12 | All eleven previews use deterministic production generators. Test bounded evidence layouts, exact save-artifact retry on failure, user-visible errors, full-split terminology, and current lifecycle actions. No fabricated preview or saved-run state. |
| P6 — Production acceptance | Tests beside each slice, App/shell integration, `tests/e2e/precision-lab-layout.spec.ts`; July task 13 | At 1437x742, 735x860, 320x844: five seconds at 50 steps/frame keep major region bounds within 1 CSS pixel; zero post-start Chromium CLS in the specified scenario; required 44px touch targets. Verify keyboard, reduced motion, 200% zoom, both browsers, zero retries, scientific invariants, and unchanged performance/gzip caps. |
| P7 — One final shell | App composition, consumer-proven obsolete-shell removal only, qualification record; July task 14 | No competing production shell. Preserve V2/scientific contracts, reviewed visual reference, all applicable gates, and exact release provenance before acceptance into main. |

P2 -> P3 -> P4 -> P5 -> P6 -> P7 follows P1 and the approved release/design gates.
Pure preview work may be reviewed separately but must not activate an incomplete
shell. Each slice must pass its focused tests and broad regression checks before
the next slice is accepted. There is no backend/account/collaboration/new-dataset
or general optimization program hidden in this roadmap.

## Review, rollback, and stop conditions

Use a fresh scoped review where available; do not label self-review independent.
Revert a specific branch commit to undo a coherent change. Do not force-update
main, delete historical branches, alter publication settings, relax assertions,
or replace failed metrics with a narrative pass. Stop at an unresolved dependency
and keep other completed work accurately recorded. After these two milestones,
use actual released-product feedback before defining another product direction.
