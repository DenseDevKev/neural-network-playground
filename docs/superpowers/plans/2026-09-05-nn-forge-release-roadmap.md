# NN.FORGE Release Roadmap and Execution Plan

> **For agentic workers:** Use Superpowers executing-plans task-by-task, test-driven-development for code, and verification-before-completion. Keep unverified work explicitly unverified.

**Goal:** Finish the current NN.FORGE release path with repeatable hosting verification, preserved scientific state, and one authoritative handoff.

**Architecture:** Keep the existing BuildRunShell, prepared V2 experiment, training hook, worker, and evidence stores. Extend test infrastructure to run the same production browser checks at either local preview or an explicit project URL. Collect source, test, and build provenance without automatically publishing the private repository's application.

**Tech stack:** Existing pnpm 9 / Node >=20 / React / TypeScript / Playwright Chromium and WebKit; dependency-free Node helpers.

**Spec:** docs/architecture/product-shell.md; docs/qa/QA_CHECKLIST.md. This document supplies the release-infrastructure design and acceptance contract below.

**Baseline:** main at ae09b9863ae90f8fb2f62545834fcc138755ba9a. Execution branch: codex/nn-forge-release-roadmap.

**Authorization:** The owner requested roadmap planning and code execution in chat. This authorizes branch code/docs work, not public publication, visibility changes, historical merges, deletion of branches, or unverified advancement of main. No pull request is opened under the existing handoff restriction.

## Global constraints

- Preserve Current recipe -> Trained snapshot -> Active run -> Evidence -> Saved runs.
- Profile/navigation/disclosure must not change recipe, model generation/revision, step, checkpoints, saved runs, code-export selection, or experiment hash.
- Keep batch/EMA separate from same-revision full-split train/test evaluation, including age and recipe drift.
- Preserve V2 schema, identities, worker protocol, checkpoint and saved-record formats, math, and evaluation cadence.
- Keep Chromium and WebKit, zero browser retries, 45-second test timeout, 10-second assertions, all existing assertions and expected fault-only skips.
- No blanket code splitting, threshold relaxation, action-runtime migration, new dependencies, backend, or mock prototype state.
- Settings and live-publication steps remain owner-gated. A blocked gate is not completed work.

## Decision and sequence

Finish the current product before replacing its shell. A broad optimization program has no reproduced failing budget; Precision Lab is a separate presentation redesign with historical prototype references. Neither is a dependency of release readiness.

The finite roadmap is R0 -> R1 -> R2 -> R3 -> R4 -> R5. R0-R4 are engineering and evidence; R5 is owner-authorized publication. A confirmed product defect discovered by these checks gets one bounded regression fix, not a new audit program.

### R0 — Establish reproducible source evidence

- [ ] Confirm private access and exact main SHA.
- [ ] Create one execution branch from the pinned commit; leave main unchanged.
- [ ] Capture the branch source with git archive and private Actions artifacts, recording the full SHA.
- [ ] Retain the existing successful CI run 33944120268 and failed deployment 33944571511 as historical, not new execution evidence.

Files: this roadmap; .github/workflows/release-verification.yml; later verification record.
Acceptance: archive identifies its SHA; no credentials, caches, dependencies, previous Mac workspace, or local-only prototypes are needed.

### R1 — Fail-closed browser targets

Files: scripts/playwright-target.mjs; scripts/playwright-target.test.mjs; playwright.config.ts.
Implement resolvePlaywrightTarget(env) with local and external variants. Default local URL/port behavior remains. PLAYWRIGHT_BASE_URL must be an absolute HTTPS URL, or HTTP loopback for a static fixture, with no credentials/query/fragment. Preserve the project subpath and add a trailing slash. Reject empty/malformed values and simultaneous explicit PLAYWRIGHT_PORT. External mode starts no preview server and cannot fall back to localhost.
Write Node tests before implementation for defaults; ports 1/65535; nonnumeric/zero/overflow ports; valid project paths; loopback; forbidden HTTP/credentials/query/fragment; and relative-navigation resolution. Test helpers outside browser discovery avoid duplicating nonbrowser tests in two browsers.
Acceptance: node --test scripts/playwright-target.test.mjs passes; config metadata records mode/baseURL; existing timeouts/projects/retries are unchanged.

### R2 — Production subpath and transport verification

Files: tests/e2e/playground-smoke.spec.ts; tests/e2e/accessibility.spec.ts; tests/e2e/worker-recovery.spec.ts; tests/e2e/deployment-contract.spec.ts; scripts/serve-release-fixture.mjs and tests as needed.
Replace root application navigation with ./, retaining existing assertions. Add two external-only cases: (1) naturally non-isolated secure-context worker load and step-0 -> step-1 paired evaluation; (2) project-path lazy Inspection/Code/History/Configuration and canonical V2 share reload into a fresh context. Collect page/console/resource failures and attach observed target and resource evidence.
Use the unchanged normal dist at a non-isolated loopback project subdirectory. Do not spoof SharedArrayBuffer, intercept application assets with fake successful responses, or publish fault-enabled output.
Acceptance: original runnable browser cases pass with their existing checks in local and external fixture modes; deployment cases pass in Chromium/WebKit externally and explicitly skip only under isolated preview.

### R3 — Repeatable verification and independent results

Files: .github/workflows/release-verification.yml; small dependency-free reporting helpers/tests where required.
Run lint/typecheck/unit/build; normal browser smoke; non-isolated subpath browser smoke; dedicated fault-enabled recovery; and a final clean normal rebuild/check. Keep normal, fixture, and recovery reports in distinct artifact paths even on failure. Record exact source SHA and build SHA-256 file manifest. Performance runs retain existing thresholds; never compare different hardware as a controlled before/after speedup. Independent gate failures remain visible and do not suppress other evidence.
Acceptance: green means actual exits and asserted results, never a pending run; normal output is rebuilt after recovery; no deploy-pages step or write permission exists in this verification workflow.

### R4 — Consolidated evidence and historical reconciliation

Files: docs/superpowers/verification/2026-09-05-release-roadmap.md; README.md; docs/deployment.md; docs/qa/QA_CHECKLIST.md; the May 22 Build/Run, July 11 Scientific Trust, and July 16/17 Precision Lab specs/plans as short pointers.
Record newly executed and earlier evidence separately. Scientific Trust has committed implementation and verification; unchecked historical boxes are not a backlog. Precision Lab is not mounted by App at the baseline. Preserve historical content and add dated pointers instead of rewriting history.
Acceptance: each claim has a source/run or is marked unverified/blocked; documentation matches actual test commands; no successful deployment receipt is invented.

### R5 — Owner-authorized publication

No application-code change is prescribed. The owner chooses public publication and verifies GitHub plan eligibility while keeping source private. Enable Pages with GitHub Actions only after that decision. Reuse a valid pinned artifact or a new verified candidate build; record the actual deployment SHA, artifact, URL, and successful run. Execute the external suite against that exact URL.
Expected default URL (not proof of publication): https://densedevkev.github.io/neural-network-playground/.
Acceptance: explicit publication consent; successful deployment; live Chromium/WebKit resource, state, and hosting tests; known product/harness SHAs. Until then status is blocked on owner publication, not completed.

## Product work after this release

Preserve the current shell as authoritative. The only already-specified later direction is Precision Lab. Before activating it, reconcile its exact file map against the released head, confirm the visual reference without importing mock state, and renew the design decision. Its internal dependency order is bundle measurement -> display adapters -> shell and focus -> real graph selection -> single pinned boundary and demand -> dataset previews -> evidence/transport -> responsive/accessibility/performance acceptance. This is a deferred design, not shipped code, and is not silently authorized by an unchecked historical plan.

## Review and rollback

Each coherent code task is a separate commit. Validate before advancing; do not force-update main. Revert a task's branch commit to undo it. Do not delete old branches or merge historical code. Source-level defects need a failing regression and the smallest correction. Owner operations, inaccessible references, pending CI, and failed gates remain explicit in the handoff.
