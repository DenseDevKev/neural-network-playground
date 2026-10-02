# Cloud Reliability and Browser Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the existing cloud session. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Check the browser modes omitted from the standard run, reproduce the highest-value remaining usability reports, and give the owner screenshots and an actionable findings report from the cloud.

**Architecture:** Use the existing production build, Playwright suites, hosting fixture, and evidence wrappers. Keep browser modes sequential because they share build and report directories. Decide source fixes only after a failure is reproduced and traced.

**Tech Stack:** pnpm 9.15.9, Node 20.19+, React/Vite, Vitest, Playwright Chromium and WebKit.

**Spec:** The user's cloud-only workflow and request for visible browser evidence; [maintenance contracts](../../maintenance/README.md); [post-release observations](../../qa/2026-10-post-release-evaluation.md); sections 20–21 of the [living plan](../../../NN-FORGE-LIVING-EXECUTION-PLAN.md).

## Global Constraints

- Start from verified current `main`; planning baseline is `a48ff98eb261688cab3a17975aa650df24195a45`. Recheck the remote before execution and preserve existing changes.
- Work in the existing cloud checkout. The owner can review screenshots in chat from a phone; a local session is unnecessary.
- Preserve V2 recipes, saved evidence, worker protocol, scientific defaults, and evaluation cadence.
- Keep zero Playwright retries and existing assertions. Retain failures and distinguish environment errors from application defects.
- Preserve gzip caps: entry 152245 bytes, InspectionPanel 7373 bytes, total JavaScript including worker 234161 bytes.
- Phone viewport emulation is cloud browser evidence, not a physical iPhone/Android or GPU certification.

## Review Focus

- Returning from Code → TF.js to Share setup: verify which tab opens and whether the link controls are reachable (Task 1).
- Training status at 320, 360, and 390 CSS pixels: inspect step/epoch spacing, long status text, touch targets, and horizontal overflow (Task 1).
- Worker startup failure: recovery must restore focus and a functioning step-zero model before a real training step succeeds (Task 2).
- Hosting without cross-origin isolation: workers and lazy assets must load below the project path, and fresh shared links must retain the recipe (Task 3).
- Save failure: retry must retain the captured artifact even after navigation or a recipe change; a failure must never appear as a successful save (Tasks 1 and 3).

## Evidence already available

The merged baseline passed 1,838 unit/integration tests, 156 helper tests, typecheck, lint, build, bundle guards, and 152 standard browser tests. Ten mode-specific skips remain: four opt-in visual cases, four external-hosting cases, and two fault-enabled cases. Preserve those receipts; repeat unchanged checks only when the source, dependencies, or build mode warrants it.

The old walkthrough predates the favicon fix and bundle reductions. Reproduce remaining observations instead of treating the historical report or its 73-byte headroom as current. The latest measured total is 233101 / 234161 gzip bytes.

### Task 1: Capture the application and verify reported friction

**Files:** Use `tests/e2e/atelier-visual-evidence.spec.ts`, `atelier-visual-gallery.ts`, `atelier-helpers.ts`, `atelier-critical-flows.spec.ts`, and `precision-acceptance.spec.ts`. Inspect `apps/web/src/App.tsx`, `components/atelier/AtelierTransport.tsx`, and `styles/atelier.css` if the reports reproduce. No source edit is predetermined.

**Interfaces:** Consumes a normal production build and the saved cloud tool/WebKit activation instructions. Produces browser-specific screenshots, console/network observations, and exact reproduction steps.

- [x] Activate the saved cloud environment, record the source SHA, and confirm the build is normal rather than fault-enabled.
- [x] Run the existing gallery separately for each engine, using an absolute writable output directory so captures cannot resolve to `/outputs` or overwrite the other engine:

```bash
ATELIER_VISUAL_EVIDENCE=1 ATELIER_VISUAL_OUTPUT=/workspace/outputs/cloud-reliability/chromium pnpm exec playwright test tests/e2e/atelier-visual-evidence.spec.ts --project=chromium --workers=1
ATELIER_VISUAL_EVIDENCE=1 ATELIER_VISUAL_OUTPUT=/workspace/outputs/cloud-reliability/webkit pnpm exec playwright test tests/e2e/atelier-visual-evidence.spec.ts --project=webkit --workers=1
```

- [x] Preserve JSON/HTML reports and test artifacts after each run, before the next Playwright invocation overwrites them. Both light/dark cases must execute in each engine; report the actual outcomes.
- [x] Reproduce Code → TF.js → close → Share setup in both engines. Record the actual selected tab, reachability of sharing controls, and whether the recipe/model changes. Classify behavior against the intended entry point before calling it a bug.
- [x] Capture trained Results and Setup at 320×844, 360×844, and 390×844 in both themes. Check non-overlapping status text, reachable controls, and no unintended page overflow. Reuse existing state-readiness assertions instead of arbitrary delays.
- [x] Share a small selection of actual desktop and phone captures in chat, with browser, viewport, and app state captions. Inspect the images before reporting visual findings.

### Task 2: Verify failure recovery and restore the normal build

**Files:** Use `tests/e2e/worker-recovery.spec.ts`, `scripts/run-with-evidence.mjs`, and the recovery sequence in `.github/workflows/precision-lab-verification.yml`.

**Interfaces:** Consumes the existing fault-injection build switch. Produces separate fault-enabled and normal-build recovery receipts; leaves a normal production build for Task 3.

- [x] Run `pnpm test:e2e:recovery`. Require the injected failure test to execute and pass in both browsers, including the error dialog, focus trap, refresh, and subsequent training step.
- [x] Archive the fault-enabled report before rebuilding; preserve diagnostics if either browser fails.
- [x] Rebuild normally even after a failed recovery test: `env -u VITE_E2E_FAULTS pnpm build`.
- [x] Run `pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --grep '@fault-disabled' --workers=2`. Both browsers must ignore the fault query and train successfully. Run `pnpm test:bundle` on the restored normal build.

### Task 3: Test deployment-style hosting within the cloud

**Files:** Use `scripts/serve-release-fixture.mjs`, `scripts/playwright-target.mjs`, `tests/e2e/deployment-contract.spec.ts`, and the existing browser suite.

**Interfaces:** Consumes Task 2's normal build. Serves `http://127.0.0.1:4174/neural-network-playground/` without isolation headers; produces hosting and full-workflow browser results.

- [x] Start `node scripts/serve-release-fixture.mjs apps/web/dist 4174` in a managed execution session. Confirm HTTP readiness and the intended absence of COOP/COEP.
- [x] Run `PLAYWRIGHT_BASE_URL=http://127.0.0.1:4174/neural-network-playground/ pnpm test:e2e --workers=2 --max-failures=5` with `PLAYWRIGHT_PORT` unset. This exercises deployment contracts plus navigation, save/retry, and concurrent persistence workflows in both browsers.
- [x] Confirm the four hosting tests execute; inspect worker/chunk/font responses, the real training step, and shared-recipe reload results. Document all remaining mode skips explicitly.
- [x] Archive reports and stop the fixture. Describe this as a hosting simulation, not a live deployment receipt.

### Task 4: Deliver findings and scope any fixes

**Files:** Create `docs/qa/2026-10-02-cloud-reliability-pass.md` after execution. Keep raw captures and logs under `/workspace/outputs/cloud-reliability/`; link the selected evidence from the report.

**Interfaces:** Consumes Tasks 1–3's exact-source results. Produces one concise report and, only for reproduced defects, a small implementation scope naming the source files and regression tests.

- [x] Record the source SHA, command outcomes, browser counts/skips, screenshots, and whether each historical observation reproduced. Distinguish usability preferences from broken contracts.
- [x] Prioritize confirmed data loss, incorrect results, and blocked workflows. For each confirmed bug, specify reproduction, root cause, minimal fix, and a regression that fails before the fix and passes afterward.
- [x] Not applicable in this pass: no source fixes were made. After any later authorized source fixes, run focused regressions and the relevant standard/browser/build gates on the final candidate. Keep evidence tied to that candidate; do not reuse baseline passes for changed code.
- [x] Report any performance question separately. A release performance claim requires the existing five-pair same-host comparator; isolated `pnpm test:perf` timings do not establish that claim. Publishing or merging a new candidate is a subsequent integration step.

**Done when:** The owner has viewable screenshots, executed results for all three omitted modes, and a prioritized list of reproducible findings or an explicit no-new-defects result. No new feature, redesign, live deployment, or performance claim is required to complete this pass.

Execution result: see [cloud reliability report](../../qa/2026-10-02-cloud-reliability-pass.md). All planned modes executed; two reproduced defects are documented with fix scopes. Application source remains unchanged.
