# Release-Ready Product Shell Implementation Plan

> **Execution record, not final evidence:** This plan remains authoritative for
> implementation order and acceptance criteria. Pre-change measurements live in
> [`../../qa/2026-07-16-product-shell-baseline.md`](../../qa/2026-07-16-product-shell-baseline.md),
> while current architecture is documented in
> [`../../architecture/product-shell.md`](../../architecture/product-shell.md).
> Final pass counts, bundle sizes, performance medians, browser results, and
> limitations must be recorded separately during Task 7 after the exact commands
> run; task text alone is not proof of completion.

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task, with test-first implementation and an independent review after each slice.

**Goal:** Ship accessible Advanced Tools, a typed terminology catalog, and shared Beginner/Explore/Lab profiles without changing experiment or training semantics, then prove the result release-ready in Chromium and WebKit.

**Architecture:** Keep one product shell and one state graph. A pure profile/capability table drives visibility; `useLayoutStore` persists local profile/disclosure navigation; a shared evidence resolver drives both rendering and worker demand; a React-free concept catalog feeds small accessible help disclosures. Existing engine, worker, V2 document, URL, checkpoint, and saved-run contracts stay unchanged.

**Tech stack:** React 19, TypeScript, Zustand, Vitest/Testing Library/jest-axe, Playwright, Vite, pnpm.

**Design reference:** `docs/superpowers/specs/2026-07-16-release-ready-product-shell-design.md`

---

## Task 1: Record the verified baseline and browser diagnosis

**Files:**

- Modify: `docs/architecture/state-ownership.md`
- Create: `docs/qa/2026-07-16-product-shell-baseline.md`
- Verify: `playwright.config.ts`, `tests/e2e/playground-smoke.spec.ts`, `.github/workflows/ci.yml`

**Steps:**

1. Record the exact pre-change lint, unit/integration, build, zlib gzip, browser, and performance outputs.
2. Document the prior timeout symptoms, host load evidence, and the clean unchanged rerun. Classify the root cause as host contention unless a later reproducible browser defect contradicts it.
3. Record current smoke scenarios and CI browser installation/command. Do not add retry, sleep, skip, or timeout changes.
4. Rerun `pnpm test:e2e` once after the audit agents stop to confirm both projects remain green.
5. Run `git diff --check` and commit only the baseline/design/plan documentation.

**Acceptance:** Evidence clearly distinguishes product behavior, test behavior, and environment behavior; exact commands and measurements are reproducible.

## Task 2: Add the pure profile and visibility model

**Files:**

- Create: `apps/web/src/productShell/audienceProfiles.ts`
- Create: `apps/web/src/productShell/audienceProfiles.test.ts`
- Create: `apps/web/src/productShell/visibleShell.ts`
- Create: `apps/web/src/productShell/visibleShell.test.ts`
- Modify: `apps/web/src/store/useLayoutStore.ts`
- Modify: `apps/web/src/store/useLayoutStore.test.ts`

**TDD steps:**

1. Write failing table tests for exact Beginner/Explore/Lab modules, evidence views, defaults, descriptions, guidance density, advanced union, and immutable profile data.
2. Write failing resolver tests for core/open visibility, legacy `history -> boundary`, hidden-target detection, Data/Boundary fallbacks, and identical rendering/demand resolution.
3. Write failing store tests for Explore default, invalid/missing persisted values, backward-compatible hydration, explicit Lab collapse, hidden-target hydration, atomic mode/disclosure transitions, and legacy alias synchronization.
4. Run:

   ```bash
   NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/productShell/audienceProfiles.test.ts src/productShell/visibleShell.test.ts src/store/useLayoutStore.test.ts --pool=forks --reporter=dot
   ```

5. Implement the smallest pure tables/resolvers and additive sanitized store fields/actions. Do not touch experiment or runtime stores.
6. Rerun focused tests and commit.

**Acceptance:** Profiles are configuration, not UI forks; switching changes no non-layout state; persisted state remains backward compatible.

## Task 3: Implement Advanced Tools and unify visualization demand

**Files:**

- Modify: `apps/web/src/components/layout/BuildRunShell.tsx`
- Create: `apps/web/src/components/layout/BuildRunShell.test.tsx`
- Modify: `apps/web/src/components/layout/Header.tsx`
- Modify: `apps/web/src/components/layout/Header.test.tsx`
- Modify: `apps/web/src/components/layout/deriveVisualizationDemand.ts`
- Modify: `apps/web/src/components/layout/deriveVisualizationDemand.test.ts`
- Modify: `apps/web/src/components/layout/ExperimentStateContext.tsx`
- Modify: `apps/web/src/components/layout/ExperimentStateContext.test.tsx`
- Modify: `apps/web/src/components/controls/InspectionPanel.tsx`
- Modify: `apps/web/src/components/controls/InspectionPanel.test.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/App.test.tsx`
- Modify: `apps/web/src/__tests__/appShell.integration.test.tsx`
- Modify: `apps/web/src/explanations/explanationActionFocus.ts`
- Modify: `apps/web/src/explanations/explanationActionFocus.test.ts`
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.tsx`
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`
- Modify: `apps/web/src/styles/forge.css`

**TDD steps:**

1. Add failing component/integration tests for exact profile modules/tabs, disclosure semantics/description, unmounting, open/collapse fallbacks, hydration without focus theft, focus restoration, Escape, and roving Arrow/Home/End navigation.
2. Add failing demand tests proving Advanced-open alone requests no diagnostic artifacts and hidden/legacy targets resolve identically for UI and demand.
3. Add failing navigation tests proving Inspect/Code, explanation actions, and lessons open Advanced Tools atomically without changing profile.
4. Search production consumers of the More drawer, legacy `history`, and direct InspectionPanel rendering. Record the result before deleting a path.
5. Replace More with the persisted disclosure and render only visible modules/evidence. Keep Presets/Lessons/History drawers.
6. Remove `historyDrawerOpen` from demand. Use `resolveVisibleEvidenceView` in BuildRunShell, ExperimentStateContext, and demand derivation.
7. If consumer searches prove InspectionPanel has no production path outside App visibility, remove its demand writer and retain wrapper compatibility tests without duplicate ownership.
8. Add responsive, focus-visible, minimum target-size, and reduced-motion styles.
9. Run focused shell, demand, navigation, and integration tests plus `pnpm lint`; commit.

**Acceptance:** One accessible disclosure owns advanced visibility; no duplicate More path; worker demand follows the same resolved visible state as rendering.

## Task 4: Add the typed terminology catalog

**Files:**

- Create: `apps/web/src/concepts/conceptCatalog.ts`
- Create: `apps/web/src/concepts/conceptCatalog.test.ts`
- Create: `apps/web/src/components/common/ConceptHelp.tsx`
- Create: `apps/web/src/components/common/ConceptHelp.test.tsx`
- Modify: `apps/web/src/styles/forge.css`

**TDD steps:**

1. Write failing tests for exact ID lookup, case-insensitive canonical/alias lookup, missing entries, profile filtering, stable order, unique aliases, valid related IDs, and entry completeness.
2. Write failing UI tests for accessible name, `aria-expanded`/`aria-controls`, click/keyboard toggling, Escape/focus restoration, plain and extended definitions, examples, related concepts, UI links, and no hover-only dependency.
3. Implement the six bounded concepts with accurate beginner-readable definitions and pure lookup/filter functions.
4. Implement `ConceptHelp` without a global provider or duplicated definitions.
5. Run focused tests and commit.

**Acceptance:** One typed source of truth supports all required metadata and lookup behaviors; accessible help works independently of hover.

## Task 5: Integrate profile switching, guidance, and terminology

**Files:**

- Modify: `apps/web/src/components/layout/Header.tsx`
- Modify: `apps/web/src/components/layout/Header.test.tsx`
- Modify: `apps/web/src/components/controls/CurrentRunCard.tsx`
- Modify: `apps/web/src/components/controls/CurrentRunCard.test.tsx`
- Modify: `apps/web/src/components/controls/TrainingControls.tsx`
- Modify: `apps/web/src/components/controls/TrainingControls.test.tsx`
- Modify: `apps/web/src/components/visualization/LossChart.tsx`
- Modify: `apps/web/src/components/visualization/LossChart.test.tsx`
- Modify: `apps/web/src/components/controls/inspection/InspectionPanelView.tsx`
- Modify: `apps/web/src/components/controls/inspection/InspectionPanelView.test.tsx`
- Modify: `apps/web/src/components/layout/ExperimentStateContext.tsx`
- Modify: `apps/web/src/components/layout/ExperimentStateContext.test.tsx`
- Modify: `apps/web/src/components/visualization/DecisionBoundary.tsx`
- Modify: `apps/web/src/components/visualization/DecisionBoundary.test.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/__tests__/appShell.integration.test.tsx`

**TDD steps:**

1. Add failing Header tests for the labeled profile select, practical-difference description, announcement, persisted change, and disclosure default.
2. Add failing copy/help tests proving repeated definitions come from the catalog and all critical labels remain visible without opening help.
3. Add integration tests that snapshot recipe, URL hash, runtime identity/step, checkpoints, saved-run count, and code tab across every profile/disclosure transition.
4. Add tests proving hidden configured values remain in Current Recipe summaries and are not reset or silently ignored.
5. Integrate catalog help at the six representative surfaces, using profile guidance density only to control optional explanatory detail.
6. Run focused and complete web tests; commit.

**Acceptance:** The profile switch is understandable and accessible; all profiles use identical scientific/business logic; user work survives every transition.

## Task 6: Expand cross-browser, accessibility, and responsive coverage

**Files:**

- Modify: `tests/e2e/playground-smoke.spec.ts`
- Create or modify: `apps/web/src/__tests__/productShell.accessibility.test.tsx`
- Modify: `apps/web/src/styles/forge.css`
- Modify: `.github/workflows/ci.yml` only if the normal command does not already execute new coverage

**TDD/verification steps:**

1. Add a cross-browser paused-run scenario that records URL/hash, model identity/step, checkpoint timeline, saved-run count, and selected code tab; cycles all profiles and disclosure states; and proves invariants unchanged.
2. Use roles, accessible names, and observable state. Do not use arbitrary sleeps, retries, browser skips, or presentation-dependent selectors.
3. Add jest-axe coverage for Build/Run in all profiles and disclosure states; resolve every serious/critical finding.
4. Add component tests for 320px responsive behavior, keyboard-only operation, logical focus order, 44px touch targets, and reduced-motion media behavior where reliably testable.
5. Run Chromium and WebKit through the normal `pnpm test:e2e` command and confirm CI invokes that same command.
6. Commit.

**Acceptance:** Existing workflows and the new profile invariant scenario pass in both engines; automated accessibility reports no serious/critical violations.

## Task 7: Documentation and release verification

**Files:**

- Modify: `README.md`
- Modify: `docs/architecture/state-ownership.md`
- Create: `docs/architecture/product-shell.md`
- Create: `docs/qa/2026-07-16-product-shell-release.md`
- Modify: `docs/qa/testing.md` or the repository's nearest testing guide if present

**Steps:**

1. Document profiles, persistence, Advanced Tools rules, catalog extension, compatibility, browser coverage, commands, accessibility behavior, and known limitations.
2. Perform and record manual keyboard, focus, screen-reader semantics, contrast, color-independent information, reduced motion, touch target, narrow viewport, 200% zoom/text resize, error association, and hover-independence checks.
3. Run exact final gates:

   ```bash
   pnpm lint
   pnpm test
   pnpm build
   pnpm test:perf
   pnpm test:perf
   pnpm test:perf
   pnpm test:e2e
   git diff --check
   ```

4. Measure exact zlib gzip chunk, entry, non-worker, worker, and total JavaScript sizes; compare with the baseline and pilot budgets.
5. Inspect build output for duplicate dependencies and retained lazy chunks. Record measured results and any host-contaminated performance samples.
6. Search for temporary debugging code, unexplained skips, disabled tests, placeholder copy, and unresolved implementation TODOs in changed files.
7. Request an independent whole-branch code review, fix every validated issue test-first, and rerun affected plus full gates.
8. Use `superpowers:verification-before-completion`; only then use `superpowers:finishing-a-development-branch` for integration.

**Acceptance:** Every completion criterion in the goal objective has current evidence; documentation and repository state are release-ready; remaining limitations are explicit and non-material.
