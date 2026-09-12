# Task 8B/E review fixes, round 1

Base: 1937d4f. Scoped UI/test-only corrections; no engine, worker, store/controller or scientific model changes.

## Changes

- Trace activation flow and primary comparison table now have named region semantics and native tabindex=0. Existing global focus-visible styling supplies an outline. Both retain full horizontal evidence and native scrolling.
- Native neuron traversal uses Option-Tab only for WebKit on Darwin; all other platforms use Tab. Linux WebKit behavior was not locally executed.
- Import regression starts at step 1, stages a distinct model seed and learning rate plus changed view flags, preserves identity through invalid input and pre-Apply review, requires generation advancement and step reset, and verifies exact exported document equality. Existing code/clipboard assertions remain.
- Grid provenance caption now renders as one interpolated text node, retaining exact text and stable block geometry. Production observer proved the separate period in this caption caused the remaining CLS (see task-8e-production-cls-diagnosis.md).
- Forced-color preservation applies only to boundary legend swatches, class legend marks and SVG sample circles. Labels/controls continue using native forced-color adjustment.

## Verification

- `pnpm --filter @nn-playground/web exec vitest run src/components/controls/inspection/InspectionPanelView.test.tsx src/components/controls/InspectionPanel.test.tsx`: 9 tests, 2 files passed; /tmp/task8be-unit.log.
- `pnpm typecheck`: passed; /tmp/task8be-typecheck.log. Scoped ESLint and git diff --check passed.
- `pnpm build` and `pnpm test:bundle`: passed. Entry gzip150890/152245; InspectionPanel6591/7373; total234137/234161 (24-byte headroom). No budgets changed. Logs /tmp/task8be-build.log, /tmp/task8be-bundle.log.
- `pnpm exec playwright test tests/e2e/accessibility.spec.ts tests/e2e/atelier-critical-flows.spec.ts tests/e2e/precision-lab-layout.spec.ts tests/e2e/playground-smoke.spec.ts --grep 'keyboard|export files roundtrip|layout stay stable at (1440|768)' --workers=1 --output=/tmp/task8be-production2-results --reporter=list`: 16/16 passed in40.7s, both engines. Includes zero-CLS Chromium at1440/768, native keyboard selection, both keyboard scroll targets, changed import, concept-help1280/390/320 (including previously timed-out WebKit390). Log /tmp/task8be-production2.log.
- `pnpm exec playwright test tests/e2e/accessibility.spec.ts --grep 'forced colors' --workers=1 --output=/tmp/task8be-forced3-results --reporter=list`: 2/2 passed in3.7s. Both engines require visible, distinct data colors in active media; property introspection is conditional on CSS.supports. WebKit matches active forced-colors media but does not support forced-color-adjust; an earlier message suggesting inactive media was incorrect. No engine was skipped. Corrected screenshots: outputs/nn-forge-implementation/display-modes/{chromium,webkit}-dark-forced-colors-fixed.png.

## Diagnostic attempts and limits

The first dev test run passed5/6; WebKit focused Trace correctly but instantaneous ArrowRight down/up did not scroll. A realistic150ms held arrow scrolls natively in both engines and both regions; regression asserts nonzero scrollLeft, overflow, actual focus, visible outline and unchanged model step. No custom key handler added. First production external-URL attempt failed because the preceding suite had exited its preview; retried with standard Playwright-managed preview lifecycle. Initial forced-color test incorrectly asserted an unsupported WebKit CSS property; the final check separates CSS support from actual visible data-color behavior without skipping supported media. Parent owns whole-suite final release qualification.

The only preexisting dirty file, task-8e-browser.mjs, remains unstaged and untouched by this fix round.
