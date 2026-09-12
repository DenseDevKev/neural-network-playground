# Task 8C — Critical browser acceptance

Scope: `tests/e2e/atelier-critical-flows.spec.ts` only, plus this report. Uses parent-owned `atelier-helpers.ts`, the actual lesson registry, the existing local Vite server, real worker execution and local browser storage. No production files changed, no scientific output mocks, no additional worker fault APIs, no build or push.

## Coverage

- Initial system light/dark after media-configured load, live media change, explicit override, persisted reload, and denied theme Storage reads/writes. Theme actions preserve generation, revision, step and URL.
- One raw setup draft across all three sections, empty numeric input preservation, combined acknowledgement at generation +1/step 0, cancel, invalid batch/population rejection without silently clamping, and dirty-leave Stay/Discard/Apply.
- XOR, three-class and regression: engine-backed train/test traces, full output tuple (three softmax values with finite/range/sum checks), source-change evidence clearing, sampled activation provenance, backprop preview, parameter-grid probe and model identity preservation. Running diagnostics expose Pause and disable one-off preview until paused. Regression exposes its task-specific loss selector.
- JSON setup download, malformed-file model retention, staged canonical-file review with no automatic apply, explicit apply, exact JSON roundtrip, code download, selectable setup-link/code fallback on clipboard denial, and learned parameter snapshot invalidation after a different recipe.
- All ten registry lessons: real generation acknowledgement for Start and Restart, target setup/results navigation, actual training steps, Previous/Continue/Show me/Finish/Restart/Exit. The noise, learning-rate and regularization journeys change actual controls and apply/retrain where instructed. Both declared completion rules produce Done. The remaining steps have no registry completion predicates; tests exercise their observable actions and finish the journey without inventing accuracy thresholds.

## Validation

- Full Chromium scoped file: 20/20 passed in 29.2s.
- Full WebKit scoped file: 20/20 passed in 46.8s.
- Self-review strengthened Start/Restart against observing a pre-reset step-0 model; both engines' changed lesson cases passed: 20/20 in 47.5s (10 lessons per engine).
- `pnpm exec eslint tests/e2e/atelier-critical-flows.spec.ts` passed.
- `pnpm exec tsc --noEmit --strict --module esnext --moduleResolution bundler --target ES2022 --skipLibCheck --allowImportingTsExtensions --typeRoots apps/web/node_modules/@types tests/e2e/atelier-critical-flows.spec.ts` passed.

Commands use `PLAYWRIGHT_BASE_URL=http://127.0.0.1:5173/`, `--workers=1 --reporter=line`, and dedicated `/tmp/nn-atelier-critical*` output/log paths. macOS sandbox initially prevented browser launch; authorized unsandboxed local browser runs succeeded. Intermediate failures were test locator/sampling setup issues, not confirmed production defects.

## Boundaries

This is deterministic automated browser acceptance, not human teaching effectiveness, model convergence, deployment, performance or visual-design signoff. Scientific race-rejection coverage remains with the existing unit suites. No confirmed production bug required coordination or a source fix.
