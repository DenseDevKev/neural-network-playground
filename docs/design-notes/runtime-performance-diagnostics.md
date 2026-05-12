# Design Note: Runtime Performance Diagnostics

## Problem

Wave 5 hardens the existing worker/runtime architecture with regression tests, but the project still relies on broad benchmark commands and manual Browser QA notes for performance evidence.

Future runtime or visualization work would benefit from repeatable diagnostics for worker cadence, frame-buffer freshness, SharedArrayBuffer fallback, WebGPU fallback, and expensive visualization updates. Those diagnostics must not destabilize training, alter hot paths, or expand the worker protocol casually.

## Proposed Change

Do not add runtime telemetry in this Wave 5 slice.

For now, keep diagnostics to existing verification surfaces:

- `pnpm test:perf` for engine benchmark smoke checks.
- `pnpm build` output for bundle-size comparisons.
- Runtime/worker regression tests for demand cadence, frame-buffer counters, SharedArrayBuffer fallback, WebGPU fallback, invalid demand handling, and stream lifecycle behavior.
- Browser QA evidence only for browser-visible UI slices.

Future diagnostics should start as test-only or dev-only helpers, not production telemetry.

## User Value

- Keeps the teaching app responsive and stable while runtime internals evolve.
- Gives maintainers a clear performance evidence trail before adding heavier Wave 6 or Wave 7 features.
- Avoids distracting learners with internal instrumentation UI.

## Affected Files

Current Wave 5 decision:

- `docs/design-notes/runtime-performance-diagnostics.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

Future implementation, if approved, may touch:

- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/worker/frameBuffer.ts`
- focused test files under `apps/web/src/worker/`
- benchmark files under `packages/engine/src/__benchmarks__/`

## Affected Runtime/Data Flow

No runtime data flow changes in this slice.

Any future diagnostic implementation must preserve:

- existing worker protocol fields,
- frame-buffer version-counter semantics,
- SharedArrayBuffer and postMessage fallback behavior,
- WebGPU CPU fallback behavior,
- deterministic engine behavior,
- large-array isolation outside React state.

## Compatibility Risks

Current docs-only decision has no compatibility risk.

Future risks if diagnostics are implemented:

- Hot-path overhead from timing or message payload expansion.
- False confidence from noisy local benchmark values.
- Accidental protocol or state shape changes.
- Confusing dev-only measurements with production telemetry.

## Accessibility Impact

No user-facing UI change in this slice.

Future diagnostics HUDs or panels must use semantic controls and text alternatives and must not rely on color alone.

## Performance Impact

No runtime impact in this slice.

Wave 5 performance evidence remains command-based. The current benchmark suite is useful for catching obvious regressions, but local runs can be noisy; repeat runs should be recorded when a threshold appears exceeded and the changed files do not plausibly affect the measured path.

## Test Plan

Current slice:

- Docs/status review.
- Final Wave 5 verification with `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf`.

Future diagnostics implementation:

- Unit tests for any test-only helper.
- Worker bridge tests for lifecycle/cadence behavior.
- Performance comparison against `docs/perf/PERFORMANCE_BASELINE.md`.

## Browser QA Plan

No Browser QA is required for this docs-only slice.

Future browser-visible diagnostic UI must record Browser QA evidence under `docs/qa/browser-qa/`.

## Rollback Plan

Remove this design note and any related docs references.

If future diagnostics are implemented, rollback must remove instrumentation and tests in the same slice without changing worker protocol or persisted user data.

## Approval Required

No approval is required for this docs-only design note.

Explicit approval would be required before:

- worker protocol changes,
- frame-buffer redesign,
- persistent telemetry,
- user-visible diagnostics HUD,
- hot-path runtime instrumentation,
- new dependencies.

## Decision

For Wave 5, keep diagnostics evidence command-based and test-based. Do not add runtime diagnostics code until a later approved slice proves the value, scope, overhead, and rollback plan.
