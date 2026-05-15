# Design Note: Multiclass URL and Config Outbound Guards

## Problem

Wave 7 multiclass foundations can create hidden, unsupported multiclass config states during internal guard work. Public URL sharing and Config JSON export must not publish `outputSize > 1`, `softmax`, or `categoricalCrossEntropy` until public multiclass URL/config migration is intentionally designed.

## Proposed Change

Guard app outbound config paths only:

- Normalize `syncToUrl()` through existing lenient config normalization before encoding the current hash.
- Validate Config Panel JSON export through existing strict import validation before creating a downloadable blob.
- Keep strict Config JSON import behavior unchanged.

## User Value

Users cannot accidentally share or export a hidden unsupported multiclass experiment that the current public app cannot safely reload.

## Affected Files

- `apps/web/src/store/usePlaygroundStore.ts`
- `apps/web/src/store/usePlaygroundStore.test.ts`
- `apps/web/src/components/controls/ConfigPanel.tsx`
- `apps/web/src/components/controls/ConfigPanel.test.tsx`

## Affected Runtime/Data Flow

Only app-local URL sync and Config Panel export are affected. No worker, engine, frame-buffer, persistence, run-history, or visualization data flow changes are introduced.

## Compatibility Risks

No URL key, JSON key, public config type, persistence schema, or import migration changes are introduced. Unsupported hidden multiclass outbound state falls back to the current scalar public contract for URL sync, and JSON export refuses with the existing strict validation error.

## Accessibility Impact

Config export failures use the existing `role="alert"` feedback surface.

## Performance Impact

URL sync and export perform bounded config normalization/validation over scalar config objects only. No hot-path training, worker, or large-array behavior is affected.

## Test Plan

- Red/green store test for hidden multiclass URL sync.
- Red/green Config Panel test for hidden multiclass export refusal.
- Existing shared strict import and lenient decode tests remain unchanged.
- Full `pnpm test`, `pnpm lint`, `pnpm build`, and `pnpm test:perf` before commit.

## Browser QA Plan

No Browser QA is required for this slice because it changes no visible controls or layout flows. The export error uses existing component feedback tested with Testing Library.

## Rollback Plan

Revert the app-local URL sync/export guard commit and its tests. No schema, persistence, protocol, or generated artifact rollback is required.

## Approval Required

Covered by the user's broad Wave 7 approval. This slice does not change URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, dependencies, or engine math.

## Decision

Proceed with app-boundary guards and defer any shared-helper serialization behavior changes until public multiclass URL/config migration is explicitly designed.
