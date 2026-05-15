# Design Note: Multiclass Persistence Eligibility

## Problem

The shared URL/import layer now has an explicit opt-in contract for exactly one bounded 3-class multiclass config: classification data, `outputSize: 3`, `softmax`, and `categoricalCrossEntropy`.

Experiment memory still deliberately rejects multiclass run records through the default strict validation path. Before public controls can expose multiclass, the shared persistence boundary needs a narrowly tested eligibility path that proves valid 3-class run records can be normalized without changing the v1 schema or accepting unsupported class counts.

## Proposed Change

Add an opt-in validation option for experiment-memory normalization that allows the same approved 3-class config already defined by shared URL/import validation.

Decision: experiment-memory v1 can preserve the exact approved 3-class config without a schema version bump because the existing record shape already stores `AppConfig`, scalar summary metrics, optional serialized network payloads, and bounded history. Any future NxN confusion matrix or richer multiclass metric payload remains out of scope and would require a separate contract decision.

The default behavior stays scalar-only:

- `validateExperimentRunRecord(record)` continues rejecting multiclass records.
- `createExperimentMemoryEnvelope(records)` continues dropping multiclass records.
- `normalizeExperimentMemoryEnvelope(envelope)` continues dropping multiclass records.

The new opt-in behavior should:

- Accept an exact 3-class config only when callers pass the explicit multiclass option.
- Validate serialized-network weights and biases against the 3-output architecture.
- Preserve `network: null` records.
- Reject unsupported output sizes, softmax/categorical-loss mismatches, regression multiclass configs, malformed summaries, and invalid serialized-network payloads.

No app call site should opt into this behavior in this slice.

Public localStorage save/load, run capture, restore, report export, and live arena initialization should continue using the default scalar-only validation path until public multiclass UI/runtime/visualization support is deliberately wired in a later slice.

## User Value

This gives the roadmap a safe persistence contract before public multiclass controls are exposed. Learners eventually get save/history support for multiclass experiments, while current binary/regression saved-run behavior remains unchanged.

## Affected Files

- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/__tests__/experimentMemory.test.ts`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

Current default flow:

1. Shared experiment-memory validation normalizes a saved run with strict scalar config validation.
2. Hidden multiclass records are rejected before persistence normalization preserves them.
3. Serialized-network payload validation validates embedded network config through the same scalar strict path.

Proposed opt-in flow:

1. A future caller explicitly passes an experiment-memory option enabling the approved 3-class config.
2. Record config validation delegates to `validateImportedConfig(..., { allowMulticlass: true })`.
3. Serialized-network embedded config validation uses the same option so weights/biases are checked against `[input, ...hidden, 3]`.
4. Envelope normalization can accept valid opted-in records without schema changes.

This slice must not wire the option into web run-history capture, import/export UI, URL state, presets, stores, worker snapshots, or deployment behavior.

## Compatibility Risks

- Accidentally changing the default validator would let hidden multiclass records enter current run history before the public app can restore or visualize them.
- Accepting partial multiclass configs would create records that cannot round-trip through the approved URL/import contract.
- Treating confusion matrices as multiclass-ready would overstate the current v1 schema. Multiclass records may omit confusion matrices; the existing binary `{ tp, tn, fp, fn }` shape remains the only accepted confusion payload in this slice.
- Validating embedded serialized-network config with different options from the record config could preserve mismatched payloads.
- A loose option on envelope creation could make current app call sites opt in unintentionally.

Mitigation:

- Keep the option defaulted to scalar-only.
- Test default rejection and opt-in acceptance side by side.
- Reuse the existing approved shared import validator.
- Keep the v1 envelope and record shapes unchanged.

## Accessibility Impact

None in this slice. There are no visible controls or browser-visible surfaces.

## Performance Impact

Expected runtime impact is limited to shared validation tests and future opt-in persistence validation. The production app does not call the opt-in path in this slice.

`pnpm test:perf` and production build-size evidence should still be recorded because the roadmap requires performance evidence for high-risk persistence work.

## Test Plan

- Add failing shared tests first:
  - Default record validation still rejects approved multiclass records.
  - Opt-in record validation accepts the exact approved 3-class config with `network: null`.
  - Opt-in record validation accepts a matching 3-output serialized-network payload unchanged.
  - Opt-in envelope creation/normalization preserves valid multiclass records only when explicitly opted in.
  - Opt-in validation rejects unsupported class counts, softmax/categorical-loss mismatches, regression multiclass configs, and malformed 3-output network payloads.
- Run targeted shared tests:
  - `pnpm --filter @nn-playground/shared exec vitest run src/__tests__/experimentMemory.test.ts src/__tests__/serialization.test.ts`
- Run full verification before code commit:
  - `pnpm test`
  - `pnpm lint`
  - `pnpm build`
  - `pnpm test:perf`
  - `git diff --check`

## Browser QA Plan

Browser QA is not required for this slice because it changes shared opt-in validation only and adds no visible UI or browser interaction path.

## Rollback Plan

Revert the persistence eligibility commit and its roadmap/performance evidence. The prior scalar-only persistence guards remain intact because default validation behavior is explicitly tested.

## Approval Required

The user has broadly approved continuing. This design note is still required because the slice touches the protected persistence boundary. No additional approval is required as long as implementation remains opt-in, schema-neutral, and does not change web call sites or public behavior.

## Decision

Proceed with a small TDD implementation of opt-in experiment-memory eligibility for the approved 3-class config only.
