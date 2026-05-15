# Design Note: Experiment Memory Serialized Network Guard

## Problem

Run-history records already validate their top-level `config` through strict public import validation, which rejects hidden unsupported multiclass state. The embedded serialized model payload, `record.network.config`, is currently shape-checked only enough to ensure it has `config`, `weights`, and `biases`, then cloned.

That means a malformed or future hidden multiclass `SerializedNetwork` could be preserved inside an otherwise scalar-compatible experiment record. Public save paths currently avoid this, but the shared validator should not rely on every caller being perfect.

## Proposed Change

Tighten `validateExperimentRunRecord()` so `network.config` is validated against the same public compatibility boundary as the top-level config before the serialized network is accepted.

The validator should:

- Keep accepting `network: null`.
- Keep accepting existing scalar-compatible serialized networks.
- Validate embedded `network.config` as a compatibility check by combining it with the already-validated record `data`, `training`, `features`, and `ui` fields.
- Reject serialized networks whose embedded config uses unsupported multiclass-only network fields such as `outputSize: 3` or `outputActivation: 'softmax'` when combined with the record's other config fields.
- Keep the serialized network payload unchanged after validation instead of replacing `network.config` with a normalized public config result.
- Leave the experiment-memory schema version unchanged.

## User Value

Saved runs stay safe to reload and compare. Learners do not encounter stale or unsupported model payloads hidden inside run history while multiclass remains private and guard-first.

## Affected Files

- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/__tests__/experimentMemory.test.ts`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

Only shared validation of run-history records changes. The public capture path already rejects top-level unsupported configs before records are created. This slice does not change worker runtime, frame-buffer transport, URL/config serialization, public config shape, or the persisted record schema.

## Compatibility Risks

This is a stricter validator. Existing valid scalar records should continue to load. A record with a scalar top-level config but hidden unsupported embedded model config will be dropped during normalization instead of preserving an unsafe payload.

That is intentional while multiclass remains unexposed, but it should be noted as a validation behavior change.

## Accessibility Impact

None. This slice changes shared validation and tests only.

## Performance Impact

Negligible. Validation runs on bounded run-history records, capped by `EXPERIMENT_MEMORY_MAX_RECORDS`, and does not touch training hot paths.

## Test Plan

- Add a failing shared regression test showing a scalar-compatible record with hidden multiclass `network.config` is rejected.
- Add compatibility assertions that `network: null` and valid scalar serialized networks continue to validate.
- Run `pnpm --filter @nn-playground/shared exec vitest run src/__tests__/experimentMemory.test.ts`.
- Run full `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and `git diff --check` before commit.

## Browser QA Plan

Browser QA is not required because this slice changes shared validation only and adds no visible UI or interaction path. Existing run-history UI tests continue to cover visible saved-run behavior.

## Rollback Plan

Revert the shared validation and regression test commit. No migration or schema cleanup is required because the schema is unchanged.

## Approval Required

This touches a persistence-validation boundary but does not change the run-history schema. The user has provided broad approval to continue; implementation should remain limited to this validator guard and stop if a schema change becomes necessary.

## Decision

Proceed with TDD for the serialized-network validation guard only.
