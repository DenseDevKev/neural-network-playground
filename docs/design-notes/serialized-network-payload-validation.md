# Design Note: Serialized Network Payload Validation

## Problem

Experiment memory now validates embedded `SerializedNetwork.config` compatibility before preserving a run record, but the serialized model payload still only checks that `weights` and `biases` are arrays before casting.

Malformed numeric contents or mismatched layer dimensions can therefore survive shared persistence validation and fail later when a saved model is inspected, compared, or restored.

## Proposed Change

Add shared validation for serialized network payload shape and numeric contents:

- Validate `SerializedNetwork.config` through the existing record-compatible `validateImportedConfig()` composition, then derive expected layer sizes from the validated embedded config.
- Require raw `SerializedNetwork.config.inputSize` to match the validated embedded input size so a self-consistent but record-incompatible payload cannot survive.
- Validate `weights.length === layerSizes.length - 1`.
- Validate each weight matrix has `nextLayerSize` rows and `previousLayerSize` finite numeric values per row.
- Validate `biases.length === layerSizes.length - 1`.
- Validate each bias vector has `nextLayerSize` finite numeric values.
- Keep accepted payloads cloned unchanged.
- Reject invalid records during experiment-memory normalization without changing the schema version.

## User Value

Saved-run comparisons and future restore workflows become more trustworthy because corrupt model payloads are filtered at the persistence boundary instead of failing deeper in engine deserialization.

## Affected Files

- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/__tests__/experimentMemory.test.ts`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

Only shared experiment-memory validation changes. The live training loop, worker protocol, frame buffer, URL/config serialization, public config shape, and run-history schema remain unchanged.

## Compatibility Risks

This is a stricter validator. Existing valid scalar records continue to load. Records with malformed serialized model payloads will be dropped during normalization instead of preserving a record that cannot safely deserialize later.

## Accessibility Impact

None. This is shared validation only.

## Performance Impact

Low. Validation is bounded by `EXPERIMENT_MEMORY_MAX_RECORDS` and the small saved model payloads currently captured by the playground. It does not run in training hot paths.

## Test Plan

- Add failing shared tests for weight layer-count mismatch, bias layer-count mismatch, non-array matrix/vector entries, non-finite weight values, non-finite bias values, wrong matrix row count, wrong row width, wrong bias width, and raw/validated input-size mismatch.
- Add compatibility assertions that valid scalar serialized networks, valid multi-hidden-layer scalar payloads, and `network: null` remain accepted.
- Run `pnpm --filter @nn-playground/shared exec vitest run src/__tests__/experimentMemory.test.ts`.
- Run full `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and `git diff --check` before commit.

## Browser QA Plan

Browser QA is not required because this slice changes shared validation only and adds no visible UI or interaction path.

## Rollback Plan

Revert the shared validation/test commit. No schema migration or data cleanup is required.

If malformed records are normalized and storage is later rewritten, rollback will not resurrect those dropped malformed records. That is acceptable because the slice targets payloads that cannot safely deserialize.

## Approval Required

The user has provided broad approval to continue, but implementation should remain limited to validation of serialized network payload shape and finite numeric values. Stop if a schema migration, payload rewrite, engine deserialization change, or public restore behavior change becomes necessary.

## Decision

Proceed only as a small TDD validation-hardening slice after review.
