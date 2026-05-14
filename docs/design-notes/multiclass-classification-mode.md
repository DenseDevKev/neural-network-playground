# Design Note: Multiclass Classification Mode

## Problem

Neural Network Playground currently teaches binary classification and regression. Several Wave 7 features now expose deeper inspection, comparison, checkpoints, backprop summaries, and local loss slices, but the product still cannot show a model separating more than two classes.

Multiclass classification would be a major educational step: learners could compare one-vs-rest intuition, class competition, decision regions, class imbalance, and confusion patterns beyond a 2x2 matrix. It is also a high-risk product bet because the current app is intentionally shaped around one output for public config, URL/import-export, worker targets, decision-boundary colors, confusion matrix layout, and many tests.

## Proposed Change

Prepare Multiclass Classification Mode as a staged Wave 7 initiative. This design note is an approval package only and does not authorize implementation.

Recommended direction:

1. Start with a bounded 3-class classification mode.
2. Prefer a softmax output with categorical cross-entropy for the final design unless a first implementation note proves one-vs-rest sigmoid is safer.
3. Add a new multiclass dataset only after schema/config migration is approved.
4. Keep binary classification and regression workflows unchanged.
5. Add migration and compatibility tests before any public config or URL/import-export changes.
6. Update visualization surfaces only after engine, shared config, and worker target encoding are proven.

Implementation must be split into small approval-gated slices. The first implementation slice should be design/test-only or engine-only, not a full product rollout.

## User Value

- Shows learners that classification can involve more than a single threshold.
- Makes class competition visible through probability distributions and multi-region boundaries.
- Teaches why binary confusion matrices do not scale to richer classification problems.
- Enables future lessons on class imbalance, calibration, separability, and representation learning.
- Gives teachers a more advanced playground without abandoning the simple binary/regression paths.

## Affected Files

Design-only package:

- `docs/design-notes/multiclass-classification-mode.md`
- `docs/roadmap/WAVE_7_PROPOSAL.md`
- `docs/roadmap/ROADMAP_STATE.md`

Likely future implementation files after separate approval:

- `packages/engine/src/types.ts`
- `packages/engine/src/datasets.ts`
- `packages/engine/src/losses.ts`
- `packages/engine/src/activations.ts`
- `packages/engine/src/network.ts`
- `packages/engine/src/__tests__/*`
- `packages/shared/src/constants.ts`
- `packages/shared/src/serialization.ts`
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/presets.ts`
- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/codeExport.ts`
- `packages/shared/src/colorScale.ts`
- `apps/web/src/worker/*`
- `apps/web/src/store/*`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/components/controls/DataPanel.tsx`
- `apps/web/src/components/controls/HyperparamPanel.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/styles/index.css`
- `docs/worker-protocol.md`
- `docs/qa/browser-qa/*`
- `docs/perf/PERFORMANCE_BASELINE.md`

## Affected Runtime/Data Flow

Current binary flow:

1. Dataset points carry one numeric `label`.
2. Worker converts labels into one-element targets such as `[point.label]`.
3. Network output size is normalized to `1` in URL/import-export handling.
4. Classification decisions threshold one scalar output at `0.5`.
5. Decision boundary colors and legends assume two classes.
6. Confusion matrix renders a fixed 2x2 summary.
7. Run history, code export, presets, and shared guards assume the current config shape.

Future multiclass flow would need:

1. Dataset metadata with a bounded class count, initially `3`.
2. Target encoding from class index to one-hot target vector.
3. Network output size matching the class count.
4. Output activation/loss compatibility for multiclass, likely softmax plus categorical cross-entropy.
5. Worker and frame-buffer paths that keep large grid arrays outside React state.
6. Decision-boundary rendering that maps class index and confidence to accessible visual and text summaries.
7. Confusion matrix generalized to NxN, with readable labels and keyboard/screen-reader-friendly summaries.
8. Backward-compatible config migration for old binary/regression URLs and imports.

## Compatibility Risks

This feature crosses mandatory approval gates:

- Public config shape.
- URL/config serialization.
- Import/export validation.
- Presets.
- Run-history compatibility.
- Worker protocol and snapshot validation if class-specific payloads are streamed.
- Frame-buffer layouts if decision-boundary or class-probability grids change shape.
- Engine output/loss semantics.
- Code export semantics.

Known current single-output assumptions:

- URL/import normalization forces `network.outputSize` to `1`.
- Strict imported-config validation rejects other output sizes.
- Worker target creation currently wraps scalar labels as one-element arrays.
- Decision boundary thresholds one scalar at `0.5`.
- Confusion matrix is fixed to binary true/false positive/negative fields.

Any implementation must include migration tests before changing these contracts.

## Accessibility Impact

Multiclass UI must not rely on color alone.

Requirements:

- Class labels must be visible text, not only colors.
- Decision-boundary summaries must state dominant class, confidence, and ambiguity where useful.
- Confusion matrix must expose row/column headers and per-cell labels.
- Class legends must be keyboard reachable and screen-reader readable.
- Probability summaries should use text values or grouped bars with names.
- Compact layouts must preserve readable legends and avoid clipping matrix labels.
- Reduced-motion concerns are low unless animated transitions are added; any animation must respect reduced motion.

## Performance Impact

Expected costs:

- Grid prediction output size grows with class count.
- Decision-boundary visualization may need class index plus confidence instead of one scalar.
- Confusion matrix and metrics need NxN computation.
- Worker message and frame-buffer payloads can grow if class-probability grids are transported.
- UI rendering can become heavier if every grid cell carries multiple probabilities.

Mitigation:

- Start with exactly 3 classes.
- Render class index and confidence first; defer full per-class probability grids.
- Keep large arrays in worker/frame-buffer domains.
- Demand-gate multiclass-heavy visualizations.
- Compare build size, fixed grid prediction, worker cadence, and memory proxy against `docs/perf/PERFORMANCE_BASELINE.md`.

Warning thresholds:

- Production build size increase greater than 10%.
- Fixed training scenario runtime increase greater than 20%.
- Decision-boundary update time increase greater than 20%.
- Worker message cadence degradation greater than 20%.
- Memory increase greater than 25% for the same scenario.

## Test Plan

Design-only package:

- `git diff --check`

Future approved implementation must use TDD and include:

- Engine tests for softmax/categorical cross-entropy or the approved one-vs-rest alternative.
- Engine tests for multi-output training determinism and shape validation.
- Dataset determinism tests for any new multiclass generator.
- Shared serialization tests proving old binary/regression URLs decode unchanged.
- Shared import/export tests for valid and invalid multiclass configs.
- Preset tests for any new multiclass preset.
- Worker tests for one-hot target encoding, metrics, snapshots, and fallback paths.
- Frame-buffer/version-counter tests if new grid payloads are added.
- Component tests for dataset controls, boundary legend, confusion matrix, code export, and run-history summaries.
- Accessibility tests for legends, matrix labels, and non-color summaries.

Required verification for code slices:

- Targeted affected tests.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

## Browser QA Plan

Browser QA is required for any UI/runtime slice:

1. App loads with no console errors.
2. Existing binary preset still trains, pauses, steps, resets, and exports config.
3. Existing regression preset still trains, pauses, steps, resets, and exports config.
4. Multiclass preset loads with the expected class count.
5. Training starts, pauses, steps, resets, and renders metrics.
6. Decision boundary shows class regions with text-accessible legend and confidence/ambiguity summary.
7. Confusion matrix renders an NxN summary with readable row/column labels.
8. Import/export and URL sharing round-trip a multiclass config.
9. Compact viewport, about `390x844`, has no overlap in legends or matrix labels.
10. Screenshots and console status are recorded under `docs/qa/browser-qa/`.

## Rollback Plan

Keep implementation slices separable:

1. Engine output/loss support.
2. Dataset and preset support.
3. Shared config/serialization migration.
4. Worker target/metric/protocol support.
5. Frame-buffer and visualization support.
6. UI controls, confusion matrix, and QA evidence.

Rollback only the current slice files. If a slice touches config or protocol contracts, rollback must also restore docs and tests, then run targeted checks plus full verification. If rollback cannot separate current work from pre-existing user work, stop and write a block report.

## Approval Required

Yes. Multiclass Mode is a Wave 7 feature and would require mandatory approval before implementation because it likely touches public config shape, URL/config serialization, import/export validation, worker protocol or snapshot guards, frame-buffer payloads, run-history compatibility, engine output/loss behavior, code export, and product direction.

This note approves no implementation.

## Decision

Design-note-only approval package prepared on 2026-05-14 after two approval-gate review passes. Both reviewers recommended Multiclass Mode as the next fresh Wave 7 target and required stopping before implementation.

Recommended first implementation request after this note is reviewed:

- Engine-only multiclass loss/output experiment behind tests, or shared serialization migration tests only.
- Do not combine engine, shared config migration, worker transport, visualization, and UI in one slice.

## Exact Approval Question

Do you approve creating only the first implementation slice for Multiclass Classification Mode, after this design note is reviewed, with the exact slice to be limited to either engine-only multiclass output/loss tests or shared serialization migration tests, and still forbidding worker protocol, frame-buffer, persistence/run-history schema, URL/config format changes beyond the approved migration tests, public config rollout, dependencies, deployment changes, and UI implementation until separately approved?
