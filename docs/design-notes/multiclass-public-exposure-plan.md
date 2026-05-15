# Design Note: Public Multiclass Exposure Plan

## Status Update

This plan is historical background as of 2026-05-15. The shared config
migration contract, opt-in persistence eligibility, hidden dataset contract,
bounded multiclass boundary transport, read-only boundary renderer, and
UI-derived confusion readout have since landed.

Use `docs/design-notes/multiclass-public-ui-controls-browser-qa.md` for the
next active gate before visible public multiclass controls.

## Problem

Wave 7 now has private multiclass foundations: engine softmax and categorical cross-entropy helpers, a private `Network` path, direct-call worker target guards, a hidden three-class dataset helper, code-export truthfulness, visualization non-exposure guards, public outbound config guards, and stricter experiment-memory validation.

The app still intentionally does not expose multiclass workflows. Public exposure cannot be a single UI toggle because it crosses URL/config format, public config compatibility, dataset and preset registries, worker/runtime snapshots, visualization payloads, run-history compatibility, code export, accessibility, Browser QA, and performance evidence.

This note supersedes the "first implementation request" guidance in `docs/design-notes/multiclass-classification-mode.md`. That older note was correct when no implementation had landed, but the repository now already contains engine, shared contract, worker, hidden dataset, code-export, visualization-guard, URL/export-guard, runtime-guard, and persistence-guard slices. This note is now historical background; future public-control work should use `docs/design-notes/multiclass-public-ui-controls-browser-qa.md` as the active gate.

## Proposed Change

Stage public Multiclass Classification Mode as a controlled 3-class rollout. The public mode should be:

- Dataset: one bounded 3-class cluster dataset.
- Problem type: `classification`.
- Network output: exactly `outputSize: 3`.
- Output activation: `softmax`.
- Loss: `categoricalCrossEntropy`.
- Visualization payloads: class index plus confidence first, not full per-class probability grids.
- UI exposure: a deliberate dataset/preset entry and compatible controls only after worker, serialization, persistence, and visualization behavior are tested.

Do not make public UI controls available until the app can train, visualize, import/export, save history, and describe the mode without falling back to binary copy or scalar assumptions.

## User Value

Learners can explore how a neural network separates more than two classes, why softmax outputs compete, how confidence differs from class identity, and why a 2x2 confusion matrix does not generalize.

The staged plan protects existing binary classification and regression workflows while turning the hidden multiclass foundations into a visible educational feature.

## Affected Files

Likely implementation areas, split across separate commits:

- `packages/engine/src/types.ts`
- `packages/engine/src/datasets.ts`
- `packages/engine/src/__tests__/datasets.test.ts`
- `packages/shared/src/constants.ts`
- `packages/shared/src/serialization.ts`
- `packages/shared/src/__tests__/serialization.test.ts`
- `packages/shared/src/presets.ts`
- `packages/shared/src/__tests__/presets.test.ts`
- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/__tests__/experimentMemory.test.ts`
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/__tests__/workerProtocol.test.ts`
- `packages/shared/src/codeExport.ts`
- `packages/shared/src/__tests__/codeExport.test.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/training.worker.test.ts`
- `apps/web/src/worker/frameBuffer.ts`
- `apps/web/src/worker/workerBridge.ts`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/store/usePlaygroundStore.ts`
- `apps/web/src/components/controls/DataPanel.tsx`
- `apps/web/src/components/controls/HyperparamPanel.tsx`
- `apps/web/src/components/controls/ConfigPanel.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/data/datasetInsights.ts`
- `docs/worker-protocol.md`
- `docs/qa/browser-qa/*`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

Existing scalar public flow:

1. Dataset labels are scalar `0` or `1` for classification.
2. Public config normalization forces `network.outputSize` to `1`.
3. URL serialization does not encode output size.
4. Strict import validation rejects `softmax`, `categoricalCrossEntropy`, and `outputSize !== 1`.
5. Worker public hook validation rejects hidden multiclass state before touching the worker.
6. Binary decision-boundary and confusion-matrix UI are guarded against hidden multiclass state.

Target public multiclass flow:

1. The selected dataset declares a bounded class count of `3`.
2. Public config validation accepts only the approved 3-class pairing: classification data, `outputSize: 3`, `softmax`, and `categoricalCrossEntropy`.
3. URL serialization round-trips an output-size key while old URLs without the key still decode as scalar `outputSize: 1`.
4. Worker target encoding produces one-hot targets for valid class labels.
5. Worker snapshots expose bounded multiclass visualization summaries without streaming raw per-class probability grids.
6. Frame-buffer storage keeps heavy grid arrays out of React state.
7. Decision boundary renders class regions plus confidence and accessible text summaries.
8. Confusion matrix renders an NxN matrix with labels, totals, and screen-reader text.
9. Run-history capture and experiment-memory validation preserve valid multiclass runs without schema changes unless a separate migration note proves one is needed.

## Compatibility Risks

High-risk surfaces:

- URL/config serialization format.
- Strict and lenient import validation.
- Public config shape and store normalization.
- Dataset and preset registries.
- Worker runtime compatibility and target encoding.
- Worker protocol/runtime guards if new snapshot fields are streamed.
- Frame-buffer version counters if multiclass grids are stored.
- Run-history validation and restore behavior.
- Visualization assumptions for binary legends and 2x2 confusion matrices.

Compatibility rules:

- Old binary/regression URLs must decode unchanged.
- Old JSON imports must remain valid.
- Existing scalar run-history records must normalize unchanged.
- New multiclass records must be bounded to class count `3`.
- Unsupported class counts must fail loudly in strict paths and fall back safely in lenient paths.

## Accessibility Impact

Public multiclass must not rely on color alone.

Requirements:

- Class names must be rendered as visible text.
- Decision-boundary summaries must report dominant class and confidence.
- Confusion matrix cells must include row and column labels.
- Keyboard users must be able to reach dataset, preset, and inspection controls.
- Compact viewport legends and matrix labels must not overlap.
- Any animated visual transitions must honor reduced motion.

## Performance Impact

Expected costs:

- Worker output vectors grow from one scalar to three logits/probabilities.
- Decision-boundary grids may need class-index and confidence arrays.
- Confusion matrices grow from 2x2 to 3x3.
- UI rendering has more legend and matrix text.

Mitigation:

- Start with class index plus confidence only.
- Keep raw probability grids out of React state.
- Use frame-buffer arrays and version counters for heavy grids.
- Demand-gate multiclass visualization work.
- Compare `pnpm test:perf` and production bundle sizes against the latest baseline.

## Implementation Slices

1. **Public Exposure Design Gate**: this note only. No code behavior changes.
2. **Shared Config Migration Contract Slice**: add TDD coverage, then update strict/lenient validation and URL encode/decode for exactly one bounded 3-class public config shape. This slice must avoid UI controls, presets, worker protocol/frame-buffer fields, and visible runtime behavior.
3. **Persistence Eligibility Slice**: decide whether valid 3-class configs can be accepted by run-history/experiment-memory v1 without schema changes. If not, write a narrower persistence migration design note before code.
4. **Dataset Contract Slice**: promote the existing deterministic three-class generator into an explicit bounded dataset contract, with tests, but keep visible controls hidden until downstream paths are ready.
5. **Worker/Protocol Visualization Slice**: add bounded multiclass snapshot fields for class index and confidence if needed, update runtime guards, frame-buffer version counters, worker tests, and `docs/worker-protocol.md`.
6. **Visualization UI Slice**: render multiclass decision-boundary legend/summary and 3x3 confusion matrix from bounded data, with accessibility tests.
7. **Public Controls and Preset Slice**: expose one dataset/preset/control path after the above can support training, visualization, import/export, and history safely.
8. **Browser QA and Evidence Slice**: run desktop and compact Mode B QA, update performance evidence and `ROADMAP_STATE.md`.

## Historical First Approved Implementation Candidate

Status: completed by the later shared config migration contract slice.

The next code slice should be the **Shared Config Migration Contract Slice** only:

- Add failing shared serialization/import tests for old binary/regression URLs decoding unchanged.
- Add failing shared tests for explicit 3-class URL encoding/decoding.
- Add failing strict import tests for exactly `classification + outputSize: 3 + softmax + categoricalCrossEntropy`.
- Add rejection tests for unsupported class counts, softmax without categorical loss, categorical loss without softmax, and regression multiclass configs.
- Implement the minimum shared validation and URL round-trip support needed for those tests.
- Keep UI controls, presets, worker protocol/frame-buffer changes, visible runtime behavior, and run-history schema changes out of this slice.

This slice is intentionally useful because it defines the public config contract before UI controls or datasets can expose it. It is also the highest-risk contract boundary, so landing it first makes later public work less ambiguous.

## Test Plan

For this design-only slice:

- `git diff --check`

For the first Shared Config Migration Contract Slice:

- Targeted shared serialization/import tests.
- Existing shared experiment-memory tests to reveal any persistence coupling.
- Existing app URL/config guard tests to reveal public outbound behavior changes.
- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`
- `git diff --check`

For later dataset, worker/protocol, or UI slices:

- Engine dataset determinism tests.
- Shared preset and lesson registry tests.
- Worker bridge/runtime guard tests.
- Frame-buffer version-counter tests.
- Browser QA Mode B for any visible UI/runtime path.

## Browser QA Plan

Browser QA is not required for this design-only slice or for hidden test-only contract guards.

Browser QA is required before any visible multiclass control is marked complete:

1. Load the app with no console errors.
2. Load a multiclass preset or dataset.
3. Train, step, pause, and reset.
4. Confirm decision boundary class regions and confidence summary.
5. Confirm 3x3 confusion matrix labels.
6. Confirm URL share and JSON import/export round-trip.
7. Confirm run-history save and report export.
8. Verify desktop and compact viewport.
9. Capture screenshots and console status under `docs/qa/browser-qa/`.

## Rollback Plan

Keep each slice independently revertible. If a slice touches URL/config, protocol, frame-buffer, or persistence behavior, revert that slice and its evidence docs together, then run targeted tests plus full verification.

The design-only slice can be rolled back by deleting this file and removing the roadmap-state entry.

## Approval Required

The user has broadly approved continuing, but each implementation slice must still respect the roadmap gates and design-note discipline. Any URL/config format, worker protocol, frame-buffer, persistence schema, public config, or visible product direction change must be covered by this plan or a narrower follow-up design note before code.

## Decision

Proceed with this design-only slice, then continue to the Shared Config Migration Contract Slice if review finds no blockers.
