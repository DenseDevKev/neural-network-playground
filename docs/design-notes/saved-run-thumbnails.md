# Design Note: Saved Run Thumbnails

## Problem

Saved runs currently list final step and train/test loss values, but learners cannot quickly scan the shape of a run. Wave 6A added numeric comparison summaries; Wave 6B should add a compact visual preview without changing persistence or storing image data.

## Proposed Change

Render a lightweight inline SVG thumbnail for each saved run from the existing bounded `record.history` points.

The first slice should:

- use only `ExperimentRunRecordV1.history` and `ExperimentRunRecordV1.summary`,
- draw train and test loss sparklines when history points exist,
- show an accessible text summary for the preview,
- fall back to a metrics-only empty state when history is unavailable,
- avoid storing generated images, canvases, or binary thumbnails.

## User Value

- Learners can distinguish smooth improvement, noisy runs, plateaus, and overfitting shapes at a glance.
- The History panel becomes more useful for comparing saved experiments before restoring or exporting them.
- Teachers can discuss training trajectories without opening the full loss chart for every saved run.

## Affected Files

Expected implementation files:

- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.test.tsx`
- `docs/qa/browser-qa/wave-6b-saved-run-thumbnails.md`
- `docs/perf/PERFORMANCE_BASELINE.md`
- `docs/roadmap/ROADMAP_STATE.md`

## Affected Runtime/Data Flow

No runtime data-flow changes.

The thumbnail is generated during render from data that is already present in saved run records. It does not request worker data, inspect frame buffers, store raw arrays in React state, or change capture behavior.

## Compatibility Risks

No persistence schema, URL/config serialization, public config, worker protocol, or run-history schema changes are proposed.

The main compatibility risk is visual clutter in dense History cards. The implementation should keep the preview compact and allow cards to wrap without overlapping controls.

## Accessibility Impact

The SVG preview should use `role="img"` with an accessible label describing the number of points and start/end train/test losses. It should not rely on color alone; visible labels and text summaries must convey the same high-level trend.

## Performance Impact

The preview should derive a bounded path from the already bounded saved history, capped by `EXPERIMENT_MEMORY_MAX_HISTORY`. It should not allocate large arrays or run per-frame work. Work happens only when the History panel renders saved records.

Expected bundle impact is limited to the lazy `RunHistoryPanel` chunk.

## Test Plan

- Component test for a saved run with history rendering an accessible thumbnail.
- Component test for a saved run without history rendering a stable fallback.
- Existing save, restore, export, and comparison tests should continue to pass.
- Full verification before commit: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`.

## Browser QA Plan

Use Browser QA Mode B:

1. Open the app locally.
2. Save at least two runs with training history.
3. Open History.
4. Verify thumbnails appear in saved run cards.
5. Verify comparison summaries remain visible.
6. Check keyboard access for existing History buttons.
7. Check current-URL console errors.
8. Capture a screenshot.

## Rollback Plan

Remove the SVG thumbnail rendering helper and associated tests/docs. Because the slice does not change stored data or schemas, rollback is file-local.

## Approval Required

No additional approval is required for generated, non-persisted SVG previews using existing stored history data.

Approval would be required before storing thumbnails, adding binary/image data to persistence, changing the run-history schema, adding dependencies, or capturing new runtime data.

## Decision

Proceed with non-persisted, generated SVG loss-history thumbnails only. Defer stored thumbnails, image export, and selectable thumbnail styles.
