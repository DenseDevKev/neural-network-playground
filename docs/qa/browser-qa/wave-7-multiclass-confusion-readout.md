# Browser QA: Wave 7 Multiclass Confusion Readout

## Date

2026-05-15

## Commit

- Code: `c219cfc`, `330f464`
- Verification harness: `624eaa6`

## Environment

- OS: macOS
- Browser: Codex in-app Browser
- Viewport: desktop default, compact `390x760`
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B

## Scenario

Verify that the public scalar Confusion panel still works after adding the UI-only hidden multiclass confusion readout path, and verify compact keyboard tab navigation around the panel. Direct hidden multiclass browser QA remains pending because public controls intentionally do not expose the hidden 3-class configuration.

## Steps

1. Started the Vite dev server with `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5177`.
2. Opened `http://127.0.0.1:5177/` in the Codex in-app Browser.
3. Checked Browser console errors for the loaded app.
4. Started training from the public scalar default flow, then paused training.
5. Opened the `Confusion` visualization tab.
6. Verified the public binary scalar matrix rendered with `TN`, `FP`, `FN`, `TP`, accuracy, precision, and recall metrics.
7. Switched to compact `390x760` viewport.
8. Used keyboard tab controls to move between `Confusion` and adjacent visualization tabs.
9. Re-checked Browser console errors.

## Expected Results

- App loads without a framework error overlay.
- Public scalar training starts and pauses.
- Public scalar Confusion panel still renders binary test-set metrics.
- Keyboard tab movement remains usable in compact viewport.
- No Browser console errors are reported.
- Hidden multiclass readout remains unexposed through public controls.

## Actual Results

- Public scalar flow loaded and the Confusion panel rendered the binary scalar matrix and metrics.
- Compact keyboard tab navigation worked for the visualization tab strip.
- Browser console error checks returned `[]`.
- Direct hidden multiclass browser QA was not executed because no public control path safely exposes hidden 3-class state. Hidden multiclass rendering is covered by component tests.

## Console Errors

None observed in Browser console checks.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-multiclass-confusion-readout-public-scalar-desktop.png`
- `docs/qa/browser-qa/wave-7-multiclass-confusion-readout-public-scalar-compact.png`

## Accessibility Notes

- Compact keyboard navigation reached the relevant visualization tabs.
- The hidden multiclass readout includes per-cell accessible labels, row/column total labels, and a screen-reader-only summary in component coverage.
- A future official worker-authored multiclass matrix should consider table/grid semantics as part of the public UI exposure slice.

## Result

Pass for public scalar Browser QA. Hidden multiclass browser QA is pending until a safe public or test-only exposure path exists.
