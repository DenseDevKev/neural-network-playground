# Browser QA: Wave 4 Activation Histogram Explorer

## Date

2026-05-11

## Commit

Implementation commit: `590114d`.

## Environment

- OS: macOS local workspace
- Browser: Codex in-app Browser plugin
- Viewport: compact dock viewport in the in-app browser
- Local URL: `http://127.0.0.1:5176/`

## QA Mode

Mode B: Agent-assisted Browser QA.

## Scenario

Verify that the Inspection panel requests and renders bounded activation histogram bins without raw activation arrays in React state.

## Steps

1. Start a fresh local Vite dev server on `http://127.0.0.1:5176/`.
2. Open the app in the Codex in-app Browser.
3. Select the `Inspection` output tab in compact dock mode.
4. Click `Run one training step` six times.
5. Confirm the Activation Histogram explorer renders a native `Histogram layer` selector.
6. Confirm the histogram chart exposes a text alternative with near-zero and activation-limit percentages.
7. Change the selected histogram layer to `Output`.
8. Reload after tightening the histogram demand path and repeat the Inspection/step flow.
9. Check current-URL console warnings and errors.
10. Capture a screenshot.

## Expected Results

- The app loads without a blank page.
- The Inspection panel appears in compact dock mode.
- The histogram explorer remains empty until training snapshots provide histogram bins.
- After stepping, the histogram explorer renders bounded layer-level bins.
- The chart has an accessible text summary.
- The native layer selector can change the active layer.
- No console errors are reported for the fresh local URL.

## Actual Results

- App title loaded as `Neural Network Playground 2.0`.
- The in-app browser rendered compact dock mode, with focus/grid/split layout buttons disabled by viewport width.
- After six manual steps, the histogram explorer rendered:
  - `Histogram layer` native combobox.
  - Hidden-layer histogram summary with `31.4% near zero`, `0.0% near activation limits`, and range `-0.555 to 0.517`.
  - Output-layer histogram summary with `0.0% near zero`, `0.0% near activation limits`, and range `0.460 to 0.536`.
- The histogram chart exposed an `img` text alternative, for example `Hidden 1 activations: 31.4% near zero, 0.0% near activation limits. Activations are spread across the sampled range.`
- The `Output` layer was selected through the native combobox.

## Console Errors

No console errors were reported for `http://127.0.0.1:5176/`.

Development performance warnings were reported for the fresh URL:

- `[perf] Slow interaction: Promise Resolved (237.38ms)`
- `[perf] Slow interaction: Promise Resolved (190.84ms)`
- `[perf] Slow interaction: Promise Resolved (59.94ms)`
- `[perf] Slow interaction: Promise Resolved (81.55ms)`

Older logs from prior HMR attempts on ports `5173`, `5174`, and `5175` included dev perf warnings and one React HMR dependency-array warning. Those did not recur on the fresh `5176` load.

## Screenshots / Recordings

- Histogram explorer: `/private/tmp/nn-playground-wave4-activation-histogram.png`

## Accessibility Notes

- The explorer uses a native `<select>` labelled `Histogram layer`.
- The histogram bars are non-interactive and hidden from assistive technology.
- The chart container uses `role="img"` with a layer-specific summary.
- The visible summary includes near-zero percentage, activation-limit percentage, and activation range.

## Result

Pass for compact Browser QA.
