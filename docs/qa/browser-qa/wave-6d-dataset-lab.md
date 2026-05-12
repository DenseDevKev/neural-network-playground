# Browser QA: Wave 6D Dataset Parameter Lab

## Date

2026-05-11

## Commit

`d6eeffa`

## Environment

- OS: macOS, local Codex workspace
- Browser: Codex in-app Browser plugin
- Viewport: compact in-app browser viewport
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B.

## Scenario

Verify bounded dataset sample-count presets and accessible dataset settings summary.

## Steps

1. Used the existing local Vite dev server at `http://127.0.0.1:5177/`.
2. Reloaded the app in the Browser plugin.
3. Opened the `Data` tab.
4. Verified the dataset settings summary exposed current samples, noise, and train ratio.
5. Verified sample-count preset buttons were visible.
6. Clicked `600 samples`.
7. Verified the dataset settings summary updated to `600 samples`.
8. Checked keyboard focus movement from the sample presets.
9. Checked current-URL console errors.
10. Captured a compact viewport screenshot.

## Expected Results

- The Data panel shows bounded sample presets only.
- Clicking a preset updates the existing `numSamples` data field through the data config path.
- The settings summary updates without schema, serialization, persistence, worker, or runtime changes.
- No current-URL console errors appear.

## Actual Results

- Browser DOM snapshot showed `Dataset settings: 300 samples, 0 noise, 50% train`.
- Browser DOM snapshot showed `100 samples`, `300 samples`, `600 samples`, and `1000 samples`.
- Clicking `600 samples` updated the settings summary to `Dataset settings: 600 samples, 0 noise, 50% train`.
- Current-URL console errors were empty.

## Console Errors

No errors were reported for `127.0.0.1:5177`.

## Screenshots / Recordings

- `/private/tmp/nn-playground-wave6d-dataset-lab.png`

## Accessibility Notes

- Sample presets are native buttons with `aria-pressed`.
- The dataset settings summary uses `aria-live="polite"`.
- Existing range controls for train/test split and noise remain native sliders.

## Result

Pass.
