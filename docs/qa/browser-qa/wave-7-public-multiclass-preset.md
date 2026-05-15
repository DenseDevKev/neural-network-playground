# Browser QA: Wave 7 Public Multiclass Preset

## Date

2026-05-15

## Commit

Pending

## Environment

- OS: macOS, local Codex desktop session
- Browser: Codex in-app Browser
- Viewport: existing desktop viewport plus `390x800` compact override
- Local URL: `http://127.0.0.1:5181/`

## QA Mode

Mode B.

## Scenario

Verify the first public multiclass entry point: the `Three-Class Softmax Lab`
preset.

## Steps

1. Opened `http://127.0.0.1:5181/`.
2. Switched from Data to Presets using the visible rail button.
3. Verified there is exactly one `Apply preset: Three-Class Softmax Lab`
   button.
4. Activated the preset.
5. Verified the selected preset button is pressed and the status bar shows
   `DATA: three-class-clusters`.
6. Verified the Network Graph text alternative shows
   `X1, X2 -> [6] -> [6] -> 3 outputs (softmax)` and the screen-reader
   description says `3 outputs`.
7. Started training, then paused.
8. Verified the Boundary panel rendered a multiclass text summary naming Class
   0, Class 1, and Class 2, and showed progress around Step 90 / Epoch 6.
9. Clicked `Reset model and data`.
10. Reloaded the app with the generated URL hash and verified the preset button
    remained available.
11. Set a `390x800` compact viewport override, reloaded the approved URL, and
    verified the preset, `3 outputs` topology text, dataset status, Start, and
    Reset remained reachable.
12. Reset the temporary viewport override.
13. Checked console warnings/errors.

## Expected Results

- The preset is visible and keyboard/mouse reachable as a native button.
- Applying it sets the complete approved tuple:
  `three-class-clusters`, `classification`, `outputSize: 3`, `softmax`, and
  `categoricalCrossEntropy`.
- Training starts and pauses without console errors.
- Decision-boundary text alternatives use multiclass language and class labels.
- URL sync includes the approved tuple without exposing arbitrary class counts.

## Actual Results

- The preset button was found and activated.
- The app URL synced to the approved tuple, including `d=three-class-clusters`,
  `os=3`, `oa=softmax`, and `l=categoricalCrossEntropy`.
- The Network Graph summary and screen-reader description reported 3 softmax
  outputs.
- Training reached Step 90 / Epoch 6 before pause.
- The Boundary panel showed a multiclass decision-boundary summary and visible
  Class 0, Class 1, and Class 2 labels.
- Reset remained available and clickable.
- Compact viewport verification at `390x800` found one selected preset button,
  the approved tuple in the URL, `3 outputs` topology text, dataset status,
  Start, and Reset.

## Console Errors

No console errors were observed for the `5181` tab. The Browser log buffer
included development-mode slow-interaction warnings for the `5181` app and older
development-mode perf warnings from a prior `5177` tab.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-public-multiclass-preset.png`
- `docs/qa/browser-qa/wave-7-public-multiclass-preset-compact.png`

## Accessibility Notes

- The preset is a native button with accessible name
  `Apply preset: Three-Class Softmax Lab`.
- The selected preset uses `aria-pressed="true"`.
- The Network Graph `role="img"` description is output-shape aware.
- Boundary text alternatives name the dominant class and list all three class
  labels.
- Compact viewport layout verification passed at `390x800`; the temporary
  viewport override was reset after capture.

## Result

Pass.
