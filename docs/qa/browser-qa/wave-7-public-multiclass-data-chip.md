# Browser QA: Wave 7 Public Multiclass Data Chip

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

Verify direct Data-panel access to the approved `three-class-clusters`
multiclass tuple.

## Steps

1. Started the local Vite dev server on `http://127.0.0.1:5181/`.
2. Opened the app in the Codex in-app Browser.
3. Selected the Data tab.
4. Verified exactly one `Three-Class` data chip exists.
5. Activated the `Three-Class` chip.
6. Verified the URL includes `d=three-class-clusters`, `os=3`,
   `oa=softmax`, and `l=categoricalCrossEntropy`.
7. Verified the chip reports `aria-pressed="true"`, the Data panel remains
   visible, the Network Graph text reports `3 outputs`, and the app shows
   `DATA: three-class-clusters`.
8. Started training, waited for progress, then paused.
9. Verified the Boundary panel showed multiclass decision-boundary copy and
   Class 0, Class 1, and Class 2 labels.
10. Captured a desktop screenshot.
11. Set a `390x800` compact viewport override, reloaded the approved URL,
    selected the Data tab, and verified the `Three-Class` chip was still visible
    and selected.
12. Captured a compact screenshot.
13. Reset the temporary viewport override.
14. Checked console warnings/errors.

## Expected Results

- The Data panel exposes exactly one direct `Three-Class` chip.
- Activating it applies only the approved tuple:
  `three-class-clusters`, `classification`, `outputSize: 3`, `softmax`, and
  `categoricalCrossEntropy`.
- The direct chip remains a native keyboard-accessible button.
- Training starts and pauses without console errors.
- Multiclass boundary text alternatives use class labels.
- Compact layout keeps the chip visible and selectable.

## Actual Results

- One Data-tab `Three-Class` chip was found and activated.
- The app URL synced to the approved tuple:
  `d=three-class-clusters`, `os=3`, `oa=softmax`, and
  `l=categoricalCrossEntropy`.
- The chip reported `aria-pressed="true"`.
- The Network Graph text reported `3 outputs` with `softmax`.
- Training reached about Step 370 / Epoch 24 before pause.
- The Boundary panel showed multiclass decision-boundary copy and visible
  Class 0, Class 1, and Class 2 labels.
- Compact viewport verification at `390x800` found the Data tab, selected
  `Three-Class` chip, dataset status, Start, and Reset.

## Console Errors

No console errors were observed for the `5181` tab. The Browser log buffer
included development-mode slow-interaction warnings for the `5181` app and older
development-mode perf warnings from a prior `5177` tab.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-public-multiclass-data-chip.png`
- `docs/qa/browser-qa/wave-7-public-multiclass-data-chip-compact.png`

## Accessibility Notes

- The direct dataset control is a native button with accessible name
  `Three-Class`.
- The selected state uses `aria-pressed="true"`.
- The existing Network Graph text alternative reports the 3-output softmax
  topology.
- The Boundary panel text alternative names the dominant class and all three
  class labels.
- The compact `390x800` pass kept the selected chip visible.

## Result

Pass.
