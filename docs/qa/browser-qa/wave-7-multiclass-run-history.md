# Browser QA: Wave 7 Multiclass Run History

## Date

2026-05-15

## Commit

`4e9d106`

## Environment

- OS: macOS, local Codex desktop session
- Browser: Codex in-app Browser
- Viewport: existing desktop viewport in the in-app Browser
- Local URL: `http://127.0.0.1:5181/#d=three-class-clusters&pt=classification&os=3&oa=softmax&l=categoricalCrossEntropy`

## QA Mode

Mode B, partial.

## Scenario

Attempt to inspect the direct hidden approved multiclass URL after the
run-history capture/localStorage slice. The code slice itself is primarily
validated by component/store/shared tests because visible public multiclass
controls and public multiclass presets are still intentionally absent.

## Steps

1. Started the web dev server on `http://127.0.0.1:5181/`.
2. Opened the approved tuple URL hash for `three-class-clusters`,
   `classification`, `outputSize: 3`, `softmax`, and
   `categoricalCrossEntropy`.
3. Captured DOM state from the in-app Browser.
4. Clicked the native `Reset model and data` button.
5. Captured DOM state again.
6. Checked browser console logs for errors and warnings.

## Expected Results

- App loads without framework overlay or console errors.
- The approved multiclass tuple can be represented without public controls.
- Run-history save/restore and live-arena guard behavior are covered by tests
  until public controls expose an end-to-end route.

## Actual Results

- The app loaded at the direct hidden URL.
- The status bar displayed `DATA: three-class-clusters`.
- The `Reset model and data` button was reachable and clickable.
- The Network Graph canvas still reported a scalar summary
  (`X1, X2 -> [4] -> [4] -> 1 output`) because that visualization currently
  has scalar text alternatives and public controls have not been exposed.
- The Boundary panel displayed the existing binary-unavailable fallback because
  the hidden direct-URL route does not provide a complete public QA path.
- End-to-end run-history save/restore was not verified in Browser because there
  is no visible public multiclass control or preset yet.

## Console Errors

No console errors were observed for the `5181` tab during the run-history QA
attempt. The Browser log buffer contained only pre-existing development-mode
performance warnings from the earlier `5177` local app tab.

## Screenshots / Recordings

No screenshot was captured for this partial QA attempt. DOM snapshot evidence is
summarized above.

## Accessibility Notes

Automated tests cover the changed live-arena guard: selected saved multiclass
runs keep the start action disabled, and the disabled button references an
explanatory status note with `aria-describedby`. Keyboard/browser verification
for the public multiclass save/restore flow remains pending until visible
controls exist.

## Result

Partial / Pending Public Controls.
