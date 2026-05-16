# Browser QA: Wave 7 Multiclass Run History

## Date

2026-05-15

## Commit

- Original slice: `4e9d106`
- Public follow-up QA: this record

## Environment

- OS: macOS, local Codex desktop session
- Browser: Codex in-app Browser
- Viewport: existing desktop viewport in the in-app Browser
- Local URL: original partial check at `http://127.0.0.1:5181/#d=three-class-clusters&pt=classification&os=3&oa=softmax&l=categoricalCrossEntropy`; follow-up public check at `http://127.0.0.1:5177/#d=three-class-clusters&pt=classification&r=0.5&n=0.05&ns=300&s=42&hl=6%2C6&os=3&a=tanh&oa=softmax&wi=xavier&ws=42&lr=0.03&bs=10&l=categoricalCrossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000`

## QA Mode

Mode B. Original direct-hidden check was partial; follow-up public check passed after public controls landed.

## Scenario

Attempt to inspect the direct hidden approved multiclass URL after the
run-history capture/localStorage slice. The code slice itself is primarily
validated by component/store/shared tests because visible public multiclass
controls and public multiclass presets are still intentionally absent.

Follow-up on 2026-05-15: verify the public approved three-class route can save
and restore a run through the visible Run History panel after the public preset,
Data chip, runtime, Config Panel, and run-history preservation slices landed.

## Steps

1. Started the web dev server on `http://127.0.0.1:5181/`.
2. Opened the approved tuple URL hash for `three-class-clusters`,
   `classification`, `outputSize: 3`, `softmax`, and
   `categoricalCrossEntropy`.
3. Captured DOM state from the in-app Browser.
4. Clicked the native `Reset model and data` button.
5. Captured DOM state again.
6. Checked browser console logs for errors and warnings.
7. Follow-up: opened the public approved three-class URL on port `5177`.
8. Opened the History tab.
9. Clicked `Save current run`.
10. Verified `three-class-clusters at step 0` appeared in saved runs.
11. Clicked `Restore config for three-class-clusters at step 0`.
12. Verified the restored state still referenced `three-class-clusters`.
13. Checked browser console errors.

## Expected Results

- App loads without framework overlay or console errors.
- The approved multiclass tuple can be represented without public controls.
- Run-history save/restore and live-arena guard behavior are covered by tests
  until public controls expose an end-to-end route.
- Follow-up: public three-class save/restore works without console errors.

## Actual Results

- The app loaded at the direct hidden URL.
- The status bar displayed `DATA: three-class-clusters`.
- The `Reset model and data` button was reachable and clickable.
- The Network Graph canvas still reported a scalar summary
  (`X1, X2 -> [4] -> [4] -> 1 output`) because that visualization currently
  has scalar text alternatives and public controls have not been exposed.
- The Boundary panel displayed the existing binary-unavailable fallback because
  the hidden direct-URL route does not provide a complete public QA path.
- At the time of the original hidden-route QA, end-to-end run-history
  save/restore was not verified in Browser because there was no visible public
  multiclass control or preset yet.
- Follow-up after public controls landed: `Save current run` saved
  `three-class-clusters at step 0`, `Restore config for three-class-clusters at
  step 0` was reachable, and the restored state remained on
  `three-class-clusters`.

## Console Errors

No console errors were observed for the `5181` tab during the run-history QA
attempt. The Browser log buffer contained only pre-existing development-mode
performance warnings from the earlier `5177` local app tab.

Follow-up public check on `5177` returned `[]` for Browser console errors.

## Screenshots / Recordings

No screenshot was captured for the original hidden-route partial QA attempt.
DOM snapshot evidence and the follow-up public Browser check are summarized
above.

## Accessibility Notes

Automated tests cover the changed live-arena guard: selected saved multiclass
runs keep the start action disabled, and the disabled button references an
explanatory status note with `aria-describedby`. Keyboard/browser verification
for the public multiclass save/restore flow is now covered by the follow-up
Browser check above.

## Result

Pass for public three-class save/restore follow-up. Original hidden-route notes
remain historical context.
