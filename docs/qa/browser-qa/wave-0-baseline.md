# Browser QA: Wave 0 Baseline

## Date

2026-05-11

## Commit

Pending Wave 0 commit.

## Environment

- OS: macOS local workspace
- Browser: Codex in-app Browser plugin
- Viewport: default browser viewport plus compact override at 390x844
- Local URL: `http://127.0.0.1:5173/`

## QA Mode

Mode B: Agent-assisted Browser QA.

## Scenario

Baseline smoke test for the current Neural Network Playground app before Wave 1 feature work.

## Steps

1. Open the local Vite dev server.
2. Confirm the app loads without a framework error overlay.
3. Check browser console errors.
4. Start training, pause training, and reset.
5. Interact with an existing preset or dataset control.
6. Confirm decision boundary, loss, and network visualization surfaces are reachable.
7. Confirm the lesson panel is reachable.
8. Confirm the compact viewport layout remains usable.

## Expected Results

- App loads with the NN-FORGE shell visible.
- No console errors are reported during the smoke flow.
- Start, pause, and reset controls respond.
- Preset or dataset interaction does not throw.
- Decision boundary, loss chart, network visualization, and lesson surfaces are reachable.
- Compact layout does not show obvious overlap in the checked viewport.

## Actual Results

- App loaded with title `Neural Network Playground 2.0`.
- The NN-FORGE shell rendered without a framework error overlay.
- Start and pause were exercised through the header training control; reset was exercised through the main training controls and returned the visible status to `Step 0`.
- Data tab was opened and the dataset was changed from Circle to XOR.
- Decision boundary was visible after reset.
- Loss tab was opened and showed loss diagnostics controls.
- Network topology controls remained visible in the compact layout.
- Guided lesson drawer was expanded and the lesson selector was visible.
- Compact viewport override at 390x844 disabled non-compact layout choices and kept the dock layout usable.

## Console Errors

No browser console errors were returned by `tab.dev.logs({ levels: ['error'] })` during the smoke path.

## Screenshots / Recordings

- Compact boundary smoke: `/private/tmp/nn-playground-wave0-compact.png`
- Compact loss/lesson smoke: `/private/tmp/nn-playground-wave0-final.png`

## Accessibility Notes

- Wave 0 has no product UI change.
- Smoke confirmed semantic buttons, tabs, tabpanels, guided lesson combobox, and the decision-boundary `img` role were reachable through the Browser DOM snapshot.
- Compact layout used disabled buttons for unavailable layout modes, preserving visible state rather than silently hiding mode affordances.

## Result

Pass.
