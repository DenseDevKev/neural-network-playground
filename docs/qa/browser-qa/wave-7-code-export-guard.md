# Browser QA: Wave 7 Code Export Guard

## Date

2026-05-15

## Commit

`f869ea2`

## Environment

- OS: macOS
- Browser: In-app browser requested, but Browser plugin tools were unavailable in this session
- Viewport: Pending human verification
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode C: Human-Verified-Only QA

## Scenario

Verify the visible Code Export panel still works for the currently public scalar app flows after the hidden multiclass code-export guard.

## Steps

1. Open `http://127.0.0.1:5177/`.
2. Open the Code panel.
3. Confirm the Pseudocode tab renders non-empty code.
4. Switch to NumPy and confirm code changes and remains readable.
5. Switch to TF.js and confirm code changes and remains readable.
6. Copy the current code and confirm the button gives feedback.
7. Check the browser console for errors.
8. Repeat the Code panel check at a compact viewport.

## Expected Results

- Code Export remains visible and usable for current scalar workflows.
- Pseudocode, NumPy, and TF.js tabs all render non-empty output.
- No visible multiclass controls are exposed.
- No console errors appear.
- Compact viewport has no Code panel text overlap beyond expected code scrolling.

## Actual Results

Browser QA status: pending human verification.

Automated/component evidence:

- `pnpm --filter @nn-playground/web test -- src/components/controls/CodeExportPanel.test.tsx` passed with 51 files and 382 tests.
- `pnpm test` passed with web 51 files and 382 tests.

## Console Errors

Pending human verification.

## Screenshots / Recordings

Pending human verification.

## Accessibility Notes

No new interactive controls were added. The existing native tab buttons and copy button remain covered by component tests.

## Result

Pending Human Verification
