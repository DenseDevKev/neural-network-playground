# Browser QA: Wave 6A Run Comparison

## Date

2026-05-11

## Commit

`aa5fff1`

## Environment

- OS: macOS, local Codex workspace
- Browser: Codex in-app Browser plugin
- Viewport: compact in-app browser viewport, screenshot captured at approximately 684 x 925
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B.

## Scenario

Verify the saved-run comparison summary in the History panel using existing saved run data.

## Steps

1. Started the local Vite dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177`.
2. Opened `http://127.0.0.1:5177/` in the Browser plugin.
3. Activated `Run one training step`.
4. Opened the `History` tab.
5. Saved the first current run.
6. Activated `Run one training step` again.
7. Saved the second current run.
8. Verified the newer saved run showed a comparison group against the previous saved run.
9. Verified the `Save current run` button remained enabled and keyboard-operable.
10. Checked current-URL console errors.

## Expected Results

- The app loads without a framework overlay.
- The History panel can be reached.
- Two saved runs can be captured from existing training snapshots.
- The newer saved run shows comparison text against the previous saved run.
- The comparison includes train loss, test loss, generalization gap, and step deltas.
- The comparison is exposed as an accessible group.
- No current-URL console errors appear.

## Actual Results

- The app loaded at `http://127.0.0.1:5177/`.
- The History panel opened successfully.
- Two saved runs were captured at step 1 and step 2.
- The comparison group was present: `Comparison for circle at step 2 against circle at step 1`.
- The visible comparison summary included `Compared with circle at step 1` and `Steps +1`.
- `Save current run` remained enabled and accepted keyboard focus/action.

## Console Errors

No errors were reported for `127.0.0.1:5177`.

An older retained Browser log entry from `127.0.0.1:5173` was ignored because it was not from the QA URL.

## Screenshots / Recordings

- `/private/tmp/nn-playground-wave6a-run-comparison.png`

## Accessibility Notes

- The comparison summary uses a named `role="group"` so assistive technology gets the run-to-baseline relationship.
- No new interactive control was added by this slice.
- Existing native buttons in the History panel remained keyboard reachable.

## Result

Pass.
