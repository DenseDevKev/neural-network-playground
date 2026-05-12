# Browser QA: Wave 6B Saved Run Thumbnails

## Date

2026-05-11

## Commit

`96ee7a4`

## Environment

- OS: macOS, local Codex workspace
- Browser: Codex in-app Browser plugin
- Viewport: compact in-app browser viewport
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B.

## Scenario

Verify generated, non-persisted saved-run thumbnails in the History panel.

## Steps

1. Used the existing local Vite dev server at `http://127.0.0.1:5177/`.
2. Reloaded the app in the Browser plugin after the code change.
3. Opened the `History` tab.
4. Verified saved run cards render SVG loss thumbnails from existing saved history.
5. Verified comparison summaries remain visible.
6. Checked current-URL console errors.
7. Captured a compact viewport screenshot.

## Expected Results

- Saved run cards with at least two history points show a compact loss thumbnail.
- The thumbnail is exposed as an accessible image with a text summary.
- Saved run comparison summaries still render.
- No current-URL console errors appear.
- No stored image or schema change is needed.

## Actual Results

- The History panel showed two saved-run thumbnails.
- Browser DOM snapshot included accessible images such as `Loss thumbnail for circle at step 2: 3 points, train loss 0.6926 to 0.6906, test loss 0.6918 to 0.6926.`
- The comparison summary remained visible under the newer saved run.
- Current-URL console errors were empty.

## Console Errors

No errors were reported for `127.0.0.1:5177`.

## Screenshots / Recordings

- `/private/tmp/nn-playground-wave6b-run-thumbnails.png`

## Accessibility Notes

- Each rendered thumbnail uses `role="img"` with an accessible summary of point count and first-to-last train/test losses.
- Missing or insufficient history falls back to visible text rather than a decorative empty chart.
- No new interactive controls were added.

## Result

Pass.
