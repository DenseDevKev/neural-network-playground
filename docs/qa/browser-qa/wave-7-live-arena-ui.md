# Browser QA: Wave 7 Live Arena UI

## Date

2026-05-12

## Commit

`0344bbe`

## Environment

- OS: macOS local workspace
- Browser: Codex in-app browser
- Viewport: default browser viewport, plus explicit compact override at 820x720
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B: Agent-assisted Browser QA with the Codex in-app browser.

## Scenario

Verify the scalar live side-by-side arena UI exposed in Run History. The UI must use existing saved runs, call the already-approved scalar worker APIs, render bounded summaries only, remain keyboard accessible, and keep compact dock layout usable.

## Steps

1. Started a fresh dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177` after confirming the previous browser tab was stale.
2. Opened `http://127.0.0.1:5177/` in the Codex in-app browser.
3. Checked console errors via Browser dev logs.
4. Confirmed the Run History tab exposed the side-by-side model arena and the new `Start live arena` / `Step live arena` controls.
5. Activated `Start live arena with selected saved runs`.
6. Confirmed `Live arena scalar summaries` appeared with Model A and Model B bounded scalar summaries.
7. Confirmed `Step live arena once` became enabled.
8. Activated `Step live arena once` by mouse.
9. Activated `Step live arena once` by keyboard Enter.
10. Set browser viewport override to 820x720, reloaded the app, and confirmed the History arena controls remained present in compact dock layout.
11. Reset the browser viewport override.

## Expected Results

- App loads without framework overlay or console errors.
- Run History shows the saved-run side-by-side arena.
- Live arena controls are native buttons with accessible names.
- Start initializes bounded live summaries for both models.
- Step remains enabled after initialization and updates without breaking the page.
- Keyboard activation works without triggering global training shortcuts.
- Compact viewport still exposes the arena controls.
- No URL/config serialization, persistence, public config shape, dependency, engine math, or training behavior changes are required.

## Actual Results

- App loaded at `http://127.0.0.1:5177/` with zero console errors.
- Browser DOM snapshot showed `Start live arena with selected saved runs` and `Step live arena once` in the `Live arena controls` group.
- Start activation produced the `Live arena scalar summaries` group.
- Step button was enabled after Start.
- Mouse activation preserved live summaries.
- Keyboard Enter activation on the Step button preserved live summaries.
- Compact 820x720 viewport reload showed the side-by-side model arena and Start live arena controls.

## Console Errors

None observed during desktop or compact checks.

## Screenshots / Recordings

Screenshot capture through the in-app Browser backend timed out twice with `Page.captureScreenshot`. No screenshot artifact was recorded for this slice. DOM snapshots and console checks were recorded in the Browser QA run.

## Accessibility Notes

- Controls are native `<button type="button">` elements.
- The controls are grouped with `aria-label="Live arena controls"`.
- Scalar summaries are grouped with `aria-label="Live arena scalar summaries"`.
- Keyboard Enter activation on `Step live arena once` worked and did not trigger global training shortcuts.

## Result

Pass, with screenshot artifact unavailable due Browser screenshot timeout.
