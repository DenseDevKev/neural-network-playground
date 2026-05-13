# Browser QA: Wave 7 Loss Landscape Probe UI

## Date

2026-05-13

## Commit

`4d4878b` (`feat(wave7,ui): add loss landscape probe panel`)

## Environment

- OS: macOS
- Browser: Codex in-app browser
- Viewport: Desktop default and compact `390x844`
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B

## Scenario

Verify the Inspection panel Loss Landscape Probe UI after the Wave 7 one-shot worker RPC slice.

## Steps

1. Started the Vite dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177`.
2. Reloaded `http://127.0.0.1:5177/` and confirmed the app loaded with zero console errors.
3. Confirmed the `Loss landscape probe` region rendered in the Inspection panel with a native `Probe loss surface` button and named status region.
4. Activated `Run one training step`, then activated `Probe loss surface` by mouse.
5. Confirmed the UI rendered a worker summary, step/epoch, `Local 2D loss slice` image alternative, 7 by 7 heatmap, center/min/max loss, sample count, and best direction text.
6. Activated `Probe loss surface` with Enter while the button was focused.
7. Set the Browser viewport to `390x844`, reloaded the app, stepped once, and confirmed the probe controls remained reachable in the compact layout.
8. Repeated compact activation and confirmed by DOM snapshot that the compact layout rendered the same status, heatmap alternative text, scalar summary, and best direction text.
9. Reset the Browser viewport to the default size.
10. Checked Browser console logs.

## Expected Results

- App loads without console errors.
- Inspection panel opens.
- `Probe loss surface` is reachable by mouse and keyboard.
- Activating the button calls the one-shot worker RPC and renders bounded scalar/grid results.
- Loading, success, and error states are announced through a small live region.
- The heatmap has a non-color text alternative and visible scalar summary.
- Compact viewport has no text overlap or broken wrapping in the probe controls.

## Actual Results

Passed. The Inspection panel remained usable after the probe request, the button stayed keyboard reachable, the live status region summarized the result, and the compact DOM exposed the bounded result without raw activation or loss arrays in React state.

Desktop screenshot capture succeeded after the probe result rendered. Compact screenshot capture succeeded for the focused native button before the repeated result check; the in-app Browser screenshot backend timed out when capturing a second compact screenshot after the result rendered, so the compact result evidence is recorded by DOM snapshot and console checks instead.

## Console Errors

No console errors observed.

Warnings observed:

- `[perf] Slow interaction: Promise Resolved (82.57ms)`
- `[perf] Slow interaction: Cascading Update (28.86ms)`
- `[perf] Slow interaction: Promise Resolved (131.53ms)`
- `[perf] Slow interaction: Promise Resolved (85.90ms)`
- `[perf] Slow interaction: Promise Resolved (106.75ms)`
- `[perf] Slow interaction: Promise Resolved (74.86ms)`

These are development-mode performance warnings and match earlier Browser QA evidence patterns.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-loss-landscape-probe-desktop.png`
- `docs/qa/browser-qa/wave-7-loss-landscape-probe-compact.png`

## Accessibility Notes

- The probe is a native `<button type="button">`.
- Mouse and Enter activation were verified in Browser QA.
- The announcement surface is a named `role="status"` live region containing loading/error/summary text.
- The heatmap renders as `role="img"` with a text alternative that includes center/min/max and best direction.
- Non-color text below the heatmap repeats the sampled grid size, scalar losses, and best direction.
- Component tests cover keyboard activation, duplicate request guarding, error status text, and accessible result rendering.

## Result

Pass
