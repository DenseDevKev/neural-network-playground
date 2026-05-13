# Browser QA: Wave 7 Slow-Motion Backprop Preview

## Date

2026-05-13

## Commit

`7dce9f9` (`feat(wave7,ui): add backprop preview panel`)

## Environment

- OS: macOS
- Browser: Codex in-app browser
- Viewport: Desktop default and compact `390x844`
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B

## Scenario

Verify the Inspection panel slow-motion backprop preview UI after the Wave 7 one-shot worker RPC slice.

## Steps

1. Started the Vite dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177`.
2. The Browser tool initially could not navigate away from Chrome's generated `data:` connection-error page, so the user manually restored the tab to `http://127.0.0.1:5177/`.
3. Confirmed the app loaded with the Inspection tab visible.
4. Confirmed the `Slow-motion backprop preview` region rendered with a native `Preview backprop` button and empty status region.
5. Activated `Preview backprop` by mouse.
6. Confirmed the UI rendered `Backprop preview found 3 healthy layer updates.`, preview step/epoch, batch size, loss, learning rate, gradient norm, clipping state, and a semantic `Backprop layer summaries` list.
7. Activated the same button with Enter while focused.
8. Set the Browser viewport to `390x844` and confirmed the preview region and layer summaries remained reachable in the compact layout.
9. Reset the Browser viewport to the default size.
10. Checked Browser console logs.

## Expected Results

- App loads without console errors.
- Inspection panel opens.
- `Preview backprop` is reachable by mouse and keyboard.
- Activating the button calls the one-shot worker RPC and renders bounded scalar layer summaries.
- Loading, success, and error states are announced through a small live region.
- Compact viewport has no text overlap or broken wrapping.

## Actual Results

Passed. The Inspection panel remained usable after the preview request, the button stayed keyboard reachable, the live status region summarized the result, and the semantic list exposed bounded layer-level summaries without raw arrays.

Compact viewport verification passed by DOM and screenshot inspection. The preview region remained present and the layer-summary text wrapped within the dense layout.

## Console Errors

No console errors observed.

Warnings observed:

- `[perf] Slow interaction: Promise Resolved (82.57ms)`
- `[perf] Slow interaction: Cascading Update (28.86ms)`

These are development-mode performance warnings and match earlier Browser QA evidence patterns.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-backprop-preview-desktop.png`
- `docs/qa/browser-qa/wave-7-backprop-preview-compact.png`

## Accessibility Notes

- The preview is a native `<button type="button">`.
- Enter activation was verified in Browser QA.
- The announcement surface is a small `role="status"` live region containing loading/error/summary text only.
- Layer details render outside the live region in a semantic list labelled `Backprop layer summaries`.
- Component tests cover keyboard activation, duplicate request guarding, error status text, and semantic list rendering.

## Result

Pass
