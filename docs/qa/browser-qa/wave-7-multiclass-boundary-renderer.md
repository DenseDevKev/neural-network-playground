# Browser QA: Wave 7 Multiclass Decision Boundary Renderer

## Date

2026-05-15

## Commit

`9414244` (`feat(wave7,visualization): render multiclass decision boundaries`)

## Environment

- OS: macOS
- Browser: Dia via Computer Use fallback
- Viewport: Desktop default
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B, partial. The dedicated Browser plugin tools were not exposed in this session, so Dia plus Computer Use was used for the visible app smoke. Playwright was not installed and was not added.

## Scenario

Verify that the visible decision-boundary workflow still loads and works after adding the hidden multiclass renderer. The direct multiclass renderer remains behind internal frame-buffer state and is covered by component tests; public controls do not expose a three-class path yet.

## Steps

1. Attempted to load the already-open `http://127.0.0.1:5177/`; `curl -I http://127.0.0.1:5177/` failed because no local server was running.
2. Started the dev server with `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5177` using approved escalation after the sandbox blocked listening on `127.0.0.1:5177`.
3. Opened a new Dia tab and navigated to `http://127.0.0.1:5177/`.
4. Confirmed the app loaded, the boundary canvas rendered, binary legend/copy remained visible, and there was no framework error overlay.
5. Activated `Start training` by mouse and confirmed metrics, status, network activations, and the decision boundary updated.
6. Activated `Pause training` by mouse and confirmed the training state changed to paused.
7. Activated `Reset model and data` by mouse and confirmed the status returned to idle with step 0.
8. Captured a desktop screenshot with `screencapture -x docs/qa/browser-qa/wave-7-decision-boundary-dia.png`.
9. Stopped the Vite dev server with Ctrl-C.

## Expected Results

- App loads without a blank page or framework overlay.
- Binary decision-boundary behavior remains unchanged for public scalar configs.
- Start, pause, and reset continue to work.
- The boundary canvas, legend, controls, network graph, lesson panel, and status bar remain visible.
- New hidden multiclass renderer does not leak public controls or change default scalar UX.

## Actual Results

Partial pass. The public scalar decision-boundary workflow loaded and the start/pause/reset flow worked through Dia. The visible binary boundary remained intact after the code change.

Direct browser verification of the hidden multiclass renderer is pending because the app has no public multiclass control path yet and the available browser fallback did not provide reliable JavaScript-console execution. Component tests cover the hidden frame-buffer path, accessible image label, text summary, class legend, scalar fallback preservation, and repainting on `multiclassBoundaryVersion`.

## Console Errors

Console inspection was not available through the Dia/Computer Use fallback. No visible framework error overlay or blank page was observed. The dev server did not report browser-runtime errors during the smoke.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-decision-boundary-dia.png`

## Accessibility Notes

- Public scalar boundary still exposes an image alternative through the existing canvas label and explanatory copy.
- The hidden multiclass renderer adds `role="img"` with an `aria-describedby` text alternative summarizing dominant class, average confidence, and low-confidence share.
- Component tests assert the accessible multiclass image name and screen-reader summary.
- Keyboard-specific hidden-renderer browser QA is pending until a public or test-only browser path can safely expose the multiclass frame without changing product behavior.

## Result

Partial Pass / Hidden Multiclass Browser QA Pending
