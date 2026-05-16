# Browser QA: Wave 7 Multiclass Decision Boundary Renderer

## Date

2026-05-15

## Commit

- Original renderer: `9414244` (`feat(wave7,visualization): render multiclass decision boundaries`)
- Public follow-up QA: this record

## Environment

- OS: macOS
- Browser: Dia via Computer Use fallback for the original scalar smoke; Codex in-app Browser for follow-up public three-class QA
- Viewport: Desktop default
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B. Original scalar smoke was partial through Dia/Computer Use; follow-up public three-class QA passed through the Codex in-app Browser after public controls landed.

## Scenario

Verify that the visible decision-boundary workflow still loads and works after adding the hidden multiclass renderer. The original direct multiclass renderer remained behind internal frame-buffer state and was covered by component tests because public controls did not expose a three-class path yet.

Follow-up on 2026-05-15: verify the public approved three-class route can move
from the safe unavailable state to rendered multiclass class-region data once
training starts and fresh worker boundary data arrives.

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
10. Follow-up: opened the public approved three-class URL in the Codex in-app Browser.
11. Opened the Boundary tab and observed the pre-training unavailable fallback.
12. Started training from the banner control with the in-app Browser visible.
13. Waited for worker boundary data, then paused training.
14. Verified the boundary panel no longer showed the unavailable fallback and the DOM included `Class 0`, `Class 1`, `Class 2`, and confidence text.
15. Checked browser console errors.
16. Captured follow-up screenshots.

## Expected Results

- App loads without a blank page or framework overlay.
- Binary decision-boundary behavior remains unchanged for public scalar configs.
- Start, pause, and reset continue to work.
- The boundary canvas, legend, controls, network graph, lesson panel, and status bar remain visible.
- New hidden multiclass renderer does not leak public controls or change default scalar UX.
- Public three-class route renders multiclass class-region data after training produces worker boundary data.

## Actual Results

Partial pass. The public scalar decision-boundary workflow loaded and the start/pause/reset flow worked through Dia. The visible binary boundary remained intact after the code change.

Direct browser verification of the hidden multiclass renderer is pending because the app has no public multiclass control path yet and the available browser fallback did not provide reliable JavaScript-console execution. Component tests cover the hidden frame-buffer path, accessible image label, text summary, class legend, scalar fallback preservation, and repainting on `multiclassBoundaryVersion`.

Follow-up after public controls landed: the public approved three-class route
loaded, Boundary initially showed the safe unavailable fallback, then training
produced bounded multiclass boundary data. The DOM no longer contained the
unavailable fallback and included `Class 0`, `Class 1`, `Class 2`, and
confidence text. Browser console error checks returned `[]`.

## Console Errors

Console inspection was not available through the Dia/Computer Use fallback. No visible framework error overlay or blank page was observed. The dev server did not report browser-runtime errors during the smoke.

Follow-up public three-class check through the Codex in-app Browser returned
`[]` for Browser console errors.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-decision-boundary-dia.png`
- `docs/qa/browser-qa/wave-7-multiclass-boundary-public-three-class.png`
- `docs/qa/browser-qa/wave-7-multiclass-boundary-public-three-class-trained.png`

## Accessibility Notes

- Public scalar boundary still exposes an image alternative through the existing canvas label and explanatory copy.
- The hidden multiclass renderer adds `role="img"` with an `aria-describedby` text alternative summarizing dominant class, average confidence, and low-confidence share.
- Component tests assert the accessible multiclass image name and screen-reader summary.
- Public three-class Browser QA now verifies the class legend and confidence
  text after worker boundary data arrives. Keyboard-specific hidden-renderer
  QA remains covered by component-level accessible labels rather than a
  separate hidden browser route.

## Result

Pass for public three-class follow-up. Original hidden-route limitations remain
historical context.
