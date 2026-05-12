# Browser QA: Wave 6E Checkpoint Timeline

## Date

2026-05-12

## Commit

Pending commit for web checkpoint timeline controls and protocol guard fix.

## Environment

- OS: macOS
- Browser: Codex in-app Browser plugin
- Viewport: 1280 x 720 for captured evidence
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B

## Scenario

Verify the approved runtime-only checkpoint timeline UI after adding worker-local checkpoint payloads and bounded checkpoint metadata.

## Steps

1. Started the local Vite dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177`.
2. Opened `http://127.0.0.1:5177/`.
3. Confirmed the app loaded with the checkpoint timeline controls visible.
4. Clicked `Start training` in the main training bar.
5. Confirmed `Pause training` and `Training...` appeared without a worker error.
6. Clicked `Pause training`.
7. Confirmed the checkpoint timeline controls and restore button were visible.
8. Focused the `Checkpoint timeline` range control and pressed `Home` and `End`.
9. Checked browser console errors through the Browser plugin.

## Expected Results

- App loads without a blank page or framework overlay.
- Training starts and pauses.
- Checkpoint timeline controls are visible after initialization and remain visible after training.
- Keyboard navigation works on the timeline range.
- No console errors are reported.
- Heavy checkpoint payloads are not surfaced in React state or browser-visible data.

## Actual Results

- App loaded successfully.
- Training started and displayed `Pause training` / `Training...`.
- Training paused successfully.
- Checkpoint timeline controls and restore action were visible.
- The `Checkpoint timeline` range accepted keyboard `Home` and `End`.
- Console errors were empty.

During the first Browser QA attempt, starting training surfaced `Worker connection lost` because the shared runtime guard treated optional activation-histogram fields with `undefined` values as malformed. A red shared protocol regression test was added, the guard was fixed to treat undefined optional histogram fields as absent, and Browser QA was rerun successfully.

## Console Errors

None after the guard fix.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-6e-checkpoint-timeline-desktop.png`

## Accessibility Notes

- Timeline uses a native labelled range input: `Checkpoint timeline`.
- Restore uses a native button with an action-specific accessible name such as `Restore checkpoint Step 0`.
- Keyboard checks covered `Home` and `End` on the range.
- Compact viewport resizing was not available through the in-app Browser plugin in this session; responsive behavior is covered by CSS constraints and component tests, but compact visual Browser QA remains pending human verification.

## Result

Pass for desktop Mode B. Compact viewport visual check pending human verification due Browser plugin viewport-resize limitation.
