# Browser QA: Wave 2 Lesson Depth

## Date

2026-05-11

## Commit

Pending Wave 2 lesson content commit.

## Environment

- OS: macOS local workspace
- Browser: Codex in-app Browser plugin
- Viewport: default browser viewport
- Local URL: `http://127.0.0.1:5173/`

## QA Mode

Mode B: Agent-assisted Browser QA.

## Scenario

Verify that a newly added guided lesson is selectable and starts through the existing lesson engine.

## Steps

1. Open the local Vite dev server.
2. Open the guided lesson selector.
3. Select `Learning Rate Tuning`.
4. Click `Start guided lesson`.
5. Confirm the first step appears.
6. Check browser console errors.

## Expected Results

- The new lesson appears in the existing selector.
- Starting the lesson applies the existing preset/start flow.
- The first step is visible and uses existing panel focus behavior.
- No console errors are reported.

## Actual Results

- `Learning Rate Tuning` was selected from the guided lesson selector.
- `Start guided lesson` opened the first step, `Find the update-size controls`.
- No browser console errors were returned by `tab.dev.logs({ levels: ['error'] })`.

## Console Errors

No console errors reported.

## Screenshots / Recordings

- Learning-rate lesson started: `/private/tmp/nn-playground-wave2-learning-rate-lesson.png`

## Accessibility Notes

- The lesson was selected through the existing native `<select>` control.
- The first step appeared in the existing guided lesson region.
- No lesson-engine or focus behavior changes were introduced in this slice.

## Result

Pass.
