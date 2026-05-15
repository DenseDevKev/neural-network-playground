# Browser QA: Wave 7 Three-Class Softmax Lesson

## Date

2026-05-15

## Commit

`65d1aec`

## Environment

- OS: macOS via Codex desktop
- Browser: Codex in-app Browser
- Viewport: `1280x720`, then `390x800`
- Local URL: `http://127.0.0.1:5182/`

## QA Mode

Mode B

## Scenario

Verify the guided lesson selector exposes and starts the public Three-Class Softmax lesson without console errors, and that the compact lesson drawer can reach the same lesson.

## Steps

1. Started the local Vite dev server with `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5182`.
2. Opened `http://127.0.0.1:5182/` at `1280x720`.
3. Selected `Three-Class Softmax Lab` from the `Guided lesson` selector.
4. Started the lesson and verified Step 1 text: `Read the three clusters`.
5. Clicked through the lesson steps and verified the final run step text: `Train three class regions`.
6. Verified the `Start training` control was visible on the final run step.
7. Repeated the lesson start at `390x800` by expanding the compact guided lesson drawer, selecting the lesson, and starting it.
8. Checked browser console errors for the QA tab.

## Expected Results

- The guided lesson selector includes `Three-Class Softmax Lab`.
- Starting the lesson applies the approved three-class tuple and shows `Step 1 of 4`.
- The final lesson step points learners to training controls.
- Compact viewport exposes the collapsed drawer, can expand it, and can start the same lesson.
- No browser console errors are reported.

## Actual Results

- Desktop lesson selection and start succeeded.
- Mouse activation reached the final run step and training controls.
- Compact lesson drawer expanded and started the same lesson.
- The URL after lesson start contained the approved tuple: `d=three-class-clusters`, `os=3`, `oa=softmax`, and `l=categoricalCrossEntropy`.
- The in-app Browser keypress APIs did not advance the focused `Next lesson step` button with Enter, even after the focused element was confirmed as the native button. Explicit `type="button"` semantics and Enter/Space activation are covered by `GuidedLessonPanel.test.tsx`.

## Console Errors

None in the QA tab.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-three-class-lesson.png`
- `docs/qa/browser-qa/wave-7-three-class-lesson-compact.png`

## Accessibility Notes

- The lesson selector is a native `<select>` with accessible name `Guided lesson`.
- Start, Back, Next, and Finish lesson controls are native buttons with explicit `type="button"`.
- Automated component coverage verifies keyboard activation for lesson navigation with Enter and Space.
- Browser-keypress activation was inconclusive because the in-app Browser backend did not trigger the focused button.

## Result

Pass for Browser mouse, compact viewport, URL tuple, and console checks. Keyboard activation is verified by component tests and recorded as inconclusive in Browser Mode B.
