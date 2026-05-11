# Browser QA: Wave 1 Explanation Actions

## Date

2026-05-11

## Commit

Pending Wave 1 QA/state commit.

## Environment

- OS: macOS local workspace
- Browser: Codex in-app Browser plugin
- Viewport: default browser viewport plus compact override at 390x844
- Local URL: `http://127.0.0.1:5173/`

## QA Mode

Mode B: Agent-assisted Browser QA.

## Scenario

Verify that contextual explanation action cards render in the live app and can be activated by mouse and keyboard without console errors.

## Steps

1. Open the local Vite dev server at `http://127.0.0.1:5173/`.
2. Open the Loss panel.
3. Start training and wait for an explanation action card to appear.
4. Pause training after the action card is visible.
5. Click the visible `Open loss & accuracy` action card.
6. Activate the same action card with Enter.
7. Check browser console errors.
8. Override viewport to 390x844 and inspect the compact layout.
9. Capture screenshots and reset the viewport override.

## Expected Results

- Explanation action cards render as native buttons in the Loss panel.
- Mouse activation focuses/selects the existing target panel or tab.
- Keyboard activation works through Enter without breaking global training shortcuts.
- Compact layout remains usable and the action card does not overflow.
- No browser console errors are reported.

## Actual Results

- A `Test metrics are catching up` explanation appeared in the Loss panel while training was paused.
- The `Open loss & accuracy` action card was visible as a button.
- Mouse activation completed and the Loss tab remained selected.
- Enter activation completed on the same action card.
- Compact viewport at 390x844 showed the action card in the Loss tab with the card focused.
- No browser console errors were returned by `tab.dev.logs({ levels: ['error'] })`.

## Console Errors

No console errors reported during the Wave 1 action-card smoke flow.

## Screenshots / Recordings

- Action card desktop/default viewport: `/private/tmp/nn-playground-wave1-action-card.png`
- Action card compact viewport: `/private/tmp/nn-playground-wave1-action-card-compact.png`

## Accessibility Notes

- Action cards rendered as native `<button type="button">` controls.
- The action-card group was exposed as `aria-label="Suggested explanation actions"` in component tests.
- Keyboard activation with Enter was verified in Browser QA and covered in component tests.
- `jest-axe` component coverage found no obvious violations for the rendered explanation action panel.

## Result

Pass.
