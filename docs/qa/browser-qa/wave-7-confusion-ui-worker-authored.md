# Browser QA: Wave 7 Worker-Authored Multiclass Confusion UI

## Date

2026-05-15

## Commit

- Code: `0dd60f9`
- Evidence: pending docs commit

## Environment

- OS: macOS
- Browser: Codex in-app Browser
- Viewport: desktop `1280x720`, compact `390x800`
- Local URL: `http://127.0.0.1:5177/#d=three-class-clusters&pt=classification&r=0.5&n=0.05&ns=300&s=42&hl=6%2C6&os=3&a=tanh&oa=softmax&wi=xavier&ws=42&lr=0.03&bs=10&l=categoricalCrossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000`

## QA Mode

Mode B

## Scenario

Verify that the public approved three-class softmax tuple can render the Confusion panel from worker-authored bounded 3x3 confusion data, while keeping data outside React state.

## Steps

1. Started the Vite dev server with `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5177`.
2. Opened the approved three-class URL in the Codex in-app Browser.
3. Verified the paused Confusion panel could render the existing derived fallback.
4. Started training from the transport control and made the in-app Browser visible so the frame loop applied streamed snapshots.
5. Verified the Confusion panel rendered `Pred Class 2` and the note `Latest worker-authored test evaluation from the training worker; stored outside React state.`
6. Checked Browser console errors.
7. Captured desktop and compact viewport screenshots.
8. Switched to compact `390x800` viewport and verified the worker-authored note and 3-class matrix remained reachable after scrolling.
9. Reset the Browser viewport override.

## Expected Results

- App loads without a framework error overlay.
- The approved three-class tuple can start/pause training.
- Worker-authored 3x3 confusion data appears in the Confusion panel during/after training.
- The panel does not show stale binary labels for the approved multiclass tuple.
- Browser console errors are empty.
- Compact viewport keeps the matrix reachable without text overlap after scrolling within the dense dock layout.

## Actual Results

- The approved three-class tuple loaded and training could start/pause.
- The Confusion panel rendered the worker-authored bounded 3x3 matrix and the worker-authored note.
- Browser console error checks returned `[]`.
- Desktop screenshot shows the matrix visible in the right dock panel.
- Compact screenshot shows the dense layout; after scrolling, the matrix summary and worker-authored note are visible.
- The first background-browser attempt did not apply worker frames until the Browser was made visible; QA then passed. This appears to be Browser/rAF visibility behavior, not an app console error.

## Console Errors

None observed in Browser console checks.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-confusion-ui-desktop.png`
- `docs/qa/browser-qa/wave-7-confusion-ui-compact.png`
- `docs/qa/browser-qa/wave-7-confusion-ui-compact-scrolled.png`

## Accessibility Notes

- The 3x3 cells retain accessible labels with actual class, predicted class, count, and percentage.
- Row totals, column totals, and grand total retain `aria-label` text.
- A screen-reader-only summary announces diagonal count, sample total, and accuracy.
- No new interactive controls were added in this slice.

## Result

Pass.
