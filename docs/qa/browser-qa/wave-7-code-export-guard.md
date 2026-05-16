# Browser QA: Wave 7 Code Export Guard

## Date

2026-05-15

## Commit

- Code: `f869ea2`
- Mode B follow-up evidence: this record

## Environment

- OS: macOS
- Browser: Codex in-app Browser
- Viewport: desktop `1280x720`, compact `390x800`
- Local URL: `http://127.0.0.1:5177/#d=three-class-clusters&pt=classification&r=0.5&n=0.05&ns=300&s=42&hl=6%2C6&os=3&a=tanh&oa=softmax&wi=xavier&ws=42&lr=0.03&bs=10&l=categoricalCrossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000`

## QA Mode

Mode B

## Scenario

Verify the visible Code Export panel works for the now-public approved three-class softmax tuple after the code-export guard and later public multiclass rollout.

## Steps

1. Started the Vite dev server with `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5177`.
2. Opened `http://127.0.0.1:5177/` in the Codex in-app Browser.
3. Applied the `Three-Class Softmax Lab` preset.
4. Opened the Code panel.
5. Confirmed the Pseudocode tab renders a 2 -> 6 -> 6 -> 3 architecture, output `Softmax`, and `Categorical Cross-Entropy`.
6. Switched to NumPy and confirmed the code renders `def softmax`, `np.exp`, and `softmax(W3 @ h + b3)`.
7. Switched to TF.js and confirmed `tf.sequential`, `units: 3`, `softmax`, and `categoricalCrossentropy`.
8. Clicked `Copy Code` and confirmed copy feedback remained available.
9. Checked browser console errors.
10. Repeated the Code panel check at compact `390x800` viewport.
11. Captured desktop and compact screenshots and reset the viewport override.

## Expected Results

- Code Export remains visible and usable for the approved public multiclass workflow.
- Pseudocode, NumPy, and TF.js tabs all render non-empty output.
- Multiclass output copy is truthful for the approved tuple.
- No console errors appear.
- Compact viewport has no Code panel text overlap beyond expected code scrolling.

## Actual Results

- Code Export opened from the public `Three-Class Softmax Lab` preset.
- Pseudocode, NumPy, and TF.js tabs rendered non-empty code for the 3-output softmax network.
- TF.js output included `tf.sequential`, `units: 3`, `softmax`, and `categoricalCrossentropy`.
- NumPy output included `def softmax`, `np.exp`, and `softmax(W3 @ h + b3)`.
- Copy Code activation completed without console errors.
- Compact `390x800` viewport retained the Code panel and TF.js output without obvious text overlap; long code remains scrollable as expected.
- Browser console error checks returned `[]`.

## Console Errors

None observed in Browser console checks.

## Screenshots / Recordings

- `docs/qa/browser-qa/wave-7-code-export-guard-desktop.png`
- `docs/qa/browser-qa/wave-7-code-export-guard-compact.png`

## Accessibility Notes

No new interactive controls were added. The existing native tab buttons and copy button remain covered by component tests and were reachable in Browser QA.

## Result

Pass
