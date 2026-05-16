# Browser QA: Wave 7 Advanced Architecture Comparison

## Date

2026-05-15

## Commit

- Design note: `dc3907d`
- Implementation: `ac565e6`

## Environment

- OS: macOS, local Codex desktop session
- Browser: Codex in-app Browser
- Viewport: normal desktop viewport plus `390x800` compact viewport override
- Local URL: `http://127.0.0.1:5178/`

## QA Mode

Mode B.

## Scenario

Verify the saved-run History panel exposes an accessible architecture
comparison for the two selected saved runs, updates when the selected model
changes, and remains reachable at compact width without console errors.

## Steps

1. Started the web dev server with
   `pnpm --filter @nn-playground/web dev --host 127.0.0.1 --port 5177`.
2. Vite selected `http://127.0.0.1:5178/` because port `5177` was already in
   use.
3. Opened `http://127.0.0.1:5178/` in the Codex in-app Browser.
4. Ran one training step and saved the current run from the History panel.
5. Used the existing topology controls and dataset chip to create a second
   saved run with a different saved config.
6. Verified the History panel rendered a `Side-by-side model arena` region and
   an accessible `Architecture comparison` group.
7. Changed the Model A selector to the saved `xor` run and verified the
   architecture text updated.
8. Verified the group included hidden layers, total hidden units, activation,
   output/loss, optimizer/learning rate, batch size, regularization, data, and
   features.
9. Checked browser console errors.
10. Repeated DOM verification at `390x800`.

## Expected Results

- The History panel remains usable with saved runs.
- Architecture comparison uses existing saved-run data only.
- The comparison is exposed as a labelled semantic group.
- Selecting a different saved run updates the comparison.
- Compact viewport text wraps without requiring horizontal scrolling.
- No console errors are logged.

## Actual Results

- The `Architecture comparison` group was present and accessible.
- The comparison text updated after selecting a different Model A saved run.
- Browser DOM verification confirmed rows for hidden layers, total hidden
  units, activation, output/loss, optimizer/learning rate, batch size,
  regularization, data, and features.
- The compact viewport exposed the same architecture group and rows.
- Native model selectors remained present as labelled comboboxes.

Observed desktop architecture text included:

```text
Hidden layers
A [6, 4] / B [6, 4]
Total hidden units
A 10 / B 10 (same)
Batch size
A 10 / B 10 (same)
Data
A xor, 300 samples, noise 0 / B circle, 300 samples, noise 0
```

The component test covers the stronger architecture-diff case with different
hidden layers, total units, optimizer, learning rate, batch size,
regularization, dataset, sample count, noise, and features.

## Console Errors

Browser console error checks returned `[]` during desktop and compact
verification.

## Screenshots / Recordings

Browser screenshot capture timed out after the UI was verified. A macOS
full-screen fallback screenshot was discarded because it captured the whole
desktop instead of a clean app-only QA artifact.

## Accessibility Notes

- The comparison is exposed as `role="group"` with
  `aria-label="Architecture comparison"`.
- Rows use text labels and values; no row relies on color alone.
- Native `select` controls remain the selector UI for Model A and Model B.
- Compact CSS stacks architecture rows to one column and allows long values to
  wrap.

## Result

Pass. Screenshot evidence is unavailable due Browser screenshot timeout, but
Mode B DOM, interaction, compact viewport, and console checks passed.
