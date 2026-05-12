# Browser QA: Wave 7 Side-by-Side Model Arena Phase 1

## Date

2026-05-12

## Commit

`76c37ae`

## Environment

- OS: macOS via Codex desktop
- Browser: Codex in-app Browser
- Viewport: default desktop viewport, plus compact viewport override `390x844`
- Local URL: `http://127.0.0.1:5177/`

## QA Mode

Mode B

## Scenario

Verify the saved-run-only side-by-side model arena in the History panel without any worker, protocol, persistence, URL/config, public config, or runtime behavior changes.

## Steps

1. Started a fresh Vite dev server with `pnpm --filter @nn-playground/web exec vite --host 127.0.0.1 --port 5177`.
2. Opened `http://127.0.0.1:5177/` in the Codex in-app Browser.
3. Confirmed the History output tab was selected and existing saved runs were available.
4. Verified the `Side-by-side model arena` region rendered above the saved-run list.
5. Verified native `Model A run` and `Model B run` comboboxes were present.
6. Verified `Model A` and `Model B` model regions exposed accessible labels and loss thumbnail text alternatives.
7. Changed `Model A run` to match `Model B run` and confirmed the comparison summary updated to equal-state copy.
8. Checked console errors after desktop interactions.
9. Applied the Browser viewport override `390x844`, reloaded the app, and confirmed the arena region, model comboboxes, model regions, and comparison summary remained present.
10. Checked console errors after compact-viewport verification and reset the Browser viewport override.

## Expected Results

- The arena appears only when at least two saved runs exist.
- Native comboboxes allow selecting the two saved runs to compare.
- Each selected model has an accessible region label and text alternative for its thumbnail.
- The summary reports test-loss, generalization-gap, and step-count differences using existing saved-run data only.
- No console errors occur.
- Compact viewport keeps the arena reachable and semantically intact.

## Actual Results

- Passed. The arena rendered with two saved runs from local run history.
- Desktop DOM exposed:
  - `region "Side-by-side model arena"`
  - `combobox "Model A run"`
  - `combobox "Model B run"`
  - `region "Model A: circle at step 2"`
  - `region "Model B: circle at step 1"`
  - `group "Arena comparison summary"`
- Selecting `circle at step 1` for Model A updated the summary to:
  - `Both models have the same test loss.`
  - `Both models have the same generalization gap.`
  - `Both models trained for the same number of steps.`
- Compact viewport DOM exposed the same arena region, comboboxes, model regions, thumbnail text alternatives, and comparison summary.

## Console Errors

No console errors were reported by `tab.dev.logs({ levels: ['error'], limit: 20 })` during desktop or compact-viewport checks.

## Screenshots / Recordings

- Desktop full-page: `docs/qa/browser-qa/wave-7-side-by-side-arena.png`
- Desktop scrolled viewport: `docs/qa/browser-qa/wave-7-side-by-side-arena-scrolled.png`
- Compact full-page: `docs/qa/browser-qa/wave-7-side-by-side-arena-compact.png`
- Compact scrolled viewport: `docs/qa/browser-qa/wave-7-side-by-side-arena-compact-scrolled.png`

## Accessibility Notes

- The arena uses a labelled `section` exposed as a region.
- Model selectors are native labelled `select` controls.
- Model comparison panes are labelled regions.
- Sparkline thumbnails keep `role="img"` labels. Arena thumbnails prefix the model side to avoid duplicate accessible names with the saved-run list.
- Empty-thumbnail fallbacks are contextualized in the arena, for example `Model A has no loss history thumbnail`.

## Result

Pass.
