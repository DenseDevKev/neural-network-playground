# Mobile training metrics and Share setup fixes — October 2, 2026

Both issues reproduced in the [cloud reliability pass](2026-10-02-cloud-reliability-pass.md) are fixed. This implementation starts from `81044f6` and changes only the compact transport layout and the Share setup entry point, with regression coverage and documentation.

## Changes

- On phones, Step and Epoch each span multiple grid columns and may wrap as their values grow. Status occupies its own full-width row. Complete values and the 44px action controls remain visible.
- Share setup selects **Setup & sharing** inside the existing draft guard. Stay preserves the draft and previous export selection; Discard and Apply open sharing after their existing behavior completes. Generic Utilities access still remembers its tab and TF.js selection.

## Verification

The new browser regressions ran against the unchanged application first: 10 failed on the reproduced overlap or wrong selected tab, and the two Stay cases passed. With the fixes, all 12 focused cases pass in Chromium and WebKit, with zero retries. Mobile cases use real training, both themes, running and paused states, Results and Setup views, and 320/360/390px widths. They check text intersections, status overflow, touch targets, page overflow, and model identity. Sharing cases cover no draft, Stay, Discard, and Apply.

| Check on the final implementation | Result |
| --- | --- |
| Unit/integration tests | 1,838 passed |
| Script helper tests | 156 passed |
| Type checks and lint | Passed |
| Normal production build | Passed |
| Focused browser regressions | 12 passed; no skips or flaky tests |
| Full standard browser suite | 164 passed, 10 mode-specific skips; no failures or flaky tests |
| Gzip budgets | Entry 150615 / 152245; InspectionPanel 6590 / 7373; total JavaScript 233119 / 234161 bytes |

The standard suite includes the existing layout, zoom, accessibility, save/retry, and concurrent-save checks. Its ten skips are four opt-in gallery cases, four external-hosting cases, and two fault-enabled cases. Those separate modes passed on the prior baseline; they were not repeated for these two fixes. [Browser result statistics and skipped test names](mobile-share-fixes-2026-10-02/browser-results.json) distinguish the failing reproduction, focused passing run, and full passing run.

The supplementary strict UI audit reports the same 49 pre-existing flags before and after the change, with no new findings; it is not a passing strict audit. An independent Astra review of the implementation and regression tests found no actionable issues.

## Captures and limits

- [WebKit, light theme, paused phone transport](mobile-share-fixes-2026-10-02/phone-webkit-light.png).
- [Chromium, dark theme, paused phone transport](mobile-share-fixes-2026-10-02/phone-chromium-dark.png).

These are cropped captures of the real trained transport at a 390px viewport. Full logs, the before/after UI audit, and archived browser reports are under `/workspace/outputs/mobile-share-fixes/`. The cloud browsers emulate phone viewports; physical devices and the live deployment were not tested. This report does not claim performance qualification or publication to main.
