# QA Checklist

Use this checklist for local verification, browser smoke tests, accessibility passes, and release handoffs. It complements automated tests; it does not replace them.

## Canonical Commands

Run from the repository root:

```bash
pnpm --filter @nn-playground/web exec tsc --noEmit
pnpm test
pnpm lint
pnpm build
```

For performance-sensitive or runtime-adjacent changes, also run:

```bash
pnpm test:perf
```

For checked-in Chromium and WebKit smoke coverage, build first and then run:

```bash
pnpm build
pnpm test:e2e
```

Release browser runs must use zero retries. When diagnosing one engine, retain
the same assertions and run it explicitly:

```bash
pnpm exec playwright test --project=chromium --retries=0
pnpm exec playwright test --project=webkit --retries=0
```

Useful targeted checks:

```bash
pnpm --filter @nn-playground/web test
pnpm --filter @nn-playground/shared test
pnpm --filter @nn-playground/engine test
pnpm --filter @nn-playground/web dev --host 127.0.0.1
```

## Browser QA Modes

- Mode A: checked-in Playwright QA through `pnpm test:e2e` (Chromium and WebKit).
- Mode B: agent-assisted browser QA with the Codex Browser plugin or equivalent visible browser tooling.
- Mode C: pending human verification when no browser tooling is available.

Do not claim browser QA passed unless Mode A or Mode B was actually executed.

## Browser Smoke Checklist

- App opens at the local dev URL without a blank page.
- No framework error overlay is visible.
- Browser console has no errors.
- Dataset controls can be opened and changed.
- A preset can be selected or a lesson can apply a preset.
- Training can start and pause.
- Step and reset controls respond.
- Decision boundary is visible after data exists.
- Loss panel is reachable.
- Network visualization is visible.
- Confusion, inspection, code export, history, and lesson surfaces remain reachable.
- Changed UI can be reached by mouse.
- Changed UI can be reached by keyboard.
- Focus remains visible and predictable after changed interactions.
- Beginner, Explore, and Lab expose exactly their documented core tools.
- Advanced Tools reveals the full applicable union without duplicating controls.
- Opening Advanced Tools alone does not request inspection/confusion artifacts.
- Profile/disclosure changes preserve the URL/hash, recipe, training/model step,
  checkpoint timeline, saved-run count, and selected code-export tab.
- Hidden configured values remain summarized and have a clear Advanced Tools
  recovery path.

## Responsive / Compact Checklist

- Check at 320px and at a compact viewport such as 390x844.
- Repeat the primary flow at 200% browser zoom or equivalent text enlargement.
- Dock layout remains usable.
- Unsupported layout buttons are disabled rather than silently failing.
- Tabs and action buttons do not overflow their containers.
- Training controls remain reachable.
- Guided lesson drawer can expand/collapse.
- No text visibly overlaps adjacent controls.
- No page-level horizontal overflow hides profile or disclosure controls.
- Pointer targets remain at least 44 by 44 CSS pixels where the design requires
  compact touch interaction.

## Accessibility Checklist

- Prefer semantic HTML controls before custom ARIA.
- New buttons use `type="button"` unless submitting a form.
- Button accessible names are action-oriented and concise.
- Extra educational copy uses descriptions or nearby text, not oversized labels.
- Keyboard activation works for mouse-clickable features.
- Global shortcuts do not fire while a focused control handles Enter, Space, or text input.
- Canvas-heavy surfaces include text alternatives or adjacent summaries.
- Status and error changes are announced through existing live regions.
- Dynamic metric updates should not create noisy live announcements.
- Reduced-motion users should not receive nonessential animation effects.
- Add `jest-axe` coverage for new accessible surfaces when practical.
- Landmark and heading order describes one coherent workspace.
- Disclosure controls expose correct expanded state and controlled-region IDs.
- Roving evidence tabs support Arrow keys, Home, and End with one tab stop.
- Closing drawers, popovers, and Advanced Tools restores focus predictably.
- Profile changes use a concise polite announcement without noisy metric updates.
- Text and controls meet contrast requirements, and status is not conveyed only
  through color.
- Validation errors identify the affected field through text and programmatic
  association.
- Critical instructions and definitions remain available without hover.

## Regression Sweep

After each wave, confirm:

- Prior Wave 1 explanation action cards still render and focus existing panels.
- Prior Wave 2 lessons can still start from the selector.
- Training workflow still runs.
- Presets still apply.
- Dataset changes still reset/rebuild safely.
- URL sharing and import/export are not touched unless the slice explicitly changes them.
- Worker protocol, frame buffer, SharedArrayBuffer, WebGPU, persistence, and serialization remain unchanged unless approved.

## Screenshot / Evidence Expectations

Record browser QA in `docs/qa/browser-qa/`.

Include:

- Date.
- Commit or pending commit.
- URL.
- Browser/QA mode.
- Exact steps.
- Expected and actual results.
- Console error status.
- Screenshot paths if captured.
- Accessibility notes.
- Browser engine/version, viewport, DPR, zoom/text size, input method, and
  reduced-motion preference.
- Before/after state invariants for navigation-only changes.
- Playwright retry count and report/trace path for automated runs.
- Host-load or tooling contamination notes when timing evidence is suspect.
- Pass, fail, or pending human verification result.

## Browser Support Policy

The app is a static Vite/React app intended for modern evergreen desktop and mobile browsers. Browser QA should prioritize:

- Chromium and WebKit through the checked-in Playwright suite for release smoke.
- Chromium-based browser via the Codex Browser plugin for visible/manual checks.
- Compact mobile-like viewport for dense layout regressions.
- Static deployment compatibility with GitHub Pages/base path behavior.

Use semantic roles, accessible names, and observable application state in
Playwright. Do not mask failures with retries, arbitrary sleeps, browser skips,
timeout inflation, or weakened assertions. A fast `curl` response does not prove
the browser completed a scenario; retain Playwright reports and traces as the
authoritative failure evidence.
