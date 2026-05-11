# QA Checklist

Use this checklist for local verification, browser smoke tests, accessibility passes, and release handoffs. It complements automated tests; it does not replace them.

## Canonical Commands

Run from the repository root:

```bash
pnpm test
pnpm lint
pnpm build
```

For performance-sensitive or runtime-adjacent changes, also run:

```bash
pnpm test:perf
```

Useful targeted checks:

```bash
pnpm --filter @nn-playground/web test
pnpm --filter @nn-playground/shared test
pnpm --filter @nn-playground/engine test
pnpm --filter @nn-playground/web dev --host 127.0.0.1
```

## Browser QA Modes

- Mode A: automated browser QA if a checked-in browser automation command exists.
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

## Responsive / Compact Checklist

- Check at a compact width such as 390x844 or the Browser plugin viewport override.
- Dock layout remains usable.
- Unsupported layout buttons are disabled rather than silently failing.
- Tabs and action buttons do not overflow their containers.
- Training controls remain reachable.
- Guided lesson drawer can expand/collapse.
- No text visibly overlaps adjacent controls.

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
- Pass, fail, or pending human verification result.

## Browser Support Policy

The app is a static Vite/React app intended for modern evergreen desktop and mobile browsers. Browser QA should prioritize:

- Chromium-based browser via the Codex Browser plugin for local smoke checks.
- Compact mobile-like viewport for dense layout regressions.
- Static deployment compatibility with GitHub Pages/base path behavior.

Do not add Playwright or other browser dependencies unless a future approved slice justifies the dependency, lockfile impact, maintenance cost, and rollback plan.
