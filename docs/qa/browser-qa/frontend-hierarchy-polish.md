# Frontend Hierarchy Polish QA

Date: 2026-05-17
Branch: `codex/frontend-hierarchy-polish`

## Target State

URL:

```text
http://127.0.0.1:5173/#d=xor&pt=classification&r=0.5&n=0&ns=300&s=42&hl=4%2C4&a=tanh&oa=sigmoid&wi=xavier&ws=42&lr=0.03&bs=10&l=crossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000
```

Intended screenshot baseline:

- Desktop: `1280x720`
- Mobile: `390x844`

## Browser QA

Browser path: Codex in-app Browser was available and used for local app navigation, DOM inspection, console inspection, and the training interaction loop.

Screenshot status: blocked by environment. `tab.screenshot({ fullPage: false })` timed out with `Page.captureScreenshot` on the local app, and OS `screencapture` produced a black frame. No screenshot artifacts were committed. Validation used DOM/browser state plus automated responsive/style guards instead of visual bitmap evidence.

Observed page identity:

- Title: `Neural Network Playground 2.0`
- URL matched the target XOR hash above.
- DOM snapshot included `NN·FORGE`, `Training metrics`, `Configuration panels`, `Network Topology`, and `Guided lesson`.
- No Vite/framework error overlay was present.
- Current-app console filter for `127.0.0.1:5173` showed no warnings or errors after the Start/Pause interaction.

Interaction proof:

- Scoped the primary training control to `.training-bar button[aria-label="Start training"]`.
- Clicked Start.
- Confirmed `.training-bar button[aria-label="Pause training"]` appeared.
- Clicked Pause.
- Confirmed the training bar returned to `Resume training`.
- Confirmed speed controls remained accessible with `aria-pressed` state, including `5 steps per frame` as active.

## Verification Commands

```bash
pnpm --filter @nn-playground/web exec vitest run src/styles/forgeResponsive.test.ts src/components/layout/Header.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/visualization/NetworkGraphCanvas.test.tsx src/components/layout/UIFlows.integration.test.tsx src/App.test.tsx --pool=forks --reporter=dot --passWithNoTests
```

Result: 9 files passed, 83 tests passed.

```bash
pnpm test
```

Result: 70 files passed, 925 tests passed.

```bash
pnpm lint
```

Result: passed.

```bash
pnpm build
```

Result: passed. Vite emitted the existing chunk-size warning for the main bundle.

## Notes

- `forgeResponsive.test.ts` now verifies hierarchy tokens, topology toolbar selectors, guided-lesson active/inactive selectors, and compact dock behavior.
- The browser screenshot gap should be revisited in a follow-up environment where `Page.captureScreenshot` works, because the implementation plan explicitly calls for before/after bitmap review.
