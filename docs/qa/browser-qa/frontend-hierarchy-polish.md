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

## Second Polish Pass

Date: 2026-05-18

Files touched:

- `apps/web/src/styles/forge.css`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/frontend-hierarchy-polish.md`

What changed:

- Reduced duplicated active navigation by making the left rail a quieter context indicator while preserving the local tab state as the clearer section cue.
- Reworked the split run-phase banner into neutral workflow status: lower contrast, no glow, no warning-like orange frame.
- Quieted panel headers by lowering grip, title, phase-tag, border, and header-surface intensity.
- Improved topology anchoring in dock, focus, grid, and split contexts with a calmer panel emphasis, hidden nested body overflow, softer graph toolbar, quieter summary badges, and lower-contrast legend/filter chrome.
- Reduced left configuration “button soup” by lowering inactive chip/feature-chip contrast and making active choices clear without heavy filled pills.
- Quieted right output chrome by softening output tabs, chart tabs, loading/empty states, and Confusion Matrix framing.
- Refined lower controls into a coherent secondary strip by grouping Step/Reset, speed, and timeline controls with subtle shared surfaces while keeping Start/Pause/Resume as the only dominant action.

Second-pass browser QA:

- In-app Browser was attempted first, including visibility recovery, but remained blocked with `No active Codex browser pane available`.
- Chrome extension fallback loaded the target URL successfully at `1500x971`.
- Screenshot capture was still blocked: Chrome fallback returned `Page.captureScreenshot returned no data`. No screenshot artifacts were committed.
- Mobile viewport override was unavailable through the fallback browser. Mobile coverage remains the automated compact/responsive CSS guard plus build/test validation.
- DOM/browser identity confirmed: title `Neural Network Playground 2.0`, target XOR hash URL, `NN·FORGE`, `Configuration panels`, `Network Topology`, and `Guided lesson` present.
- Local console filter for `127.0.0.1:5173` returned no warnings or errors. Chrome-extension React perf warnings were ignored as non-app-local noise.
- Interaction proof passed: Start training -> Pause training -> Resume training -> Pause back to a stable paused state.
- Grid topology review: topology panel measured `738x395`, body overflow was `hidden`, and toolbar/summary/legend surfaces used subdued borders/backgrounds.
- Split topology review: topology panel measured `823x373`, body overflow was `hidden`, and graph toolbar stayed attached to the graph surface.
- Run banner review: `Run phase — observe training` measured `189x30`, used neutral border/background, and had no box shadow.
- Left control review: active chip retained clear cyan state while inactive chips fell back to subtle border/background treatment.
- Focus/readout review: `:focus-visible` rules were present, top metric/readout text remained visible, and the secondary training strip controls were grouped under low-contrast shared surfaces.

Second-pass verification commands:

```bash
pnpm --filter @nn-playground/web exec vitest run src/styles/forgeResponsive.test.ts src/components/layout/Header.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/visualization/NetworkGraphCanvas.test.tsx src/components/layout/UIFlows.integration.test.tsx src/App.test.tsx --pool=forks --reporter=dot --passWithNoTests
```

Result: 9 files passed, 83 tests passed.

```bash
pnpm lint
```

Result: passed.

```bash
pnpm test
```

Result: 70 files passed, 925 tests passed.

```bash
pnpm build
```

Result: passed. Vite emitted the existing chunk-size warning for the main bundle.

## Focused Flaw-Fix Pass

Date: 2026-05-18

Files touched:

- `apps/web/src/styles/forge.css`
- `apps/web/src/styles/index.css`
- `docs/qa/browser-qa/frontend-hierarchy-polish.md`

What changed:

- Compressed the lower training region by preventing the transport cluster from stretching the training bar to the guided lesson height.
- Tightened mobile training controls into a denser existing-control grid so Start remains primary while Step, Reset, speed, timeline, Restore, and status read as one secondary strip.
- Reworked the topology toolbar into a compact attached graph-edge control with two rows, then moved the architecture summary below it and narrowed the summary frame to avoid fighting layer controls.
- Quieted the Start shortcut treatment so the label dominates; the `Space` hint remains metadata and still follows the existing shortcut behavior.
- Raised small-text legibility through the shared tertiary text token and targeted readout/empty-state/legend color mixes without restoring heavy badges or borders.

Focused browser QA:

- In-app Browser was available at `http://localhost:5174/`.
- Desktop dock screenshot captured: `/private/tmp/nn-polish-flawfix-desktop.png`.
- Mobile screenshot captured: `/private/tmp/nn-polish-flawfix-mobile.png`.
- Desktop dock measured no horizontal overflow: `scrollWidth=1280` at `1280x720`.
- Desktop dock bottom zone measured `155px` high, with the training bar no longer stretched to the guided lesson height.
- Desktop dock, split, and grid topology checks showed no overlap between `.network-graph-toolbar` and layer controls.
- Grid topology measured graph `592x219`; split topology measured graph `567x278`; both kept the toolbar attached and away from graph layer shortcuts.
- Mobile measured no horizontal overflow: `scrollWidth=390` at `390x844`.
- Mobile bottom zone measured `160px` high with a compact `86px` training bar and collapsed guided lesson below it.
- Mobile topology toolbar measured `133x62`, attached to the graph edge and clear of the `+ neuron` controls.
- Start -> Pause -> Resume -> Pause interaction passed. Labels advanced through `Start`, `Pause`, `Resume`, and back to paused state while training metrics updated.
- Local console filter for `localhost:5174` returned no warnings or errors.
- Focus/readout review passed: primary action retained visible focus treatment, footer/timeline/legend text stayed readable, disabled/hover/focus selector rules remained intact, and shortcut text no longer read as a nested button.

Focused flaw-fix verification commands:

```bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/TrainingControls.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/visualization/NetworkGraphCanvas.test.tsx src/components/layout/UIFlows.integration.test.tsx src/App.test.tsx --pool=forks --reporter=dot --passWithNoTests
```

Result: 5 files passed, 51 tests passed.

```bash
pnpm lint
```

Result: passed.

```bash
pnpm test
```

Result: 70 files passed, 925 tests passed.

```bash
pnpm build
```

Result: passed. Vite emitted the existing chunk-size warning for the main bundle.

Skipped fixes:

- No behavior, worker protocol, URL/config, serialization, persistence, lazy loading, graph interaction, accessibility label, keyboard shortcut, dependency, or layout-mode changes were made.
- No markup hooks were needed; the flaw-fix pass stayed CSS-first.
- No new fake timeline, console, or placeholder content was added to fill the bottom zone.
