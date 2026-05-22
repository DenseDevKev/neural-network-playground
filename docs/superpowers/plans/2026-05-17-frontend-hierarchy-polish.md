# Frontend Hierarchy Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NN.FORGE workspace easier to scan through a CSS-first hierarchy polish pass without changing app behavior.

**Architecture:** Add a small set of forge-local hierarchy tokens, then apply them consistently to shell surfaces, top bar metrics/actions, configuration controls, topology chrome, output tabs, and the transport/lesson area. Keep state wiring and layout variants intact; add only one markup hook in `NetworkGraphCanvas.tsx` if needed to attach graph controls to a single toolbar surface.

**Tech Stack:** React 19, TypeScript, Vite 6, Vitest, Testing Library, CSS modules by convention through global `index.css` and `forge.css`, Browser QA for rendered validation.

---

## File Structure

- Modify: `apps/web/src/styles/forge.css`
  - Owns shell-level polish tokens, top bar, dock/focus/grid/split surfaces, panel chrome, tab strips, and transport-cluster refinements.
- Modify: `apps/web/src/styles/index.css`
  - Owns shared control primitives, training bar, guided lesson, and network graph chrome that predates the forge shell.
- Modify: `apps/web/src/styles/forgeResponsive.test.ts`
  - Adds text-level guards for hierarchy tokens, active/inactive tab treatment, topology toolbar, and transport/lesson visual boundaries.
- Modify: `apps/web/src/components/visualization/NetworkGraphCanvas.tsx`
  - Optional small markup hook: wrap graph zoom controls and mode toggle in a single toolbar container so they visually attach to the topology panel.
- Modify: `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
  - Verifies the toolbar wrapper preserves existing graph controls and accessible names.
- Create: `docs/qa/browser-qa/frontend-hierarchy-polish.md`
  - Records before/after screenshot baseline, tested URL/config, viewports, console status, and interaction proof.

Do not modify worker protocol, serialization, URL/config format, persistence, run-history data, shared contracts, training behavior, or layout store behavior.

---

## Baseline URL And Viewports

Use the same state for before/after screenshots:

```text
http://127.0.0.1:<vite-port>/#d=xor&pt=classification&r=0.5&n=0&ns=300&s=42&hl=4%2C4&a=tanh&oa=sigmoid&wi=xavier&ws=42&lr=0.03&bs=10&l=crossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000
```

Capture these viewports:

- Desktop: `1280x720`
- Mobile: `390x844`

Save screenshots outside the repo while iterating:

- `/private/tmp/nn-polish-before-desktop.png`
- `/private/tmp/nn-polish-before-mobile.png`
- `/private/tmp/nn-polish-after-desktop.png`
- `/private/tmp/nn-polish-after-mobile.png`

Only copy final QA screenshots into `docs/qa/browser-qa/` if the existing repo convention for that QA entry needs committed images.

---

### Task 1: Add CSS Tests For Hierarchy Tokens And Boundaries

**Files:**
- Modify: `apps/web/src/styles/forgeResponsive.test.ts`

- [ ] **Step 1: Write failing CSS guard tests**

Append these tests inside the existing `describe('forge compact dock CSS', () => { ... })` block:

```ts
    it('defines local hierarchy polish tokens without overriding base tokens', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('--forge-surface-muted');
        expect(css).toContain('--forge-surface-active');
        expect(css).toContain('--forge-border-muted');
        expect(css).toContain('--forge-shadow-muted');
        expect(css).toContain('--forge-control-gap');
    });

    it('keeps topology controls attached to a single graph toolbar surface', () => {
        const css = readFileSync(resolve(__dirname, 'index.css'), 'utf8');

        expect(css).toContain('.network-graph-toolbar');
        expect(css).toContain('.network-graph-toolbar .network-graph-controls');
        expect(css).toContain('.network-graph-toolbar .network-graph-mode-toggle');
    });

    it('keeps guided lesson visually secondary until active lesson state', () => {
        const css = readFileSync(resolve(__dirname, 'index.css'), 'utf8');

        expect(css).toContain('.guided-lesson--active');
        expect(css).toContain('.guided-lesson:not(.guided-lesson--active)');
    });
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/styles/forgeResponsive.test.ts
```

Expected: FAIL because the new hierarchy tokens, topology toolbar selectors, and `guided-lesson--active` styling do not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add apps/web/src/styles/forgeResponsive.test.ts
git commit -m "test: add frontend hierarchy polish guards"
```

---

### Task 2: Add Forge Hierarchy Tokens And Surface Cleanup

**Files:**
- Modify: `apps/web/src/styles/forge.css`
- Test: `apps/web/src/styles/forgeResponsive.test.ts`

- [ ] **Step 1: Add local polish tokens**

Add this block immediately after the opening comment in `apps/web/src/styles/forge.css`, before `.forge-shell`:

```css
/* ─── Hierarchy polish tokens ─────────────────────────────────────
   Scoped through forge surfaces. These complement index.css tokens
   without replacing base design-system values. */
.forge-shell {
    --forge-surface-muted: rgba(16, 18, 22, 0.72);
    --forge-surface-panel: rgba(19, 21, 26, 0.88);
    --forge-surface-active: rgba(129, 236, 255, 0.075);
    --forge-border-muted: rgba(255, 255, 255, 0.055);
    --forge-border-active: rgba(129, 236, 255, 0.28);
    --forge-shadow-muted: 0 8px 22px rgba(0, 0, 0, 0.18);
    --forge-shadow-panel: 0 1px 0 rgba(255, 255, 255, 0.035) inset;
    --forge-control-gap: 8px;
}
```

- [ ] **Step 2: Quiet global shell and dock surfaces**

Update these existing selectors in `apps/web/src/styles/forge.css`:

```css
.forge-shell {
    display: grid;
    grid-template-rows: 52px 1fr 24px;
    height: 100vh;
    overflow: hidden;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.014), transparent 34%),
        var(--bg-base);
    color: var(--text-primary);
}

.forge-dock__left,
.forge-dock__right {
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: var(--forge-surface-muted);
    overflow: hidden;
}

.forge-dock__center {
    grid-column: 3;
    grid-row: 1;
    min-height: 0;
    min-width: 0;
    background:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px) 0 0 / 28px 28px,
        var(--bg-base);
    padding: var(--space-sm);
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    overflow: hidden;
}

.forge-dock__bottom {
    grid-column: 2 / span 3;
    grid-row: 2;
    background: rgba(13, 14, 17, 0.96);
    border-top: 1px solid var(--forge-border-muted);
    padding: 8px var(--space-md);
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    flex-shrink: 0;
    box-shadow: var(--forge-shadow-muted);
}
```

- [ ] **Step 3: Quiet panel chrome and phase tags**

Replace the existing `.forge-panel`, `.forge-panel__head`, and phase-tag rules with:

```css
.forge-panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: var(--forge-surface-panel);
    border: 1px solid var(--forge-border-muted);
    border-radius: var(--radius-md);
    overflow: hidden;
    flex-shrink: 0;
    box-shadow: var(--forge-shadow-panel);
}

.forge-panel__head {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 var(--space-md);
    height: 34px;
    background: rgba(255, 255, 255, 0.018);
    border-bottom: 1px solid var(--forge-border-muted);
    user-select: none;
    flex-shrink: 0;
}

.forge-panel__title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.forge-panel__phase-tag {
    display: inline-flex;
    align-items: center;
    padding: 1px 6px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    flex-shrink: 0;
    color: var(--text-tertiary);
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.forge-panel__phase-tag--build {
    color: var(--color-primary);
}

.forge-panel__phase-tag--run {
    color: var(--color-positive);
}
```

- [ ] **Step 4: Run CSS guard tests**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/styles/forgeResponsive.test.ts
```

Expected: still FAIL on topology toolbar and guided lesson tests, PASS on token test.

- [ ] **Step 5: Commit token and surface cleanup**

```bash
git add apps/web/src/styles/forge.css
git commit -m "style: add forge hierarchy surface tokens"
```

---

### Task 3: Polish Top Bar, Actions, And Left Configuration Rhythm

**Files:**
- Modify: `apps/web/src/styles/forge.css`
- Modify: `apps/web/src/styles/index.css`
- Test: `apps/web/src/styles/forgeResponsive.test.ts`

- [ ] **Step 1: Quiet top bar metrics and preserve Start as primary**

Replace the existing `.forge-topbar`, `.forge-metric`, `.forge-metric__value`, and `@keyframes forge-flash` blocks in `apps/web/src/styles/forge.css` with:

```css
.forge-topbar {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: 0 var(--space-md);
    background: rgba(12, 13, 15, 0.97);
    border-bottom: 1px solid var(--forge-border-muted);
    box-shadow: var(--forge-shadow-muted);
    flex-shrink: 0;
    z-index: 100;
    position: relative;
    overflow: hidden;
}

.forge-metric {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 5px 10px 4px;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.055);
    border-radius: var(--radius-sm);
    min-width: 72px;
    box-shadow: none;
}

.forge-metric__value {
    font-family: var(--font-mono);
    font-size: var(--text-base);
    font-weight: 650;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    transition: color 200ms;
}

.forge-metric__value--primary { color: color-mix(in srgb, var(--color-primary) 82%, white); }
.forge-metric__value--accent  { color: color-mix(in srgb, var(--color-accent) 82%, white); }
.forge-metric__value--updated {
    animation: forge-flash 180ms var(--ease-out);
}

.forge-topbar .btn--play {
    box-shadow: 0 0 18px rgba(188, 135, 254, 0.28);
}

@keyframes forge-flash {
    0%, 100% { transform: scale(1); text-shadow: 0 0 0 transparent; }
    50%       { transform: scale(1.05); text-shadow: 0 0 8px currentColor; }
}
```

- [ ] **Step 2: Normalize left-panel control rhythm**

Append this block near the tab/content section in `apps/web/src/styles/forge.css`, after `.forge-tabs__content`:

```css
.forge-dock__left .forge-tabs__content,
.forge-focus__left .forge-panel__body,
.forge-grid__config .forge-panel__body,
.forge-split__col:first-child .forge-panel__body {
    display: grid;
    gap: var(--forge-control-gap);
}

.forge-dock__left .control-row,
.forge-focus__left .control-row,
.forge-grid__config .control-row,
.forge-split__col:first-child .control-row {
    margin-bottom: 0;
}

.forge-dock__left .chip-group,
.forge-focus__left .chip-group,
.forge-grid__config .chip-group,
.forge-split__col:first-child .chip-group {
    gap: 6px;
}

.forge-dock__left .control-label,
.forge-focus__left .control-label,
.forge-grid__config .control-label,
.forge-split__col:first-child .control-label {
    color: var(--text-tertiary);
}

.forge-dock__left .control-value,
.forge-focus__left .control-value,
.forge-grid__config .control-value,
.forge-split__col:first-child .control-value {
    color: color-mix(in srgb, var(--color-accent) 78%, white);
}
```

- [ ] **Step 3: Reduce secondary action weight without weakening focus**

In `apps/web/src/styles/index.css`, update shared button/chip styles:

```css
.btn--ghost {
    background: rgba(255, 255, 255, 0.018);
    color: var(--text-secondary);
    border: 1px solid rgba(255, 255, 255, 0.075);
}

.btn--ghost:hover {
    background: rgba(255, 255, 255, 0.045);
    color: var(--text-primary);
    border-color: rgba(129, 236, 255, 0.24);
}

.chip {
    display: inline-flex;
    align-items: center;
    height: var(--control-height-sm);
    padding: 0 12px;
    border-radius: 999px;
    font-size: var(--text-xs);
    font-weight: 600;
    cursor: pointer;
    transition: background var(--duration-fast) var(--ease-out),
        color var(--duration-fast) var(--ease-out),
        border-color var(--duration-fast) var(--ease-out),
        box-shadow var(--duration-fast) var(--ease-out);
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.014);
    color: var(--text-secondary);
}

.chip:hover {
    background: rgba(129, 236, 255, 0.055);
    border-color: rgba(129, 236, 255, 0.32);
    color: var(--color-primary);
}

.chip.active {
    background: rgba(129, 236, 255, 0.105);
    border-color: rgba(129, 236, 255, 0.42);
    color: var(--color-primary);
    box-shadow: inset 0 0 0 1px rgba(129, 236, 255, 0.12);
}
```

Keep the existing focus-visible selector unchanged:

```css
.btn:focus-visible,
.chip:focus-visible,
.feature-chip:focus-visible,
.speed-btn:focus-visible,
.chart-tab:focus-visible,
.preset-card:focus-visible,
.panel__header:focus-visible,
.select:focus-visible,
.checkbox-row:focus-within {
    outline: none;
    box-shadow: 0 0 0 2px var(--bg-base), 0 0 0 4px var(--border-focus);
}
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/styles/forgeResponsive.test.ts src/components/layout/Header.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx
```

Expected: Header/Data/Network tests PASS. CSS guard suite still FAILS only on topology toolbar and guided lesson active selectors.

- [ ] **Step 5: Commit top bar and left-panel polish**

```bash
git add apps/web/src/styles/forge.css apps/web/src/styles/index.css
git commit -m "style: polish top bar and config hierarchy"
```

---

### Task 4: Attach Network Topology Controls To A Single Toolbar

**Files:**
- Modify: `apps/web/src/components/visualization/NetworkGraphCanvas.tsx`
- Modify: `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
- Modify: `apps/web/src/styles/index.css`
- Test: `apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx`
- Test: `apps/web/src/styles/forgeResponsive.test.ts`

- [ ] **Step 1: Write toolbar wrapper test**

Add this test after `renders graph viewport controls and updates the zoom label` in `NetworkGraphCanvas.test.tsx`:

```ts
    it('groups graph viewport and mode controls in one attached toolbar', () => {
        const { container } = render(<NetworkGraphCanvas />);

        const toolbar = container.querySelector('.network-graph-toolbar');
        expect(toolbar).not.toBeNull();
        expect(toolbar?.querySelector('.network-graph-controls')).not.toBeNull();
        expect(toolbar?.querySelector('.network-graph-mode-toggle')).not.toBeNull();
        expect(screen.getByRole('button', { name: 'Fit graph to view' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Weights' })).toHaveAttribute('aria-pressed', 'true');
    });
```

- [ ] **Step 2: Run the toolbar test and verify it fails**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/components/visualization/NetworkGraphCanvas.test.tsx -t "groups graph viewport and mode controls"
```

Expected: FAIL because `.network-graph-toolbar` does not exist.

- [ ] **Step 3: Add the toolbar markup hook**

In `NetworkGraphCanvas.tsx`, replace the sibling `network-graph-controls` and `network-graph-mode-toggle` blocks with this wrapper:

```tsx
            <div className="network-graph-toolbar" aria-label="Graph toolbar">
                <div className="network-graph-controls" aria-label="Graph view controls">
                    <button
                        type="button"
                        aria-label="Zoom out graph"
                        title="Zoom out"
                        onClick={() => zoomGraph(-1)}
                    >
                        -
                    </button>
                    <span className="network-graph-controls__zoom">{zoomLabel(viewport.zoom)}</span>
                    <button
                        type="button"
                        aria-label="Zoom in graph"
                        title="Zoom in"
                        onClick={() => zoomGraph(1)}
                    >
                        +
                    </button>
                    <button
                        type="button"
                        aria-label="Fit graph to view"
                        title="Fit graph"
                        onClick={fitGraphToView}
                    >
                        Fit
                    </button>
                </div>

                <div className="network-graph-mode-toggle" role="group" aria-label="Topology view mode">
                    {(['weights', 'activations'] as const).map((mode) => (
                        <button
                            key={mode}
                            type="button"
                            className={viewMode === mode ? 'network-graph-mode-toggle__button network-graph-mode-toggle__button--active' : 'network-graph-mode-toggle__button'}
                            aria-pressed={viewMode === mode}
                            onClick={() => setViewMode(mode)}
                        >
                            {mode === 'weights' ? 'Weights' : 'Activations'}
                        </button>
                    ))}
                </div>
            </div>
```

- [ ] **Step 4: Replace topology control CSS**

In `apps/web/src/styles/index.css`, replace `.network-graph-controls` and `.network-graph-mode-toggle` positioning with:

```css
.network-graph-toolbar {
    position: absolute;
    top: var(--space-sm);
    right: var(--space-sm);
    z-index: 24;
    display: grid;
    gap: 4px;
    justify-items: end;
}

.network-graph-controls,
.network-graph-mode-toggle {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 3px;
    border: 1px solid rgba(255, 255, 255, 0.075);
    border-radius: var(--radius-sm);
    background: rgba(17, 20, 28, 0.72);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.16);
}

.network-graph-mode-toggle {
    position: static;
}
```

Delete the old standalone `position: absolute; right: var(--space-sm); top: 48px;` rule from `.network-graph-mode-toggle`.

- [ ] **Step 5: Quiet layer pills and legend**

Update the existing graph styles:

```css
.network-graph-layer-controls {
    position: absolute;
    z-index: 30;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    transform: translate(-50%, calc(-100% - 10px));
    opacity: 0.72;
}

.network-graph-layer-controls__pill,
.network-graph-ghost-layer__button {
    min-height: 22px;
    padding: 0 8px;
    border: 1px solid rgba(129, 236, 255, 0.22);
    border-radius: 999px;
    background: rgba(18, 24, 35, 0.78);
    color: var(--color-primary);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 800;
    cursor: pointer;
    white-space: nowrap;
    box-shadow: 0 5px 14px rgba(0, 0, 0, 0.14);
}

.network-graph-legend {
    position: absolute;
    left: var(--space-sm);
    bottom: var(--space-sm);
    display: flex;
    align-items: center;
    gap: 10px;
    max-width: calc(100% - var(--space-lg));
    padding: 5px 7px;
    border: 1px solid rgba(255, 255, 255, 0.075);
    border-radius: var(--radius-sm);
    background: rgba(17, 20, 28, 0.72);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--text-secondary);
    font-size: 10px;
    z-index: 20;
}
```

- [ ] **Step 6: Run topology and CSS tests**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/components/visualization/NetworkGraphCanvas.test.tsx src/styles/forgeResponsive.test.ts
```

Expected: Network graph tests PASS. CSS guard suite still FAILS only on guided lesson active selectors.

- [ ] **Step 7: Commit topology toolbar polish**

```bash
git add apps/web/src/components/visualization/NetworkGraphCanvas.tsx apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx apps/web/src/styles/index.css
git commit -m "style: attach topology controls to graph toolbar"
```

---

### Task 5: Polish Output Tabs, Training Controls, And Guided Lesson

**Files:**
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.tsx`
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`
- Modify: `apps/web/src/styles/forge.css`
- Modify: `apps/web/src/styles/index.css`
- Test: `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`
- Test: `apps/web/src/styles/forgeResponsive.test.ts`

- [ ] **Step 1: Add guided lesson active-state test**

In `GuidedLessonPanel.test.tsx`, add a test that starts the lesson and checks the active class. Use existing setup helpers in that file. If there is no helper, add this direct test near the existing behavior tests:

```tsx
    it('marks the drawer active only after a lesson starts', async () => {
        const user = userEvent.setup();
        render(<GuidedLessonPanel onReset={vi.fn()} />);

        const lesson = screen.getByLabelText('Guided lesson mode');
        expect(lesson).not.toHaveClass('guided-lesson--active');

        await user.click(screen.getByRole('button', { name: 'Start guided lesson' }));

        expect(lesson).toHaveClass('guided-lesson--active');
    });
```

- [ ] **Step 2: Run the guided lesson test and verify it fails**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/components/controls/GuidedLessonPanel.test.tsx -t "marks the drawer active"
```

Expected: FAIL because the component does not set `guided-lesson--active`.

- [ ] **Step 3: Add active class hook**

In `GuidedLessonPanel.tsx`, change the `<aside>` className to:

```tsx
            className={`guided-lesson ${activeStep ? 'guided-lesson--active' : ''} ${isDrawerOpen ? 'guided-lesson--open' : 'guided-lesson--collapsed'}`}
```

- [ ] **Step 4: Quiet shared tab strips**

In `apps/web/src/styles/forge.css`, replace `.forge-tabs`, `.forge-tab`, `.forge-tab:hover`, `.forge-tab--active`, and dot active styles with:

```css
.forge-tabs {
    display: flex;
    align-items: stretch;
    background: rgba(255, 255, 255, 0.012);
    border-bottom: 1px solid var(--forge-border-muted);
    padding: 0 var(--space-sm);
    gap: 0;
    flex-shrink: 0;
    overflow-x: auto;
    scrollbar-width: none;
    min-height: 34px;
}

.forge-tab {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0 10px;
    height: 32px;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    color: rgba(255, 255, 255, 0.48);
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: color var(--duration-fast) var(--ease-out),
                background var(--duration-fast) var(--ease-out),
                border-color var(--duration-fast) var(--ease-out);
}

.forge-tab:hover {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.018);
}

.forge-tab--active {
    color: var(--text-primary);
    border-bottom-color: var(--color-primary);
    background: transparent;
}

.forge-tab--active .forge-tab__dot {
    background: var(--color-primary);
    box-shadow: none;
}
```

- [ ] **Step 5: Quiet training container while preserving play action**

In `apps/web/src/styles/forge.css`, replace `.forge-transport-cluster .training-bar` with:

```css
.forge-transport-cluster .training-bar {
    height: 100%;
    min-height: 74px;
    border: 1px solid var(--forge-border-muted);
    background: rgba(9, 10, 12, 0.42);
    border-radius: var(--radius-md);
    padding: 8px 10px;
    box-shadow: var(--forge-shadow-panel);
}
```

In `apps/web/src/styles/index.css`, update speed buttons:

```css
.speed-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: var(--control-height-md);
    padding: 0 10px;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.075);
    background: rgba(255, 255, 255, 0.014);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: background var(--duration-fast) var(--ease-out),
        color var(--duration-fast) var(--ease-out),
        border-color var(--duration-fast) var(--ease-out);
    white-space: nowrap;
}

.speed-btn:hover {
    border-color: rgba(188, 135, 254, 0.36);
    color: var(--color-accent);
}

.speed-btn.active {
    background: rgba(188, 135, 254, 0.1);
    border-color: rgba(188, 135, 254, 0.42);
    color: var(--color-accent);
}
```

- [ ] **Step 6: Make guided lesson contextual**

In `apps/web/src/styles/index.css`, replace the guided lesson base/active styles with:

```css
.guided-lesson {
    position: relative;
    min-width: 0;
    border: 1px solid rgba(255, 255, 255, 0.075);
    border-radius: var(--radius-md);
    background: rgba(12, 12, 13, 0.74);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    overflow: hidden;
    transition: border-color var(--duration-fast) var(--ease-out),
        background var(--duration-fast) var(--ease-out),
        box-shadow var(--duration-fast) var(--ease-out);
}

.guided-lesson:not(.guided-lesson--active) .guided-lesson__eyebrow,
.guided-lesson:not(.guided-lesson--active) .guided-lesson__meta {
    color: var(--text-tertiary);
}

.guided-lesson--active {
    border-color: rgba(234, 179, 8, 0.34);
    background:
        linear-gradient(135deg, rgba(234, 179, 8, 0.095), transparent 44%),
        rgba(12, 12, 13, 0.9);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 0 0 1px rgba(234, 179, 8, 0.05);
}

.guided-lesson--active .guided-lesson__eyebrow {
    color: var(--color-warning);
}
```

Keep `.guided-lesson__toggle:hover`, `.guided-lesson__step-chip`, and `.lesson-target--active` warning accents strong because they indicate active lesson context.

- [ ] **Step 7: Run focused tests**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/components/controls/GuidedLessonPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/components/layout/UIFlows.integration.test.tsx src/styles/forgeResponsive.test.ts
```

Expected: PASS.

- [ ] **Step 8: Commit output/transport polish**

```bash
git add apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/index.css
git commit -m "style: polish outputs and lesson hierarchy"
```

---

### Task 6: Rendered QA, Baseline Comparison, And Final Fixes

**Files:**
- Create: `docs/qa/browser-qa/frontend-hierarchy-polish.md`
- Modify: `apps/web/src/styles/forge.css` if responsive fixes are needed
- Modify: `apps/web/src/styles/index.css` if responsive fixes are needed

- [ ] **Step 1: Start the app**

Run:

```bash
pnpm dev --host 127.0.0.1
```

Expected: Vite reports a local URL. If sandbox blocks localhost with `listen EPERM`, rerun the same command with approval. If `5173` is occupied, use the port Vite selects.

- [ ] **Step 2: Capture before/after screenshots on the same state**

Use the baseline URL from this plan with the active Vite port:

```text
http://127.0.0.1:<vite-port>/#d=xor&pt=classification&r=0.5&n=0&ns=300&s=42&hl=4%2C4&a=tanh&oa=sigmoid&wi=xavier&ws=42&lr=0.03&bs=10&l=crossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000
```

Capture:

- Desktop `1280x720`
- Mobile `390x844`

Expected:

- `NN·FORGE` is visible.
- `XOR` is selected.
- Network Topology is visibly the anchor.
- Start is the strongest action.
- No horizontal body overflow.
- No framework overlay.

- [ ] **Step 3: Exercise one training interaction**

In the rendered app:

1. Click the transport Start/Resume button.
2. Verify it changes to Pause or Resume state.
3. Pause it.
4. Verify Step/Reset remain available when not blocked.

Expected: behavior matches pre-polish flow; no console errors. Dev-only `[perf] Slow interaction` warnings may appear and should be documented rather than treated as app failure.

- [ ] **Step 4: Check focus, hover, disabled, and readout legibility**

Use keyboard Tab and pointer hover to check:

- Topbar layout segmented controls.
- Left panel dataset chips.
- Sliders.
- Graph toolbar buttons.
- Speed buttons.
- Guided lesson toggle/start.

Expected:

- Focus ring remains visible.
- Disabled states remain distinguishable.
- Metric/readout text remains legible.
- No text clips inside buttons.

- [ ] **Step 5: Fix any visual regressions**

If rendered QA reveals clipping, overlap, hidden focus, or poor contrast, make only targeted CSS fixes in the relevant file. Do not add new behavior or redesign the layout.

After each fix, rerun:

```bash
pnpm --filter @nn-playground/web test -- src/styles/forgeResponsive.test.ts
```

Expected: PASS.

- [ ] **Step 6: Create QA note**

Create `docs/qa/browser-qa/frontend-hierarchy-polish.md`:

```md
# Frontend Hierarchy Polish Browser QA

Date: 2026-05-17

## State

- URL: `http://127.0.0.1:<vite-port>/#d=xor&pt=classification&r=0.5&n=0&ns=300&s=42&hl=4%2C4&a=tanh&oa=sigmoid&wi=xavier&ws=42&lr=0.03&bs=10&l=crossEntropy&o=sgd&m=0.9&rg=none&rr=0&f=110000000`
- Desktop viewport: `1280x720`
- Mobile viewport: `390x844`

## Checks

- Page identity: PASS
- Nonblank render: PASS
- Framework overlay: PASS
- Console errors: PASS
- Console warnings: documented if dev-only perf observer warnings appear
- Interaction proof: PASS
- Focus/hover/disabled/readout legibility: PASS
- Horizontal overflow: PASS

## Notes

- Start/Resume/Pause remains the only primary action.
- Network Topology reads as the visual anchor.
- Left configuration panel reads as a clearer setup flow.
- Right output tabs read as one shared output system.
- Guided Lesson is calmer by default and stronger when active.
```

- [ ] **Step 7: Commit QA note and final visual fixes**

```bash
git add docs/qa/browser-qa/frontend-hierarchy-polish.md apps/web/src/styles/forge.css apps/web/src/styles/index.css
git commit -m "docs: record frontend hierarchy polish qa"
```

---

### Task 7: Final Verification

**Files:**
- Verify only; no planned edits.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pnpm --filter @nn-playground/web test -- src/styles/forgeResponsive.test.ts src/components/layout/Header.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/visualization/NetworkGraphCanvas.test.tsx src/components/layout/UIFlows.integration.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run production build**

Run:

```bash
pnpm build
```

Expected: PASS. Vite may warn about the existing large app chunk; document it if unchanged.

- [ ] **Step 3: Check git diff**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

- Only intentional tracked changes remain.
- `.superpowers/` may be untracked from the visual companion and should not be committed.
- Recent commits show the test, styling, topology, output/lesson, and QA commits.

- [ ] **Step 4: Final review against spec**

Open `docs/superpowers/specs/2026-05-17-frontend-hierarchy-polish-design.md` and confirm:

- Behavior/state wiring preserved.
- No new dependencies.
- Start/Resume/Pause strongest.
- Topology anchor.
- Left panel clearer.
- Output tabs sibling system.
- Guided lesson contextual.
- Desktop/mobile screenshots have no clipping, overlap, unreadable text, or horizontal overflow.

Expected: All success criteria satisfied.
