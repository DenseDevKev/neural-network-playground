# NN Playground Thirty Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify all thirty approved quick wins across correctness, onboarding, accessibility, resilience, performance, and release validation.

**Architecture:** Keep the current React/Zustand/worker boundaries and deliver each win as a focused, independently reviewed TDD slice. Shared definitions are introduced only where a later task consumes the same contract; browser and CI work exercise the built production app rather than replacing unit coverage.

**Tech Stack:** React 19, TypeScript 5.7, Zustand 5, Vite 6, Vitest 3, Testing Library, Playwright 1.61, Node 20+, pnpm 9+, ESLint 9.

## Global Constraints

- The controlling design is docs/superpowers/specs/2026-08-10-thirty-quick-wins-design.md.
- Preserve the unrelated Scientific Trust plan edit, V0.1 BUILD DOCU.md, responsive-polish draft, and entire prototypes directory.
- Preserve engine math, worker scientific authority, experiment schema versions, checkpoint guarantees, compatibility behavior, and recoverable saved-run bytes.
- Audience profiles change visible tools and guidance only; no task may change the experiment because a profile or workspace view changed.
- Add no production dependency. A test-only dependency is allowed only when a numbered acceptance criterion cannot be met with the existing stack.
- Use existing CSS tokens, component patterns, accessible names, and responsive breakpoints. Add no new design system or parallel shell.
- Every production behavior change follows RED, GREEN, REFACTOR. The implementer report must contain the failing and passing command output.
- Configuration tasks use a failing executable gate as RED and the same successful gate as GREEN.
- Keep Playwright retries at zero and do not weaken assertions, performance budgets, accessibility checks, or timeouts.
- Stage and commit only files named by the active task.

---

## Correctness and green gates

### Task 1: Protect global training shortcuts

**Quick win:** 1 — Safe global shortcuts.

**Files:**
- Create: apps/web/src/shortcuts/trainingShortcuts.ts
- Create: apps/web/src/shortcuts/trainingShortcuts.test.ts
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx

**Interfaces:**
- Produces TRAINING_SHORTCUTS, a readonly list containing Space, ArrowRight, and KeyR with labels and descriptions for Task 16.
- Produces resolveTrainingShortcut(event), returning play-pause, step, reset, or null after rejecting repeat, modifier, and editable-target events.

- [ ] **Step 1: Write failing shortcut-resolution tests**

~~~ts
expect(resolveTrainingShortcut(keyboardEvent('KeyR', { metaKey: true }))).toBeNull();
expect(resolveTrainingShortcut(keyboardEvent('KeyR', { ctrlKey: true }))).toBeNull();
expect(resolveTrainingShortcut(keyboardEvent('KeyR', { altKey: true }))).toBeNull();
expect(resolveTrainingShortcut(keyboardEvent('KeyR', { shiftKey: true }))).toBeNull();
expect(resolveTrainingShortcut(keyboardEvent('KeyR', { repeat: true }))).toBeNull();
expect(resolveTrainingShortcut(keyboardEvent('KeyR'))).toBe('reset');
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/App.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the shortcut module does not exist and App still intercepts modified R.

- [ ] **Step 3: Implement the resolver and route App through it**

~~~ts
export function resolveTrainingShortcut(event: KeyboardEvent): TrainingShortcutAction | null {
    if (event.repeat || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return null;
    if (isEditableShortcutTarget(event.target)) return null;
    return SHORTCUT_ACTION_BY_CODE[event.code] ?? null;
}
~~~

App prevents default only after the resolver returns an action and dispatches exactly one matching training command.

- [ ] **Step 4: Run GREEN and web typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec tsc --noEmit

Expected: PASS with pristine output.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/shortcuts/trainingShortcuts.ts apps/web/src/shortcuts/trainingShortcuts.test.ts apps/web/src/App.tsx apps/web/src/App.test.tsx
git commit -m "fix(web): protect global training shortcuts"
~~~

### Task 2: Make worker stream setup retry-safe

**Quick win:** 2 — Retry-safe stream setup.

**Files:**
- Modify: apps/web/src/worker/workerBridge.ts
- Modify: apps/web/src/worker/workerBridge.test.ts

**Interfaces:**
- Preserves setupStreamChannel(): Promise<void>.
- Installs the module stream port only after setStreamPort succeeds; a failed attempt owns and closes both temporary ports.

- [ ] **Step 1: Write a rejection-then-retry regression test**

~~~ts
setStreamPort.mockRejectedValueOnce(new Error('transfer failed')).mockResolvedValueOnce(undefined);
await expect(setupStreamChannel()).rejects.toThrow('transfer failed');
await expect(setupStreamChannel()).resolves.toBeUndefined();
expect(setStreamPort).toHaveBeenCalledTimes(2);
~~~

Also assert the first channel ports are closed and its port never receives a listener.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/worker/workerBridge.test.ts --pool=forks --reporter=dot

Expected: FAIL because the first rejected setup leaves _streamPort non-null and suppresses the retry.

- [ ] **Step 3: Keep the channel local until installation succeeds**

~~~ts
const channel = new MessageChannel();
try {
    await api.setStreamPort(Comlink.transfer(channel.port2, [channel.port2]));
    installStreamPort(channel.port1);
} catch (error) {
    channel.port1.close();
    channel.port2.close();
    throw error;
}
~~~

Guard against a concurrent successful setup before assigning the local port.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/worker/workerBridge.test.ts --pool=forks --reporter=dot

Expected: PASS with the retry and existing termination tests green.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/worker/workerBridge.ts apps/web/src/worker/workerBridge.test.ts
git commit -m "fix(web): make stream setup retry safe"
~~~

### Task 3: Preserve error fallback until async retry succeeds

**Quick win:** 3 — Truthful async error recovery.

**Files:**
- Modify: apps/web/src/components/common/ErrorBoundary.tsx
- Modify: apps/web/src/components/common/ErrorBoundary.test.tsx

**Interfaces:**
- ErrorBoundaryProps.onRetry becomes () => void | Promise<void>.
- The boundary exposes retry progress and retains an error message when retry rejects.

- [ ] **Step 1: Write failing async retry tests**

~~~tsx
const retry = vi.fn().mockRejectedValue(new Error('reset failed'));
await user.click(screen.getByRole('button', { name: 'Try again' }));
expect(await screen.findByText(/reset failed/i)).toBeInTheDocument();
expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
~~~

Add the successful promise case and assert the fallback clears only after resolution.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/ErrorBoundary.test.tsx --pool=forks --reporter=dot

Expected: FAIL because handleRetry clears synchronously and does not handle rejection.

- [ ] **Step 3: Await retry and retain actionable state on failure**

~~~ts
handleRetry = async () => {
    this.setState({ retrying: true, retryError: null });
    try {
        await this.props.onRetry?.();
        this.setState({ error: null, retrying: false, retryError: null });
    } catch (error) {
        this.setState({ retrying: false, retryError: toErrorMessage(error) });
    }
};
~~~

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/ErrorBoundary.test.tsx --pool=forks --reporter=dot

Expected: PASS with no unhandled-rejection noise.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/common/ErrorBoundary.tsx apps/web/src/components/common/ErrorBoundary.test.tsx
git commit -m "fix(web): await error boundary recovery"
~~~

### Task 4: React to hash navigation after startup

**Quick win:** 4 — Live URL navigation.

**Files:**
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/store/usePlaygroundStore.test.ts

**Interfaces:**
- Reuses usePlaygroundStore.getState().loadFromUrl().
- One App lifecycle effect owns the hashchange listener and removes it on unmount.

- [ ] **Step 1: Write failing valid, invalid, and cleanup tests**

~~~ts
window.location.hash = validExperimentHash;
window.dispatchEvent(new HashChangeEvent('hashchange'));
await waitFor(() => expect(usePlaygroundStore.getState().access.status).toBe('ready'));
~~~

Add an invalid hash assertion for compatibility state and an unmount assertion proving later events do not call the loader.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/App.test.tsx src/store/usePlaygroundStore.test.ts --pool=forks --reporter=dot

Expected: FAIL because production reads location only during module initialization.

- [ ] **Step 3: Add the lifecycle listener**

~~~ts
useEffect(() => {
    const loadLocation = () => { void usePlaygroundStore.getState().loadFromUrl(); };
    window.addEventListener('hashchange', loadLocation);
    return () => window.removeEventListener('hashchange', loadLocation);
}, []);
~~~

Do not add a popstate listener unless a failing browser-backed test proves hashchange is insufficient for the supported hash URL format.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/App.test.tsx src/store/usePlaygroundStore.test.ts --pool=forks --reporter=dot

Expected: PASS for valid, incompatible, and cleanup paths.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/store/usePlaygroundStore.test.ts
git commit -m "fix(web): load experiments on hash navigation"
~~~

### Task 5: Clear stale configuration errors on success

**Quick win:** 5 — No stale configuration error.

**Files:**
- Modify: apps/web/src/components/controls/ConfigPanel.tsx
- Modify: apps/web/src/components/controls/ConfigPanel.test.tsx

**Interfaces:**
- reportSuccess(message) clears the prior local error before setting timed success.

- [ ] **Step 1: Write three error-then-success tests**

~~~tsx
await produceConfigError(user);
expect(screen.getByRole('alert')).toBeInTheDocument();
await completeSuccessfulCopy(user);
expect(screen.queryByRole('alert')).not.toBeInTheDocument();
expect(screen.getByRole('status')).toHaveTextContent('URL copied');
~~~

Repeat from a fresh rendered error state for successful JSON import and successful JSON export. Assert the old alert disappears and the operation-specific success status appears in each case; do not cover the shared helper only through copy.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/ConfigPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the prior error remains rendered beside success.

- [ ] **Step 3: Clear error inside reportSuccess**

~~~ts
const reportSuccess = useCallback((message: string) => {
    if (!mounted.current) return;
    setError(null);
    setStatus(message);
}, [setStatus]);
~~~

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/ConfigPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS for copy, import, and export feedback.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/controls/ConfigPanel.tsx apps/web/src/components/controls/ConfigPanel.test.tsx
git commit -m "fix(web): clear stale config feedback"
~~~

### Task 6: Disclose lesson reset before starting

**Quick win:** 6 — Lesson reset disclosure.

**Files:**
- Modify: apps/web/src/components/controls/GuidedLessonPanel.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.test.tsx

**Interfaces:**
- The pre-start lesson state renders persistent consequence copy and the action label Start lesson and reset.

- [ ] **Step 1: Write a failing accessible-copy test**

~~~tsx
expect(screen.getByText(/replaces the current recipe and resets training/i)).toBeVisible();
expect(screen.getByRole('button', { name: 'Start lesson and reset' })).toBeEnabled();
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the button is named Start guided lesson and no visible reset note exists.

- [ ] **Step 3: Add visible consequence copy and rename the action**

~~~tsx
<p className="guided-lesson__consequence">
    Changes: replaces the current recipe and resets training. Preserves: saved runs.
</p>
~~~

Use the exact button accessible name Start lesson and reset.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS without changing startLesson behavior.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx
git commit -m "feat(web): disclose lesson reset"
~~~

### Task 7: Name saved runs

**Quick win:** 7 — Human-readable run names.

**Files:**
- Create: apps/web/src/components/controls/runTitle.ts
- Create: apps/web/src/components/controls/runTitle.test.ts
- Modify: apps/web/src/components/controls/RunHistoryPanel.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.test.tsx

**Interfaces:**
- Produces createDefaultRunTitle(recipe, snapshot): string with dataset label, compact architecture, and step.
- saveCurrentRun passes a trimmed optional title to captureRunArtifact; blank input uses the deterministic default.

- [ ] **Step 1: Write failing default-title and persistence tests**

~~~ts
expect(createDefaultRunTitle(circleRecipe, step400Snapshot)).toBe('Circle · 2-4-4-1 · step 400');
~~~

In the component test enter XOR baseline, save, remount with the persisted record, and assert the article is named XOR baseline.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/runTitle.test.ts src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the helper and title input do not exist.

- [ ] **Step 3: Add the title field and default helper**

~~~tsx
<label>
    <span>Run name</span>
    <input value={runTitle} onChange={(event) => setRunTitle(event.currentTarget.value)} />
</label>
~~~

Pass title: runTitle.trim() || createDefaultRunTitle(prepared.document.recipe, snapshot) while honoring the existing schema title limit.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/runTitle.test.ts src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS for custom, blank-default, bounded, and remount cases.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/controls/runTitle.ts apps/web/src/components/controls/runTitle.test.ts apps/web/src/components/controls/RunHistoryPanel.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx
git commit -m "feat(web): add names to saved runs"
~~~

### Task 8: Add a repository-wide typecheck gate

**Quick win:** 8 — One repository typecheck.

**Files:**
- Modify: package.json
- Modify: apps/web/package.json
- Create: apps/web/tsconfig.test.json
- Modify: packages/engine/package.json
- Modify: packages/engine/tsconfig.json
- Modify: packages/shared/tsconfig.json
- Modify: packages/shared/package.json
- Modify: packages/engine/src/__benchmarks__/grid_performance.bench.ts
- Modify: packages/engine/src/__benchmarks__/performance.bench.ts
- Modify: packages/shared/src/__tests__/codeExport.test.ts
- Modify: packages/shared/src/__tests__/sessionCheckpoint.test.ts
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx
- Modify: apps/web/src/__tests__/frameBuffer.test.ts
- Modify: apps/web/src/components/controls/CodeExportPanel.test.tsx
- Modify: apps/web/src/components/controls/ConfigPanel.test.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx
- Modify: apps/web/src/components/controls/FeaturesPanel.test.tsx
- Modify: apps/web/src/components/controls/HyperparamPanel.test.tsx
- Modify: apps/web/src/components/controls/InspectionPanel.test.tsx
- Modify: apps/web/src/components/controls/PresetPanel.test.tsx
- Modify: apps/web/src/components/controls/RecipeSummaryCard.test.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.test.tsx
- Modify: apps/web/src/components/controls/TrainingControls.test.tsx
- Modify: apps/web/src/components/controls/inspection/useInspectionPanelController.test.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/components/layout/MainArea.test.tsx
- Modify: apps/web/src/components/layout/UIFlows.integration.test.tsx
- Modify: apps/web/src/components/visualization/ConfusionMatrix.test.tsx
- Modify: apps/web/src/components/visualization/DecisionBoundary.test.tsx
- Modify: apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx
- Modify: apps/web/src/components/visualization/NetworkGraphSVG.test.tsx
- Modify: apps/web/src/components/visualization/TrainingExplanationPanel.test.tsx
- Modify: apps/web/src/components/visualization/decisionBoundaryModel.test.ts
- Modify: apps/web/src/components/visualization/useDecisionBoundaryModel.test.tsx
- Modify: apps/web/src/hooks/useTraining.test.tsx
- Modify: apps/web/src/lessons/lessonRegistry.test.ts
- Modify: apps/web/src/store/usePlaygroundStore.test.ts
- Modify: apps/web/src/worker/frameBuffer.test.ts
- Modify: apps/web/src/worker/sharedSnapshot.test.ts
- Modify: apps/web/src/worker/training.worker.v2.test.ts
- Modify: apps/web/src/worker/workerBridge.test.ts
- Modify: apps/web/src/test/jest-axe.d.ts
- Modify: apps/web/src/test/playgroundStoreTestUtils.ts
- Modify: apps/web/src/test/scientificTrustFixtures.ts
- Modify: .github/workflows/ci.yml
- Modify: pnpm-lock.yaml

**Interfaces:**
- Produces pnpm typecheck, checking web, engine, shared, and their tests/benchmarks.
- CI exposes Typecheck as a named gate before unit tests.
- Scope amendment (2026-08-11): the first strict web test program exposed 96
  pre-existing semantic diagnostics across 31 test files. Those test-only
  fixture repairs are part of this task because 22 files have no later owner.
  Keep strict checking and the full include set; do not use `noCheck`, new
  exclusions, test-only production-interface augmentation, blanket `any`
  casts, or weakened production contracts to manufacture a green gate.

- [ ] **Step 1: Capture the current failing executable RED gate**

Run: pnpm --filter @nn-playground/engine exec tsc --noEmit

Expected: FAIL on missing performance and console globals.

Run: pnpm --filter @nn-playground/shared exec tsc --noEmit

Expected: FAIL on stale NetworkSnapshot, missing node:vm types, and readonly shuffledIndices mutation.

Create only the strict `apps/web/tsconfig.test.json` harness described in Step 2,
then capture both halves directly before repairing any fixtures.

Run: pnpm --filter @nn-playground/web exec tsc --noEmit -p tsconfig.json

Expected: PASS for production source.

Run: pnpm --filter @nn-playground/web exec tsc --noEmit -p tsconfig.test.json

Expected: FAIL with the captured 96-test-diagnostic baseline.

- [ ] **Step 2: Add root typecheck orchestration and correct package contracts**

~~~json
{
  "typecheck": "pnpm -r --if-present typecheck"
}
~~~

Each workspace package defines a `typecheck` script. Web runs its existing production config plus `tsc --noEmit -p tsconfig.test.json`; the test config includes `src/**/*.test.ts`, `src/**/*.test.tsx`, and `src/test/**/*` with Vitest, Vite, jest-dom, DOM, and Node types. Add `@types/node` as an explicit development dependency in web, engine, and shared. Import performance from node:perf_hooks and console from node:console in benchmarks, update the stale snapshot fixture to the current contract, and construct a mutable checkpoint fixture instead of mutating readonly production data.

Remediate the web diagnostics at their test-fixture sources. Prefer current V2
store/evidence helpers and typed protocol builders over repeated legacy state
writes. Repair branded fingerprints through validated scientific fixtures,
construct malformed readonly artifacts before assigning their protocol type,
narrow discriminated unions once, and type local mocks against their actual
browser/Atomics contracts. Keep the worker WebGPU test bound to the repository's
existing local device contract rather than adding broad ambient globals. Update
the stale guided-lesson integration assertion so the full runtime suite accepts
the Task 6 action name `Start lesson and reset`.

- [ ] **Step 3: Add the named CI step**

~~~yaml
      - name: Typecheck
        run: pnpm typecheck
~~~

- [ ] **Step 4: Run GREEN**

Run: pnpm typecheck

Expected: PASS across all workspaces with no TypeScript errors.

Run: pnpm --filter @nn-playground/web typecheck

Expected: PASS for the production config and every included web test/helper file.

Run: pnpm test

Expected: PASS with existing and amended fixtures.

- [ ] **Step 5: Commit**

~~~bash
git add package.json apps/web/package.json apps/web/tsconfig.test.json packages/engine/package.json packages/engine/tsconfig.json packages/shared/package.json packages/shared/tsconfig.json packages/engine/src/__benchmarks__/grid_performance.bench.ts packages/engine/src/__benchmarks__/performance.bench.ts packages/shared/src/__tests__/codeExport.test.ts packages/shared/src/__tests__/sessionCheckpoint.test.ts apps/web/src/App.test.tsx apps/web/src/__tests__/appShell.integration.test.tsx apps/web/src/__tests__/frameBuffer.test.ts apps/web/src/components/controls/CodeExportPanel.test.tsx apps/web/src/components/controls/ConfigPanel.test.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/FeaturesPanel.test.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/components/controls/InspectionPanel.test.tsx apps/web/src/components/controls/PresetPanel.test.tsx apps/web/src/components/controls/RecipeSummaryCard.test.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/components/controls/inspection/useInspectionPanelController.test.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/components/layout/MainArea.test.tsx apps/web/src/components/layout/UIFlows.integration.test.tsx apps/web/src/components/visualization/ConfusionMatrix.test.tsx apps/web/src/components/visualization/DecisionBoundary.test.tsx apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx apps/web/src/components/visualization/NetworkGraphSVG.test.tsx apps/web/src/components/visualization/TrainingExplanationPanel.test.tsx apps/web/src/components/visualization/decisionBoundaryModel.test.ts apps/web/src/components/visualization/useDecisionBoundaryModel.test.tsx apps/web/src/hooks/useTraining.test.tsx apps/web/src/lessons/lessonRegistry.test.ts apps/web/src/store/usePlaygroundStore.test.ts apps/web/src/worker/frameBuffer.test.ts apps/web/src/worker/sharedSnapshot.test.ts apps/web/src/worker/training.worker.v2.test.ts apps/web/src/worker/workerBridge.test.ts apps/web/src/test/jest-axe.d.ts apps/web/src/test/playgroundStoreTestUtils.ts apps/web/src/test/scientificTrustFixtures.ts .github/workflows/ci.yml pnpm-lock.yaml
git commit -m "build: add repository typecheck gate"
~~~

### Task 9: Restore green root lint

**Quick win:** 9 — Green repository lint.

**Files:**
- Modify: eslint.config.js

**Interfaces:**
- Node .mjs tests receive Node globals without granting Node globals to browser source.

- [ ] **Step 1: Run the failing lint gate**

Run: pnpm lint

Expected: FAIL with four URL is not defined errors in the two prototype Node tests.

- [ ] **Step 2: Add the narrow Node-test environment**

~~~js
{
    files: ['prototypes/**/tests/**/*.mjs'],
    languageOptions: { globals: globals.node },
}
~~~

Use the ESLint override. Do not edit the untracked prototype tests.

- [ ] **Step 3: Run GREEN and a browser-global negative check**

Run: pnpm lint

Expected: PASS with zero errors and zero warnings.

Run: pnpm exec eslint apps/web/src

Expected: PASS while still using browser rather than Node globals.

- [ ] **Step 4: Commit**

~~~bash
git add eslint.config.js
git commit -m "build: scope prototype lint globals"
~~~

### Task 10: Make grid performance evidence robust and complete

**Quick win:** 10 — Trustworthy grid performance gate.

**Files:**
- Create: packages/engine/src/__benchmarks__/performanceStatistics.ts
- Create: packages/engine/src/__benchmarks__/performanceStatistics.test.ts
- Modify: packages/engine/src/__benchmarks__/grid_performance.bench.ts
- Modify: docs/perf/PERFORMANCE_BASELINE.md

**Interfaces:**
- Produces median(values), measureMedianMsPerIteration(run, options), and assertPerformanceBudgets(results), which reports every over-budget path in one failure.
- The four existing prediction APIs retain their current caps unless controlled evidence and code profiling justify an implementation optimization or a documented baseline change.

- [ ] **Step 1: Write failing statistics and aggregate-failure tests**

~~~ts
expect(median([9, 1, 5, 3, 7])).toBe(5);
expect(() => assertPerformanceBudgets([
    { name: 'predictGrid', measured: 12, limit: 11 },
    { name: 'predictGridInto', measured: 13, limit: 10 },
])).toThrow(/predictGrid.*predictGridInto/s);
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/engine exec vitest run src/__benchmarks__/performanceStatistics.test.ts --pool=forks --reporter=dot

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement median sampling and aggregate reporting**

~~~ts
export function median(values: readonly number[]): number {
    if (values.length === 0) throw new Error('median requires at least one value');
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle];
}
~~~

Measure at least five warmed rounds per path and collect all four results before one aggregate budget assertion.

- [ ] **Step 4: Run helper GREEN and controlled performance samples**

Run: pnpm --filter @nn-playground/engine exec vitest run src/__benchmarks__/performanceStatistics.test.ts --pool=forks --reporter=dot

Run three times: pnpm --filter @nn-playground/engine test:perf

Expected: helper PASS; each controlled run measures and reports all four paths. If a path remains over budget, profile and optimize the path before proceeding; do not merely raise its cap.

- [ ] **Step 5: Run engine unit tests and commit**

Run: pnpm --filter @nn-playground/engine test

Expected: PASS.

~~~bash
git add packages/engine/src/__benchmarks__/performanceStatistics.ts packages/engine/src/__benchmarks__/performanceStatistics.test.ts packages/engine/src/__benchmarks__/grid_performance.bench.ts docs/perf/PERFORMANCE_BASELINE.md
git commit -m "test(engine): stabilize grid performance evidence"
~~~

## Onboarding, usability, and accessibility

### Task 11: Add a first-visit lesson cue

**Quick win:** 11 — First-visit lesson cue.

**Files:**
- Modify: apps/web/src/store/useLayoutStore.ts
- Modify: apps/web/src/store/useLayoutStore.test.ts
- Create: apps/web/src/components/controls/FirstVisitLessonCue.tsx
- Create: apps/web/src/components/controls/FirstVisitLessonCue.test.tsx
- Modify: apps/web/src/components/layout/BuildRunShell.tsx
- Modify: apps/web/src/components/layout/BuildRunShell.test.tsx

**Interfaces:**
- useLayoutStore persists `lessonCueDismissed` and monotonic `hasStartedLesson` without adding either to shared experiment URLs. Calling the existing `setActiveLessonStep` marks `hasStartedLesson` true; finishing or clearing a lesson never resets it.
- FirstVisitLessonCue receives hasSavedRuns, hasStartedLesson, hasActiveLesson, onOpenLessons, and onDismiss; it renders only when all first-visit conditions are true.

- [ ] **Step 1: Write failing visibility and persistence tests**

~~~tsx
render(<FirstVisitLessonCue hasSavedRuns={false} hasStartedLesson={false} hasActiveLesson={false} onOpenLessons={open} onDismiss={dismiss} />);
expect(screen.getByRole('button', { name: 'Start a 3-minute lesson' })).toBeVisible();
~~~

Add cases for a saved run, active lesson, previously started/finished lesson, and persisted dismissal, each of which hides the cue. Store tests must prove `hasStartedLesson` survives persistence sanitization and stays true after `clearActiveLessonStep()`.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/FirstVisitLessonCue.test.tsx src/store/useLayoutStore.test.ts --pool=forks --reporter=dot

Expected: FAIL because the cue and persisted flag do not exist.

- [ ] **Step 3: Implement the bounded cue and shell integration**

~~~tsx
if (lessonCueDismissed || hasSavedRuns || hasStartedLesson || hasActiveLesson) return null;
return <aside aria-label="Getting started">...</aside>;
~~~

Place it beside Current Recipe in Build view and open the existing Lessons surface without changing audience mode or recipe state.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/FirstVisitLessonCue.test.tsx src/store/useLayoutStore.test.ts src/components/layout/BuildRunShell.test.tsx --pool=forks --reporter=dot

Expected: PASS for fresh, dismissed, active-lesson, and existing-user states.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/store/useLayoutStore.ts apps/web/src/store/useLayoutStore.test.ts apps/web/src/components/controls/FirstVisitLessonCue.tsx apps/web/src/components/controls/FirstVisitLessonCue.test.tsx apps/web/src/components/layout/BuildRunShell.tsx apps/web/src/components/layout/BuildRunShell.test.tsx
git commit -m "feat(web): add first visit lesson cue"
~~~

### Task 12: Give every lesson an observable action

**Quick win:** 12 — Observable lesson steps.

**Files:**
- Modify: apps/web/src/lessons/types.ts
- Modify: apps/web/src/lessons/lessonRegistry.ts
- Modify: apps/web/src/lessons/lessonRegistry.test.ts
- Modify: apps/web/src/components/controls/GuidedLessonPanel.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.test.tsx

**Interfaces:**
- LessonStep adds required tryThis: string and optional completion: LessonCompletionRule.
- LessonCompletionRule is a closed union derived from state already available to GuidedLessonPanel, initially training-step-at-least and view-is.
- Next remains enabled regardless of completion.

- [ ] **Step 1: Write failing registry-contract and UI tests**

~~~ts
for (const lesson of LESSON_DEFINITIONS) {
    for (const step of lesson.steps) expect(step.tryThis.trim()).not.toBe('');
}
~~~

Render a training completion rule before and after its threshold; assert Try this copy and Done appear after satisfaction while Next remains enabled in both states.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/lessons/lessonRegistry.test.ts src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because tryThis and completion rendering do not exist.

- [ ] **Step 3: Extend lesson definitions and derive completion without a new store**

~~~ts
export type LessonCompletionRule =
    | { kind: 'training-step-at-least'; step: number }
    | { kind: 'view-is'; view: 'build' | 'run' };
~~~

Populate every existing step with concise action or observation copy. Add completion only where the current view or training step can prove it.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/lessons/lessonRegistry.test.ts src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS, with no step trapping navigation.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/lessons/types.ts apps/web/src/lessons/lessonRegistry.ts apps/web/src/lessons/lessonRegistry.test.ts apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx
git commit -m "feat(web): add observable lesson actions"
~~~

### Task 13: Let users choose the two runs to compare

**Quick win:** 13 — Explicit run-pair comparison.

**Files:**
- Create: apps/web/src/components/controls/runComparisonSelection.ts
- Create: apps/web/src/components/controls/runComparisonSelection.test.ts
- Modify: apps/web/src/components/controls/RunHistoryPanel.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.test.tsx

**Interfaces:**
- Produces reconcileComparisonSelection(selectedIds, records, toggledId), returning at most two existing record IDs in stable user order.
- SavedRunComparison receives the two selected records explicitly and names both in its heading.

- [ ] **Step 1: Write failing selection reducer tests**

~~~ts
expect(reconcileComparisonSelection(['a'], records, 'b')).toEqual(['a', 'b']);
expect(reconcileComparisonSelection(['a', 'b'], records, 'c')).toEqual(['b', 'c']);
expect(reconcileComparisonSelection(['a', 'missing'], records, null)).toEqual(['a']);
~~~

Component tests select two nonadjacent runs, delete one, and preserve the existing incomparable warning.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/runComparisonSelection.test.ts src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because comparison always consumes records zero and one.

- [ ] **Step 3: Add two-run selection controls**

~~~tsx
<input
    type="checkbox"
    checked={selectedIds.includes(record.id)}
    onChange={() => setSelectedIds((ids) => reconcileComparisonSelection(ids, records, record.id))}
/>
~~~

Use the human record label in each control and comparison heading.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/runComparisonSelection.test.ts src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS for selection cap, deletion reconciliation, labels, and comparability guard.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/controls/runComparisonSelection.ts apps/web/src/components/controls/runComparisonSelection.test.ts apps/web/src/components/controls/RunHistoryPanel.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx
git commit -m "feat(web): choose runs for comparison"
~~~

### Task 14: Explain Build and Run at the switch

**Quick win:** 14 — Build/Run explanation.

**Files:**
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx

**Interfaces:**
- Workspace view references a persistent or help-triggered description available by pointer, focus, and touch.
- Switching view retains its existing state-only callback.

- [ ] **Step 1: Write a failing accessible-description test**

~~~tsx
expect(screen.getByRole('group', { name: 'Workspace view' })).toHaveAccessibleDescription(
    'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.',
);
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the view group has no explanation.

- [ ] **Step 3: Attach concise description and visible help**

Use the exact tested copy in an element referenced by aria-describedby and expose it through the existing touch-safe help pattern.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx --pool=forks --reporter=dot

Expected: PASS and existing view-switch mutation tests remain green.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx
git commit -m "feat(web): explain build and run views"
~~~

### Task 15: Rename Mode to Workspace

**Quick win:** 15 — Unambiguous workspace label.

**Files:**
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/productShell/audienceProfiles.test.ts

**Interfaces:**
- Visible label and accessible name become Workspace profile.
- Stored audience values beginner, explore, and lab remain unchanged.

- [ ] **Step 1: Write the failing label and invariance test**

~~~tsx
expect(screen.getByRole('combobox', { name: 'Workspace profile' })).toHaveValue('explore');
await user.selectOptions(screen.getByRole('combobox', { name: 'Workspace profile' }), 'lab');
expect(onAudienceModeChange).toHaveBeenCalledWith('lab');
expect(onExperimentChange).not.toHaveBeenCalled();
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the current accessible name is Audience mode and visible label is Mode.

- [ ] **Step 3: Update labels without changing enum or persistence keys**

Render Workspace as the visible label, Workspace profile as the accessible name, and Profiles change visible tools and guidance only as the descriptive copy.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/productShell/audienceProfiles.test.ts --pool=forks --reporter=dot

Expected: PASS with the same stored values and state behavior.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/productShell/audienceProfiles.test.ts
git commit -m "feat(web): clarify workspace profiles"
~~~


### Task 16: Expose a shortcut reference

**Quick win:** 16 — Discoverable shortcut reference.

**Files:**
- Modify: apps/web/src/shortcuts/trainingShortcuts.ts
- Modify: apps/web/src/shortcuts/trainingShortcuts.test.ts
- Modify: apps/web/src/components/controls/TrainingControls.tsx
- Modify: apps/web/src/components/controls/TrainingControls.test.tsx
- Modify: apps/web/src/styles/forge.css
- Modify: apps/web/src/styles/forgeResponsive.test.ts

**Interfaces:**
- Consumes TRAINING_SHORTCUTS from Task 1 as the sole list of shortcut labels and descriptions.
- Renders a native details disclosure named Keyboard shortcuts that remains visible at compact breakpoints.

- [ ] **Step 1: Write failing shared-list and responsive UI tests**

~~~tsx
await user.click(screen.getByText('Keyboard shortcuts'));
expect(screen.getByText('Space')).toBeVisible();
expect(screen.getByText('Play or pause training')).toBeVisible();
expect(screen.getByText('R')).toBeVisible();
~~~

Assert the rendered shortcut count equals TRAINING_SHORTCUTS.length.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/components/controls/TrainingControls.test.tsx --pool=forks --reporter=dot

Expected: FAIL because no disclosure exists.

- [ ] **Step 3: Render the shared definitions in a compact disclosure**

~~~tsx
<details className="training-shortcuts">
    <summary>Keyboard shortcuts</summary>
    <dl>{TRAINING_SHORTCUTS.map(renderShortcutDefinition)}</dl>
</details>
~~~

Do not hide .training-shortcuts in the max-width 900px rules that hide inline badges.

- [ ] **Step 4: Run GREEN and static responsive test**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/components/controls/TrainingControls.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Expected: PASS with definitions and responsive visibility aligned.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/shortcuts/trainingShortcuts.ts apps/web/src/shortcuts/trainingShortcuts.test.ts apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts
git commit -m "feat(web): expose training shortcuts"
~~~

### Task 17: Add meaningful range labels and values

**Quick win:** 17 — Meaningful range controls.

**Files:**
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx

**Interfaces:**
- Train ratio input has a stable ID, label, output association, and value text such as 70 percent training, 30 percent test.
- Noise input has a stable ID, label, output association, and value text such as 15 percent noise.

- [ ] **Step 1: Write failing accessible-value tests**

~~~tsx
expect(screen.getByRole('slider', { name: 'Train ratio' })).toHaveAttribute(
    'aria-valuetext',
    '50 percent training, 50 percent test',
);
expect(screen.getByRole('slider', { name: 'Noise' })).toHaveAttribute('aria-valuetext', '0 percent noise');
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/DataPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the sliders use standalone aria-label values without associated outputs or value text.

- [ ] **Step 3: Add label, output, and value text**

~~~tsx
<label htmlFor={trainRatioId}>Train ratio</label>
<output id={trainRatioOutputId} htmlFor={trainRatioId}>{trainPercent}%</output>
<input id={trainRatioId} aria-describedby={trainRatioOutputId} aria-valuetext={`${trainPercent} percent training, ${testPercent} percent test`} />
~~~

Mirror the structure for noise.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/DataPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS for initial and changed values.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx
git commit -m "feat(web): improve data slider semantics"
~~~

### Task 18: Stop live-announcing every metric update

**Quick win:** 18 — Quiet metric announcements.

**Files:**
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/components/layout/AccessibilityAnnouncer.tsx
- Modify: apps/web/src/components/layout/AccessibilityAnnouncer.test.tsx

**Interfaces:**
- Header metric group remains named Training metrics but loses status and aria-live semantics.
- AccessibilityAnnouncer retains one announcement per meaningful transition: training start/reset lifecycle, pause, configuration start, successful configuration completion, and new configuration error. A loading flag transition from true to false with no error is the completion signal; unchanged rerenders do not repeat a message.

- [ ] **Step 1: Write a failing live-region boundary test**

~~~tsx
const metrics = screen.getByLabelText('Training metrics');
expect(metrics).not.toHaveAttribute('role', 'status');
expect(metrics).not.toHaveAttribute('aria-live');
~~~

Add transition-table tests proving `Training started`, `Training paused`, `Training reset`, configuration start, configuration completion, and configuration error are each written once for one transition and are not rewritten on an unchanged rerender. Cover at least data plus one non-data configuration scope so the generic completion logic is exercised.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/components/layout/AccessibilityAnnouncer.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the metric group is currently one polite status region.

- [ ] **Step 3: Remove metric live semantics and preserve meaningful completion announcements**

Keep aria-label="Training metrics" and ordinary readable text; do not add per-value live regions. In AccessibilityAnnouncer, compare previous and current loading flags in one deterministic priority order. Announce `${scope} update complete` only for true-to-false transitions when the matching operation has no current configuration error, then update all refs so rerenders cannot repeat it.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/components/layout/AccessibilityAnnouncer.test.tsx --pool=forks --reporter=dot

Expected: PASS with the metric group quiet and lifecycle, configuration start/completion, pause, reset, and error transitions announced exactly once.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/components/layout/AccessibilityAnnouncer.tsx apps/web/src/components/layout/AccessibilityAnnouncer.test.tsx
git commit -m "fix(web): quiet live metric announcements"
~~~

### Task 19: Contain focus in the worker error modal

**Quick win:** 19 — Complete worker-error modal.

**Files:**
- Create: apps/web/src/hooks/useModalFocusContainment.ts
- Create: apps/web/src/hooks/useModalFocusContainment.test.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx

**Interfaces:**
- useModalFocusContainment(active, dialogRef, backgroundRef) focuses the dialog, traps Tab and Shift+Tab, toggles inert plus aria-hidden on background, and restores prior focus on deactivation.
- The worker alertdialog remains named and described by its current title and description.

- [ ] **Step 1: Write failing focus-cycle and background tests**

~~~tsx
triggerWorkerError();
expect(dialog).toHaveFocus();
expect(workspace).toHaveAttribute('inert');
await user.tab();
expect(within(dialog).getByRole('button', { name: 'Refresh page' })).toHaveFocus();
~~~

Add Shift+Tab wrapping and prior-focus restoration after clearing the error.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useModalFocusContainment.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Expected: FAIL because only initial container focus exists.

- [ ] **Step 3: Implement the focused hook and background wrapper**

~~~ts
useEffect(() => {
    if (!active) return;
    const previous = document.activeElement as HTMLElement | null;
    const background = backgroundRef.current;
    background?.setAttribute('inert', '');
    background?.setAttribute('aria-hidden', 'true');
    return () => {
        background?.removeAttribute('inert');
        background?.removeAttribute('aria-hidden');
        previous?.focus();
    };
}, [active, backgroundRef]);
~~~

The keydown handler cycles only focusable elements inside dialogRef.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useModalFocusContainment.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Expected: PASS for forward, reverse, background, and restoration behavior.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/hooks/useModalFocusContainment.ts apps/web/src/hooks/useModalFocusContainment.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx
git commit -m "feat(web): contain worker error focus"
~~~

### Task 20: Enlarge compact graph and evidence controls

**Quick win:** 20 — Mobile graph and evidence targets.

**Files:**
- Modify: apps/web/src/styles/forge.css
- Modify: apps/web/src/styles/forgeResponsive.test.ts
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- At 390 by 844, graph toolbar buttons, topology-mode buttons, edge filters, and decision-overlay buttons have at least 44px bounding width and height.
- At that viewport the document and forge shell retain no horizontal overflow.

- [ ] **Step 1: Add failing compact target assertions**

Add a focused `390px graph and evidence targets` browser test with viewport 390 by 844. Its target inventory includes Zoom out graph, Zoom in graph, Fit graph to view, Weights, Activations, all four edge filters, and Output, Uncertain, Errors, Split.

~~~ts
for (const target of graphAndEvidenceTargets) await expectMinimumTouchTarget(target);
~~~

Add a static CSS test requiring a compact selector block for these controls.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "390px graph and evidence targets"

Expected: FAIL because current visible controls measure 24 to 28px high.

- [ ] **Step 3: Add compact hit-area styles without scaling graph content**

~~~css
@media (max-width: 900px) {
    .network-graph-toolbar button,
    .network-graph-legend button,
    .decision-boundary__overlay-controls button {
        min-width: 44px;
        min-height: 44px;
    }
}
~~~

Use the actual existing class selectors found in the components; preserve visual label density with padding and wrapping rather than transforms.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "390px graph and evidence targets"

Expected: PASS in both browsers with every target at least 44 by 44 and no overflow at 390 by 844.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts tests/e2e/playground-smoke.spec.ts
git commit -m "fix(web): enlarge compact graph controls"
~~~

### Task 21: Keep the primary outcome reachable on compact screens

**Quick win:** 21 — Compact mobile outcome disclosure.

**Files:**
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/styles/forge.css
- Modify: apps/web/src/styles/forgeResponsive.test.ts

**Interfaces:**
- At max-width 900px, expose one compact native disclosure named `Evaluation outcome` while the desktop metric cluster remains unchanged.
- The disclosure summary names the full-evaluation step and primary held-out outcome: test accuracy for classification when available, otherwise test data loss. Its body includes train data loss, test data loss, accuracy when applicable, and evaluation provenance.
- Missing evidence is reported as `Not evaluated yet`; the disclosure never substitutes the live batch EMA for full-split evidence.

- [ ] **Step 1: Write failing semantic and compact-CSS tests**

~~~tsx
expect(screen.getByRole('group', { name: 'Evaluation outcome' })).toHaveTextContent('Test accuracy 87.5%');
expect(screen.getByRole('group', { name: 'Evaluation outcome' })).toHaveTextContent('Full evaluation at step 40');
~~~

Cover regression/test-loss fallback and the no-evaluation state. Add a static responsive assertion that the disclosure is hidden above 900px, visible at the compact breakpoint, and does not add fixed width.

- [ ] **Step 2: Run the failing coverage-gate RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Expected: FAIL because compact view currently hides the complete metric cluster with no replacement.

- [ ] **Step 3: Add the compact disclosure from existing evidence**

~~~tsx
<details className="forge-compact-outcome" aria-label="Evaluation outcome">
    <summary>{primaryOutcomeLabel}</summary>
    <span>{fullEvaluation ? `Full evaluation at step ${fullEvaluation.step}` : 'Not evaluated yet'}</span>
</details>
~~~

Derive strings from the same `fullEvaluation` values as the desktop metrics. Keep the summary concise at 320px and allow the details body to wrap.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Expected: PASS for classification, regression, empty evidence, and responsive visibility.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts
git commit -m "feat(web): surface compact evaluation outcome"
~~~

### Task 22: Explain learning rate, train/test split, and epoch in context

**Quick win:** 22 — Three high-frequency learner concepts.

**Files:**
- Modify: apps/web/src/concepts/conceptCatalog.ts
- Modify: apps/web/src/concepts/conceptCatalog.test.ts
- Modify: apps/web/src/components/controls/HyperparamPanel.tsx
- Modify: apps/web/src/components/controls/HyperparamPanel.test.tsx
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx
- Modify: apps/web/src/components/controls/TrainingControls.tsx
- Modify: apps/web/src/components/controls/TrainingControls.test.tsx

**Interfaces:**
- Extend `ConceptId` with `learning-rate`, `train-test-split`, and `epoch`.
- Every entry includes a plain definition, extended explanation, aliases, at least one example, related concepts, difficulty, profiles, and an existing valid UI target where applicable.
- Place `ConceptHelp` beside the visible Learning rate label, Train ratio label, and Epoch label; guidance remains controlled by the current audience profile.

- [ ] **Step 1: Write failing catalog and placement tests**

~~~ts
for (const id of ['learning-rate', 'train-test-split', 'epoch'] as const) {
    const concept = getConceptById(id);
    expect(concept?.plainDefinition).toBeTruthy();
    expect(concept?.examples?.length).toBeGreaterThan(0);
    expect(concept?.related.length).toBeGreaterThan(0);
}
~~~

In each component test, open the adjacent help control and assert the matching canonical term plus plain definition.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/concepts/conceptCatalog.test.ts src/components/controls/HyperparamPanel.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/TrainingControls.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the concept IDs and contextual help controls are absent.

- [ ] **Step 3: Add scientifically bounded copy and contextual controls**

Define learning rate as update scale, split as fixed membership used to separate fitting from held-out evaluation, and epoch as examples processed divided by training-set size. State that an epoch is progress accounting, not a guarantee that every example was visited exactly once under sampling with replacement.

~~~tsx
<span className="control-label">
    Learning rate
    <ConceptHelp conceptId="learning-rate" guidanceLevel={guidanceLevel} />
</span>
~~~

Use the existing component props/store selectors for `guidanceLevel`; do not introduce a second help implementation.

- [ ] **Step 4: Run GREEN and typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/concepts/conceptCatalog.test.ts src/components/controls/HyperparamPanel.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/TrainingControls.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec tsc --noEmit

Expected: PASS with all concept metadata and placements typed.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/concepts/conceptCatalog.ts apps/web/src/concepts/conceptCatalog.test.ts apps/web/src/components/controls/HyperparamPanel.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx
git commit -m "feat(web): explain core training concepts"
~~~

### Task 23: Standardize action state-effects copy

**Quick win:** 23 — Shared Changes/Preserves copy pattern.

**Files:**
- Create: apps/web/src/copy/stateEffects.ts
- Create: apps/web/src/copy/stateEffects.test.ts
- Modify: apps/web/src/components/controls/TrainingControls.tsx
- Modify: apps/web/src/components/controls/TrainingControls.test.tsx
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.test.tsx
- Modify: apps/web/src/components/controls/PresetCard.tsx
- Modify: apps/web/src/components/controls/PresetCard.test.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.test.tsx

**Interfaces:**
- Export a readonly `STATE_EFFECTS` map for `training-reset`, `reshuffle-split`, `lesson-start`, `preset-apply`, and `saved-recipe-apply`.
- Every value is one sentence beginning `Changes:` and containing `Preserves:`. Copy must match actual store/worker behavior and distinguish recipe, generated data/split, model weights/optimizer, checkpoints, lesson progress, and stored run evidence.
- Use the shared strings in visible notes or accessible descriptions at each corresponding action; do not duplicate literals in components.

- [ ] **Step 1: Write failing contract and integration tests**

~~~ts
for (const copy of Object.values(STATE_EFFECTS)) {
    expect(copy).toMatch(/^Changes: .+ Preserves: .+$/);
}
~~~

Assert each action is associated with its shared description using `aria-describedby`, tooltip content, or visible text appropriate to the existing control.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/copy/stateEffects.test.ts src/components/controls/TrainingControls.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/controls/PresetCard.test.tsx src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the shared contract does not exist.

- [ ] **Step 3: Define and consume one authoritative copy map**

~~~ts
export const STATE_EFFECTS = Object.freeze({
    'training-reset': 'Changes: model weights, optimizer state, training progress, and checkpoints. Preserves: the current recipe, generated data, split membership, and saved runs.',
    // remaining actions use verified behavior-specific wording
} as const);
~~~

Inspect each action before finalizing its string. If code behavior and existing copy disagree, keep production behavior unchanged and describe the behavior the tests prove.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/copy/stateEffects.test.ts src/components/controls/TrainingControls.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/controls/PresetCard.test.tsx src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: PASS with all five controls tied to shared accurate descriptions.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/copy/stateEffects.ts apps/web/src/copy/stateEffects.test.ts apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx apps/web/src/components/controls/PresetCard.tsx apps/web/src/components/controls/PresetCard.test.tsx apps/web/src/components/controls/RunHistoryPanel.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx
git commit -m "feat(web): standardize state effect disclosures"
~~~

### Task 24: Make timed state use current inputs with a stable setter

**Quick win:** 24 — Stable `useTimedState`.

**Files:**
- Modify: apps/web/src/hooks/useTimedState.ts
- Create: apps/web/src/hooks/useTimedState.test.tsx

**Interfaces:**
- The setter function retains identity across rerenders.
- A pending timeout resets to the latest `defaultValue` when it fires.
- Each call schedules using the `duration` current at call time; changing duration does not retroactively reschedule an already pending timer.
- Unmount and replacement calls clear the pending timer exactly once.

- [ ] **Step 1: Write failing fake-timer rerender tests**

~~~tsx
const firstSetter = result.current[1];
act(() => firstSetter('saved'));
rerender({ defaultValue: 'ready', duration: 50 });
expect(result.current[1]).toBe(firstSetter);
act(() => vi.advanceTimersByTime(100));
expect(result.current[0]).toBe('ready');
~~~

Also prove that a call made after duration changes uses the new duration and that replacing a timed value cancels the older timer.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useTimedState.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the setter is recreated and timeout closure holds the old default.

- [ ] **Step 3: Synchronize refs and memoize the setter**

~~~ts
const defaultValueRef = useRef(defaultValue);
const durationRef = useRef(duration);
defaultValueRef.current = defaultValue;
durationRef.current = duration;

const setTimed = useCallback((next: T) => {
    clearPendingTimeout();
    setValue(next);
    timeoutRef.current = setTimeout(() => setValue(defaultValueRef.current), durationRef.current);
}, []);
~~~

Use stable internal cleanup without placing changing props in the callback dependency list.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useTimedState.test.tsx --pool=forks --reporter=dot

Expected: PASS with deterministic timer counts and latest-input semantics.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/hooks/useTimedState.ts apps/web/src/hooks/useTimedState.test.tsx
git commit -m "fix(web): stabilize timed state resets"
~~~

### Task 25: Give collapsible panels stable persisted identities

**Quick win:** 25 — Stable panel storage IDs with old-key compatibility.

**Files:**
- Modify: apps/web/src/components/common/CollapsiblePanel.tsx
- Modify: apps/web/src/components/common/CollapsiblePanel.test.tsx

**Interfaces:**
- Add required prop `storageId: string`, validated against `^[a-z0-9]+(?:-[a-z0-9]+)*$` in development, and persist under `panel-v2-${storageId}`.
- When the v2 key is absent, read the former title-derived `panel-${normalizedTitle}` key once, write the valid value to the v2 key, and remove the legacy key only after the new write succeeds.
- A visible title change never changes persisted state, and unique call-site IDs prevent collisions.

- [ ] **Step 1: Write failing key, migration, and rename tests**

~~~tsx
localStorage.setItem('panel-old-title', 'false');
const { rerender } = render(<CollapsiblePanel storageId="network" title="Old Title">x</CollapsiblePanel>);
expect(localStorage.getItem('panel-v2-network')).toBe('false');
rerender(<CollapsiblePanel storageId="network" title="New Title">x</CollapsiblePanel>);
expect(screen.getByRole('button', { name: /New Title/ })).toHaveAttribute('aria-expanded', 'false');
~~~

Test an invalid legacy value, localStorage write failure, and two panels with similar titles but distinct IDs.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/CollapsiblePanel.test.tsx src/components/layout/Sidebar.test.tsx --pool=forks --reporter=dot

Expected: FAIL because storage remains title-derived and call sites have no IDs.

- [ ] **Step 3: Implement v2 identity and verify the call-site inventory**

Search `rg -n 'CollapsiblePanel' apps/web/src --glob '*.tsx'`. The current production inventory has no rendered call sites, so update all test fixtures with durable IDs and record that finding in the implementer report. If a production call site appears because an earlier task introduced one, add its owning file to this task brief before editing and assign a durable semantic ID. Keep the legacy-key helper private for migration only.

~~~tsx
<CollapsiblePanel storageId="data" title="Data" defaultExpanded>
    <DataPanel />
</CollapsiblePanel>
~~~

Do not reset React state when only `title` changes. Continue treating storage failures as recoverable in-memory behavior.

- [ ] **Step 4: Run GREEN and typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/CollapsiblePanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec tsc --noEmit

Expected: PASS and no call site omits `storageId`.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/common/CollapsiblePanel.tsx apps/web/src/components/common/CollapsiblePanel.test.tsx
git commit -m "feat(web): stabilize panel persistence keys"
~~~

### Task 26: Rehydrate saved-run memory after cross-tab changes

**Quick win:** 26 — Cross-tab run-memory storage synchronization.

**Files:**
- Create: apps/web/src/hooks/useExperimentMemoryStorageSync.ts
- Create: apps/web/src/hooks/useExperimentMemoryStorageSync.test.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx

**Interfaces:**
- The mounted app listens for `storage` events whose `storageArea` is localStorage and whose key is `EXPERIMENT_MEMORY_STORAGE_KEY`, `LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY`, or `null` from `localStorage.clear()`.
- Each accepted event invokes the store's existing queued `hydrate()` exactly once. Unrelated/session-storage events are ignored, and cleanup removes the listener.
- Native same-document localStorage writes do not emit `storage`; do not synthesize events in production or create a second persistence queue.

- [ ] **Step 1: Write failing lifecycle and filtering tests**

~~~tsx
renderHook(() => useExperimentMemoryStorageSync());
window.dispatchEvent(new StorageEvent('storage', {
    key: EXPERIMENT_MEMORY_STORAGE_KEY,
    storageArea: window.localStorage,
}));
await waitFor(() => expect(hydrate).toHaveBeenCalledTimes(1));
~~~

Cover legacy key, `key: null`, unrelated key, sessionStorage, and unmount cleanup. In App test, prove the hook is mounted once.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useExperimentMemoryStorageSync.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the lifecycle hook does not exist.

- [ ] **Step 3: Add one app-owned storage listener**

~~~ts
export function useExperimentMemoryStorageSync(): void {
    useEffect(() => {
        const onStorage = (event: StorageEvent) => {
            if (event.storageArea !== window.localStorage) return;
            if (event.key !== null && !MEMORY_KEYS.has(event.key)) return;
            void useExperimentMemoryStore.getState().hydrate();
        };
        window.addEventListener('storage', onStorage);
        return () => window.removeEventListener('storage', onStorage);
    }, []);
}
~~~

Mount it at App lifecycle scope, not inside the conditional History drawer.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useExperimentMemoryStorageSync.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Expected: PASS with exact filtering and cleanup.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/hooks/useExperimentMemoryStorageSync.ts apps/web/src/hooks/useExperimentMemoryStorageSync.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx
git commit -m "feat(web): sync saved runs across tabs"
~~~

### Task 27: Always run and report both performance suites

**Quick win:** 27 — Independent engine and web performance conclusions.

**Files:**
- Create: scripts/run-performance-gates.mjs
- Create: scripts/run-performance-gates.test.mjs
- Modify: package.json

**Interfaces:**
- Export `runPerformanceGates(runCommand)` for deterministic Node tests. It executes named `engine` and `web` commands sequentially, even after a non-zero result or thrown spawn error.
- The CLI streams child output, prints a final line for each suite (`PASS` or `FAIL`), and exits non-zero when either suite fails.
- Root `pnpm test:perf` delegates to this script; package-specific perf commands remain directly runnable.

- [ ] **Step 1: Write a failing Node orchestration test**

~~~js
const calls = [];
const result = await runPerformanceGates(async (gate) => {
    calls.push(gate.name);
    return gate.name === 'engine' ? 1 : 0;
});
assert.deepEqual(calls, ['engine', 'web']);
assert.equal(result.exitCode, 1);
~~~

Also cover both pass, web-only failure, and thrown runner failure while still reaching the other suite.

- [ ] **Step 2: Run RED**

Run: node --test scripts/run-performance-gates.test.mjs

Expected: FAIL because the orchestration module does not exist and the root script short-circuits with `&&`.

- [ ] **Step 3: Implement child-process aggregation**

Use `spawn` with `stdio: 'inherit'`, `shell: false`, and the current platform's pnpm executable. Resolve each exit/error into a result instead of rejecting the overall loop. Guard CLI execution with an `import.meta.url` entrypoint check so tests can import without spawning real suites.

~~~js
export const PERFORMANCE_GATES = Object.freeze([
    { name: 'engine', args: ['--filter', '@nn-playground/engine', 'test:perf'] },
    { name: 'web', args: ['--filter', '@nn-playground/web', 'test:perf'] },
]);
~~~

- [ ] **Step 4: Run GREEN and exercise the real combined gate**

Run: node --test scripts/run-performance-gates.test.mjs

Run: pnpm test:perf

Expected: Unit test PASS. The real command must print both suite conclusions; its overall status reflects the current calibrated budgets from Task 10.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/run-performance-gates.mjs scripts/run-performance-gates.test.mjs package.json
git commit -m "test: aggregate independent performance gates"
~~~

### Task 28: Scan the built app with Axe in desktop and compact browsers

**Quick win:** 28 — Production browser Axe scans.

**Files:**
- Create: tests/e2e/accessibility.spec.ts
- Modify: package.json
- Modify: pnpm-lock.yaml

**Interfaces:**
- Add `@axe-core/playwright` as a direct root development dependency; do not depend on the transitive `axe-core` bundled under `jest-axe`.
- Scan the built app after worker readiness at desktop and 390 by 844 compact viewports in both Chromium and WebKit.
- Fail on any Axe violation whose impact is `serious` or `critical`. Any future exception must name a rule, a tightly scoped selector, rationale, and a repository issue URL in the test; this task adds no blanket exclusions.

- [ ] **Step 1: Add the direct test dependency and a failing production scan**

Run: pnpm add -Dw @axe-core/playwright@^4.10.2

Create the spec with its expected-zero assertion before changing application markup:

~~~ts
const results = await new AxeBuilder({ page }).analyze();
const blocking = results.violations.filter(({ impact }) => impact === 'serious' || impact === 'critical');
expect(blocking, formatViolations(blocking)).toEqual([]);
~~~

- [ ] **Step 2: Run RED against a fresh production build**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/accessibility.spec.ts --project=chromium --project=webkit

Expected: The new executable gate runs all four browser/viewport cases and FAILS on any discovered serious/critical production violation. If it is already green, record the pre-implementation scan as the executable characterization RED exception and continue only after review confirms the test would fail for an injected serious violation.

- [ ] **Step 3: Fix only violations proven by the scan**

Prefer semantic HTML, names, relationships, and token-level contrast corrections in the owning components/styles. Add each required source/test file to this task's file list in the implementer report before staging. Verify one controlled mutation causes the Axe assertion to fail, then revert the mutation.

- [ ] **Step 4: Run GREEN in the required matrix**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/accessibility.spec.ts --project=chromium --project=webkit

Expected: PASS for desktop Chromium, compact Chromium, desktop WebKit, and compact WebKit, with zero serious/critical violations and no page/console errors.

- [ ] **Step 5: Commit**

~~~bash
git add tests/e2e/accessibility.spec.ts package.json pnpm-lock.yaml
# Add only any source and focused unit-test files required by observed Axe failures.
git commit -m "test(e2e): scan production accessibility"
~~~

### Task 29: Verify both narrow and common phone widths

**Quick win:** 29 — 390 by 844 responsive E2E coverage alongside 320.

**Files:**
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Run the existing compact reachability journey unchanged at both 320 by 844 and 390 by 844 using generated, uniquely named tests.
- At both widths assert no document/shell horizontal overflow, all critical and graph/evidence controls from Task 20 are at least 44 by 44, active evidence tabs remain visible, compact evaluation outcome from Task 21 is reachable, drawers close and restore focus, and browser errors remain empty.
- Do not use conditional assertions that weaken one viewport.

- [ ] **Step 1: Parameterize the current test and require 390-specific evidence**

~~~ts
for (const width of [320, 390] as const) {
    test.describe(`${width}px touch shell`, () => {
        test.use({ viewport: { width, height: 844 }, hasTouch: true, isMobile: true });
        test('keeps critical controls reachable, sized, focused, and unclipped', compactShellJourney);
    });
}
~~~

Before refactoring, run the exact missing scenario as a failing executable gate.

- [ ] **Step 2: Run RED**

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "390px touch shell"

Expected: FAIL with `No tests found` because no 390px scenario exists. This RED proves the missing automated coverage, not a product-layout regression; the GREEN matrix below is the behavioral acceptance gate.

- [ ] **Step 3: Extract one shared compact journey and instantiate both viewports**

Keep locators and assertions in one function to prevent drift. Extend it with the Task 20 graph/evidence targets, Task 21 disclosure, active-tab-in-viewport measurement, and the existing focus restoration checks.

- [ ] **Step 4: Run GREEN in both browser engines**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "touch shell"

Expected: Four tests PASS at zero retries, with both named widths represented for each browser.

- [ ] **Step 5: Commit**

~~~bash
git add tests/e2e/playground-smoke.spec.ts
git commit -m "test(e2e): cover two phone widths"
~~~

### Task 30: Prove production worker failure and recovery

**Quick win:** 30 — Production worker failure/recovery E2E with test-only fault injection.

**Files:**
- Create: apps/web/src/testing/e2eFaults.ts
- Create: apps/web/src/testing/e2eFaults.test.ts
- Create: tests/e2e/worker-recovery.spec.ts
- Modify: apps/web/src/worker/workerBridge.ts
- Modify: apps/web/src/worker/workerBridge.test.ts
- Modify: apps/web/src/hooks/useTraining.ts
- Modify: apps/web/src/hooks/useTraining.test.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/vite-env.d.ts
- Modify: package.json
- Modify: playwright.config.ts

**Interfaces:**
- Add compile-time boolean `VITE_E2E_FAULTS`; normal `pnpm build` leaves all URL-controlled fault behavior disabled.
- In an enabled E2E build, query `e2eWorkerFault=startup-once` causes one deterministic worker startup failure per tab session, records consumption in sessionStorage, and then routes through `workerBridge`'s existing `emitWorkerError` message, the installed `onSnapshot` subscriber, `useTraining`'s error handler, and the real store/modal path.
- The existing recovery action reloads the page; the consumed session marker prevents a second injected failure, and the test proves worker readiness plus a real training step afterward.
- The seam exposes no visible production control, does not alter scientific code, and ignores unknown fault names.

- [ ] **Step 1: Write failing seam, bridge, and browser tests**

~~~ts
expect(consumeE2EWorkerFault(new URL('https://example.test/?e2eWorkerFault=startup-once'), storage, true))
    .toBe('startup-once');
expect(consumeE2EWorkerFault(url, storage, true)).toBeNull();
expect(consumeE2EWorkerFault(url, storage, false)).toBeNull();
~~~

Name the enabled recovery test with tag `@fault-enabled` and the normal-build guard with tag `@fault-disabled`. The enabled test navigates with the query, expects the named alertdialog to receive focus with inert background, activates `Refresh page`, waits for ready state, runs one training step, and asserts no unexpected console/page errors beyond the deliberately injected labeled error.

- [ ] **Step 2: Run RED unit tests**

Run: pnpm --filter @nn-playground/web exec vitest run src/testing/e2eFaults.test.ts src/worker/workerBridge.test.ts src/hooks/useTraining.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Expected: FAIL because no compile-time-gated fault seam exists.

- [ ] **Step 3: Implement the smallest gated startup seam**

Add a narrowly named bridge export `emitE2EWorkerError(message): boolean` that delegates to private `emitWorkerError` and returns false when no `onSnapshot` subscriber is installed. In `useTraining`, add an `e2eStartupFaultActiveRef` checked by both the mount-initialization effect and the immediately following prepared-document synchronization effect. The mount effect—which runs after its preceding snapshot-subscription effect—consumes the gated URL fault, sets the suppression ref before queueing the bridge error, and skips `initializeWorker`. The prepared-document effect returns while that ref is active, so it cannot re-enter initialization through the replacement path. Ordinary builds never set the ref.

~~~ts
const fault = consumeE2EWorkerFault(
    new URL(window.location.href),
    window.sessionStorage,
    import.meta.env.VITE_E2E_FAULTS === '1',
);
if (fault === 'startup-once') {
    e2eStartupFaultActiveRef.current = true;
    queueMicrotask(() => emitE2EWorkerError(E2E_WORKER_FAULT_MESSAGE));
    return;
}
initializeWorker(prepared).catch((error) => {
    reportWorkerError(error, 'Failed to initialize training worker.');
});
~~~

Bridge tests prove the helper returns false without a subscriber and delivers a protocol-v2 `error` message through `onSnapshot` when subscribed. Hook tests prove the injected branch is reached only after subscription, that neither the mount initializer nor prepared-document synchronization invokes worker initialization, and that the same worker error as a native bridge error reaches the store. Place fault selection outside worker scientific logic. Add root scripts `build:e2e` and `test:e2e:recovery`; configure the recovery command to build with `VITE_E2E_FAULTS=1` and run only `--grep @fault-enabled` against the exact generated dist.

- [ ] **Step 4: Run GREEN units and production recovery**

Run: pnpm --filter @nn-playground/web exec vitest run src/testing/e2eFaults.test.ts src/worker/workerBridge.test.ts src/hooks/useTraining.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm run build:e2e

Run: pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-enabled"

Expected: Unit tests PASS. Both browsers show the genuine focused modal, reload through the real recovery control, reach ready state, and advance training with zero retries.

- [ ] **Step 5: Prove the normal build ignores the query**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-disabled"

Expected: PASS: the query cannot inject a fault in a normal production build.

- [ ] **Step 6: Commit**

~~~bash
git add apps/web/src/testing/e2eFaults.ts apps/web/src/testing/e2eFaults.test.ts tests/e2e/worker-recovery.spec.ts apps/web/src/worker/workerBridge.ts apps/web/src/worker/workerBridge.test.ts apps/web/src/hooks/useTraining.ts apps/web/src/hooks/useTraining.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/vite-env.d.ts package.json playwright.config.ts
git commit -m "test(e2e): prove worker failure recovery"
~~~

---

## Final whole-program verification

After all task reviews are clean, run the following from the repository root on the committed branch:

~~~bash
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm test:perf
pnpm build
pnpm exec playwright test --project=chromium --project=webkit --grep-invert "@fault-enabled"
pnpm run build:e2e
pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-enabled"
pnpm build
pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-disabled"
git diff --check 047c341..HEAD
git diff --check
git status --short
~~~

The final reviewer must also inspect the cumulative range from the design commit through Task 30, confirm every numbered acceptance criterion is covered by committed code or executable evidence, confirm the four unrelated pre-existing paths remain untouched, and confirm the normal production build does not honor E2E fault parameters.
