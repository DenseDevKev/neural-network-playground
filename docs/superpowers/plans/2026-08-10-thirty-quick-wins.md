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
- Modify: packages/engine/src/network.ts
- Modify: packages/engine/src/__tests__/network.test.ts
- Modify: docs/perf/PERFORMANCE_BASELINE.md

**Interfaces:**
- Produces median(values), measureMedianMsPerIteration(run, options), and assertPerformanceBudgets(results), which reports every over-budget path in one failure.
- The four existing prediction APIs retain their current caps unless controlled evidence and code profiling justify an implementation optimization or a documented baseline change.
- Scope amendment (2026-08-11): three controlled robust runs left
  `predictGridWithNeuronsInto` over its unchanged cap in two runs, with a
  median-of-run-medians of 12.3304 ms/iteration versus 12.2388. A five-trial
  isolated A/B showed that capacity guards, hoisted target-kind decisions, and
  one pre-write representability check improve the path median by 5.99% while
  preserving Float64 large-finite values and the exact Float32 overflow error.
  The two production/test paths above are authorized only for that profiled
  optimization; preserve engine math, evaluation order, flat neuron layout,
  target precision, and non-finite error path/value.

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

- [ ] **Step 3a: Capture the profiled production-path RED and protect safety semantics**

Run three times before the production optimization:
pnpm --filter @nn-playground/engine test:perf

Expected performance RED: all four paths are measured and reported, but the
unchanged `predictGridWithNeuronsInto` cap fails in runs 1 and 3 at 12.3841 and
12.3304 ms/iteration (run 2 measures 12.2321), for a 12.3304
median-of-run-medians versus the 12.2388 cap.

Add unit tests that require undersized output and neuron targets to fail before
any write, prove exact Float32 output/neuron-grid equality and layout against
the allocating API, and cover all-Float32, all-Float64, and both mixed target
permutations (output32/neuron64 and output64/neuron32). Require exact
`NonFiniteNumericalError` path/value assertions for Float32 overflow in both
optimized destination branches: `predictionGrid.output[0]`/`Infinity` and
`predictionGrid.neurons[0][0]`/`Infinity`. Run the existing
`numericalError.test.ts` cases unchanged to preserve large-finite Float64 values
and their current overflow diagnostics.

Add minimum-capacity guards before removing the old post-write readback (typed
array out-of-bounds writes are otherwise ignored). Hoist the output/neuron
Float32 decisions once per call, convert once, reject a non-finite converted
value before its write, store that converted value, and increment the flattened
neuron offset without changing write order. Do not change forward arithmetic,
buffer layout, API shape, or budgets.

- [ ] **Step 4: Run helper GREEN and controlled performance samples**

Run: pnpm --filter @nn-playground/engine exec vitest run src/__benchmarks__/performanceStatistics.test.ts --pool=forks --reporter=dot

Run three times after the production optimization:
pnpm --filter @nn-playground/engine test:perf

Expected: helper PASS; all three controlled commands PASS the unchanged caps,
and every run measures and reports all four paths. Record all raw medians and
the median-of-run-medians. If a path remains over budget, continue profiling or
stop; do not merely raise its cap.

- [ ] **Step 5: Run engine unit tests and commit**

Run: pnpm --filter @nn-playground/engine test

Expected: PASS.

Run: pnpm --filter @nn-playground/engine exec vitest run src/__tests__/network.test.ts src/__tests__/numericalError.test.ts --pool=forks --reporter=dot

Expected: PASS for exact layout/precision/capacity and numerical-error contracts.

~~~bash
git add packages/engine/src/__benchmarks__/performanceStatistics.ts packages/engine/src/__benchmarks__/performanceStatistics.test.ts packages/engine/src/__benchmarks__/grid_performance.bench.ts packages/engine/src/network.ts packages/engine/src/__tests__/network.test.ts docs/perf/PERFORMANCE_BASELINE.md
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
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx
- Modify: apps/web/src/components/layout/BuildRunShell.tsx
- Modify: apps/web/src/components/layout/BuildRunShell.test.tsx

**Interfaces:**
- useLayoutStore persists `lessonCueDismissed` and monotonic `hasStartedLesson` without adding either to shared experiment URLs. `dismissLessonCue()` only sets `lessonCueDismissed` true. Calling the existing `setActiveLessonStep` marks `hasStartedLesson` true; finishing or clearing a lesson never resets it. Production actions never reset either sticky boolean.
- FirstVisitLessonCue receives historyReady, lessonCueDismissed, hasSavedRuns, hasStartedLesson, hasActiveLesson, onOpenLessons, and onDismiss; it renders only when history hydration is ready and all first-visit conditions are true.
- Scope amendment (2026-08-11): App owns the drawer state and is the only layer
  that can idempotently open Lessons with `setOpenSurface('lessons')`.
  BuildRunShell must stay presentation-focused; App derives accepted saved-run
  readiness, renders/passes the cue, and supplies one stable open callback.
  Do not DOM-click a Header trigger or reuse the toggle callback, because those
  approaches can close the requested surface or couple components through DOM.

- [ ] **Step 1: Write failing visibility and persistence tests**

~~~tsx
render(<FirstVisitLessonCue historyReady={true} lessonCueDismissed={false} hasSavedRuns={false} hasStartedLesson={false} hasActiveLesson={false} onOpenLessons={open} onDismiss={dismiss} />);
expect(screen.getByRole('button', { name: 'Start a 3-minute lesson' })).toBeVisible();
~~~

Add cases for loading history, a saved run, active lesson, previously started/finished lesson, and persisted dismissal, each of which hides the cue. Store tests must prove `hasStartedLesson` survives persistence sanitization and stays true after `clearActiveLessonStep()`.

Add an App-shell integration RED proving the cue waits for history hydration,
then opens the existing Lessons dialog without changing audience, workspace,
recipe/prepared identity, training state, or URL hash.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/FirstVisitLessonCue.test.tsx src/store/useLayoutStore.test.ts src/components/layout/BuildRunShell.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the cue and persisted flag do not exist.

- [ ] **Step 3: Implement the bounded cue and shell integration**

~~~tsx
if (!historyReady || lessonCueDismissed || hasSavedRuns || hasStartedLesson || hasActiveLesson) return null;
return <aside aria-label="Getting started">...</aside>;
~~~

Place it beside Current Recipe in Build view and open the existing Lessons surface without changing audience mode or recipe state.

App should pass a stable `openLessons` callback backed by
`setOpenSurface('lessons')`; cue dismissal updates only layout persistence. Use
accepted experiment-memory records only after `hydrationStatus === 'ready'` so
established users never see a loading-time flash. Keep Zustand persistence
version `0`; sanitize/partialize only actual booleans, leave active lesson fields
transient, and reset the new singleton fields explicitly in affected tests.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/FirstVisitLessonCue.test.tsx src/store/useLayoutStore.test.ts src/components/layout/BuildRunShell.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Expected: PASS for fresh, dismissed, active-lesson, and existing-user states.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/store/useLayoutStore.ts apps/web/src/store/useLayoutStore.test.ts apps/web/src/components/controls/FirstVisitLessonCue.tsx apps/web/src/components/controls/FirstVisitLessonCue.test.tsx apps/web/src/App.tsx apps/web/src/__tests__/appShell.integration.test.tsx apps/web/src/components/layout/BuildRunShell.tsx apps/web/src/components/layout/BuildRunShell.test.tsx
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
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx

**Interfaces:**
- Workspace view references a persistent or help-triggered description available by pointer, focus, and touch.
- Switching view retains its existing state-only callback.
- Scope amendment (2026-08-11): the keyboard-accessible help trigger is a
  legitimate tab stop between the Build/Run switch and Audience control. Update
  the existing app-shell tab-order assertion; do not hide the trigger with a
  negative tab index or weaken keyboard access to preserve stale order.

- [ ] **Step 1: Write a failing accessible-description test**

~~~tsx
expect(screen.getByRole('group', { name: 'Workspace view' })).toHaveAccessibleDescription(
    'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.',
);
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the view group has no explanation.

- [ ] **Step 3: Attach concise description and visible help**

Use the exact tested copy in an element referenced by aria-describedby and expose it through the existing touch-safe help pattern.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Expected: PASS; the help trigger is present in keyboard order and existing view-switch mutation tests remain green.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/__tests__/appShell.integration.test.tsx
git commit -m "feat(web): explain build and run views"
~~~

### Task 15: Rename Mode to Workspace

**Quick win:** 15 — Unambiguous workspace label.

**Files:**
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/productShell/audienceProfiles.test.ts
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Visible label and accessible name become Workspace profile.
- Stored audience values beginner, explore, and lab remain unchanged.
- The app-shell keyboard/selector assertion uses the new accessible name; its
  prior `Audience mode` query is part of the executable RED.
- The Playwright audience-mode helper uses the new accessible name while its
  internal helper/storage terminology and stored values remain unchanged.

- [ ] **Step 1: Write the failing label and invariance test**

~~~tsx
expect(screen.getByRole('combobox', { name: 'Workspace profile' })).toHaveValue('explore');
await user.selectOptions(screen.getByRole('combobox', { name: 'Workspace profile' }), 'lab');
expect(onAudienceModeChange).toHaveBeenCalledWith('lab');
expect(onExperimentChange).not.toHaveBeenCalled();
~~~

Update the existing app-shell selectors and Playwright helper to query
`Workspace profile`; those selector changes are part of RED and must not alter
the profile-cycle scientific-state assertions.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/productShell/audienceProfiles.test.ts src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm test:e2e -- tests/e2e/playground-smoke.spec.ts

Expected: FAIL because the current accessible name is Audience mode and visible label is Mode; the updated Playwright helper cannot yet resolve Workspace profile.

- [ ] **Step 3: Update labels without changing enum or persistence keys**

Render Workspace as the visible label, Workspace profile as the accessible name, and Profiles change visible tools and guidance only as the descriptive copy.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/productShell/audienceProfiles.test.ts src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm test:e2e -- tests/e2e/playground-smoke.spec.ts

Expected: PASS with the same stored values/state behavior, updated app-shell selectors, and the paused-run/compact Playwright invariants still green.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/productShell/audienceProfiles.test.ts apps/web/src/__tests__/appShell.integration.test.tsx tests/e2e/playground-smoke.spec.ts
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
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Extends `TRAINING_SHORTCUTS` from Task 1 into the authoritative
  code/label/description/action registry and derives shortcut resolution plus
  rendered definitions from it; do not maintain a second code/action map.
- Renders a default-closed native `details` disclosure named Keyboard shortcuts
  as a direct child of `.training-bar`, immediately after
  `.training-bar__controls`. It remains visible at compact breakpoints and its
  summary has a 44-pixel compact target.
- Global training shortcuts ignore the implicitly focusable native `summary`
  (and other focusable descendants), preserving native disclosure keyboard
  behavior instead of preventing Space or triggering training.
- Author styles apply the definition-list grid only while details is open; the
  closed native disclosure keeps its `dl` hidden in the built app.

- [ ] **Step 1: Write failing shared-list and responsive UI tests**

~~~tsx
// Extend the existing helper to accept `code = 'KeyR'` instead of hard-coding KeyR.
const nativeDetails = document.createElement('details');
const nativeSummary = document.createElement('summary');
nativeDetails.append(nativeSummary);
expect(resolveFromTarget(nativeSummary, 'Space')).toBeNull();

const details = screen.getByRole('group', { name: 'Keyboard shortcuts' });
const summary = within(details).getByText('Keyboard shortcuts');
expect(details).not.toHaveAttribute('open');
await user.click(summary);
expect(details).toHaveAttribute('open');

const terms = within(details).getAllByRole('term');
const definitions = within(details).getAllByRole('definition');
expect(terms.map((term) => term.textContent))
    .toEqual(TRAINING_SHORTCUTS.map(({ label }) => label));
expect(definitions.map((definition) => definition.textContent))
    .toEqual(TRAINING_SHORTCUTS.map(({ description }) => description));
~~~

Assert all supported codes resolve to their registry action on the page
background, all three resolve to null from a native summary, and both rendered
term/definition arrays have `TRAINING_SHORTCUTS.length`. The responsive
stylesheet has a positive full-width rule plus a 44-pixel summary target. Scope
duplicate `Space` and `R` queries within the disclosure because inline badges
already use those strings.

Add a compact built-app browser test named `keyboard shortcuts disclosure hides
definitions when closed`: at 800 by 844, assert the `dl` is hidden while details
lacks `open`, visible after pointer activation, and hidden again after closing.
This computed-visibility assertion guards against author `display` rules
overriding the user-agent closed-details behavior.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/components/controls/TrainingControls.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "keyboard shortcuts disclosure hides definitions when closed"

Expected: FAIL because no disclosure or compact rule exists, the shortcut
resolver intercepts keys from a native summary, and the built-app browser gate
cannot observe correct closed/open visibility.

- [ ] **Step 3: Render the shared definitions in a compact disclosure**

~~~tsx
<details className="training-shortcuts" aria-label="Keyboard shortcuts">
    <summary>Keyboard shortcuts</summary>
    <dl>{TRAINING_SHORTCUTS.map(renderShortcutDefinition)}</dl>
</details>
~~~

Render the disclosure as a direct child of `.training-bar`, immediately after
`.training-bar__controls`. Within its `dl`, render one `dt > kbd` and one `dd`
per registry item. Keep native disclosure state and activation—do not add custom
key handlers, roles, `tabIndex`, or React open state. Preserve the existing explicit `tabindex` guard, add the implicit
`HTMLElement.tabIndex >= 0` focusability guard, and derive the handler lookup
and existing inline shortcut labels from the registry. Do not hide
`.training-shortcuts` in the max-width 900px rules that hide inline badges.
Scope the popover layout rule to `.training-shortcuts[open] dl`; never apply an
author `display` value to the closed definition list.

- [ ] **Step 4: Run GREEN and static responsive test**

Run: pnpm --filter @nn-playground/web exec vitest run src/shortcuts/trainingShortcuts.test.ts src/components/controls/TrainingControls.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "keyboard shortcuts disclosure hides definitions when closed"

Expected: PASS with definitions and responsive visibility aligned.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/shortcuts/trainingShortcuts.ts apps/web/src/shortcuts/trainingShortcuts.test.ts apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts tests/e2e/playground-smoke.spec.ts
git commit -m "feat(web): expose training shortcuts"
~~~

### Task 17: Add meaningful range labels and values

**Quick win:** 17 — Meaningful range controls.

**Files:**
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx

**Interfaces:**
- Each DataPanel instance derives stable, unique input/output IDs from an
  unconditional `useId` call made before the recipe guard; multiple panels must
  not collide.
- Train ratio has a native label, associated output, and value text such as 70
  percent training, 30 percent test. Its rounded train value and complementary
  test value always total 100.
- Noise has a native label, associated output, and value text such as 15 percent
  noise. Preserve the controlled recipe value rather than independently
  rounding it.
- Keep native range behavior and existing bounds; remove overriding aria-labels
  so accessible names are exactly Train ratio and Noise.

- [ ] **Step 1: Write failing accessible-value tests**

~~~tsx
const trainSlider = screen.getByRole('slider', { name: 'Train ratio' });
const noiseSlider = screen.getByRole('slider', { name: 'Noise' });
expect(trainSlider).toHaveAttribute(
    'aria-valuetext',
    '50 percent training, 50 percent test',
);
expect(noiseSlider).toHaveAttribute('aria-valuetext', '0 percent noise');

const trainId = trainSlider.id;
expect(document.querySelector(`label[for="${trainId}"]`)).toHaveTextContent('Train ratio');
expect(document.getElementById(trainSlider.getAttribute('aria-describedby') ?? '')?.tagName)
    .toBe('OUTPUT');
~~~

Assert both sliders have nonempty IDs, matching native labels, outputs whose
`htmlFor` points back to their input, exact initial visible/value text, and the
same IDs after sequential changes to 70/30 and 15 percent. Update the existing
queries from `Train/test split percentage`/`Noise level`. Render two DataPanels
in one root and prove all input/output IDs are distinct and correctly owned.
Because `output` has implicit status semantics, narrow the existing singular
loading-status assertion to the `Generating data...` status instead of removing
output semantics.

~~~tsx
const loadingStatus = screen.getByText('Generating data...').closest('[role="status"]');
expect(loadingStatus).toHaveTextContent('Generating data...');
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/DataPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the sliders use standalone aria-label values without associated outputs or value text.

- [ ] **Step 3: Add label, output, and value text**

~~~tsx
<label className="control-label" htmlFor={trainRatioId}>Train ratio</label>
<output className="control-value" id={trainRatioOutputId} htmlFor={trainRatioId}>{trainPercent}%</output>
<input id={trainRatioId} aria-describedby={trainRatioOutputId} aria-valuetext={`${trainPercent} percent training, ${testPercent} percent test`} />
~~~

Mirror the structure for noise. Derive `trainPercent` once with `Math.round` and
derive `testPercent` as its complement. Keep the inputs controlled by the recipe
with no local state, custom role, key handler, or tab index. Preserve the same
`control-label` and `control-value` classes for Noise.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/DataPanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Expected: PASS for initial/changed values, stable unique ownership, and native range behavior.

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
- Modify: apps/web/src/components/layout/ExperimentStateContext.tsx
- Modify: apps/web/src/components/layout/ExperimentStateContext.test.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/components/common/LoadingState.tsx
- Modify: apps/web/src/components/common/LoadingState.test.tsx
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx
- Modify: apps/web/src/components/controls/FeaturesPanel.tsx
- Modify: apps/web/src/components/controls/FeaturesPanel.test.tsx
- Modify: apps/web/src/components/controls/NetworkConfigPanel.tsx
- Modify: apps/web/src/components/controls/NetworkConfigPanel.test.tsx
- Modify: apps/web/src/components/controls/HyperparamPanel.tsx
- Modify: apps/web/src/components/controls/HyperparamPanel.test.tsx
- Modify: apps/web/src/components/controls/PresetPanel.tsx
- Modify: apps/web/src/components/controls/PresetPanel.test.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.tsx
- Modify: apps/web/src/components/controls/GuidedLessonPanel.test.tsx

**Interfaces:**
- Rapidly changing Header metrics, evidence context, diagnostic cockpit metrics,
  and the bottom status-bar step remain ordinary readable content outside every
  `aria-live`, status, or alert ancestor. Named metric/status containers use
  `role="group"`; removing only explicit `aria-live` is insufficient because
  `role="status"` is implicitly polite.
- `AccessibilityAnnouncer` is the single live owner for training/configuration
  transitions and has the stable name Training and configuration announcements.
  Its one polite atomic region starts empty.
- Training start is idle/paused to running. Pause requires paused status plus a
  non-null non-error pause reason. A successful reset is a changed evidence
  generation whose trained recipe source is reset, including idle-to-idle;
  status alone must not mislabel config-sync initialization as a reset.
- Reset publication is split across store writes, and a later config sync can
  advance generation while the prior source still says reset. Announce only
  when an unannounced generation is paired with a new trained-recipe publication
  whose source is reset. Use the cloned `trainedRecipe` object identity (not a
  timestamp alone) as the publication token, then latch that reset generation;
  unchanged rerenders cannot miss/repeat it and stale source cannot create a
  false reset.
- Configuration scopes are data, network, features, training, and preset. One
  owned busy interval emits one start and then either one completion or one
  `(source,message)` error. Same-scope rapid edits coalesce. Superseding A with
  B emits B start without A completion. Retry may announce the same error again
  only after the error identity was cleared by a new owned interval.
- Deterministic per-render priority is new error, new/superseding configuration
  start, successful configuration completion, reset token, non-error pause,
  then training start. Every previous-state field advances even when an event
  loses priority so it cannot replay.
- Config-sync's internal null-reason pause/idle lifecycle is not announced as a
  user pause/reset. Worker errors remain owned by App's focused alertdialog;
  suppress the announcer's generic pause when pause reason is error so users do
  not hear two error surfaces.
- Add an `announce` (or equivalently named) LoadingState option that defaults to
  true. The five config panels opt out because the central announcer owns those
  starts; Sidebar/MainArea and other independent loading feedback retain current
  live behavior. The same panels keep config errors visible/retryable but
  non-live; incompatibility/no-recipe alerts that are not store config errors
  remain alerts.
- GuidedLessonPanel's preset-backed lesson-start failure is the same central
  config error. Keep its persistent retry/action copy visible but non-live;
  preserve unrelated lesson step/progress/completion live semantics.

- [ ] **Step 1: Write a failing live-region boundary test**

~~~tsx
const metrics = screen.getByRole('group', { name: 'Training metrics' });
expect(metrics).not.toHaveAttribute('aria-live');
expect(metrics.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
~~~

Repeat the non-live-ancestor assertion after metric/evidence step updates for the
Header, EvidenceContextLine, DiagnosticCockpitStrip, and StatusBar. Assert the
cockpit/status bar remain named groups.

Add a transition table for all five config scopes covering start, success,
failure without false completion, same-scope coalescing, A-to-B supersession,
retry of the same error, and unchanged rerenders. Observe the named live region
with `MutationObserver` (childList plus characterData) so an unchanged rerender
proves zero new DOM writes rather than merely ending with the same text. Cover
training start, manual/automatic pause, idle-to-idle reset generation, config
internal pause/idle suppression, and worker-error pause suppression. Drive at
least one complete start/finish/fail flow through the real training store in App
tests instead of relying only on prop rerenders.

For reset, test the real split-publication order and stale-source hazard:

- generation advances while the prior source/publication is non-reset: zero;
  a new trained-recipe publication with reset source: one;
- after that reset, a config-sync generation advances while the stale source is
  still reset but publication identity is unchanged: zero; its new config-sync
  publication: still zero;
- a later real reset (including reset-after-reset) advances generation alone:
  zero; its new reset publication identity: one; unchanged rerender: zero.

For duplicate ownership, assert `LoadingState announce={false}` and each of the
five config panels' active loading/config-error feedback has no live/status/alert
ancestor while remaining visible and retryable. Preserve default live LoadingState
coverage for independent callers and the panels' distinct compatibility alerts.
Prove GuidedLessonPanel's matching preset failure is visible/non-live/retryable
without changing its separate lesson-progress announcements.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/components/layout/AccessibilityAnnouncer.test.tsx src/components/layout/ExperimentStateContext.test.tsx src/App.test.tsx src/components/common/LoadingState.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/FeaturesPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx src/components/controls/HyperparamPanel.test.tsx src/components/controls/PresetPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because multiple volatile metric/config surfaces are live, reset
ownership is status-only, completions/supersession are not modeled, and config
panels duplicate the central announcer.

- [ ] **Step 3: Remove metric live semantics and preserve meaningful completion announcements**

Keep the named metric/context/status copy as ordinary content; do not add
per-value live regions. Use one deterministic AccessibilityAnnouncer transition
effect/reducer, not independent effects that can clobber each other. Track the
previous full snapshot: status, pause reason, worker error, active config scope,
error source/message, evidence generation, trained recipe object/publication
identity, and trained recipe source. Announce
`${Scope} update complete` only when the previously owned scope ends without a
matching current error. Always update the full previous snapshot.

Use the non-live LoadingState option only at the five central-owned config panel
call sites. Remove `aria-live` from DataPanel's dataset/split summaries and
HyperparamPanel's configuration summaries as part of the same ownership rule;
retain their readable copy and Task 17 label/output semantics.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/components/layout/AccessibilityAnnouncer.test.tsx src/components/layout/ExperimentStateContext.test.tsx src/App.test.tsx src/components/common/LoadingState.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/FeaturesPanel.test.tsx src/components/controls/NetworkConfigPanel.test.tsx src/components/controls/HyperparamPanel.test.tsx src/components/controls/PresetPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useTraining.test.tsx src/store/useTrainingStore.test.ts src/__tests__/training.integration.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Expected: PASS with volatile metrics quiet, one central DOM write per observed
meaningful transition, no panel duplicate, and lifecycle/store integrations
unchanged. This guarantees the app-owned live-region mutation contract; actual
speech coalescing remains assistive-technology behavior.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/components/layout/AccessibilityAnnouncer.tsx apps/web/src/components/layout/AccessibilityAnnouncer.test.tsx apps/web/src/components/layout/ExperimentStateContext.tsx apps/web/src/components/layout/ExperimentStateContext.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/components/common/LoadingState.tsx apps/web/src/components/common/LoadingState.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/FeaturesPanel.tsx apps/web/src/components/controls/FeaturesPanel.test.tsx apps/web/src/components/controls/NetworkConfigPanel.tsx apps/web/src/components/controls/NetworkConfigPanel.test.tsx apps/web/src/components/controls/HyperparamPanel.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/components/controls/PresetPanel.tsx apps/web/src/components/controls/PresetPanel.test.tsx apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx
git commit -m "fix(web): quiet live metric announcements"
~~~

### Task 19: Contain focus in the worker error modal

**Quick win:** 19 — Complete worker-error modal.

**Files:**
- Create: apps/web/src/hooks/useModalFocusContainment.ts
- Create: apps/web/src/hooks/useModalFocusContainment.test.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/__tests__/training.integration.test.tsx
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx

**Interfaces:**
- `useModalFocusContainment(active, dialogRef, backgroundRef)` is scoped to the
  app's one worker-error modal; simultaneous active modal instances are not a
  supported contract. It performs DOM work only in effects.
- Render the unchanged named/described alertdialog through a body portal outside
  the `.forge-shell`; keep the entire three-row shell under `backgroundRef` so
  no wrapper breaks its CSS grid.
- On activation, capture prior focus, focus the dialog with `preventScroll`, then
  lease exact `inert` and `aria-hidden` state for every background body sibling,
  including Header's current/future body portals. Observe body children while
  active so a newly mounted portal is also leased. Exclude the modal portal and
  its ancestors.
- Query enabled/visible focusables inside the dialog fresh for every Tab. Always
  prevent Tab and wrap forward/reverse; with none, focus the dialog. A document
  `focusin` guard redirects programmatic escape because jsdom and some fallback
  environments do not enforce inert.
- Escape is a contained no-op for this fatal modal; it cannot close a drawer or
  Advanced Tools behind the dialog. While worker error is active, Space,
  ArrowRight, and R do not reach global training shortcuts. Refresh page remains
  the only recovery action.
- On deactivate/unmount, remove only attributes leased by this hook, restore all
  preexisting inert/aria-hidden values exactly, then restore prior focus only if
  its element is connected and focusable. StrictMode setup/cleanup/setup must
  not capture the dialog as the opener or leak attributes/listeners.

- [ ] **Step 1: Write failing focus-cycle and background tests**

~~~tsx
triggerWorkerError();
expect(dialog).toHaveFocus();
expect(shell).toHaveAttribute('inert');
expect(shell).toHaveAttribute('aria-hidden', 'true');
await user.tab();
expect(within(dialog).getByRole('button', { name: 'Refresh page' })).toHaveFocus();
~~~

In a hook harness cover two controls, zero controls, disabled/hidden exclusion,
dynamic insertion/removal, focus outside redirected by `focusin`, forward and
reverse wrap, exact preservation of preexisting attributes, deactivation and
unmount restoration, detached opener without throw, and StrictMode. Assert the
portal dialog is not inside the shell and Header's body-level help target is
also inert/hidden while the modal is open.

Explicitly activate the hook, append/render a new direct body portal, wait for
it to receive inert plus the owned aria-hidden value, then deactivate/unmount
and prove its exact preexisting values are restored. In appShell integration,
mount the real Header and prove its actual body help portal is leased; App and
training integration mocks alone do not cover that path.

In App/integration tests focus the skip link before the worker error, assert the
named/described dialog receives focus, then prove Escape leaves the error and
drawer/Advanced Tools state unchanged and all three global shortcuts invoke no
training callback. Clearing the store error restores the skip link and removes
only owned background attributes. Extend the real worker-error integration path
with dialog focus/background assertions.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useModalFocusContainment.test.tsx src/App.test.tsx src/__tests__/training.integration.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Expected: FAIL because only initial container focus exists; background/portals
remain exposed, focus can escape, and global handlers still act behind it.

- [ ] **Step 3: Implement the focused hook and background wrapper**

Use one activation effect that owns prior focus, background leases, the body
child observer, keydown containment, and `focusin` containment. Preserve each
node's `hasAttribute` state and exact value before changing it. The focusable
query includes enabled links/buttons/inputs/selects/textarea/contenteditable and
nonnegative tabindex, excluding hidden, disabled, inert, or aria-hidden
ancestors. Do not use cached focusable lists.

The lease set is `backgroundRef.current` plus every direct body child that does
not contain the dialog; collapse duplicate/nested ownership safely and exclude
the modal portal/ancestors. All `document`, `window`, and `MutationObserver`
access in the hook remains inside the active effect.

Move the existing alertdialog to a render-safe portal guarded as
`typeof document === 'undefined' ? null : createPortal(..., document.body)` and
wrap the normal skip link, announcer, Header, main workspace, and StatusBar in
the ref-owned `.forge-shell`. Guard both App global key handlers when
`workerError` is active; the hook's capture handler independently contains
Escape/Tab.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useModalFocusContainment.test.tsx src/App.test.tsx src/__tests__/training.integration.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Expected: PASS for dynamic forward/reverse containment, programmatic escape,
single-modal background/portal ownership, StrictMode, restoration, App handler
suppression, and the real worker-error path.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/hooks/useModalFocusContainment.ts apps/web/src/hooks/useModalFocusContainment.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/__tests__/training.integration.test.tsx apps/web/src/__tests__/appShell.integration.test.tsx
git commit -m "feat(web): contain worker error focus"
~~~

### Task 20: Enlarge compact graph and evidence controls

**Quick win:** 20 — Mobile graph and evidence targets.

**Files:**
- Modify: apps/web/src/styles/forge.css
- Modify: apps/web/src/styles/forgeResponsive.test.ts
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Validate the default Canvas graph path (`canvasNetworkGraph: true`) in Run with
  the Boundary tab active and no drawer open; this task does not claim SVG
  fallback control parity.
- At 390 by 844, all 13 unique graph toolbar, topology-mode, edge-filter, and
  decision-overlay controls have at least 44px bounding width and height after
  each is scrolled fully into the viewport.
- The document and forge shell retain no horizontal overflow. The toolbar and
  edge legend may remain intentional inner scrollers, but every target must be
  reachable through that scrolling.
- Enlarging the toolbar must retain the existing six-pixel gap to
  `.network-graph-summary`; the two boxes never overlap.
- Define one module-level graph/evidence locator and assertion helper in the
  smoke spec for Task 29 to reuse; do not duplicate the 13-name inventory.

- [ ] **Step 1: Add failing compact target assertions**

Add a focused `390px graph and evidence targets` browser test with viewport 390
by 844. After `loadPlayground(page)`, explicitly select/assert Boundary and
assert no drawer dialog is open. Scope exact locators to Network graph toolbar,
Topology view mode, Edge weight legend, the Boundary tabpanel, and its Decision
overlay controls.

~~~ts
function graphAndEvidenceTargets(page: Page): readonly Locator[] {
    const toolbar = page.getByRole('toolbar', { name: 'Network graph toolbar' });
    const modes = toolbar.getByRole('group', { name: 'Topology view mode' });
    const legend = page.getByLabel('Edge weight legend', { exact: true });
    const boundaryPanel = page.getByRole('tabpanel', { name: 'Boundary', exact: true });
    const overlays = boundaryPanel.getByLabel('Decision overlay controls', { exact: true });
    return [
        toolbar.getByRole('button', { name: 'Zoom out graph', exact: true }),
        toolbar.getByRole('button', { name: 'Zoom in graph', exact: true }),
        toolbar.getByRole('button', { name: 'Fit graph to view', exact: true }),
        modes.getByRole('button', { name: 'Weights', exact: true }),
        modes.getByRole('button', { name: 'Activations', exact: true }),
        legend.getByRole('button', { name: 'Show all edges', exact: true }),
        legend.getByRole('button', { name: 'Show only strong edges', exact: true }),
        legend.getByRole('button', { name: 'Show positive edges', exact: true }),
        legend.getByRole('button', { name: 'Show negative edges', exact: true }),
        overlays.getByRole('button', { name: 'Output', exact: true }),
        overlays.getByRole('button', { name: 'Uncertain', exact: true }),
        overlays.getByRole('button', { name: 'Errors', exact: true }),
        overlays.getByRole('button', { name: 'Split', exact: true }),
    ];
}

async function expectGraphAndEvidenceTargets(page: Page): Promise<void> {
    for (const target of graphAndEvidenceTargets(page)) {
        await expect(target).toHaveCount(1);
        await target.scrollIntoViewIfNeeded();
        await expectFullyInViewport(page, target);
        await expectMinimumTouchTarget(target);
    }
}

test.describe('390px graph and evidence targets', () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
    test('keeps every compact graph and evidence target reachable', async ({ page }) => {
        await loadPlayground(page);
        const boundaryTab = page.getByRole('tab', { name: 'Boundary', exact: true });
        await boundaryTab.click();
        await expect(boundaryTab).toHaveAttribute('aria-selected', 'true');
        await expect(page.getByRole('dialog')).toHaveCount(0);
        await expectGraphAndEvidenceTargets(page);

        const overflow = await page.evaluate(() => {
            const shell = document.querySelector<HTMLElement>('.forge-shell');
            if (!shell) throw new Error('forge shell is missing');
            return {
                document: document.documentElement.scrollWidth
                    <= document.documentElement.clientWidth + 1,
                shell: shell.scrollWidth <= shell.clientWidth + 1,
            };
        });
        expect(overflow).toEqual({ document: true, shell: true });

        const toolbar = page.getByRole('toolbar', { name: 'Network graph toolbar' });
        const summary = page.locator(
            '.forge-buildrun__topology-stage .network-graph-summary',
        );
        const toolbarBox = await toolbar.boundingBox();
        const summaryBox = await summary.boundingBox();
        expect(toolbarBox).not.toBeNull();
        expect(summaryBox).not.toBeNull();
        if (toolbarBox && summaryBox) {
            expect(summaryBox.y - (toolbarBox.y + toolbarBox.height))
                .toBeGreaterThanOrEqual(6);
        }
    });
});
~~~

Use a 1px tolerance only for viewport-edge math, never for the 44px minimum.
Assert document/shell client and scroll widths match after inner scrolling, and
require `summary.y - (toolbar.y + toolbar.height) >= 6`.

Add a brace-balanced `extractMediaBlocks` helper to the static CSS test because
forge.css has multiple max-width 900px blocks. Require the exact grouped target
rule and summary offset in the same block; a broad cross-block regex is a
false-green.

~~~ts
function extractMediaBlocks(css: string, header: string): string[] {
    const blocks: string[] = [];
    let cursor = 0;
    while (true) {
        const headerIndex = css.indexOf(header, cursor);
        if (headerIndex < 0) return blocks;
        const open = css.indexOf('{', headerIndex);
        if (open < 0) throw new Error(`Missing block for ${header}`);
        let depth = 1;
        let index = open + 1;
        for (; index < css.length && depth > 0; index += 1) {
            if (css[index] === '{') depth += 1;
            if (css[index] === '}') depth -= 1;
        }
        if (depth !== 0) throw new Error(`Unclosed block for ${header}`);
        blocks.push(css.slice(open + 1, index - 1));
        cursor = index;
    }
}

it('touch-sizes compact graph and evidence controls in one owning media block', () => {
    const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');
    const compactBlocks = extractMediaBlocks(css, '@media (max-width: 900px)');
    const owningBlock = compactBlocks.find((block) => {
        const targets = block.match(
            /\.forge-buildrun__topology-stage \.network-graph-toolbar button\s*,\s*\.forge-buildrun__topology-stage \.network-graph-legend__filter\s*,\s*\.forge-buildrun__evidence-body \.decision-overlay-controls button\s*\{([^}]*)\}/,
        );
        const summary = block.match(
            /\.forge-buildrun__topology-stage \.network-graph-summary\s*\{([^}]*)\}/,
        );
        return Boolean(
            targets
            && /min-width:\s*44px/.test(targets[1])
            && /min-height:\s*44px/.test(targets[1])
            && summary
            && /top:\s*62px/.test(summary[1]),
        );
    });
    expect(owningBlock).toBeDefined();
});
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "390px graph and evidence targets"

Expected: the new static test FAILS because the scoped rule/offset are absent;
build PASSES and refreshes dist; Chromium E2E FAILS because controls measure 24
to 28px. The existing small toolbar still has its six-pixel gap in RED; the gap
assertion is the regression guard that would fail if controls were enlarged
without the scoped summary offset.

- [ ] **Step 3: Add compact hit-area styles without scaling graph content**

~~~css
@media (max-width: 900px) {
    .forge-buildrun__topology-stage .network-graph-toolbar button,
    .forge-buildrun__topology-stage .network-graph-legend__filter,
    .forge-buildrun__evidence-body .decision-overlay-controls button {
        min-width: 44px;
        min-height: 44px;
    }
    .forge-buildrun__topology-stage .network-graph-summary { top: 62px; }
}
~~~

Keep the current six-pixel toolbar/summary gap when the toolbar grows from 32px
to 56px. Preserve visual label density with padding/wrapping and intentional
inner overflow; do not transform or scale graph content.

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
- Preserve Task 17's native Train ratio label/output association. The help
  button is a sibling of its `<label>`, never nested inside or substituted for
  the label.
- Call `useAudienceGuidanceLevel()` unconditionally before DataPanel and
  HyperparamPanel recipe/prepared early returns; TrainingControls already calls
  it unconditionally. Readiness transitions must not change hook order.

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

For DataPanel, wrap a sibling label and help control rather than replacing the
Task 17 label:

~~~tsx
<span className="control-label">
    <label htmlFor={trainRatioId}>Train ratio</label>
    <ConceptHelp conceptId="train-test-split" guidanceLevel={guidanceLevel} />
</span>
~~~

- [ ] **Step 4: Run GREEN and typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/concepts/conceptCatalog.test.ts src/components/controls/HyperparamPanel.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/TrainingControls.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

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
- Run one shared compact reachability journey at both 320 by 844 and 390 by 844 using generated, uniquely named tests.
- At both widths assert no document/shell horizontal overflow, all critical and graph/evidence controls from Task 20 are at least 44 by 44, active evidence tabs remain visible, compact evaluation outcome from Task 21 is reachable, drawers close and restore focus, and browser errors remain empty.
- Reuse Task 20's one module-level graph/evidence locator/assertion helper. After
  any Code/drawer/profile journey state, explicitly return to Run + Boundary
  before calling it. Replace Task 20's standalone 390 test when the matrix
  absorbs it so the locator inventory does not drift or run twice.
- At both widths reach the Task 16 Keyboard shortcuts summary, prove touch
  activation opens the native disclosure without starting/resetting training,
  close it, then focus it and prove native Space activation opens it without
  the global training shortcut intercepting the key.
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

Keep locators and assertions in one function to prevent drift. Before activating
the Task 16 shortcut summary, call `scrollIntoViewIfNeeded()`, assert it is fully
inside the viewport, and run `expectMinimumTouchTarget(summary)`. Snapshot the
status bar `data-status`, current step, current-run `data-model-generation`, and
current-run `data-model-revision`; require that exact snapshot after touch open,
touch close, and native Space open. For the keyboard action use
`await summary.focus()`, assert focus, then `await summary.press('Space')`.
Extend the same journey with Task 20 graph/evidence targets, Task 21 outcome
disclosure, active-tab-in-viewport measurement, and the existing focus
restoration checks.

Before the Task 20 assertions, explicitly select Run and the exact Boundary tab,
assert no drawer dialog is open, then call the existing shared helper. Remove
the prior standalone `390px graph and evidence targets` test rather than copying
its 13 accessible names into this journey.

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
- Modify: apps/web/src/vite-env.d.ts
- Modify: package.json
- Modify: playwright.config.ts

**Interfaces:**
- This task runs after and consumes Task 19's worker-modal containment; do not
  reimplement focus trapping, background leases, or shortcut suppression in the
  fault seam.
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

Name the enabled recovery test with tag `@fault-enabled` and the normal-build
guard with tag `@fault-disabled`. The enabled test navigates with the query,
expects the named/described alertdialog to receive focus and the shell to be
inert/aria-hidden, proves Tab and Shift+Tab remain on the Refresh page action,
then activates it. After the real reload, wait for the modal to disappear,
background attributes to clear, and real evidence convergence: status step 0,
full-evaluation step 0, and checkpoint timeline step 0 must agree. Then click
`Run one training step` and wait for step 1; do not assert old-document focus
restoration across reload. The disabled-build test navigates with the same query,
proves no alertdialog through initial evidence convergence, and also completes
one real step, so an early idle state cannot false-pass. Assert no unexpected
console/page errors beyond the deliberately injected labeled error.

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

Wrap the hook harness in StrictMode and prove effect replay queues exactly one
bridge error after the replacement subscription, while the sticky suppression
ref short-circuits both replayed mount/prepared effects and initialization stays
at zero until the page reloads. Consuming the session marker on the replay must
not re-enable initialization in the same document.

- [ ] **Step 4: Run GREEN units and production recovery**

Run: pnpm --filter @nn-playground/web exec vitest run src/testing/e2eFaults.test.ts src/worker/workerBridge.test.ts src/hooks/useTraining.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm run build:e2e

Run: pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-enabled"

Expected: Unit tests PASS. Both browsers show the genuine focused modal, reload through the real recovery control, reach ready state, and advance training with zero retries.

- [ ] **Step 5: Prove the normal build ignores the query**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-disabled"

Expected: PASS: the query cannot inject a fault in a normal production build;
both browsers converge on step-0 evidence with no modal and complete one real
step.

- [ ] **Step 6: Commit**

~~~bash
git add apps/web/src/testing/e2eFaults.ts apps/web/src/testing/e2eFaults.test.ts tests/e2e/worker-recovery.spec.ts apps/web/src/worker/workerBridge.ts apps/web/src/worker/workerBridge.test.ts apps/web/src/hooks/useTraining.ts apps/web/src/hooks/useTraining.test.tsx apps/web/src/vite-env.d.ts package.json playwright.config.ts
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
