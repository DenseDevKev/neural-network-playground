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
  prior legacy profile-selector query is part of the executable RED.
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

Expected: FAIL because the current legacy accessible/visible labels do not match;
the updated Playwright helper cannot yet resolve Workspace profile.

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
- Modify: tests/e2e/playground-smoke.spec.ts

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

Update the shared Playwright `statusBar(page)` helper from role `status` to role
`group`. In the existing `training can pause, single-step, and restore the
initial checkpoint` test, assert the Status bar is readable and has no
`[aria-live]`, `[role="status"]`, or `[role="alert"]` ancestor. This makes the
app-wide role change executable in the built artifact rather than leaving the
next smoke task with a stale locator.

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

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "training can pause"

Expected: FAIL because multiple volatile metric/config surfaces are live, reset
ownership is status-only, completions/supersession are not modeled, config
panels duplicate the central announcer, and the built Status bar is not a group.

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

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "training can pause"

Expected: PASS with volatile metrics quiet, one central DOM write per observed
meaningful transition, no panel duplicate, and lifecycle/store integrations
unchanged in units and both built browsers. This guarantees the app-owned
live-region mutation contract; actual speech coalescing remains
assistive-technology behavior.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/components/layout/AccessibilityAnnouncer.tsx apps/web/src/components/layout/AccessibilityAnnouncer.test.tsx apps/web/src/components/layout/ExperimentStateContext.tsx apps/web/src/components/layout/ExperimentStateContext.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/components/common/LoadingState.tsx apps/web/src/components/common/LoadingState.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/FeaturesPanel.tsx apps/web/src/components/controls/FeaturesPanel.test.tsx apps/web/src/components/controls/NetworkConfigPanel.tsx apps/web/src/components/controls/NetworkConfigPanel.test.tsx apps/web/src/components/controls/HyperparamPanel.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/components/controls/PresetPanel.tsx apps/web/src/components/controls/PresetPanel.test.tsx apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx tests/e2e/playground-smoke.spec.ts
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
- Enlarging the toolbar must establish at least a six-pixel gap to
  `.network-graph-summary`; the current compact baseline is only two pixels and
  the two boxes must never overlap.
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
to 28px and the compact baseline has only a two-pixel toolbar/summary gap. The
gap assertion proves the new layout establishes six pixels and guards against
enlarging controls without the scoped summary offset.

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

Establish a six-pixel toolbar/summary gap when the toolbar grows to 56px.
Preserve visual label density with padding/wrapping and intentional
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
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- At max-width 900px, expose one compact native disclosure named `Evaluation outcome` while the desktop metric cluster remains unchanged.
- Derive the disclosure only from the existing `evidence.fullEvaluation`. Its
  summary is `Step {localized step} · Test accuracy {one-decimal percent}` when
  `testAccuracy != null`, otherwise `Step {localized step} · Test data loss
  {four decimals}`. A real zero-percent accuracy remains classification.
- Its body names `Full evaluation at step …`, train and test full-split data
  losses, and test accuracy only when available. It never duplicates the live
  batch EMA as full-split evidence.
- Missing evidence is reported as `Not evaluated yet`; the disclosure never substitutes the live batch EMA for full-split evidence.
- The disclosure is ordinary non-live content, preserving Task 18's single
  announcement owner. It is closed by default, keyboard/touch operable, at
  least 44px high at compact widths, fully reachable at 320px, and opening it
  does not create document or shell horizontal overflow.

- [ ] **Step 1: Write failing semantic and compact-CSS tests**

Use `within(...)` because desktop and compact markup intentionally repeat metric
values. Cover classification, regression, zero-percent accuracy, and a non-null
live batch EMA with null full evaluation. Assert the last case says `Not
evaluated yet` and contains neither the EMA value nor `Batch trend`. Prove the
native disclosure begins closed, its body becomes visible only after activation,
and neither it nor an ancestor has live/status/alert semantics.

~~~tsx
const metrics = screen.getByRole('group', { name: 'Training metrics' });
const outcome = screen.getByRole('group', { name: 'Evaluation outcome' });
const summary = within(outcome).getByText(
    'Step 1,230 · Test accuracy 49.3%',
    { selector: 'summary' },
);
expect(within(metrics).getByText('0.2345')).toBeInTheDocument();
expect(outcome).not.toHaveAttribute('open');
expect(outcome.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
await user.click(summary);
expect(outcome).toHaveAttribute('open');
expect(outcome).toHaveTextContent('Full evaluation at step 1,230');
expect(outcome).toHaveTextContent('Train data loss (full split) 0.2345');
expect(outcome).toHaveTextContent('Test data loss (full split) 0.5678');
expect(outcome).toHaveTextContent('Test accuracy 49.3%');
~~~

Add a brace-balanced CSS assertion using Task 20's `extractMediaBlocks`. Require
the base disclosure rule to be `display: none`; require one owning max-width
900px block to contain the complete compact disclosure, summary, and `[open]`
body rules below. Rule-local assertions must prove the 44px target, wrapping,
full-width flex behavior, and absence of a fixed pixel/rem/em width.

Extend the existing `320px touch shell` Playwright journey after
`loadPlayground(page)`: the outcome and summary are visible/fully in viewport,
the desktop `Training metrics` group is hidden, the summary meets the 44px
target, its step-0 full-evaluation copy opens, and the existing document/shell
overflow assertion is repeated while open. Task 29 later parameterizes these
same assertions at 390px rather than duplicating them.

- [ ] **Step 2: Run the failing coverage-gate RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "320px touch shell"

Expected: Vitest FAILS because compact view hides the complete metric cluster
with no replacement. The build succeeds and refreshes `dist`; Chromium FAILS
the new missing compact-outcome assertion.

- [ ] **Step 3: Add the compact disclosure from existing evidence**

~~~tsx
const fullEvaluation = evidence.fullEvaluation;
const compactPrimaryOutcome = fullEvaluation === null
    ? 'Not evaluated yet'
    : accuracy != null
        ? `Test accuracy ${accStr}`
        : `Test data loss ${testLoss}`;
const compactOutcomeSummary = fullEvaluation === null
    ? compactPrimaryOutcome
    : `Step ${fullEvaluation.step.toLocaleString()} · ${compactPrimaryOutcome}`;

<details className="forge-compact-outcome" aria-label="Evaluation outcome">
    <summary>{compactOutcomeSummary}</summary>
    {fullEvaluation && (
        <div className="forge-compact-outcome__body">
            <span>{`Full evaluation at step ${fullEvaluation.step.toLocaleString()}`}</span>
            <span>{`Train data loss (full split) ${trainLoss}`}</span>
            <span>{`Test data loss (full split) ${testLoss}`}</span>
            {accuracy != null && <span>{`Test accuracy ${accStr}`}</span>}
        </div>
    )}
</details>
~~~

Place it next to the desktop metrics and before the topbar spacer. Reuse the
existing four-decimal loss and one-decimal accuracy formatting. Add these exact
rule bodies; keeping author `display: grid` behind `[open]` is required so CSS
cannot defeat native closed-details hiding:

~~~css
.forge-compact-outcome { display: none; }
.forge-compact-outcome > summary:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
}
@media (max-width: 900px) {
    .forge-compact-outcome {
        display: block;
        flex: 1 0 100%;
        min-width: 0;
        max-width: 100%;
        box-sizing: border-box;
    }
    .forge-compact-outcome > summary {
        display: flex;
        align-items: center;
        min-width: 0;
        min-height: 44px;
        box-sizing: border-box;
        padding: 6px 10px;
        cursor: pointer;
        overflow-wrap: anywhere;
    }
    .forge-compact-outcome[open] > .forge-compact-outcome__body {
        display: grid;
        gap: 4px;
        min-width: 0;
        padding: 6px 10px 10px;
        overflow-wrap: anywhere;
    }
}
~~~

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "320px touch shell"

Expected: PASS for classification, regression, zero accuracy, empty evidence,
native disclosure semantics, compact reachability, and zero global overflow in
both browser engines.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts tests/e2e/playground-smoke.spec.ts
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
- Modify: apps/web/src/styles/forge.css
- Modify: apps/web/src/styles/forgeResponsive.test.ts
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Extend `ConceptId` with `learning-rate`, `train-test-split`, and `epoch`.
- Append the three IDs after the existing six IDs so catalog and profile order
  remain stable. Every entry includes exact reviewed copy, aliases, an example,
  related concepts, difficulty, all three profiles, and a valid UI target where
  applicable: `learning-rate` targets `hyperparams`, `train-test-split` targets
  `data`, and `epoch` deliberately has no target.
- Place `ConceptHelp` beside the visible Learning rate label, Train ratio label,
  and Epoch label; guidance remains controlled by the current audience profile.
  Give all three the dedicated `concept-help--viewport-overlay` modifier so the
  disclosure is fixed to the viewport instead of clipped by its control module.
- Preserve Task 17's native Train ratio label/output association. The help
  button is a sibling of its `<label>`, never nested inside or substituted for
  the label.
- Call `useAudienceGuidanceLevel()` unconditionally before DataPanel and
  HyperparamPanel recipe/prepared early returns; TrainingControls already calls
  it unconditionally. Readiness transitions must not change hook order.
- One epoch in this worker is one complete pass through the current training
  set: indices are shuffled without replacement, every example is consumed
  once, the final batch may be short, and the integer epoch increments only
  after that pass. Do not describe sampling with replacement.
- The Epoch help retains `concept-help--above concept-help--end` alongside the
  new viewport-overlay modifier in the bottom transport. A sufficiently
  specific later CSS rule fixes viewport overlays 12px from the right and 36px
  above the bottom; it overrides label-relative `above`/`end` offsets without
  changing other concept help. The desktop and 320px browser journeys verify
  all three new help surfaces, 44px compact triggers, viewport bounds, and
  ancestor-clipping safety through corner hit-testing.

- [ ] **Step 1: Write failing catalog and placement tests**

Update the catalog's exact `EXPECTED_ORDER` and six-ID assertion, then lock the
complete reviewed metadata rather than checking only truthiness:

~~~ts
expect(CONCEPT_IDS).toEqual([
    'data-loss', 'training-objective', 'decision-boundary',
    'activation', 'gradient', 'checkpoint',
    'learning-rate', 'train-test-split', 'epoch',
]);
expect(getConceptById('learning-rate')).toMatchObject({
    canonicalTerm: 'Learning rate',
    plainDefinition:
        'The learning rate sets the scale of each optimizer update to the model’s parameters.',
    extendedExplanation:
        'The optimizer uses the learning rate to scale parameter updates. Larger values can move faster but may overshoot or make loss unstable; smaller values can be steadier but slower. A schedule can change the rate as training proceeds.',
    aliases: ['step size', 'optimizer learning rate', 'update scale'],
    related: ['gradient', 'training-objective'],
    profiles: ['beginner', 'explore', 'lab'],
    difficulty: 'beginner',
    examples: [
        'With plain SGD, learning rate 0.1 moves a parameter ten times as far as 0.01 for the same gradient.',
    ],
    uiTarget: 'hyperparams',
});
expect(getConceptById('train-test-split')).toMatchObject({
    canonicalTerm: 'Train/test split',
    plainDefinition:
        'The train/test split assigns generated examples to a training set used for fitting and a held-out test set used only for evaluation.',
    extendedExplanation:
        'Membership is deterministic and stays fixed while the current prepared experiment trains. Data-recipe changes or Reshuffle split rebuild membership. Test examples never drive weight updates; full-split test evidence measures held-out performance at the same model step.',
    aliases: ['data split', 'training test split', 'held-out split'],
    related: ['data-loss', 'training-objective'],
    profiles: ['beginner', 'explore', 'lab'],
    difficulty: 'beginner',
    examples: [
        'With 200 generated examples and a 70% train ratio, 140 train and 60 are held out for test.',
    ],
    uiTarget: 'data',
});
expect(getConceptById('epoch')).toMatchObject({
    canonicalTerm: 'Epoch',
    plainDefinition: 'One epoch is one complete pass through the current training set.',
    extendedExplanation:
        'The worker shuffles the training examples, processes each one once in mini-batches, and increments Epoch only after the full training set is consumed. The final batch may be smaller than the configured batch size. Epoch counts data passes, not convergence or model quality.',
    aliases: ['training epoch', 'data pass', 'full training pass'],
    related: ['learning-rate', 'checkpoint'],
    profiles: ['beginner', 'explore', 'lab'],
    difficulty: 'beginner',
    examples: [
        'With 150 training examples and batch size 64, one epoch completes after batches of 64, 64, and 22 examples.',
    ],
});
expect(getConceptById('epoch')?.uiTarget).toBeUndefined();
~~~

In each component test, open the exact adjacent controls `Learn about Learning
rate`, `Learn about Train/test split`, and `Learn about Epoch`; assert their
named regions, canonical terms, and exact plain definitions. Prove DataPanel
and HyperparamPanel can rerender from incompatible/not-ready to ready without a
hook-order error. Explicitly reset `useLayoutStore` audience mode in both test
suites. Prove Beginner guidance includes extended copy and the example while
Lab retains the plain definition but omits optional extended/example content.
For DataPanel, reassert the Task 17 label/output IDs, `for` ownership,
`aria-describedby`, `aria-valuetext`, and label/help sibling relationship.
For TrainingControls, retain an exact inner `Epoch N` text span and assert the
help sibling carries `concept-help--above concept-help--end`.

Extend `expectConceptHelpInViewport` in `playground-smoke.spec.ts` rather than
replacing its current terms; its union becomes `Data loss | Checkpoint | Epoch |
Train/test split | Learning rate`. Keep the existing trigger, open/close,
focus, Escape, and bounding-box checks for all five. For the three new
viewport-overlay concepts only, select the requirement from an explicit concept
name set (never by detecting the implementation class), assert the parent owns
`concept-help--viewport-overlay`, then sample every disclosure corner four
pixels inward with `document.elementFromPoint`. This proves the three new boxes
are not clipped by an overflow ancestor without redefining the existing static
Data loss or timeline Checkpoint surfaces:

~~~ts
const viewportOverlayConcepts = new Set([
    'Epoch',
    'Train/test split',
    'Learning rate',
]);
if (viewportOverlayConcepts.has(concept)) {
    await expect(panel.locator('..')).toHaveClass(/concept-help--viewport-overlay/);
    const cornersAreExposed = await panel.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        const points = [
            [rect.left + 4, rect.top + 4],
            [rect.right - 4, rect.top + 4],
            [rect.left + 4, rect.bottom - 4],
            [rect.right - 4, rect.bottom - 4],
        ];
        return points.every(([x, y]) => {
            const hit = document.elementFromPoint(x, y);
            return hit !== null && (hit === element || element.contains(hit));
        });
    });
    expect(cornersAreExposed).toBe(true);
}
~~~

In the desktop concept-help test, verify Epoch in Run, then switch to Build and
verify Train/test split and Learning rate. In the 320px touch-shell journey do
the same, require each trigger to meet the existing 44px target and each open
region to be fully in the viewport; require the three new overlay regions to be
corner-exposed, then return to Run before the journey's existing Run-only
assertions. Leave Data loss's existing block placement and Checkpoint's existing
timeline placement unchanged.

Add a static CSS regression proving the exact scoped modifier owns fixed
positioning and the complete inset. The selector must be more specific than the
existing `concept-help--above` and `concept-help--end` rules and appear after
them:

~~~ts
expect(css).toMatch(
    /\.forge-shell \.concept-help\.concept-help--viewport-overlay \.concept-help__content\s*\{[^}]*position:\s*fixed;[^}]*inset:\s*auto 12px 36px auto;/,
);
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/concepts/conceptCatalog.test.ts src/components/controls/HyperparamPanel.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "concept help|320px touch shell"

Expected: Vitest FAILS because the concept IDs, contextual controls, and scoped
viewport-overlay rule are absent. The build succeeds and refreshes `dist`;
Chromium FAILS on the missing new help triggers. A runtime-only probe of the
planned label-relative placement measured Train/test split and Learning rate
outside or clipped by their modules at both desktop and 320px, so GREEN must
also satisfy corner hit-testing rather than bounding boxes alone.

- [ ] **Step 3: Add scientifically bounded copy and contextual controls**

Add the exact catalog entries above. Define learning rate as optimizer-update
scale, split as deterministic membership separating fitting from held-out
evaluation, and epoch as the worker's full without-replacement pass. State that
epoch counts data passes rather than convergence or model quality.

~~~tsx
<span className="control-label">
    Learning rate
    <ConceptHelp
        conceptId="learning-rate"
        guidanceLevel={guidanceLevel}
        className="concept-help--viewport-overlay"
    />
</span>
~~~

Use the existing component props/store selectors for `guidanceLevel`; do not introduce a second help implementation.

For DataPanel, wrap a sibling label and help control rather than replacing the
Task 17 label:

~~~tsx
<span className="control-label">
    <label htmlFor={trainRatioId}>Train ratio</label>
    <ConceptHelp
        conceptId="train-test-split"
        guidanceLevel={guidanceLevel}
        className="concept-help--viewport-overlay"
    />
</span>
~~~

Keep that label and help as siblings; do not remove or rename Task 17's input
and output IDs or accessible value text. In TrainingControls, wrap the exact
visible `Epoch {currentModel.epoch}` text and an adjacent help control:

~~~tsx
<span>
    <span>Epoch {currentModel.epoch}</span>
    <ConceptHelp
        conceptId="epoch"
        guidanceLevel={guidanceLevel}
        className="concept-help--above concept-help--end concept-help--viewport-overlay"
    />
</span>
~~~

Add the dedicated overlay rule after the existing `above`/`end` modifiers so
the stronger selector wins on desktop and compact screens while leaving every
other help surface unchanged:

~~~css
.forge-shell .concept-help.concept-help--viewport-overlay .concept-help__content {
    position: fixed;
    inset: auto 12px 36px auto;
}
~~~

- [ ] **Step 4: Run GREEN and typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/concepts/conceptCatalog.test.ts src/components/controls/HyperparamPanel.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/TrainingControls.test.tsx src/styles/forgeResponsive.test.ts --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "concept help|320px touch shell"

Expected: PASS with all concept metadata and placements typed, readiness-safe
hook order, profile-specific guidance, and fully visible help in both browser
engines at desktop and 320px, including explicit corner exposure for the three
new viewport-overlay concepts.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/concepts/conceptCatalog.ts apps/web/src/concepts/conceptCatalog.test.ts apps/web/src/components/controls/HyperparamPanel.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/styles/forge.css apps/web/src/styles/forgeResponsive.test.ts tests/e2e/playground-smoke.spec.ts
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
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Export a readonly `STATE_EFFECTS` map for `training-reset`, `reshuffle-split`, `lesson-start`, `preset-apply`, and `saved-recipe-apply`.
- Every value is one exact reviewed sentence beginning `Changes:` and containing
  `Preserves:`. Copy matches actual store/worker behavior: reset preserves the
  deterministic generated examples and split; reshuffle increments the data
  seed and replaces them; lesson/preset/saved-recipe application starts a fresh
  runtime; none of the five actions deletes experiment-memory records.
- Each action button directly owns a persistent `aria-describedby` reference to
  its shared string. Tooltip relationships on wrapper elements do not count.
  Descriptions are ordinary non-live content, unique across component
  instances, and components do not duplicate the literals.
- Rename both inaccurate legacy reset labels to the exact accessible and visible
  name `Reset training`.
- Replace RunHistoryPanel's hard-coded `run-name-guidance` ID with an
  instance-owned `useId` value while adding one visible saved-recipe effects
  note per panel. Apply buttons may share that note within one panel, but two
  mounted panels must not share IDs.

- [ ] **Step 1: Write failing contract and integration tests**

Lock key order, frozen identity, and the full exact map; a regex-only check is
insufficient:

~~~ts
const EXPECTED_STATE_EFFECTS = {
    'training-reset': 'Changes: reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: the current recipe, generated examples, train/test membership, lesson progress, and stored run evidence.',
    'reshuffle-split': "Changes: increments the recipe's data seed, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: every other recipe setting, lesson progress, and stored run evidence.",
    'lesson-start': "Changes: replaces the current recipe with the lesson recipe, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, replaces in-session checkpoints with a new step-0 checkpoint, and starts lesson progress at step 1; Preserves: the document's test-data and discretization options and stored run evidence.",
    'preset-apply': "Changes: applying a different preset replaces the recipe, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: the document's test-data and discretization options, lesson state, and stored run evidence.",
    'saved-recipe-apply': "Changes: replaces the current recipe with the saved recipe and starts a fresh step-0 run with regenerated examples and train/test membership, reinitialized model weights and optimizer state, new step-0 evaluation evidence, and a new in-session checkpoint; Preserves: the document's test-data and discretization options, lesson state, and every stored run record, but does not restore the saved run's trained parameters or evidence into the live run.",
} as const;
expect(Object.keys(STATE_EFFECTS)).toEqual(Object.keys(EXPECTED_STATE_EFFECTS));
expect(Object.isFrozen(STATE_EFFECTS)).toBe(true);
expect(STATE_EFFECTS).toEqual(EXPECTED_STATE_EFFECTS);
~~~

For each component, assert the action itself has `aria-describedby`, every
referenced ID resolves, and `toHaveAccessibleDescription` equals the matching
shared string. Render two component instances and prove their owned IDs are
unique. RunHistoryPanel may share one effects ID among Apply buttons within a
panel, but must use a different effects/run-name ID in the second panel.
Assert each resolved description is outside every `aria-live`, status, or alert
ancestor. Assert both TrainingControls and DataPanel visibly render the exact
button text `Reset training`, not merely that an overriding aria-label supplies
that name. Assert both DataPanel action tooltips expose the matching shared
constants so stale Cause/Effect copy cannot contradict their descriptions.
Preserve Zustand/localStorage isolation and update the existing Guided Lesson
consequence assertion and reset accessible-name queries. Include
`PresetPanel.test.tsx` in the gate because it proves the actual preset-apply
behavior even though it need not change.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/copy/stateEffects.test.ts src/components/controls/TrainingControls.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/controls/PresetCard.test.tsx src/components/controls/PresetPanel.test.tsx src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the map, direct action descriptions, unique panel IDs,
and accurate reset names do not exist.

- [ ] **Step 3: Define and consume one authoritative copy map**

~~~ts
export const STATE_EFFECTS = Object.freeze({
    'training-reset': 'Changes: reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: the current recipe, generated examples, train/test membership, lesson progress, and stored run evidence.',
    'reshuffle-split': "Changes: increments the recipe's data seed, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: every other recipe setting, lesson progress, and stored run evidence.",
    'lesson-start': "Changes: replaces the current recipe with the lesson recipe, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, replaces in-session checkpoints with a new step-0 checkpoint, and starts lesson progress at step 1; Preserves: the document's test-data and discretization options and stored run evidence.",
    'preset-apply': "Changes: applying a different preset replaces the recipe, regenerates examples and train/test membership, reinitializes model weights and optimizer state, resets training progress and live evidence, and replaces in-session checkpoints with a new step-0 checkpoint; Preserves: the document's test-data and discretization options, lesson state, and stored run evidence.",
    'saved-recipe-apply': "Changes: replaces the current recipe with the saved recipe and starts a fresh step-0 run with regenerated examples and train/test membership, reinitialized model weights and optimizer state, new step-0 evaluation evidence, and a new in-session checkpoint; Preserves: the document's test-data and discretization options, lesson state, and every stored run record, but does not restore the saved run's trained parameters or evidence into the live run.",
} as const);
~~~

Keep production behavior unchanged. In TrainingControls, make Reset directly
describe a persistent hidden `training-reset` string and let its Tooltip consume
the same constant. In DataPanel, derive separate reshuffle/reset effect IDs from
the existing unconditional Task 17 ID stem; the Reshuffle button references
`reshuffle-split`, while its reset button is renamed `Reset training` and
references `training-reset`. Both DataPanel action Tooltips must consume those
same matching constants; remove their contradictory bespoke Cause/Effect text.
In GuidedLessonPanel, give the visible shared
lesson-start consequence paragraph an instance ID and reference it from Start.
PresetCard uses `useId`, describes its button directly, and feeds the same
`preset-apply` constant to its Tooltip. RunHistoryPanel uses an instance-owned
visible `saved-recipe-apply` note referenced by every Apply button, plus an
instance-owned run-name guidance ID.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/copy/stateEffects.test.ts src/components/controls/TrainingControls.test.tsx src/components/controls/DataPanel.test.tsx src/components/controls/GuidedLessonPanel.test.tsx src/components/controls/PresetCard.test.tsx src/components/controls/PresetPanel.test.tsx src/components/controls/RunHistoryPanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/engine exec vitest run src/__tests__/datasets.test.ts --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec vitest run src/worker/training.worker.v2.test.ts src/hooks/useTraining.test.tsx src/store/usePlaygroundStore.test.ts --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "curated presets|saved runs survive|touch shell"

Expected: PASS with every action directly tied to shared accurate descriptions,
reset/reshuffle/runtime invariants unchanged, the compact reset locator renamed
to `Reset training`, and no saved-run loss.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/copy/stateEffects.ts apps/web/src/copy/stateEffects.test.ts apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/components/controls/GuidedLessonPanel.tsx apps/web/src/components/controls/GuidedLessonPanel.test.tsx apps/web/src/components/controls/PresetCard.tsx apps/web/src/components/controls/PresetCard.test.tsx apps/web/src/components/controls/RunHistoryPanel.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx tests/e2e/playground-smoke.spec.ts
git commit -m "feat(web): standardize state effect disclosures"
~~~

### Task 24: Make timed state use current inputs with a stable setter

**Quick win:** 24 — Stable `useTimedState`.

**Files:**
- Modify: apps/web/src/hooks/useTimedState.ts
- Create: apps/web/src/hooks/useTimedState.test.tsx

**Interfaces:**
- Preserve the exact public return type `[T, (value: T) => void]`; the setter
  does not accept React functional updates.
- The setter retains identity across rerenders, including React StrictMode.
- `defaultValue` is the initial value and reset target. Changing it does not
  immediately overwrite the visible value; an active timeout reads the latest
  rendered `defaultValue` when it fires.
- Each call captures the current `duration`. Changing duration never reschedules
  an active timeout, while a later call through the same stable setter uses the
  new duration.
- A replacement call clears the prior timer exactly once and leaves one active
  timer. Unmount clears once, nulls the timer ref, and permanently makes a
  retained setter a no-op so late async completions cannot schedule orphaned
  work. A timeout updates state only while it still owns the active ref.

- [ ] **Step 1: Write failing fake-timer rerender tests**

~~~tsx
const firstSetter = result.current[1];
act(() => firstSetter('saved'));
rerender({ defaultValue: 'ready', duration: 50 });
expect(result.current[1]).toBe(firstSetter);
act(() => vi.advanceTimersByTime(100));
expect(result.current[0]).toBe('ready');
~~~

Wrap every hook render in StrictMode. Prove changing duration does not move an
active 100ms deadline, then a call made through the retained setter uses the
new 10ms duration. Spy on `clearTimeout` to prove replacement and unmount each
clear once, timer count returns to zero, and a retained setter invoked after
unmount creates no timer. After enabling fake timers, spy on
`globalThis.setTimeout` while delegating to the fake-timer implementation and
capture the hook timer deterministically by installing/clearing the spy
immediately before the first setter call or selecting the call with the expected
duration. Replace its timer with a second setter call, manually invoke the
captured stale callback, and prove it cannot reset or overwrite the newer
visible value; advancing the owned second timer still performs the one valid
reset. Assert the exact tuple type with `expectTypeOf`. In `afterEach`, use the
exact order `vi.clearAllTimers(); vi.restoreAllMocks(); vi.useRealTimers();` so
restoring the spy cannot reinstall an already-uninstalled fake `setTimeout`.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useTimedState.test.tsx --pool=forks --reporter=dot

Expected: FAIL for unstable setter identity/stale default, stale duration through
the retained setter, a retained post-unmount setter creating a timer, and the
captured stale callback overwriting the replacement value.

- [ ] **Step 3: Synchronize refs and memoize the setter**

~~~ts
export function useTimedState<T>(
    defaultValue: T,
    duration: number,
): [T, (value: T) => void] {
    const [value, setValue] = useState<T>(defaultValue);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const defaultValueRef = useRef(defaultValue);
    const durationRef = useRef(duration);
    const mountedRef = useRef(true);

    defaultValueRef.current = defaultValue;
    durationRef.current = duration;

    const clearPendingTimeout = useCallback(() => {
        if (timeoutRef.current === null) return;
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
    }, []);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            clearPendingTimeout();
        };
    }, [clearPendingTimeout]);

    const setTimed = useCallback((next: T) => {
        if (!mountedRef.current) return;
        clearPendingTimeout();
        setValue(next);
        const timeoutId = setTimeout(() => {
            if (timeoutRef.current !== timeoutId) return;
            timeoutRef.current = null;
            if (mountedRef.current) setValue(defaultValueRef.current);
        }, durationRef.current);
        timeoutRef.current = timeoutId;
    }, [clearPendingTimeout]);

    return [value, setTimed];
}
~~~

Import `useCallback`. Keep all changing values behind refs, re-arm
`mountedRef` in effect setup for StrictMode, and null ownership on every clear
or successful timeout.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useTimedState.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web exec vitest run src/components/controls/CodeExportPanel.test.tsx src/components/controls/ConfigPanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Expected: PASS with deterministic timer ownership, latest-input semantics,
StrictMode-safe cleanup, post-unmount no-op behavior, and unchanged consumers.

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
- Add required prop `storageId: string`. In development, reject values not
  matching `^[a-z0-9]+(?:-[a-z0-9]+)*$` with a deterministic error before any
  storage access. Production skips that assertion and uses the supplied ID
  verbatim; never normalize it. Persist under `panel-v2-${storageId}`.
- `storageId` is immutable for a mounted component. Capture the initial value;
  a development rerender with a different value throws, while production keeps
  the captured identity. A caller changing identity must remount with
  `key={storageId}`. Changing only `title` retains React state and the v2 key.
- On initial mount, read the v2 key first. Exact `true` and `false` are valid. A
  valid v2 value wins without reading/touching legacy state. Invalid non-null v2
  falls back to `defaultExpanded`, is best-effort removed after commit, and
  never revives the legacy key.
- Only when v2 is absent, read the exact former key
  `panel-${initialTitle.toLowerCase().replace(/\s+/g, '-')}`. Valid legacy state
  initializes memory, then migrates after commit: write v2 first and remove
  legacy only if that write succeeds. A failed v2 write retains legacy; a failed
  legacy removal retains both. Invalid legacy state falls back to the default,
  writes no v2 value, and is best-effort removed.
- Throwing/unavailable `localStorage`, `getItem`, `setItem`, or `removeItem`
  never escapes; reads fall back and toggles still update memory. No storage
  write occurs during render. Guard the initial read so it is not re-executed
  as a `useRef(read(...))` argument on rerenders, and clear pending migration
  ownership before the effect attempt so StrictMode replay cannot repeat it.
- A storage/getItem exception is not the same as an absent key: stop that
  initialization/migration attempt, use `defaultExpanded`, and perform no
  further storage read/write/remove. Apply the same stop rule if legacy read
  itself throws.
- Unique IDs prevent simultaneous v2 collisions; duplicate IDs intentionally
  share storage and remain a caller error. This component remains client-only
  because its Tooltip already portals to `document.body`; SSR is not added here.

- [ ] **Step 1: Write failing key, migration, and rename tests**

~~~tsx
localStorage.setItem('panel-old-title', 'false');
const { rerender } = render(<CollapsiblePanel storageId="network" title="Old Title">x</CollapsiblePanel>);
expect(localStorage.getItem('panel-v2-network')).toBe('false');
rerender(<CollapsiblePanel storageId="network" title="New Title">x</CollapsiblePanel>);
expect(screen.getByRole('button', { name: /New Title/ })).toHaveAttribute('aria-expanded', 'false');
~~~

Add exact tests for:

- empty, uppercase, underscore, leading/trailing-hyphen, and doubled-hyphen IDs
  throwing before any storage call in development;
- valid v2 precedence with no legacy read/removal; invalid v2 fallback/removal
  with no legacy revival;
- successful migration call order (`setItem(v2)` before `removeItem(legacy)`),
  invalid legacy default/no v2 write/removal, v2 write failure retaining legacy,
  and removal failure retaining both keys;
- throwing storage getter/getItem falling back, and throwing toggle setItem
  still updating React state;
- title rerender retaining state, performing no new legacy lookup, and writing
  the original v2 key; storageId rerender following the mount-stable contract;
- two same/similar-title panels with distinct IDs toggling independently;
- every existing lazy-mount, accessibility, ResizeObserver, and performance
  case updated with a durable semantic ID.

Capture the shared `window.localStorage` property descriptor outside the suite,
restore it after every test, then clear it. `vi.restoreAllMocks()` does not undo
`Object.defineProperty`; use prototype method spies for operation failures and
restore a throwing property getter in `finally` or through the descriptor.

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/CollapsiblePanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Expected: FAIL because the component neither accepts nor uses `storageId`,
persistence remains title-derived, and migration ordering/ownership plus
mount-stable identity are absent.

- [ ] **Step 3: Implement v2 identity and verify the call-site inventory**

Search rendered call sites with
`rg -n '<CollapsiblePanel\b' apps/web/src --glob '*.tsx'`, then separately audit
all symbol imports. The current inventory is zero production renderers and 11
test fixtures, all in `CollapsiblePanel.test.tsx`; update every fixture with a
durable ID and record the inventory. If a production renderer appears, amend
Files/staging before editing it.

Capture initial `storageId`/title, validate before storage I/O, and use guarded
initial-read metadata containing the expanded value plus at most one pending
post-commit cleanup/migration. In the effect, clear pending ownership before
attempting any operation. Use safe storage helpers for getter and method errors.
Keep the legacy normalizer private and only for the initial-title fallback.

~~~tsx
<CollapsiblePanel storageId="data" title="Data" defaultExpanded>
    <DataPanel />
</CollapsiblePanel>
~~~

Do not reset React state when only `title` changes. Continue treating storage failures as recoverable in-memory behavior.

- [ ] **Step 4: Run GREEN and typecheck**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/common/CollapsiblePanel.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm --filter @nn-playground/web build

Expected: PASS for the migration/error/identity matrix, both production and test
TypeScript configs, the production build, and no rendered fixture omitting
`storageId`.

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
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- The mounted app listens for `storage` events whose `storageArea` is localStorage and whose key is `EXPERIMENT_MEMORY_STORAGE_KEY`, `LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY`, or `null` from `localStorage.clear()`.
- Each accepted event invokes the store's existing queued `hydrate()` exactly once. Unrelated/session-storage events are ignored, and cleanup removes the listener.
- Native same-document localStorage writes do not emit `storage`; do not synthesize events in production or create a second persistence queue.
- Treat `storageArea: null` as unrelated. Safely capture `window.localStorage`
  once inside the effect; if its getter throws, install no listener.
- The singleton store already performs initial hydration at module construction;
  this hook handles later external changes. Accepted rapid events each enqueue
  one existing `hydrate`, but queued runs may all observe the newest bytes.
  Per-tab queues do not solve cross-tab last-writer-wins or distributed merge.
- Apply after Task 19. Mount the hook unconditionally at root App scope before
  the incompatible-document early return. Task 25 `panel-v2-*` keys remain
  ignored, while key null stays accepted because clear removes memory keys too.
- Import both exported memory-key constants; never duplicate strings. Native
  Chromium/WebKit coverage uses two pages to prove other-document delivery and
  the writing document's intentional lack of a storage event.

- [ ] **Step 1: Write failing lifecycle and filtering tests**

~~~tsx
renderHook(() => useExperimentMemoryStorageSync());
window.dispatchEvent(new StorageEvent('storage', {
    key: EXPERIMENT_MEMORY_STORAGE_KEY,
    storageArea: window.localStorage,
}));
expect(hydrate).toHaveBeenCalledTimes(1);
~~~

Wrap hook tests in StrictMode and prove exactly one active listener: current key,
legacy key, and localStorage `key: null` each invoke hydrate once. Unrelated key,
sessionStorage, and null storage area do not. After unmount, an otherwise valid
event does nothing. In one mounted StrictMode instance, dispatch current,
legacy, and null-key events and assert three hydrate calls, proving accepted
events are not coalesced. Except for the safe-getter test, use the browser-shaped
jsdom storage from `src/test/setup.ts` without redefining it.

For the safe-getter test, capture the exact `window.localStorage` property
descriptor, replace it with a throwing `SecurityError` getter only inside
`try/finally`, and restore that exact descriptor before cleanup. Assert rendering
does not throw, no storage listener is installed, and hydrate is never called.

In App tests, hoist/mock the hook, clear the mock in `beforeEach`, and assert it
was called in the incompatible-document branch. Do not assert render-call count:
hooks run on every render and StrictMode probes twice.

Add a real two-page Playwright test. After both pages reach worker readiness,
save one run in the owner and require the peer History drawer to hydrate from
zero to one. Then clear localStorage in the owner: owner memory stays at one
(same-document writes emit no event), while peer rehydrates to zero.

~~~ts
test('cross-tab saved-run memory follows native localStorage events', async ({ context, page }) => {
    await loadPlayground(page);
    const peer = await context.newPage();
    await loadPlayground(peer);

    const peerHistory = await openDrawer(peer, 'History');
    await expect(peerHistory.getByRole('article')).toHaveCount(0);
    const ownerHistory = await openDrawer(page, 'History');
    await ownerHistory.getByRole('button', { name: 'Save current run' }).click();
    await expect(ownerHistory.getByRole('article')).toHaveCount(1);
    await expect(peerHistory.getByRole('article')).toHaveCount(1);

    await page.evaluate(() => window.localStorage.clear());
    await expect(ownerHistory.getByRole('article')).toHaveCount(1);
    await expect(peerHistory.getByRole('article')).toHaveCount(0);
});
~~~

- [ ] **Step 2: Run RED**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useExperimentMemoryStorageSync.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --grep "cross-tab saved-run memory"

Expected: Vitest FAILS because the hook/root wiring do not exist. The build
succeeds and Chromium FAILS because peer memory never rehydrates.

- [ ] **Step 3: Add one app-owned storage listener**

~~~ts
export function useExperimentMemoryStorageSync(): void {
    useEffect(() => {
        let localStorageArea: Storage;
        try {
            localStorageArea = window.localStorage;
        } catch {
            return;
        }
        const onStorage = (event: StorageEvent) => {
            if (event.storageArea !== localStorageArea) return;
            if (event.key !== null && !MEMORY_KEYS.has(event.key)) return;
            void useExperimentMemoryStore.getState().hydrate();
        };
        window.addEventListener('storage', onStorage);
        return () => window.removeEventListener('storage', onStorage);
    }, []);
}
~~~

Mount it at App lifecycle scope before reading/branching on document
compatibility, not inside the conditional History drawer. Keep `MEMORY_KEYS`
module-local and source it from the two store exports.

- [ ] **Step 4: Run GREEN**

Run: pnpm --filter @nn-playground/web exec vitest run src/hooks/useExperimentMemoryStorageSync.test.tsx src/store/experimentMemoryStore.test.ts src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "cross-tab saved-run memory"

Expected: PASS with exact filtering/StrictMode cleanup, unchanged memory-store
queue semantics, root incompatible-state wiring, and native two-page delivery
in both browsers.

- [ ] **Step 5: Commit**

~~~bash
git add apps/web/src/hooks/useExperimentMemoryStorageSync.ts apps/web/src/hooks/useExperimentMemoryStorageSync.test.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx tests/e2e/playground-smoke.spec.ts
git commit -m "feat(web): sync saved runs across tabs"
~~~

### Task 27: Always run and report both performance suites

**Quick win:** 27 — Independent engine and web performance conclusions.

**Files:**
- Create: scripts/run-performance-gates.mjs
- Create: scripts/run-performance-gates.test.mjs
- Modify: package.json

**Interfaces:**
- Export frozen `PERFORMANCE_GATES`, `runPerformanceGates(runCommand)`,
  `resolvePnpmInvocation(gate, platform, env)`, and injectable
  `main({ runCommand = runPnpmGate, writeLine = console.log } = {})` so CLI
  calls with no argument and tests inject dependencies through the same API.
- The runner receives one frozen gate and resolves `{ code: number | null,
  signal: string | null }`; it may throw/reject for synchronous or emitted
  spawn errors. Execute engine then web strictly sequentially and always try
  both after non-zero, signal, null status, or runner error.
- A suite passes only for `{ code: 0, signal: null }`. Preserve each result's
  name/code/signal and normalized single-line error; return aggregate
  `{ exitCode: 0 | 1, results }`, normalizing every aggregate failure to 1.
- Stream children with `stdio: 'inherit'`. After both attempts, print exactly
  one ordered summary line per suite with PASS or FAIL plus an exit, signal, or
  spawn-error reason. Set `process.exitCode = await main()` rather than calling
  `process.exit`, so inherited output and summaries flush.
- Literal summaries are `[perf] engine: PASS`, `[perf] web: FAIL (exit 2)`,
  `[perf] engine: FAIL (signal SIGTERM)`, `[perf] engine: FAIL (spawn error:
  <single-line message>)`, or `[perf] engine: FAIL (no exit status)`.
  For defensively injected combinations, choose exactly one reason in this
  precedence: normalized spawn error, then signal, then numeric exit, then no
  exit status.
- Guard CLI execution with `process.argv[1] !== undefined` before resolving it,
  then compare `import.meta.url` to
  `pathToFileURL(resolve(process.argv[1])).href`; imports without argv never
  spawn or throw. Import process, console, and other globals from `node:`
  modules; Task 9 does not grant Node globals to this directory.
- Root `test:perf` is exactly `node scripts/run-performance-gates.mjs`;
  package-specific perf commands remain runnable. The resolver is exact:
  non-Windows returns `{ command: 'pnpm', args: [...gate.args] }`; win32 returns
  `{ command: env.ComSpec ?? 'cmd.exe', args: ['/d', '/s', '/c', 'pnpm.cmd',
  ...gate.args] }`. Both use `shell: false`; never spawn `pnpm.cmd` directly.
  Do not claim complete Windows support because the current web package perf
  script itself remains POSIX-only.
- CI performance execution is outside this task; final whole-program
  verification already invokes the root gate.

- [ ] **Step 1: Write a failing Node orchestration test**

Use dynamic imports inside individual tests so the independent root-script
assertion still runs while the implementation module is missing. The injected
runner returns the real child-close shape:

~~~js
const calls = [];
const result = await runPerformanceGates(async (gate) => {
    calls.push(gate.name);
    return gate.name === 'engine'
        ? { code: 1, signal: null }
        : { code: 0, signal: null };
});
assert.deepEqual(calls, ['engine', 'web']);
assert.equal(result.exitCode, 1);
~~~

Cover both pass; engine-only, web-only, and both failure; strict sequential
completion; thrown/rejected engine still reaching web; signal with null code;
defensive null-code/null-signal; emitted spawn error followed by close settling
once; exact POSIX argv/options and explicit Windows cmd invocation;
exact ordered two-line summaries for pass/exit/signal/spawn failure; import
without default-runner invocation; and package.json containing exactly
`"test:perf": "node scripts/run-performance-gates.mjs"` with no `&&`. Inject
an impossible error+signal+exit combination and lock the documented single
failure-reason precedence rather than concatenating reasons.

- [ ] **Step 2: Run RED**

Run: node --test scripts/run-performance-gates.test.mjs

Expected: FAIL for the missing module; the independent package-script assertion
also fails because the current root command contains `&&`. The missing-module
failure alone does not prove short-circuiting.

- [ ] **Step 3: Implement child-process aggregation**

Listen for child `close(code, signal)`, not just `exit`, so reporting follows
stdio closure. Convert synchronous spawn throws and emitted errors into results;
guard settlement because Node may emit `close` after `error`. Resolve platform
invocation separately and keep argv tokenized.

~~~js
export const PERFORMANCE_GATES = Object.freeze([
    Object.freeze({
        name: 'engine',
        args: Object.freeze(['--filter', '@nn-playground/engine', 'test:perf']),
    }),
    Object.freeze({
        name: 'web',
        args: Object.freeze(['--filter', '@nn-playground/web', 'test:perf']),
    }),
]);
~~~

- [ ] **Step 4: Run GREEN and exercise the real combined gate**

Run: node --check scripts/run-performance-gates.mjs

Run: node --test scripts/run-performance-gates.test.mjs

Run: pnpm exec eslint scripts/run-performance-gates.mjs scripts/run-performance-gates.test.mjs

Run: pnpm test:perf

Expected: syntax, unit, and lint gates PASS. The real command streams both
children and prints both final conclusions even when one fails. Status is zero
only when both calibrated suites pass, otherwise normalized one after both were
attempted.

- [ ] **Step 5: Review exact scope**

Run: git add package.json scripts/run-performance-gates.mjs scripts/run-performance-gates.test.mjs

Run: git diff --cached --check

Run: git diff --cached --name-only

Expected staged set exactly `package.json`,
`scripts/run-performance-gates.mjs`, and
`scripts/run-performance-gates.test.mjs`. Do not edit/stage CI, package-specific
perf configurations, plans, or prototypes.

- [ ] **Step 6: Commit**

~~~bash
git commit -m "test: aggregate independent performance gates"
~~~

### Task 28: Scan the built app with Axe in desktop and compact browsers

**Quick win:** 28 — Production browser Axe scans.

**Files:**
- Create: tests/e2e/accessibility.spec.ts
- Modify: package.json
- Modify: pnpm-lock.yaml
- Modify: apps/web/src/components/visualization/NetworkGraphCanvas.tsx
- Modify: apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx

**Interfaces:**
- Add `@axe-core/playwright` as a direct root development dependency; do not depend on the transitive `axe-core` bundled under `jest-axe`.
- A self-contained spec explicitly generates desktop 1280x720 and compact
  390x844 cases; Chromium/WebKit projects then produce exactly four scans.
- Scan only after the Run workspace reaches real step-0 evidence convergence:
  idle status, full evaluation step 0, and checkpoint timeline Step 0.
- Fail on any Axe violation whose impact is `serious` or `critical`. Any future exception must name a rule, a tightly scoped selector, rationale, and a repository issue URL in the test; this task adds no blanket exclusions.
- Collect page errors and console errors for every case and fail after each
  test. Format violations with impact, rule/help URL, exact targets, and failure
  summaries so browser output is actionable.
- The genuine pre-implementation production scan fails all four cases on one
  serious `role-img-alt` violation at
  `canvas[aria-describedby="network-graph-desc"]`. Give that canvas the exact
  accessible name `Neural network graph` while retaining its dynamic
  `aria-describedby` summary; add no exception.
- Prohibit Axe `.exclude()`, `.include()`, `.options()`, `.disableRules()`,
  `.withRules()`, `.withTags()`, and `runOnly`. Every scan must retain the
  literal whole-page `new AxeBuilder({ page }).analyze()` shape. A future
  exception must filter only a single rule's single
  tight node selector, retain all unmatched nodes, explain why, and link
  `https://github.com/DenseDevKev/neural-network-playground/issues/<number>`.

- [ ] **Step 1: Add the direct test dependency and a failing production scan**

Run: pnpm add -Dw @axe-core/playwright@^4.10.2

Create the spec with its expected-zero assertion before changing application markup:

~~~ts
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

type AxeViolations = Awaited<ReturnType<AxeBuilder['analyze']>>['violations'];
const browserErrors = new WeakMap<Page, string[]>();

test.beforeEach(({ page }) => {
    const errors: string[] = [];
    browserErrors.set(page, errors);
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
});

test.afterEach(({ page }) => {
    const errors = browserErrors.get(page) ?? [];
    expect(errors, `Unexpected browser errors:\n${errors.join('\n')}`).toEqual([]);
});

function formatViolations(violations: AxeViolations): string {
    return violations.map((violation) => [
        `[${violation.impact?.toUpperCase() ?? 'UNKNOWN'}] ${violation.id}: ${violation.help}`,
        violation.helpUrl,
        ...violation.nodes.map((node, index) => (
            `  ${index + 1}. ${JSON.stringify(node.target)}`
            + `${node.failureSummary ? `\n     ${node.failureSummary}` : ''}`
        )),
    ].join('\n')).join('\n\n');
}

async function loadReadyPlayground(page: Page): Promise<void> {
    await page.goto('/');
    await expect(page.getByRole('main', {
        name: 'Neural network playground workspace',
    })).toBeVisible();
    const run = page.getByRole('button', { name: 'run', exact: true });
    if (await run.getAttribute('aria-pressed') !== 'true') await run.click();
    await expect(page.getByRole('group', { name: 'Status bar' }))
        .toHaveAttribute('data-status', 'idle');
    await expect(page.locator('section[role="region"][aria-label="Current run"]'))
        .toContainText(/Full evaluation \d+ at step 0\b/);
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' }))
        .toHaveAttribute('aria-valuetext', 'Step 0');
}

const SCAN_CASES = [
    { name: 'desktop accessibility', width: 1280, height: 720 },
    { name: '390px compact accessibility', width: 390, height: 844 },
] as const;

for (const scanCase of SCAN_CASES) {
    test.describe(scanCase.name, () => {
        test.use({ viewport: { width: scanCase.width, height: scanCase.height } });
        test('has no serious or critical Axe violations', async ({ page }) => {
            await loadReadyPlayground(page);
            const results = await new AxeBuilder({ page }).analyze();
            const blocking = results.violations.filter(
                ({ impact }) => impact === 'serious' || impact === 'critical',
            );
            expect(blocking, formatViolations(blocking)).toEqual([]);
        });
    });
}
~~~

Add a focused owner test before changing the canvas. It must require both the
new exact accessible name and the existing dynamic description, so replacing
the scientific summary with a generic label is not accepted:

~~~tsx
const graph = screen.getByRole('img', { name: 'Neural network graph' });
expect(graph).toHaveAccessibleDescription(/Neural network:/);
expect(graph).toHaveAttribute('aria-describedby', 'network-graph-desc');
~~~

- [ ] **Step 2: Run RED against a fresh production build**

Run: pnpm --filter @nn-playground/web exec vitest run src/components/visualization/NetworkGraphCanvas.test.tsx --pool=forks --reporter=dot

Expected: FAIL because the canvas has `role="img"` and a description but no
accessible name.

Temporarily insert an unlabeled image immediately before `analyze()`:

~~~ts
await page.evaluate(() => {
    const probe = document.createElement('img');
    probe.src = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
    probe.width = 1;
    probe.height = 1;
    probe.dataset.axeRedProbe = '';
    document.body.prepend(probe);
});
~~~

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/accessibility.spec.ts --project=chromium --grep "desktop accessibility"

Expected: FAIL containing the `image-alt` rule with critical impact. Remove the
probe manually and verify no probe diff remains. The unmodified production
baseline also reports the genuine serious `role-img-alt` canvas violation in
all four browser/viewport cases; the controlled mutation must still prove the
independent critical `image-alt` gate.

- [ ] **Step 3: Fix the proven graph-canvas violation without an exception**

Add `aria-label="Neural network graph"` to the existing `role="img"` canvas and
keep `aria-describedby="network-graph-desc"` unchanged. Do not replace, hide,
or flatten the current dynamic network-shape summary. Record the observed
`role-img-alt` rule, target, owner, and focused test in the Task 28 report. Do
not change other application source and do not add an Axe exception.

- [ ] **Step 4: Run GREEN in the required matrix**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/accessibility.spec.ts --project=chromium --project=webkit --list

Run: pnpm exec playwright test tests/e2e/accessibility.spec.ts --project=chromium --project=webkit

Run: pnpm exec eslint tests/e2e/accessibility.spec.ts

Run: pnpm --filter @nn-playground/web exec vitest run src/components/visualization/NetworkGraphCanvas.test.tsx --pool=forks --reporter=dot

Run: pnpm --filter @nn-playground/web typecheck

Run: pnpm exec eslint apps/web/src/components/visualization/NetworkGraphCanvas.tsx apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx tests/e2e/accessibility.spec.ts

Run: pnpm install --frozen-lockfile

Run: pnpm list -w @axe-core/playwright --depth 0

Run: test -f tests/e2e/accessibility.spec.ts

Run: ! rg -n '\.(exclude|include|options|disableRules|withRules|withTags)\(|\brunOnly\b' tests/e2e/accessibility.spec.ts

Run: rg -n 'new AxeBuilder\(\{ page \}\)\.analyze\(\)' tests/e2e/accessibility.spec.ts

Expected: `--list` reports exactly four Task 28 cases. Desktop Chromium,
compact Chromium, desktop WebKit, and compact WebKit PASS with zero
serious/critical violations and no browser errors; lint/frozen lock/direct-root
dependency checks pass, the exact whole-page builder shape is present, and the
executable prohibited-method scan returns no match. The graph canvas has the
exact accessible name plus its existing dynamic description.

- [ ] **Step 5: Commit**

~~~bash
git add tests/e2e/accessibility.spec.ts package.json pnpm-lock.yaml apps/web/src/components/visualization/NetworkGraphCanvas.tsx apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx
git commit -m "test(e2e): scan production accessibility"
~~~

### Task 29: Verify both narrow and common phone widths

**Quick win:** 29 — 390 by 844 responsive E2E coverage alongside 320.

**Files:**
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Run one shared compact reachability journey at both 320 by 844 and 390 by 844 using generated, uniquely named tests.
- At both widths assert no document/shell horizontal overflow, all critical and graph/evidence controls from Task 20 are at least 44 by 44, active evidence tabs remain visible, compact evaluation outcome from Task 21 is reachable, drawers close and restore focus, and browser errors remain empty.
- Reuse Task 23's exact compact reset locator `Reset training`; do not restore
  legacy reset wording when extracting the shared journey.
- Reuse Task 20's one module-level graph/evidence locator/assertion helper. After
  any Code/drawer/profile journey state, explicitly return to Run + Boundary
  before calling it. Replace Task 20's standalone 390 test when the matrix
  absorbs it so the locator inventory does not drift or run twice. Move its
  runtime toolbar-to-summary assertion into the shared helper: at both widths,
  `summary.y - (toolbar.y + toolbar.height) >= 6` remains required.
- At both widths reach the Task 16 Keyboard shortcuts summary, prove touch
  activation opens the native disclosure without starting/resetting training,
  close it, then focus it and prove native Space activation opens it without
  the global training shortcut intercepting the key.
- Do not use conditional assertions that weaken one viewport.
- Preserve the independent Task 16 800px disclosure test, Task 22 desktop
  concept-help test, and Task 26 cross-tab test; parameterize only the existing
  compact journey and remove only Task 20's absorbed standalone 390 graph test.

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
its 13 accessible names into this journey. Extend that helper after its existing
target loop with the absorbed geometry gate so it runs once per width:

~~~ts
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
~~~

Exercise Boundary, Loss, and Confusion as active tabs: scroll each into view,
click it, assert `aria-selected="true"`, require the 44px target, then use
`expectFullyInViewport` without a second corrective scroll so the assertion
proves the active tab remains visible. For Keyboard shortcuts, use the existing real `touchTap` helper for
open and close, then focus the summary and press Space. Before the snapshot,
require exact status `idle`, exact current step `0`, and non-null, nonempty
generation/revision strings; compare that complete object exactly after all
three activations.

- [ ] **Step 4: Run GREEN in both browser engines**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "touch shell" --list

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --list --grep "cross-tab saved-run memory|concept help remains fully visible|keyboard shortcuts disclosure hides definitions"

Run: pnpm exec playwright test tests/e2e/playground-smoke.spec.ts --project=chromium --project=webkit --grep "touch shell"

Run: pnpm exec eslint tests/e2e/playground-smoke.spec.ts

Run: ! rg -n '390px graph and evidence targets' tests/e2e/playground-smoke.spec.ts

Run: git diff --check

Expected: `--list` reports exactly four tests; four tests PASS at zero retries,
with both named widths represented for each browser. The absorbed standalone
name is absent, the independent Task 16/22/26 list reports exactly three
Chromium tests, and lint/diff checks pass.

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
const url = new URL('https://example.test/?e2eWorkerFault=startup-once');
expect(consumeE2EWorkerFault(url, storage, true))
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
console or page errors in either build. The deliberately injected worker fault
is expected application state delivered through the bridge/store/modal path;
it must not be logged or thrown as a browser error.

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

Bridge tests prove the helper returns false without a subscriber and delivers a protocol-v2 `error` message through `onSnapshot` when subscribed. Hook tests prove the injected branch is reached only after subscription, that neither the mount initializer nor prepared-document synchronization invokes worker initialization, and that the same worker error as a native bridge error reaches the store. Place fault selection outside worker scientific logic. Add exact root scripts `build:e2e` and `test:e2e:recovery`; the latter runs `build:e2e` with `VITE_E2E_FAULTS=1`, then the two-browser worker-recovery spec with only `--grep @fault-enabled` against that generated dist.

Use these exact root script values:

~~~json
"build:e2e": "VITE_E2E_FAULTS=1 pnpm build",
"test:e2e:recovery": "pnpm run build:e2e && playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep \"@fault-enabled\""
~~~

Wrap the hook harness in StrictMode and prove effect replay queues exactly one
bridge error after the replacement subscription, while the sticky suppression
ref short-circuits both replayed mount/prepared effects and initialization stays
at zero until the page reloads. Consuming the session marker on the replay must
not re-enable initialization in the same document.

- [ ] **Step 4: Run GREEN units and production recovery**

Run: pnpm --filter @nn-playground/web exec vitest run src/testing/e2eFaults.test.ts src/worker/workerBridge.test.ts src/hooks/useTraining.test.tsx src/App.test.tsx --pool=forks --reporter=dot

Run: pnpm run test:e2e:recovery

Expected: Unit tests PASS. Both browsers show the genuine focused modal, reload through the real recovery control, reach ready state, and advance training with zero retries.

- [ ] **Step 5: Prove the normal build ignores the query**

Run: pnpm build

Run: pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-disabled"

Expected: PASS: the query cannot inject a fault in a normal production build;
both browsers converge on step-0 evidence with no modal and complete one real
step.

- [ ] **Step 6: Commit**

~~~bash
git add apps/web/src/testing/e2eFaults.ts apps/web/src/testing/e2eFaults.test.ts tests/e2e/worker-recovery.spec.ts apps/web/src/worker/workerBridge.ts apps/web/src/worker/workerBridge.test.ts apps/web/src/hooks/useTraining.ts apps/web/src/hooks/useTraining.test.tsx apps/web/src/vite-env.d.ts package.json
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
pnpm run test:e2e:recovery
pnpm build
pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --project=chromium --project=webkit --grep "@fault-disabled"
git diff --check 047c341..HEAD
git diff --check
git status --short
~~~

The final reviewer must also inspect the cumulative range from the design commit through Task 30, confirm every numbered acceptance criterion is covered by committed code or executable evidence, confirm the four unrelated pre-existing paths remain untouched, and confirm the normal production build does not honor E2E fault parameters.
