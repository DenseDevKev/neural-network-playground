# Precision Lab Production Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Replace the production Build and Run presentation with one responsive Precision Lab workspace that keeps the real topology and the single live decision boundary visible together without changing scientific, runtime, persistence, or compatibility behavior.

**Architecture:** App remains the orchestration boundary and sole useTraining owner. Focused feature adapters derive immutable display-safe models and commands from existing stores, frame buffers, and selectors; PrecisionLabShell receives only those values, commands, and prepared React content nodes. The work lands through independently reviewable TDD slices, with the old shell retained until the new production path passes consumer searches, semantic workflows, bundle limits, performance gates, accessibility checks, and cross-browser layout-stability tests.

**Tech Stack:** React 19, TypeScript, Zustand 5, Vite, Vitest, Testing Library, Playwright, Canvas 2D, @tabler/icons-react 3.44.x, pnpm 9, Node 20, GitHub Actions.

## Global Constraints

- Preserve usePlaygroundStore as owner of the shareable V2 experiment document, preparation/import/URL compatibility, and visualization-demand delivery cache.
- Preserve useTrainingStore as owner of runtime status, generation/revision identity, evidence references, frame versions, configuration synchronization, checkpoints, errors, and training speed.
- Preserve useTraining as the only training and worker orchestration owner. PrecisionLabShell and its children must not instantiate it.
- Preserve the worker as the authoritative producer of parameters and scientific artifacts.
- Preserve frameBuffer typed-array identity. Display adapters may hold subarray views but must not copy live grids or weights into React state.
- Preserve metricHistoryBuffer as the bounded metric-series owner and experimentMemoryStore as the saved-run artifact owner.
- No engine-math, optimizer, objective, dataset-generation, worker-protocol, V2 schema, URL, checkpoint, export, or saved-run migration.
- Use one shell and one state graph for Beginner, Explore, and Lab. Audience mode changes visibility and guidance only.
- The pinned boundary rail owns the only live decision-boundary canvas in the production shell. Boundary evidence renders controls, provenance, summaries, and explanations only.
- Graph fallback parity covers commands, data accuracy, persistent selection, selected-path meaning, labels, status, keyboard operation, and accessible summaries. Its geometry may differ.
- Do not introduce a generic view-model framework. Add only focused pure models and hooks at store-heavy feature boundaries.
- Build context disclosure and neuron selection are ephemeral. Do not add either to V2 state, the URL, checkpoints, saved runs, or persisted layout state.
- Keep the main landmark accessible name Neural network playground workspace and preserve the existing Workspace view, Audience mode, Advanced Tools, dialog, evidence-tab, timeline, status, and training-action names.
- Preserve generation and revision data attributes used by scientific workflow tests.
- Preserve existing control validation, prepare/pause/acknowledge/initialize/commit ordering, stale-preparation fencing, worker-error recovery, incompatible import/URL recovery, checkpoint restore revision fencing, and saved-run byte retention.
- Retain the production Inter and Space Grotesk type system, token system, and base styles. Add no global reset.
- At 1437x742, 735x860, and 320x844, training at 50 steps per frame for five seconds must keep each major region x, y, width, and height within 1 CSS pixel of baseline.
- Chromium cumulative layout shift during that post-start sample, excluding hadRecentInput entries, must equal zero.
- Chromium and WebKit document and workspace overflow must remain no more than 1 CSS pixel before, during, and after the sample.
- Main entry gzip must be at most 152,245 bytes, InspectionPanel gzip at most 7,373 bytes, and total JavaScript gzip at most 234,161 bytes.
- Forced scientific evaluation must remain at most 250ms and saved-run capture at most 500ms by repeated idle-host medians.
- If the always-visible boundary causes a performance failure, optimize only visual artifact cadence or paint frequency; do not lower scientific evaluation cadence or accuracy.
- Playwright retries remain zero, timeout remains 45 seconds, and assertions, cadence, performance thresholds, and accessibility checks may not be weakened to obtain a pass.
- Use direct, tree-shaken Tabler icon imports. Do not ship prototype screenshots, prototype mock state, handcrafted interface SVGs, or CSS-drawn data/neurons.
- Preserve the unrelated Scientific Trust plan edit, V0.1 BUILD DOCU.md, the responsive-polish draft, and prototypes directory. Stage only files named by the active task.

---

## File and responsibility map

### New shell and recipe files

- **apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx** — presentation-only composition, evidence tab keyboard behavior, context-sheet focus management, drawer placement, and stable region hooks.
- **apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx** — store-free composition, visibility, focus, semantic tab, drawer, and single-boundary-slot tests.
- **apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.ts** — React-free recipe-strip derivation from primitive identity/evidence inputs.
- **apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.test.ts** — architecture, drift, update, stale-evaluation, and unavailable-state tests.
- **apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.ts** — narrow store selectors that adapt canonical recipe/trained-recipe/evidence state into the pure model input.
- **apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx** — exact subscription and store-to-model integration tests.
- **apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.tsx** — store-free Recipe summary region and Edit recipe command.
- **apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx** — accessible rendering and command forwarding.
- **apps/web/src/styles/precisionLab.css** — isolated precision-* shell, responsive tracks, transport, evidence, dataset-preview, and graph-tile styling.
- **apps/web/src/styles/precisionLabResponsive.test.ts** — static CSS guards for breakpoints, containment, reduced motion, minimum targets, and overflow rules.

### New dataset-preview files

- **apps/web/src/components/controls/datasetPreviewModel.ts** — deterministic preview generation through generateDatasetV2.
- **apps/web/src/components/controls/datasetPreviewModel.test.ts** — all eleven dataset IDs, determinism, task kind, and point-count tests.
- **apps/web/src/components/controls/DatasetPreviewCanvas.tsx** — presentation-only canvas painter for real generated points.
- **apps/web/src/components/controls/DatasetPreviewCanvas.test.tsx** — canvas sizing, neutral empty state, and accessibility-boundary tests.

### New network-selection files

- **apps/web/src/components/visualization/networkSelectionModel.ts** — React-free selected-neuron statistics and strongest-path derivation using accepted typed-array views.
- **apps/web/src/components/visualization/networkSelectionModel.test.ts** — offsets, ranking, sign, summary, neutral state, and typed-array identity tests.
- **apps/web/src/components/visualization/useNetworkSelectionController.ts** — generation/architecture-scoped selection state plus narrow frame-version subscriptions.
- **apps/web/src/components/visualization/useNetworkSelectionController.test.tsx** — persistence and reset-fence tests.
- **apps/web/src/components/visualization/NetworkSelectionDeck.tsx** — store-free selected-neuron detail surface.
- **apps/web/src/components/visualization/NetworkSelectionDeck.test.tsx** — labels, statistics, influence lists, and clear-command tests.

### New decision-boundary files

- **apps/web/src/components/visualization/useDecisionBoundaryController.ts** — one focused adapter for boundary display model, rail summary, overlay state, document-view controls, and commands.
- **apps/web/src/components/visualization/useDecisionBoundaryController.test.tsx** — store/model command integration and snapshot-coherence tests.
- **apps/web/src/components/visualization/DecisionBoundaryCanvas.tsx** — store-free canvas sizing and painting from one DecisionBoundaryDisplayModel snapshot.
- **apps/web/src/components/visualization/DecisionBoundaryCanvas.test.tsx** — scalar, discrete, multiclass, unavailable, DPR, and ResizeObserver tests.
- **apps/web/src/components/visualization/PinnedBoundaryRail.tsx** — the only production canvas plus freshness, compact metric, and expand action.
- **apps/web/src/components/visualization/PinnedBoundaryRail.test.tsx** — display-safe rendering and expand forwarding.
- **apps/web/src/components/visualization/BoundaryEvidencePanel.tsx** — no-canvas controls, provenance, summaries, and explanation.
- **apps/web/src/components/visualization/BoundaryEvidencePanel.test.tsx** — single-boundary contract, control forwarding, and accessible summaries.

### New saved-run and verification files

- **apps/web/src/components/controls/useSaveCurrentRun.ts** — shared current-run capture/persistence state machine extracted from RunHistoryPanel.
- **apps/web/src/components/controls/useSaveCurrentRun.test.tsx** — exact pending-artifact retry, blocked, saving, success, and error tests.
- **scripts/check-web-bundle-gzip.mjs** — deterministic entry, Inspection chunk, and total-JavaScript gzip budget check.
- **scripts/check-web-bundle-gzip.test.mjs** — Node test fixtures for passing and failing each budget.
- **tests/e2e/precision-lab-layout.spec.ts** — cross-browser wide, compact, and phone stability/overflow/touch/zoom checks.

### Existing files modified

- **package.json**, **apps/web/package.json**, **pnpm-lock.yaml**, and **.github/workflows/ci.yml** — icon dependency and automated bundle gate.
- **apps/web/src/main.tsx** — import precisionLab.css after forge.css.
- **apps/web/src/App.tsx** — compose display adapters, mount one boundary controller/canvas, and replace only the shell call.
- **apps/web/src/store/useLayoutStore.ts** and its test — nonpersisted Build context disclosure and focus-safe target actions.
- **apps/web/src/components/layout/Header.tsx** and its test — Precision Lab chrome while preserving production semantics.
- **apps/web/src/components/layout/MainArea.tsx** and its tests — split boundary controls from its canvas while retaining the legacy wrapper until deletion is proven safe.
- **apps/web/src/components/layout/deriveVisualizationDemand.ts** and its test — request the boundary whenever the ready Precision Lab workspace is mounted.
- **apps/web/src/components/controls/DataPanel.tsx** and its test — real dataset preview choices.
- **apps/web/src/components/controls/TrainingControls.tsx** and its test — compact transport with shared Save run command.
- **apps/web/src/components/controls/RunHistoryPanel.tsx** and its test — consume the shared save controller without duplicating capture logic.
- **apps/web/src/components/visualization/NetworkGraph.tsx**, **NetworkGraphCanvas.tsx**, **NetworkGraphSVG.tsx**, **networkGraphPainter.ts**, and focused tests — responsive activation tiles, persistent selection, selected paths, and fallback parity.
- **apps/web/src/components/visualization/DecisionBoundary.tsx**, **decisionBoundaryModel.ts**, **useDecisionBoundaryModel.ts**, and focused tests — retain the public wrapper while extracting one store-free painter.
- **apps/web/src/components/visualization/LossChart.tsx**, **ConfusionMatrix.tsx**, and focused tests — bounded responsive evidence layouts without horizontal scroll.
- **apps/web/src/App.test.tsx**, **apps/web/src/__tests__/appShell.integration.test.tsx**, **apps/web/src/styles/forgeResponsive.test.ts**, and **tests/e2e/playground-smoke.spec.ts** — production integration, accessibility, scientific invariants, and existing workflow coverage.

## Checkpoint policy

After Tasks 4, 8, 10, and 13, run the focused suites named by those tasks plus:

~~~bash
pnpm lint
pnpm build
pnpm test:bundle
git diff --check
~~~

Expected: every command exits 0; bundle output reports all three measured values at or below their fixed caps. Do not proceed past a checkpoint with a failing command.

### Task 1: Make bundle budgets executable

**Files:**
- Create: scripts/check-web-bundle-gzip.mjs
- Create: scripts/check-web-bundle-gzip.test.mjs
- Modify: package.json
- Modify: .github/workflows/ci.yml

**Interfaces:**
- Consumes: apps/web/dist/index.html and apps/web/dist/assets after pnpm build.
- Produces:

~~~js
export const BUNDLE_LIMITS = Object.freeze({
    entry: 152_245,
    inspection: 7_373,
    totalJavaScript: 234_161,
});

export function measureWebBundle(distDir) {
    return {
        entry: { file, gzipBytes },
        inspection: { file, gzipBytes },
        totalJavaScript: { files, gzipBytes },
    };
}

export function assertBundleWithinLimits(measurement, limits = BUNDLE_LIMITS) {
    return measurement;
}
~~~

- Produces root command pnpm test:bundle, which fails on missing/ambiguous chunks or any cap overrun.

- [ ] **Step 1: Write Node tests for bundle discovery and all three caps**

Create a temporary Vite-like dist fixture with one module entry, one InspectionPanel chunk, and one additional JavaScript chunk. Assert deterministic gzip measurements, rejection of zero or multiple InspectionPanel chunks, and separate failures for entry, inspection, and total budgets.

~~~js
import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import {
    BUNDLE_LIMITS,
    assertBundleWithinLimits,
    measureWebBundle,
} from './check-web-bundle-gzip.mjs';

const roots = [];
afterEach(() => {
    for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function makeFixture(files) {
    const root = mkdtempSync(join(tmpdir(), 'precision-bundle-'));
    roots.push(root);
    for (const [relativePath, contents] of Object.entries(files)) {
        const absolutePath = join(root, relativePath);
        mkdirSync(dirname(absolutePath), { recursive: true });
        writeFileSync(absolutePath, contents);
    }
    return root;
}

test('measures the module entry, one InspectionPanel chunk, and all JavaScript', () => {
    const dist = makeFixture({
        'index.html': '<script type="module" src="/assets/index-a.js"></script>',
        'assets/index-a.js': 'export const entry = 1;',
        'assets/InspectionPanel-b.js': 'export const inspect = 1;',
        'assets/vendor-c.js': 'export const vendor = 1;',
    });
    const measured = measureWebBundle(dist);
    assert.equal(measured.entry.file, 'assets/index-a.js');
    assert.equal(measured.inspection.file, 'assets/InspectionPanel-b.js');
    assert.equal(measured.totalJavaScript.files, 3);
});

test('reports the exact budget that failed', () => {
    assert.throws(
        () => assertBundleWithinLimits({
            entry: { file: 'assets/index.js', gzipBytes: BUNDLE_LIMITS.entry + 1 },
            inspection: { file: 'assets/InspectionPanel.js', gzipBytes: 1 },
            totalJavaScript: { files: 2, gzipBytes: 2 },
        }),
        /entry.*152245/i,
    );
});
~~~

- [ ] **Step 2: Run the Node test and verify the module is absent**

Run:

~~~bash
node --test scripts/check-web-bundle-gzip.test.mjs
~~~

Expected: FAIL with ERR_MODULE_NOT_FOUND for scripts/check-web-bundle-gzip.mjs.

- [ ] **Step 3: Implement deterministic bundle measurement and cap reporting**

Walk assets recursively, parse the type=module script source from index.html, require exactly one basename matching /^InspectionPanel-[^.]+[.]js$/, gzip every .js file with zlib.gzipSync, and throw one Error containing the measured and allowed bytes for every failed dimension. In CLI mode, print one line per dimension and set process.exitCode = 1 on failure.

~~~js
const measurement = measureWebBundle(resolve('apps/web/dist'));
assertBundleWithinLimits(measurement);
console.log('entry gzip ' + measurement.entry.gzipBytes + ' / ' + BUNDLE_LIMITS.entry);
console.log(
    'InspectionPanel gzip '
    + measurement.inspection.gzipBytes
    + ' / '
    + BUNDLE_LIMITS.inspection,
);
console.log(
    'total JavaScript gzip '
    + measurement.totalJavaScript.gzipBytes
    + ' / '
    + BUNDLE_LIMITS.totalJavaScript,
);
~~~

- [ ] **Step 4: Add the root script and CI gate**

Add this package script:

~~~json
{
  "test:bundle": "node scripts/check-web-bundle-gzip.mjs"
}
~~~

Add this CI step immediately after Build production bundle:

~~~yaml
      - name: Check production bundle budgets
        run: pnpm test:bundle
~~~

- [ ] **Step 5: Verify tests, build, and the real baseline**

Run:

~~~bash
node --test scripts/check-web-bundle-gzip.test.mjs
pnpm build
pnpm test:bundle
~~~

Expected: Node tests PASS; build exits 0; the bundle command reports entry at or below 152,245, InspectionPanel at or below 7,373, and total JavaScript at or below 234,161.

- [ ] **Step 6: Commit the executable budget gate**

~~~bash
git add scripts/check-web-bundle-gzip.mjs scripts/check-web-bundle-gzip.test.mjs package.json .github/workflows/ci.yml
git commit -m "test: enforce Precision Lab bundle budgets"
~~~

### Task 2: Derive and render a display-safe recipe strip

**Files:**
- Create: apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.ts
- Create: apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.test.ts
- Create: apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.ts
- Create: apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx
- Create: apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.tsx
- Create: apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx

**Interfaces:**
- Consumes: the prepared document/compiled network, getRecipeDrift, selectScientificEvidence, prepared recipe fingerprint, trained recipe fingerprint, and pendingConfigSource.
- Produces:

~~~ts
export type PrecisionLabRecipeTone = 'ready' | 'updating' | 'drift' | 'stale' | 'unavailable';

export interface PrecisionLabRecipeModelInput {
    readonly dataset: string;
    readonly architecture: string;
    readonly hiddenActivation: string;
    readonly output: string;
    readonly seed: number;
    readonly hasRecipeDrift: boolean;
    readonly pendingConfiguration: boolean;
    readonly evaluationAgeSteps: number | null;
}

export interface PrecisionLabRecipeModel {
    readonly dataset: string;
    readonly architecture: string;
    readonly hiddenActivation: string;
    readonly output: string;
    readonly seed: string;
    readonly evaluationLabel: string;
    readonly tone: PrecisionLabRecipeTone;
}

export function derivePrecisionLabRecipeModel(
    input: PrecisionLabRecipeModelInput | null,
): PrecisionLabRecipeModel;

export function usePrecisionLabRecipeModel(): PrecisionLabRecipeModel;

export interface PrecisionLabRecipeStripProps {
    readonly model: PrecisionLabRecipeModel;
    readonly onEditRecipe: () => void;
}
~~~

- [ ] **Step 1: Write pure model tests for all identity and freshness states**

Assert unavailable input, pending configuration precedence, trained-recipe drift, aged full evaluation, and ready state. Use an architecture label of 2 -> 6 -> 6 -> 3 and ensure changing evaluation age never changes recipe values.

~~~ts
expect(derivePrecisionLabRecipeModel({
    dataset: 'three-class-clusters',
    architecture: '2 -> 6 -> 6 -> 3',
    hiddenActivation: 'tanh',
    output: 'softmax',
    seed: 42,
    hasRecipeDrift: false,
    pendingConfiguration: false,
    evaluationAgeSteps: 12,
})).toEqual({
    dataset: 'three-class-clusters',
    architecture: '2 -> 6 -> 6 -> 3',
    hiddenActivation: 'tanh',
    output: 'softmax',
    seed: '42',
    evaluationLabel: 'Evaluation 12 steps behind',
    tone: 'stale',
});
~~~

- [ ] **Step 2: Run the pure test and verify the model is absent**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/precisionLab/precisionLabRecipeModel.test.ts --pool=forks --reporter=dot
~~~

Expected: FAIL because precisionLabRecipeModel.ts does not exist.

- [ ] **Step 3: Implement the pure precedence and formatting rules**

Use this precedence: null input -> unavailable, pending -> updating, drift -> drift, positive evaluationAgeSteps -> stale, otherwise ready. Keep labels stable-width by returning strings only; CSS owns widths.

~~~ts
const EMPTY_MODEL: PrecisionLabRecipeModel = Object.freeze({
    dataset: 'Unavailable',
    architecture: 'Unavailable',
    hiddenActivation: 'Unavailable',
    output: 'Unavailable',
    seed: 'Unavailable',
    evaluationLabel: 'No evaluation',
    tone: 'unavailable',
});

export function derivePrecisionLabRecipeModel(
    input: PrecisionLabRecipeModelInput | null,
): PrecisionLabRecipeModel {
    if (input === null) return EMPTY_MODEL;
    const tone = input.pendingConfiguration
        ? 'updating'
        : input.hasRecipeDrift
            ? 'drift'
            : (input.evaluationAgeSteps ?? 0) > 0
                ? 'stale'
                : 'ready';
    return Object.freeze({
        dataset: input.dataset,
        architecture: input.architecture,
        hiddenActivation: input.hiddenActivation,
        output: input.output,
        seed: String(input.seed),
        evaluationLabel: tone === 'updating'
            ? 'Updating'
            : tone === 'drift'
                ? 'Recipe differs from trained model'
                : tone === 'stale'
                    ? 'Evaluation ' + input.evaluationAgeSteps + ' steps behind'
                    : 'Evaluation fresh',
        tone,
    });
}
~~~

- [ ] **Step 4: Write hook tests that prove canonical sources and narrow updates**

Seed the existing vanilla stores with one ready V2 document, trained recipe/fingerprint, pending source, live signal, and evaluation. Assert the hook reports the prepared recipe values, uses drift from fingerprints, updates when latestEvaluation changes, and does not mutate either store.

~~~tsx
const { result, rerender } = renderHook(() => usePrecisionLabRecipeModel());
expect(result.current.dataset).toBe('three-class-clusters');
expect(result.current.architecture).toBe('2 -> 6 -> 6 -> 3');

act(() => useTrainingStore.setState({ pendingConfigSource: 'network' }));
rerender();
expect(result.current.tone).toBe('updating');
~~~

- [ ] **Step 5: Implement the focused store adapter**

Select only the prepared document/compiled network, prepared fingerprint, trained recipe, trained fingerprint, pendingConfigSource, latestLiveSignal, and latestEvaluation. Reuse getRecipeDrift and selectScientificEvidence. Read output activation from the prepared compiled network; do not derive scientific claims in the shell.

~~~ts
const evidence = useMemo(
    () => selectScientificEvidence({ latestLiveSignal, latestEvaluation }),
    [latestEvaluation, latestLiveSignal],
);
const drift = useMemo(
    () => getRecipeDrift(
        trainedRecipe,
        currentRecipe,
        3,
        trainedRecipeFingerprint === null
            ? undefined
            : { trainedRecipeFingerprint, currentRecipeFingerprint },
    ),
    [currentRecipe, currentRecipeFingerprint, trainedRecipe, trainedRecipeFingerprint],
);
return derivePrecisionLabRecipeModel(prepared === null ? null : {
    dataset: prepared.document.recipe.task.dataset,
    architecture: [
        prepared.compiled.network.inputSize,
        ...prepared.compiled.network.hiddenLayers,
        prepared.compiled.network.outputSize,
    ].join(' -> '),
    hiddenActivation: prepared.document.recipe.model.hiddenActivation,
    output: prepared.compiled.network.outputActivation,
    seed: prepared.document.recipe.model.seed,
    hasRecipeDrift: drift.hasDrift,
    pendingConfiguration: pendingConfigSource !== null,
    evaluationAgeSteps: evidence.evaluationAgeSteps,
});
~~~

- [ ] **Step 6: Write and implement the store-free recipe strip**

The view must expose role=region and aria-label=Recipe summary, six definition-list values, one status text that is not color-only, and an Edit recipe button. It receives no store, evidence, frame, or worker imports.

~~~tsx
export const PrecisionLabRecipeStrip = memo(function PrecisionLabRecipeStrip({
    model,
    onEditRecipe,
}: PrecisionLabRecipeStripProps) {
    return (
        <section
            className="precision-recipe"
            role="region"
            aria-label="Recipe summary"
            data-tone={model.tone}
            data-precision-region="recipe"
        >
            <dl className="precision-recipe__facts">
                <div><dt>Data</dt><dd>{model.dataset}</dd></div>
                <div><dt>Architecture</dt><dd>{model.architecture}</dd></div>
                <div><dt>Activation</dt><dd>{model.hiddenActivation}</dd></div>
                <div><dt>Output</dt><dd>{model.output}</dd></div>
                <div><dt>Seed</dt><dd>{model.seed}</dd></div>
                <div><dt>Evaluation</dt><dd>{model.evaluationLabel}</dd></div>
            </dl>
            <button type="button" onClick={onEditRecipe}>Edit recipe</button>
        </section>
    );
});
~~~

Test that each value renders, the status remains textual, and clicking Edit recipe calls onEditRecipe once.

- [ ] **Step 7: Run all recipe-strip tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/precisionLab/precisionLabRecipeModel.test.ts src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS with no act warnings.

- [ ] **Step 8: Commit the adapter and presentational strip**

~~~bash
git add apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.ts apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.test.ts apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.ts apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.tsx apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx
git commit -m "refactor: add Precision Lab recipe adapter"
~~~

### Task 3: Add ephemeral Build context disclosure

**Files:**
- Modify: apps/web/src/store/useLayoutStore.ts
- Modify: apps/web/src/store/useLayoutStore.test.ts

**Interfaces:**
- Consumes: existing view, activeRecipeSection, activeEvidenceView, audienceMode, Advanced Tools, and compatibility aliases.
- Produces:

~~~ts
export interface LayoutStore {
    readonly buildContextOpen: boolean;
    setBuildContextOpen(open: boolean): void;
    selectBuildContext(section: RecipeSectionId): void;
}
~~~

- buildContextOpen is deliberately omitted from persist.partialize and resets to false on hydration.

- [ ] **Step 1: Add failing tests for ephemeral disclosure and atomic navigation**

Cover direct selection, close, lesson/advanced selection, view switching, persistence, and compatibility aliases:

~~~ts
store.getState().selectBuildContext('network');
expect(store.getState()).toMatchObject({
    view: 'build',
    phase: 'build',
    activeRecipeSection: 'network',
    activeTabLeft: 'network',
    buildContextOpen: true,
});

const persisted = JSON.parse(storage.getItem(LAYOUT_STORAGE_KEY)!);
expect(persisted.state).not.toHaveProperty('buildContextOpen');
~~~

Hydrate a stored advanced target with buildContextOpen absent and assert Advanced Tools opens as it does now while the Build context sheet stays closed. Assert setView('run') closes Build context without changing the selected recipe section.

- [ ] **Step 2: Run the store test and verify the new actions are missing**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/store/useLayoutStore.test.ts --pool=forks --reporter=dot
~~~

Expected: FAIL because buildContextOpen and selectBuildContext do not exist.

- [ ] **Step 3: Implement atomic, nonpersisted disclosure state**

Add buildContextOpen: false to defaults and return it from every sanitizer result as false. Use a functional setView so Build preserves the current value and Run sets false:

~~~ts
setView: (view) => set((state) => ({
    view,
    phase: view,
    buildContextOpen: view === 'run' ? false : state.buildContextOpen,
})),
setBuildContextOpen: (buildContextOpen) => set({ buildContextOpen }),
selectBuildContext: (activeRecipeSection) => set((state) => ({
    view: 'build',
    phase: 'build',
    activeRecipeSection,
    activeTabLeft: activeRecipeSection,
    buildContextOpen: true,
    advancedToolsOpen: state.advancedToolsOpen
        || !isRecipeSectionVisible(state.audienceMode, false, activeRecipeSection),
})),
~~~

Route setActiveRecipeSection, setActiveTabLeft, and openAdvancedRecipeSection through the same field updates so direct lessons and explanation actions open the required context atomically. Do not add buildContextOpen to partialize.

- [ ] **Step 4: Re-run layout, profile, and visibility tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/store/useLayoutStore.test.ts src/productShell/audienceProfiles.test.ts src/productShell/visibleShell.test.ts src/productShell/shellTypes.test.ts --pool=forks --reporter=dot
~~~

Expected: PASS; legacy aliases remain synchronized.

- [ ] **Step 5: Commit the ephemeral layout state**

~~~bash
git add apps/web/src/store/useLayoutStore.ts apps/web/src/store/useLayoutStore.test.ts
git commit -m "refactor: add ephemeral Build context disclosure"
~~~

### Task 4: Build the store-free Precision Lab shell and header chrome

**Files:**
- Create: apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx
- Create: apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx
- Create: apps/web/src/styles/precisionLab.css
- Create: apps/web/src/styles/precisionLabResponsive.test.ts
- Modify: apps/web/src/components/layout/Header.tsx
- Modify: apps/web/src/components/layout/Header.test.tsx
- Modify: apps/web/src/main.tsx
- Modify: apps/web/package.json
- Modify: pnpm-lock.yaml

**Interfaces:**
- Consumes: existing audience visibility helpers, shell IDs, prepared content nodes, and explicit navigation commands.
- Produces:

~~~ts
export interface PrecisionLabShellProps {
    readonly view: WorkspaceView;
    readonly status: TrainingStatus;
    readonly activeRecipeSection: RecipeSectionId;
    readonly activeEvidenceView: EvidenceViewId;
    readonly audienceMode: AudienceMode;
    readonly advancedToolsOpen: boolean;
    readonly buildContextOpen: boolean;
    readonly openSurface: DrawerSurfaceId | null;
    readonly onSelectRecipeSection: (section: RecipeSectionId) => void;
    readonly onCloseRecipeSection: () => void;
    readonly onSelectEvidence: (view: EvidenceViewId) => void;
    readonly onCloseSurface: () => void;
    readonly recipeStripContent: ReactNode;
    readonly runSummaryContent: ReactNode;
    readonly buildContent: Readonly<Record<BuildModuleId, ReactNode>>;
    readonly topologyContent: ReactNode;
    readonly boundaryRailContent: ReactNode;
    readonly selectionContent: ReactNode;
    readonly evidenceContent: Readonly<Record<ShellEvidenceViewId, ReactNode>>;
    readonly transportContent: ReactNode;
    readonly presetContent: ReactNode;
    readonly lessonContent: ReactNode;
    readonly historyContent: ReactNode;
}
~~~

- PrecisionLabShell imports no stores, worker bridge, frame buffer, scientific selector, or training hook.
- Stable browser hooks:

~~~html
data-precision-workspace
data-precision-region="header"
data-precision-region="recipe"
data-precision-region="topology"
data-precision-region="boundary"
data-precision-region="evidence"
data-precision-region="transport"
~~~

- [ ] **Step 1: Write shell contract tests before implementation**

Render the shell with uniquely labelled nodes and assert:

1. topology and boundary slots render exactly once in Build and Run;
2. changing evidence content never replaces either primary slot;
3. only profile-visible rail items and tabs mount;
4. Advanced Tools exposes the complete Build/Run union;
5. Boundary/ Loss/ Confusion/ Inspect/ Code use one tabpanel and roving focus;
6. Arrow keys, Home, and End select and focus the expected evidence tab;
7. selecting a Build module opens the supplied context content;
8. close and Escape call onCloseRecipeSection and restore focus to the selected rail trigger;
9. Presets/Lessons/History drawers keep their existing dialog labels and close command;
10. no element inside the Boundary evidence content has data-decision-boundary-canvas.

~~~tsx
render(<PrecisionLabShell {...props} />);
expect(screen.getByLabelText('Neural network')).toBeVisible();
expect(screen.getByLabelText('Pinned decision boundary')).toBeVisible();
expect(screen.getAllByTestId('live-boundary')).toHaveLength(1);

await user.click(screen.getByRole('tab', { name: 'Loss' }));
expect(screen.getByTestId('live-boundary')).toBeVisible();
expect(screen.getByRole('tabpanel')).toHaveTextContent('loss details');
~~~

- [ ] **Step 2: Run the shell test and verify the module is absent**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/precisionLab/PrecisionLabShell.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL because PrecisionLabShell.tsx does not exist.

- [ ] **Step 3: Implement one stable shell DOM**

Use getVisibleBuildModules, getVisibleEvidenceViews, and resolveVisibleEvidenceView. Keep topology, boundary, selection, evidence, and transport containers mounted for the lifetime of a ready shell. Only the inner evidence panel changes. Derive rail items as tagged unions so Build actions call onSelectRecipeSection and Run actions call onSelectEvidence. Derive buildContext from buildContent[selectedBuildModule], activeDrawerContent from openSurface, and evidenceTablist from the visible evidence list before the return. Treat the legacy presets recipe section as Data because Presets is now a header surface.

~~~tsx
return (
    <div className={'precision-shell precision-shell--' + view} data-precision-workspace>
        <nav className="precision-rail" aria-label={view === 'build' ? 'Build tools' : 'Run tools'}>
            {visibleRailItems.map((item) => (
                <button
                    key={item.id}
                    id={'precision-rail-' + item.id}
                    type="button"
                    aria-pressed={isSelected(item.id)}
                    onClick={() => selectRailItem(item.id)}
                >
                    {item.label}
                </button>
            ))}
        </nav>
        {recipeStripContent}
        {view === 'run' && (
            <aside className="precision-run-summary" aria-label="Current run">
                {runSummaryContent}
            </aside>
        )}
        <section className="precision-topology" aria-label="Neural network" data-precision-region="topology">
            {topologyContent}
        </section>
        <aside className="precision-boundary" aria-label="Pinned decision boundary" data-precision-region="boundary">
            {boundaryRailContent}
        </aside>
        <section className="precision-selection" aria-label="Neuron selection">
            {selectionContent}
        </section>
        <section className="precision-evidence" aria-label="Evidence" data-precision-region="evidence">
            {evidenceTablist}
            <div role="tabpanel">{evidenceContent[visibleEvidenceView]}</div>
        </section>
        <footer className="precision-transport" data-status={status} data-precision-region="transport">
            {transportContent}
        </footer>
        {buildContext}
        {drawer}
    </div>
);
~~~

On context close, capture the trigger ID before unmount and focus it in requestAnimationFrame. Handle Escape inside the shell by closing the most local context panel before delegating to App-level Advanced Tools behavior.

- [ ] **Step 4: Write failing Header tests for Precision Lab semantics**

Retain Workspace view, Audience mode, Advanced Tools, Presets, Lessons, History, and lifecycle action names. Add assertions for PRECISION LAB branding, textual status, fixed step cell, the mobile utility-menu disclosure, Escape close/focus restore, and no detailed loss/accuracy values in the header.

~~~tsx
expect(screen.getByRole('banner')).toHaveAttribute('data-precision-region', 'header');
expect(screen.getByText('PRECISION LAB')).toBeVisible();
expect(screen.getByRole('group', { name: 'Workspace view' })).toBeVisible();
expect(screen.getByRole('combobox', { name: 'Audience mode' })).toBeVisible();
expect(screen.queryByLabelText('Training metrics')).not.toBeInTheDocument();
~~~

- [ ] **Step 5: Install direct Tabler imports and implement compact Header chrome**

Add the dependency:

~~~json
{
  "@tabler/icons-react": "^3.44.0"
}
~~~

Run pnpm install so pnpm-lock.yaml records the exact resolved package. Import only used icons by name:

~~~tsx
import {
    IconAdjustments,
    IconBook2,
    IconHistory,
    IconPlayerPause,
    IconPlayerPlay,
    IconSparkles,
} from '@tabler/icons-react';
~~~

Replace functional text symbols in the new Header chrome with those components and aria-hidden=true when adjacent text names the action. Keep Build and Run text visible. Show status and current step in fixed inline-size cells; leave detailed loss and accuracy in evidence surfaces. Add a phone-only utilities trigger with aria-expanded, aria-controls, Escape close, and focus restoration.

- [ ] **Step 6: Write responsive CSS guard tests**

Read precisionLab.css as text and assert:

~~~ts
expect(css).toContain('@media (min-width: 1180px)');
expect(css).toContain('@media (min-width: 680px) and (max-width: 1179px)');
expect(css).toContain('@media (max-width: 679px)');
expect(css).toContain('@media (prefers-reduced-motion: reduce)');
expect(css).toMatch(/min-height:[ ]*44px/);
expect(css).toMatch(/overflow-x:[ ]*(clip|hidden)/);
expect(css).toContain('contain: layout paint');
~~~

Also assert forge.css is imported before precisionLab.css in main.tsx.

- [ ] **Step 7: Implement isolated stable tracks**

Import precisionLab.css after forge.css in main.tsx. Define:

~~~css
.precision-shell {
    min-width: 0;
    overflow-x: clip;
    display: grid;
    grid-template-areas:
        "recipe recipe recipe"
        "rail topology boundary"
        "rail evidence boundary"
        "transport transport transport";
    grid-template-columns: 52px minmax(0, 1fr) clamp(250px, 20vw, 280px);
    grid-template-rows: auto minmax(300px, 1fr) clamp(180px, 22vh, 220px) auto;
    contain: layout paint;
}

.precision-topology,
.precision-boundary,
.precision-evidence {
    min-width: 0;
    overflow-x: clip;
}

@media (min-width: 680px) and (max-width: 1179px) {
    .precision-shell {
        grid-template-columns: 52px minmax(0, 1fr) clamp(190px, 27vw, 230px);
    }
    .precision-context {
        position: absolute;
        inset: 0 auto 0 52px;
        width: min(360px, calc(100% - 52px));
    }
}

@media (max-width: 679px) {
    .precision-shell {
        grid-template-areas:
            "recipe"
            "rail"
            "topology"
            "boundary"
            "evidence"
            "transport";
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: auto auto minmax(300px, 48vh) auto auto auto;
    }
    .precision-rail button,
    .precision-transport button,
    .precision-header button,
    .precision-header select {
        min-height: 44px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .precision-shell *,
    .precision-shell *::before,
    .precision-shell *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
~~~

Do not style any unnamespaced element selector and do not import prototype CSS.

- [ ] **Step 8: Run the shell, Header, CSS, layout, and profile suites**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/precisionLab/PrecisionLabShell.test.tsx src/components/layout/Header.test.tsx src/styles/precisionLabResponsive.test.ts src/store/useLayoutStore.test.ts src/productShell/audienceProfiles.test.ts src/productShell/visibleShell.test.ts --pool=forks --reporter=dot
pnpm lint
pnpm build
pnpm test:bundle
git diff --check
~~~

Expected: all tests PASS; lint/build/bundle/diff checks exit 0.

- [ ] **Step 9: Commit the shell contract and chrome**

~~~bash
git add apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx apps/web/src/styles/precisionLab.css apps/web/src/styles/precisionLabResponsive.test.ts apps/web/src/components/layout/Header.tsx apps/web/src/components/layout/Header.test.tsx apps/web/src/main.tsx apps/web/package.json pnpm-lock.yaml
git commit -m "feat: add Precision Lab shell and chrome"
~~~

### Task 5: Add deterministic production dataset previews

**Files:**
- Create: apps/web/src/components/controls/datasetPreviewModel.ts
- Create: apps/web/src/components/controls/datasetPreviewModel.test.ts
- Create: apps/web/src/components/controls/DatasetPreviewCanvas.tsx
- Create: apps/web/src/components/controls/DatasetPreviewCanvas.test.tsx
- Modify: apps/web/src/components/controls/DataPanel.tsx
- Modify: apps/web/src/components/controls/DataPanel.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Consumes: DatasetId, DatasetContract, DataPoint, generateDatasetV2, current data seed, and current noise.
- Produces:

~~~ts
export interface DatasetPreviewModel {
    readonly datasetId: DatasetId;
    readonly taskKind: DatasetContract['taskKind'];
    readonly points: readonly DataPoint[];
}

export function deriveDatasetPreviewModel(input: {
    readonly datasetId: DatasetId;
    readonly seed: number;
    readonly noise: number;
    readonly sampleCount?: number;
}): DatasetPreviewModel;

export interface DatasetPreviewCanvasProps {
    readonly model: DatasetPreviewModel;
}
~~~

- [ ] **Step 1: Write pure preview tests for all production datasets**

Use the eleven IDs from DataPanel. Assert each result has exactly 72 points by default, the contract taskKind matches, all x/y/label values are finite, and repeated inputs serialize identically. Assert different IDs or seeds change the point sequence.

~~~ts
const DATASET_IDS = [
    'circle',
    'xor',
    'gauss',
    'spiral',
    'moons',
    'checkerboard',
    'rings',
    'heart',
    'three-class-clusters',
    'reg-plane',
    'reg-gauss',
] as const satisfies readonly DatasetId[];

for (const datasetId of DATASET_IDS) {
    const preview = deriveDatasetPreviewModel({ datasetId, seed: 42, noise: 5 });
    expect(preview.datasetId).toBe(datasetId);
    expect(preview.points).toHaveLength(72);
    expect(preview.points.every((point) => (
        Number.isFinite(point.x)
        && Number.isFinite(point.y)
        && Number.isFinite(point.label)
    ))).toBe(true);
}
~~~

- [ ] **Step 2: Run the model test and verify it fails before implementation**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/datasetPreviewModel.test.ts --pool=forks --reporter=dot
~~~

Expected: FAIL because datasetPreviewModel.ts does not exist.

- [ ] **Step 3: Generate preview points only through the engine**

Use generateDatasetV2 with trainFraction 0.5, concatenate train and test without transformation, and freeze the outer model. Use the engine dataset contract lookup for taskKind and domain bounds; do not duplicate dataset equations.

~~~ts
export function deriveDatasetPreviewModel({
    datasetId,
    seed,
    noise,
    sampleCount = 72,
}: {
    datasetId: DatasetId;
    seed: number;
    noise: number;
    sampleCount?: number;
}): DatasetPreviewModel {
    const split = generateDatasetV2({
        dataset: datasetId,
        sampleCount,
        noise,
        seed,
        trainFraction: 0.5,
    });
    return Object.freeze({
        datasetId,
        taskKind: getDatasetContract(datasetId).taskKind,
        points: Object.freeze([...split.train, ...split.test]),
    });
}
~~~

- [ ] **Step 4: Write Canvas tests before the painter**

Mock getContext and ResizeObserver. Assert the canvas is aria-hidden because the wrapping dataset button owns the accessible name, CSS size remains 100% by 100%, backing-store size follows DPR, classification uses class colors, regression uses a continuous value scale, and an empty point list paints only the neutral background.

- [ ] **Step 5: Implement the store-free preview canvas**

Use one canvas, one ResizeObserver with a less-than-1px no-op guard, and requestAnimationFrame batching. Scale x/y from the production contract domain into the measured rectangle. Paint real points only; no SVG, CSS shapes, or image files.

~~~tsx
return (
    <canvas
        ref={canvasRef}
        className="precision-dataset-preview"
        aria-hidden="true"
    />
);
~~~

- [ ] **Step 6: Replace DataPanel chips with labelled preview buttons**

Keep the same candidate arrays, commitRecipeEdit call, disabled/loading behavior, tooltip copy, labels, and aria-pressed state. Memoize one preview model per candidate from current seed/noise.

~~~tsx
<button
    type="button"
    className={'precision-dataset-choice ' + (dataset === candidate.id ? 'is-active' : '')}
    onClick={() => chooseDataset(candidate.id)}
    aria-pressed={dataset === candidate.id}
    disabled={isLoading}
>
    <DatasetPreviewCanvas model={previews.get(candidate.id)!} />
    <span>{candidate.label}</span>
</button>
~~~

Extend DataPanel tests to click Gaussian, Checker, Three-Class, Plane, and Multi-Gauss by accessible name and assert the same production recipe transaction occurs once.

- [ ] **Step 7: Add bounded responsive preview-grid styles**

Use auto-fit grid tracks with minmax(76px, 1fr), aspect-ratio: 1, min-width: 0, and no horizontal scroller. On phone, use two columns and preserve 44px button targets.

- [ ] **Step 8: Run preview and DataPanel tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/datasetPreviewModel.test.ts src/components/controls/DatasetPreviewCanvas.test.tsx src/components/controls/DataPanel.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS; no existing recipe-edit assertion changes.

- [ ] **Step 9: Commit real dataset previews**

~~~bash
git add apps/web/src/components/controls/datasetPreviewModel.ts apps/web/src/components/controls/datasetPreviewModel.test.ts apps/web/src/components/controls/DatasetPreviewCanvas.tsx apps/web/src/components/controls/DatasetPreviewCanvas.test.tsx apps/web/src/components/controls/DataPanel.tsx apps/web/src/components/controls/DataPanel.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "feat: add deterministic dataset previews"
~~~

### Task 6: Derive persistent neuron selection and strongest influences

**Files:**
- Create: apps/web/src/components/visualization/networkSelectionModel.ts
- Create: apps/web/src/components/visualization/networkSelectionModel.test.ts
- Create: apps/web/src/components/visualization/useNetworkSelectionController.ts
- Create: apps/web/src/components/visualization/useNetworkSelectionController.test.tsx
- Create: apps/web/src/components/visualization/NetworkSelectionDeck.tsx
- Create: apps/web/src/components/visualization/NetworkSelectionDeck.test.tsx

**Interfaces:**
- Consumes: accepted frameBuffer weight/bias/neuron-grid arrays and layouts, paramsVersion, neuronGridsVersion, evidenceGenerationId, and compiled architecture key.
- Produces:

~~~ts
export interface NetworkNodeRef {
    readonly layerIdx: number;
    readonly nodeIdx: number;
}

export interface NetworkInfluence {
    readonly edgeKey: string;
    readonly direction: 'incoming' | 'outgoing';
    readonly peer: NetworkNodeRef;
    readonly peerLabel: string;
    readonly weight: number;
    readonly magnitude: number;
    readonly sign: 'positive' | 'negative' | 'zero';
}

export type NetworkSelectionDisplayModel =
    | { readonly kind: 'empty' }
    | {
        readonly kind: 'selected';
        readonly node: NetworkNodeRef;
        readonly nodeLabel: string;
        readonly bias: number | null;
        readonly grid: Float32Array | null;
        readonly gridSize: number;
        readonly activation: {
            readonly minimum: number;
            readonly maximum: number;
            readonly mean: number;
            readonly standardDeviation: number;
        } | null;
        readonly incoming: readonly NetworkInfluence[];
        readonly outgoing: readonly NetworkInfluence[];
        readonly highlightedEdgeKeys: ReadonlySet<string>;
    };

export interface NetworkSelectionController {
    readonly model: NetworkSelectionDisplayModel;
    readonly selectedNode: NetworkNodeRef | null;
    readonly commands: {
        selectNode(node: NetworkNodeRef): void;
        clearSelection(): void;
    };
}
~~~

- [ ] **Step 1: Write pure-model tests using packed buffers**

Construct a 2 -> 3 -> 2 packed network with known weights, biases, and 4x4 neuron grids. Assert correct layerWeightOffset/layerBiasOffset indexing, top-three absolute ranking per direction, signed labels, output-node handling, input-node null bias/grid, invalid-node empty state, finite zero-spread statistics, and exact grid subarray identity:

~~~ts
const model = deriveNetworkSelectionModel(snapshot, { layerIdx: 1, nodeIdx: 1 });
expect(model.kind).toBe('selected');
if (model.kind === 'selected') {
    expect(model.grid!.buffer).toBe(snapshot.neuronGrids!.buffer);
    expect([...model.highlightedEdgeKeys]).toEqual([
        '1:1:0',
        '1:1:1',
        '2:0:1',
        '2:1:1',
    ]);
}
~~~

- [ ] **Step 2: Run the model test and verify the module is absent**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/networkSelectionModel.test.ts --pool=forks --reporter=dot
~~~

Expected: FAIL because networkSelectionModel.ts does not exist.

- [ ] **Step 3: Implement the React-free derivation**

Read each accepted typed array once from the input snapshot. Return at most three incoming and three outgoing influences sorted by descending magnitude with edgeKey from networkGraphPainter.edgeRefKey. Compute activation statistics in one pass without allocating a copied grid.

~~~ts
for (let index = 0; index < grid.length; index += 1) {
    const value = grid[index];
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    sumSquares += value * value;
}
const mean = sum / grid.length;
const variance = Math.max(0, sumSquares / grid.length - mean * mean);
~~~

- [ ] **Step 4: Write controller tests for persistence and reset fences**

Assert selection survives paramsVersion and neuronGridsVersion changes, survives status/metric changes, and clears only when commands.clearSelection runs, evidenceGenerationId changes, or architectureKey changes. Assert selecting a second node replaces the first. Assert one rerender reads one coherent frame-buffer snapshot.

- [ ] **Step 5: Implement the focused selection controller**

Subscribe only to paramsVersion, neuronGridsVersion, evidenceGenerationId, and primitive architectureKey. Hold NetworkNodeRef in local React state. Read getFrameBuffer once inside useMemo after touching both version values. Do not put weights, biases, or grids into state.

~~~ts
const snapshot = useMemo(() => {
    void paramsVersion;
    void neuronGridsVersion;
    return getFrameBuffer();
}, [neuronGridsVersion, paramsVersion]);

useEffect(() => {
    setSelectedNode(null);
}, [architectureKey, evidenceGenerationId]);
~~~

- [ ] **Step 6: Write and implement the store-free selection deck**

The empty state explains how to select a neuron. The selected state shows node label, real activation tile or neutral unavailable state, activation range/mean/standard deviation, bias, strongest inputs, strongest outputs, and a Clear selection button. Use text Positive/Negative/Zero in addition to color.

~~~tsx
if (model.kind === 'empty') {
    return <p>Select a neuron to inspect its activation and strongest influences.</p>;
}

function formatFinite(value: number | null | undefined): string {
    return typeof value === 'number' && Number.isFinite(value)
        ? value.toFixed(4)
        : 'Unavailable';
}

function InfluenceList({
    label,
    items,
}: {
    label: string;
    items: readonly NetworkInfluence[];
}) {
    return (
        <section aria-label={label}>
            <h3>{label}</h3>
            {items.length === 0
                ? <p>None</p>
                : (
                    <ol>
                        {items.map((item) => (
                            <li key={item.edgeKey}>
                                {item.peerLabel + ' · ' + item.sign + ' · ' + item.weight.toFixed(4)}
                            </li>
                        ))}
                    </ol>
                )}
        </section>
    );
}

return (
    <section className="precision-selection-deck" aria-label="Selected neuron details">
        <h2>{model.nodeLabel}</h2>
        <dl>
            <div><dt>Bias</dt><dd>{formatFinite(model.bias)}</dd></div>
            <div><dt>Mean activation</dt><dd>{formatFinite(model.activation?.mean)}</dd></div>
        </dl>
        <InfluenceList label="Strongest inputs" items={model.incoming} />
        <InfluenceList label="Strongest outputs" items={model.outgoing} />
        <button type="button" onClick={onClear}>Clear selection</button>
    </section>
);
~~~

- [ ] **Step 7: Run all selection tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/networkSelectionModel.test.ts src/components/visualization/useNetworkSelectionController.test.tsx src/components/visualization/NetworkSelectionDeck.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS with no worker mocks in pure/view tests and no typed-array copies.

- [ ] **Step 8: Commit the selection adapter and deck**

~~~bash
git add apps/web/src/components/visualization/networkSelectionModel.ts apps/web/src/components/visualization/networkSelectionModel.test.ts apps/web/src/components/visualization/useNetworkSelectionController.ts apps/web/src/components/visualization/useNetworkSelectionController.test.tsx apps/web/src/components/visualization/NetworkSelectionDeck.tsx apps/web/src/components/visualization/NetworkSelectionDeck.test.tsx
git commit -m "feat: add persistent neuron selection model"
~~~

### Task 7: Render responsive activation tiles and selected paths in Canvas

**Files:**
- Modify: apps/web/src/components/visualization/networkGraphPainter.ts
- Create: apps/web/src/components/visualization/networkGraphPainter.test.ts
- Modify: apps/web/src/components/visualization/NetworkGraphCanvas.tsx
- Modify: apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx
- Modify: apps/web/src/components/visualization/NetworkGraph.tsx
- Create: apps/web/src/components/visualization/NetworkGraph.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Consumes: NetworkSelectionController from Task 6.
- Produces:

~~~ts
export interface NodeGeometry {
    readonly width: number;
    readonly height: number;
    readonly cornerRadius: number;
    readonly hitPadding: number;
}

export function deriveNodeGeometry(
    availableHeight: number,
    largestLayer: number,
): NodeGeometry;

export interface NetworkGraphRendererProps {
    readonly controller: NetworkSelectionController;
}

export interface NetworkGraphProps {
    readonly selectionController?: NetworkSelectionController;
}
~~~

- [ ] **Step 1: Add painter tests for bounded tile geometry and selected paths**

Assert deriveNodeGeometry clamps tiles to 30–64 CSS pixels, fits six nodes within 360px without overlap, produces a 6–12px corner radius, and increases hit area without changing visual size. Assert unselected edges remain visible at reduced alpha, selected positive edges are solid, selected negative edges are dashed, and magnitude controls width.

~~~ts
expect(deriveNodeGeometry(360, 6)).toEqual({
    width: 44,
    height: 44,
    cornerRadius: 9,
    hitPadding: 6,
});
~~~

Use exact expected geometry values produced by the final formula; keep the clamp tests independent of that one representative value.

- [ ] **Step 2: Run painter and Canvas tests to verify new contracts fail**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/networkGraphPainter.test.ts src/components/visualization/NetworkGraphCanvas.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL because NodeGeometry and renderer selection props are missing.

- [ ] **Step 3: Implement tile geometry and selected-edge paint passes**

Replace fixed NODE_RADIUS geometry in Canvas painter inputs with NodeGeometry. Keep batched rendering for ordinary edges. Paint selected edges in a final small pass so positive and negative styles remain distinguishable without sacrificing the normal batched path.

~~~ts
export function deriveNodeGeometry(availableHeight: number, largestLayer: number): NodeGeometry {
    const safeCount = Math.max(1, largestLayer);
    const rawSize = Math.floor((Math.max(160, availableHeight) - 56 - (safeCount - 1) * 8) / safeCount);
    const size = Math.max(30, Math.min(64, rawSize));
    return {
        width: size,
        height: size,
        cornerRadius: Math.max(6, Math.min(12, Math.round(size * 0.2))),
        hitPadding: 6,
    };
}
~~~

Pass highlightedEdgeKeys into paintEdges. For a selected negative edge call ctx.setLineDash([6, 4]); for positive call ctx.setLineDash([]); reset line dash after the final pass.

- [ ] **Step 4: Replace broad frame subscription and no-op resize churn**

NetworkGraphCanvas must subscribe to paramsVersion and neuronGridsVersion instead of frameVersion. Its ResizeObserver batches the last contentRect in requestAnimationFrame and returns the previous size when both dimensions differ by less than 1 CSS pixel.

~~~ts
setContainerSize((previous) => (
    Math.abs(previous.width - width) < 1 && Math.abs(previous.height - height) < 1
        ? previous
        : { width, height }
));
~~~

Read getFrameBuffer once per derived model. Retain grid subarrays and paint them into rounded rectangular tiles. Paint a labelled neutral tile when a grid is unavailable; never manufacture a heatmap.

- [ ] **Step 5: Add persistent selection interaction**

Split NetworkGraph into NetworkGraphWithLocalController and a renderer component that requires a controller. The public no-prop compatibility path renders NetworkGraphWithLocalController; the Precision Lab path passes App's selectionController and therefore does not create a second controller. Both Canvas and SVG receive the same controller. Canvas hit-testing uses NodeGeometry and calls selectNode on click, Enter, or Space. Selected tiles expose aria-pressed=true through the existing keyboard overlay controls; Escape clears only graph selection when focus is inside the graph.

~~~tsx
export const NetworkGraph = memo(function NetworkGraph({
    selectionController,
}: NetworkGraphProps) {
    return selectionController
        ? <NetworkGraphRenderer controller={selectionController} />
        : <NetworkGraphWithLocalController />;
});

function NetworkGraphWithLocalController() {
    const controller = useNetworkSelectionController();
    return <NetworkGraphRenderer controller={controller} />;
}
~~~

- [ ] **Step 6: Add bounded topology styles**

Keep the graph container min-width: 0, block internal horizontal scrolling, preserve toolbar controls, and size the visual canvas from its stable parent. Add selected tile ring, reduced ordinary-edge emphasis, and explicit negative dash legend. Disable flow transitions under reduced motion.

- [ ] **Step 7: Run Canvas, wrapper, painter, lesson, and accessibility tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/networkGraphPainter.test.ts src/components/visualization/NetworkGraphCanvas.test.tsx src/components/visualization/NetworkGraph.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS; architecture summary, zoom, pan, fit, view mode, edge filters, health states, and lesson callouts remain covered.

- [ ] **Step 8: Commit responsive Canvas tiles**

~~~bash
git add apps/web/src/components/visualization/networkGraphPainter.ts apps/web/src/components/visualization/networkGraphPainter.test.ts apps/web/src/components/visualization/NetworkGraphCanvas.tsx apps/web/src/components/visualization/NetworkGraphCanvas.test.tsx apps/web/src/components/visualization/NetworkGraph.tsx apps/web/src/components/visualization/NetworkGraph.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "feat: render responsive activation tiles"
~~~

### Task 8: Preserve functional parity in the SVG fallback

**Files:**
- Modify: apps/web/src/components/visualization/NetworkGraphSVG.tsx
- Modify: apps/web/src/components/visualization/NetworkGraphSVG.test.tsx
- Modify: apps/web/src/components/visualization/NetworkGraph.tsx
- Modify: apps/web/src/components/visualization/NetworkGraph.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Consumes: the same NetworkGraphRendererProps and NetworkSelectionController from Tasks 6–7.
- Produces: one shared selection meaning across Canvas and SVG; visual node shape and spacing may remain renderer-specific.

- [ ] **Step 1: Add fallback parity tests**

Switch the production feature flag to SVG and assert:

1. weight/bias/grid labels match packed frame data;
2. clicking and keyboard-activating a node selects it;
3. selection persists across new parameter and grid versions;
4. the same strongest edge keys receive selected styling;
5. negative selected paths have a non-color cue;
6. switching Canvas -> SVG -> Canvas retains the selected node;
7. generation or architecture change clears selection;
8. zoom/pan/fit, mode, filter, health, status, tooltips, and Architecture summary remain available.

~~~tsx
await user.click(screen.getByRole('button', { name: 'Hidden 1, Neuron 2' }));
expect(screen.getByRole('button', { name: 'Hidden 1, Neuron 2' })).toHaveAttribute(
    'aria-pressed',
    'true',
);
expect(screen.getByLabelText('Selected neuron details')).toHaveTextContent('Hidden 1, Neuron 2');
~~~

- [ ] **Step 2: Run fallback tests and verify parity assertions fail**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/NetworkGraphSVG.test.tsx src/components/visualization/NetworkGraph.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL on selection props/semantics while existing SVG tests continue to execute.

- [ ] **Step 3: Wire SVG nodes and paths to the shared controller**

Keep existing SVG geometry. Add aria-pressed and onClick/onKeyDown to node hit targets. Use selection.model.highlightedEdgeKeys to reduce ordinary paths and emphasize selected paths; add strokeDasharray for selected negative weights. Use the same node and edge key helpers as Canvas.

~~~tsx
<g
    role="button"
    tabIndex={0}
    aria-label={describeGraphNode(layerIdx, nodeIdx, layers.length)}
    aria-pressed={isSelected}
    onClick={() => onSelectNode({ layerIdx, nodeIdx })}
    onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSelectNode({ layerIdx, nodeIdx });
        }
    }}
>
    {nodeVisual}
</g>
~~~

- [ ] **Step 4: Run the complete graph test set and checkpoint gates**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/networkSelectionModel.test.ts src/components/visualization/useNetworkSelectionController.test.tsx src/components/visualization/NetworkSelectionDeck.test.tsx src/components/visualization/networkGraphPainter.test.ts src/components/visualization/NetworkGraphCanvas.test.tsx src/components/visualization/NetworkGraphSVG.test.tsx src/components/visualization/NetworkGraph.test.tsx --pool=forks --reporter=dot
pnpm lint
pnpm build
pnpm test:bundle
git diff --check
~~~

Expected: all graph tests PASS and every checkpoint gate exits 0.

- [ ] **Step 5: Commit fallback parity**

~~~bash
git add apps/web/src/components/visualization/NetworkGraphSVG.tsx apps/web/src/components/visualization/NetworkGraphSVG.test.tsx apps/web/src/components/visualization/NetworkGraph.tsx apps/web/src/components/visualization/NetworkGraph.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "feat: preserve graph fallback selection parity"
~~~

### Task 9: Split one decision-boundary controller into one canvas and one details panel

**Files:**
- Create: apps/web/src/components/visualization/useDecisionBoundaryController.ts
- Create: apps/web/src/components/visualization/useDecisionBoundaryController.test.tsx
- Create: apps/web/src/components/visualization/DecisionBoundaryCanvas.tsx
- Create: apps/web/src/components/visualization/DecisionBoundaryCanvas.test.tsx
- Create: apps/web/src/components/visualization/PinnedBoundaryRail.tsx
- Create: apps/web/src/components/visualization/PinnedBoundaryRail.test.tsx
- Create: apps/web/src/components/visualization/BoundaryEvidencePanel.tsx
- Create: apps/web/src/components/visualization/BoundaryEvidencePanel.test.tsx
- Modify: apps/web/src/components/visualization/DecisionBoundary.tsx
- Modify: apps/web/src/components/visualization/DecisionBoundary.test.tsx
- Modify: apps/web/src/components/visualization/useDecisionBoundaryModel.ts
- Modify: apps/web/src/components/visualization/useDecisionBoundaryModel.test.tsx
- Modify: apps/web/src/components/layout/MainArea.tsx
- Modify: apps/web/src/components/layout/MainArea.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Consumes: DecisionBoundaryDisplayModel, existing outputGridVersion/multiclassBoundaryVersion, one accepted frameBuffer snapshot, scientific evidence, recipe identity, document view, and editView.
- Produces:

~~~ts
export interface BoundaryRailDisplayModel {
    readonly freshnessLabel: string;
    readonly tone: 'fresh' | 'stale' | 'drift' | 'unavailable';
    readonly metric: {
        readonly label: 'Accuracy' | 'Test loss';
        readonly value: string;
    } | null;
}

export interface DecisionBoundaryController {
    readonly model: DecisionBoundaryDisplayModel;
    readonly rail: BoundaryRailDisplayModel;
    readonly controls: {
        readonly showTestData: boolean;
        readonly discretize: boolean;
        readonly overlayMode: DecisionOverlayMode;
        readonly overlayCopy: DecisionOverlayCopy;
    };
    readonly commands: {
        setShowTestData(value: boolean): Promise<void>;
        setDiscretize(value: boolean): Promise<void>;
        setOverlayMode(value: DecisionOverlayMode): void;
    };
}

export function useDecisionBoundaryController(): DecisionBoundaryController;

export interface DecisionBoundaryCanvasProps {
    readonly model: DecisionBoundaryDisplayModel;
    readonly density?: 'standard' | 'compact';
}

export interface PinnedBoundaryRailProps {
    readonly model: DecisionBoundaryDisplayModel;
    readonly rail: BoundaryRailDisplayModel;
    readonly onExpand: () => void;
}

export interface BoundaryEvidencePanelProps {
    readonly controller: DecisionBoundaryController;
}
~~~

- [ ] **Step 1: Write controller tests for coherent snapshots and commands**

Seed real playground/training stores and mock getFrameBuffer once per derivation. Assert:

1. outputGridVersion refreshes scalar model and no unrelated frame version does;
2. multiclassBoundaryVersion refreshes multiclass model;
3. one derivation reads one accepted frame snapshot;
4. train/test arrays are passed through without copies;
5. setShowTestData and setDiscretize use editView and preserve unrelated document.view fields;
6. overlay state is local and never enters the V2 document;
7. classification uses full test accuracy when fresh;
8. regression uses full test loss when fresh;
9. trained-recipe drift and evaluation age produce textual drift/stale rail states.

~~~tsx
await act(async () => controller.result.current.commands.setShowTestData(true));
expect(usePlaygroundStore.getState().access.prepared.document.view).toMatchObject({
    showTestData: true,
    discretizeOutput: false,
});
expect(mockGetFrameBuffer).toHaveBeenCalledTimes(1);
~~~

- [ ] **Step 2: Run the controller test and verify the module is absent**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/useDecisionBoundaryController.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL because useDecisionBoundaryController.ts does not exist.

- [ ] **Step 3: Implement the controller as the only production boundary adapter**

Reuse useDecisionBoundaryModel semantics, getRecipeDrift, and selectScientificEvidence. Keep overlayMode in the controller. Select only trainPoints, testPoints, latestLiveSignal, latestEvaluation, trained/current identity, and view booleans. Return frozen controls and rail values; expose no stores or raw evidence.

~~~ts
const model = useDecisionBoundaryModel({
    trainPoints,
    testPoints,
    showTestData,
    discretize,
    overlayMode,
});
const commands = useMemo(() => ({
    setShowTestData: (value: boolean) => editView((view) => ({
        ...view,
        showTestData: value,
    })),
    setDiscretize: (value: boolean) => editView((view) => ({
        ...view,
        discretizeOutput: value,
    })),
    setOverlayMode,
}), [editView]);
~~~

Keep useDecisionBoundaryModel subscribed only to outputGridVersion and multiclassBoundaryVersion and reading getFrameBuffer exactly once inside its memo.

- [ ] **Step 4: Write store-free canvas tests before extraction**

Move current DecisionBoundary painting assertions into DecisionBoundaryCanvas.test.tsx. Cover scalar smooth/discrete, uncertainty, misclassification, train/test points, multiclass, empty/unavailable, accessible summaries, exact typed-array identity, compact/standard density, DPR, and a ResizeObserver no-op below 1px.

- [ ] **Step 5: Extract the canvas without changing scientific derivation**

Move all DOM measurement, off-screen canvas reuse, ImageData reuse, and paint helpers from DecisionBoundary.tsx into DecisionBoundaryCanvas.tsx. Accept only model and density. Mark the real canvas:

~~~tsx
const canvasLabel = model.kind === 'multiclass'
    ? 'Multiclass decision boundary visualization showing predicted class regions and confidence'
    : 'Decision boundary visualization showing model output and data points';

return <canvas
    ref={canvasRef}
    data-decision-boundary-canvas
    role="img"
    aria-label={canvasLabel}
    aria-describedby={descriptionId}
/>
~~~

Use aspect-ratio: 1 in CSS, never a measured height fed back into layout, and a requestAnimationFrame-batched less-than-1px resize guard. Retain getDecisionOverlayCopy, classifyPointFromGrid, and DecisionBoundaryProps exports from DecisionBoundary.tsx. Keep DecisionBoundary as a compatibility wrapper that calls useDecisionBoundaryModel and renders DecisionBoundaryCanvas.

- [ ] **Step 6: Write rail and evidence-panel view tests**

PinnedBoundaryRail must render exactly one DecisionBoundaryCanvas, freshness text, optional metric, and Expand boundary button. BoundaryEvidencePanel must render zero canvases, all four overlay buttons, test/discretize checkboxes, accessible summary/provenance text from the display model, and existing decision-boundary concept help. Every event forwards to the supplied commands.

~~~tsx
render(<BoundaryEvidencePanel controller={controller} />);
expect(document.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(0);
await user.click(screen.getByRole('button', { name: 'Uncertain' }));
expect(controller.commands.setOverlayMode).toHaveBeenCalledWith('uncertainty');
~~~

- [ ] **Step 7: Implement the rail and details panel**

PinnedBoundaryRail remains store-free:

~~~tsx
return (
    <section className="precision-boundary-rail" aria-label="Decision boundary">
        <DecisionBoundaryCanvas model={model} density="compact" />
        <div role="status" data-tone={rail.tone}>{rail.freshnessLabel}</div>
        {rail.metric && (
            <dl><div><dt>{rail.metric.label}</dt><dd>{rail.metric.value}</dd></div></dl>
        )}
        <button type="button" onClick={onExpand}>Expand boundary details</button>
    </section>
);
~~~

BoundaryEvidencePanel reads only controller values and commands. Move the production no-canvas details composition into MainArea as BoundaryEvidenceContent({ controller }) while retaining BoundaryContent as a legacy direct-render wrapper until Task 14 proves it has no production consumer.

- [ ] **Step 8: Run every boundary suite**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/decisionBoundaryModel.test.ts src/components/visualization/useDecisionBoundaryModel.test.tsx src/components/visualization/useDecisionBoundaryController.test.tsx src/components/visualization/DecisionBoundaryCanvas.test.tsx src/components/visualization/PinnedBoundaryRail.test.tsx src/components/visualization/BoundaryEvidencePanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx src/components/layout/MainArea.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS; the public wrapper and helper exports remain compatible.

- [ ] **Step 9: Commit the single-boundary building blocks**

~~~bash
git add apps/web/src/components/visualization/useDecisionBoundaryController.ts apps/web/src/components/visualization/useDecisionBoundaryController.test.tsx apps/web/src/components/visualization/DecisionBoundaryCanvas.tsx apps/web/src/components/visualization/DecisionBoundaryCanvas.test.tsx apps/web/src/components/visualization/PinnedBoundaryRail.tsx apps/web/src/components/visualization/PinnedBoundaryRail.test.tsx apps/web/src/components/visualization/BoundaryEvidencePanel.tsx apps/web/src/components/visualization/BoundaryEvidencePanel.test.tsx apps/web/src/components/visualization/DecisionBoundary.tsx apps/web/src/components/visualization/DecisionBoundary.test.tsx apps/web/src/components/visualization/useDecisionBoundaryModel.ts apps/web/src/components/visualization/useDecisionBoundaryModel.test.tsx apps/web/src/components/layout/MainArea.tsx apps/web/src/components/layout/MainArea.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "refactor: split pinned boundary from evidence details"
~~~

### Task 10: Switch App to Precision Lab and make pinned-boundary demand authoritative

**Files:**
- Modify: apps/web/src/components/layout/deriveVisualizationDemand.ts
- Modify: apps/web/src/components/layout/deriveVisualizationDemand.test.ts
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx
- Modify: apps/web/src/components/layout/MainArea.tsx
- Modify: apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx

**Interfaces:**
- Consumes: PrecisionLabShell, recipe model/strip, selection controller/deck, boundary controller/rail/details, existing content components, and one useTraining result.
- Produces revised demand input:

~~~ts
export function deriveVisualizationDemand(args: {
    readonly view: WorkspaceView;
    readonly activeEvidenceView: EvidenceViewId;
    readonly audienceMode: AudienceMode;
    readonly advancedToolsOpen: boolean;
    readonly graphRenderer: 'canvas' | 'svg';
    readonly boundaryRailMounted: boolean;
}): VisualizationDemand;
~~~

- Produces one production call to useTraining, one useDecisionBoundaryController, one useNetworkSelectionController, and exactly one data-decision-boundary-canvas.

- [ ] **Step 1: Add failing demand tests for the pinned rail**

Assert needDecisionBoundary is true in Build and every Run evidence tab when boundaryRailMounted is true, false when it is false, and independent of confusion/inspection demand. Assert Advanced Tools alone requests no confusion or inspection artifact.

~~~ts
expect(deriveVisualizationDemand({
    view: 'build',
    activeEvidenceView: 'loss',
    audienceMode: 'explore',
    advancedToolsOpen: false,
    graphRenderer: 'canvas',
    boundaryRailMounted: true,
}).needDecisionBoundary).toBe(true);
~~~

- [ ] **Step 2: Run the demand test and verify the old visibility rule fails**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/deriveVisualizationDemand.test.ts --pool=forks --reporter=dot
~~~

Expected: FAIL because Build/Loss currently do not request a boundary.

- [ ] **Step 3: Make rail mount state the only graphical boundary-demand input**

Set needDecisionBoundary: args.boundaryRailMounted. Keep confusion and inspection dependent on the resolved visible Run evidence view. Keep needNeuronGrids based on the active graph renderer. Return a fresh immutable demand object through the existing App effect; do not alter scientific evaluation cadence or cached-domain behavior.

- [ ] **Step 4: Write App integration assertions before switching shells**

Extend App and appShell integration tests to assert:

1. Build and Run both render data-precision-workspace;
2. main landmark keeps Neural network playground workspace;
3. topology and pinned boundary are simultaneously visible;
4. exactly one data-decision-boundary-canvas exists after cycling every evidence tab;
5. Boundary tab contains controls and no second canvas;
6. Expand boundary details selects/focuses Boundary without replacing topology;
7. Build rail/context actions mount existing production controls;
8. direct lesson/explanation hidden-tool actions open Advanced Tools and Build context atomically;
9. audience changes alter visibility only;
10. URL/hash, prepared recipe, generation, revision, checkpoint timeline, saved-run count, and training state remain unchanged while cycling mode/disclosure/shell navigation;
11. App calls the mocked useTraining hook once;
12. pending configuration, worker failure, incompatible access, empty evidence, stale trained recipe, and unavailable visualization states retain their existing recovery commands and provenance-aware copy;
13. rapid Data/Network edits still compose against the latest candidate document and stale preparation completions stay fenced;
14. all six Beginner/Explore/Lab x Build/Run axe states report zero violations.

- [ ] **Step 5: Compose adapters in CompatiblePlayground**

Add:

~~~ts
const recipeModel = usePrecisionLabRecipeModel();
const boundary = useDecisionBoundaryController();
const networkSelection = useNetworkSelectionController();
const buildContextOpen = useLayoutStore((state) => state.buildContextOpen);
const selectBuildContext = useLayoutStore((state) => state.selectBuildContext);
const setBuildContextOpen = useLayoutStore((state) => state.setBuildContextOpen);
~~~

Pass display-safe components into PrecisionLabShell:

~~~tsx
<PrecisionLabShell
    view={view}
    status={status}
    activeRecipeSection={activeRecipeSection}
    activeEvidenceView={activeEvidenceView}
    audienceMode={audienceMode}
    advancedToolsOpen={advancedToolsOpen}
    buildContextOpen={buildContextOpen}
    openSurface={openSurface}
    onSelectRecipeSection={selectBuildContext}
    onCloseRecipeSection={() => setBuildContextOpen(false)}
    onSelectEvidence={setActiveEvidenceView}
    onCloseSurface={closeSurface}
    recipeStripContent={(
        <PrecisionLabRecipeStrip
            model={recipeModel}
            onEditRecipe={() => selectBuildContext('data')}
        />
    )}
    runSummaryContent={<CurrentRunCard />}
    buildContent={buildContent}
    topologyContent={<CanvasContent selectionController={networkSelection} />}
    boundaryRailContent={(
        <PinnedBoundaryRail
            model={boundary.model}
            rail={boundary.rail}
            onExpand={() => {
                setActiveEvidenceView('boundary');
                requestAnimationFrame(() => {
                    document.getElementById('precision-evidence-tab-boundary')?.focus();
                });
            }}
        />
    )}
    selectionContent={(
        <NetworkSelectionDeck
            model={networkSelection.model}
            onClear={networkSelection.commands.clearSelection}
        />
    )}
    evidenceContent={{
        boundary: <BoundaryEvidencePanel controller={boundary} />,
        loss: <LossContent />,
        confusion: <ConfusionContent />,
        inspection: <InspectContent />,
        code: <CodeContent />,
    }}
    transportContent={transport}
    presetContent={presetContent}
    lessonContent={lessonContent}
    historyContent={historyContent}
/>
~~~

CanvasContent may accept an optional selectionController prop by passing it to NetworkGraph. Its no-prop compatibility path creates an internal controller in a separate wrapper component so hooks are never conditional.

- [ ] **Step 6: Preserve error recovery, shortcuts, focus ordering, and status region**

Leave CompatibilityState, AccessibilityAnnouncer, worker-error alertdialog, global shortcuts, ErrorBoundary, StatusBar, openSurface state, and Advanced Tools Escape handling in App. Change only shell composition and the display adapters named above.

- [ ] **Step 7: Run integration and checkpoint gates**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/deriveVisualizationDemand.test.ts src/components/layout/precisionLab/PrecisionLabShell.test.tsx src/App.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot
pnpm lint
pnpm build
pnpm test:bundle
git diff --check
~~~

Expected: focused tests PASS; exactly one boundary canvas assertion passes in both views; every checkpoint gate exits 0.

- [ ] **Step 8: Commit the production shell switch**

~~~bash
git add apps/web/src/components/layout/deriveVisualizationDemand.ts apps/web/src/components/layout/deriveVisualizationDemand.test.ts apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/__tests__/appShell.integration.test.tsx apps/web/src/components/layout/MainArea.tsx apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx
git commit -m "feat: switch production to Precision Lab"
~~~

### Task 11: Make Loss and Confusion responsive inside the bounded evidence deck

**Files:**
- Modify: apps/web/src/components/visualization/LossChart.tsx
- Modify: apps/web/src/components/visualization/LossChart.test.tsx
- Modify: apps/web/src/components/visualization/ConfusionMatrix.tsx
- Modify: apps/web/src/components/visualization/ConfusionMatrix.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Produces:

~~~ts
export interface LossChartViewport {
    readonly width: number;
    readonly height: number;
}

export function deriveLossChartViewport(containerWidth: number): LossChartViewport;
~~~

- The chart height clamps to 96–140 CSS pixels and all draw functions receive viewport.height rather than a module-level fixed height.

- [ ] **Step 1: Add Loss viewport and ResizeObserver tests**

Assert widths are rounded and at least 1, heights clamp at 96 and 140, a repeated/sub-pixel resize returns the same state object, DPR changes backing dimensions only, chart CSS dimensions do not change when new metrics arrive, and the empty state uses the same bounded container.

~~~ts
expect(deriveLossChartViewport(320)).toEqual({ width: 320, height: 112 });
expect(deriveLossChartViewport(160).height).toBe(96);
expect(deriveLossChartViewport(900).height).toBe(140);
~~~

- [ ] **Step 2: Run Loss tests and verify fixed-height expectations fail**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/LossChart.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL because the current chart uses a fixed 220px height and updates on every observed width.

- [ ] **Step 3: Implement a responsive chart viewport**

Use:

~~~ts
export function deriveLossChartViewport(containerWidth: number): LossChartViewport {
    const width = Math.max(1, Math.round(containerWidth));
    const height = Math.max(96, Math.min(140, Math.round(width * 0.35)));
    return { width, height };
}
~~~

Pass width and height through drawChart and drawSeries. Position tab controls, legend, and basis outside the plot. Use requestAnimationFrame-batched ResizeObserver updates with a less-than-1px no-op guard. Keep metricHistoryBuffer reads keyed only by trainingTrendVersion and evaluationHistoryVersion.

- [ ] **Step 4: Add Confusion structure tests for binary and multiclass layouts**

Assert matrix and metrics are siblings in precision-confusion-layout, no inline min-width or overflow style exists, metrics use a definition list, provenance follows both, and 3-class labels/cells retain their current accessible names and values.

~~~tsx
expect(screen.getByLabelText('Confusion matrix layout')).toContainElement(
    screen.getByLabelText('Confusion metrics'),
);
expect(screen.getByLabelText('Confusion matrix layout')).not.toHaveStyle({
    overflowX: 'auto',
});
~~~

- [ ] **Step 5: Implement responsive matrix-and-metrics composition**

Do not change calculations. Wrap current matrix and metrics in:

~~~tsx
<div className="precision-confusion-layout" aria-label="Confusion matrix layout">
    <div className="precision-confusion-layout__matrix">{matrixGrid}</div>
    <dl className="precision-confusion-layout__metrics" aria-label="Confusion metrics">
        {metricItems}
    </dl>
    <ProvenanceCaption {...provenance} />
</div>
~~~

Use minmax(0, 1fr) tracks, clamp cell sizing, and stack metrics below the matrix under 520px container width. Permit vertical overflow only on the evidence deck; set overflow-x: clip on Loss and Confusion roots.

- [ ] **Step 6: Run evidence tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/visualization/LossChart.test.tsx src/components/visualization/ConfusionMatrix.test.tsx src/components/layout/precisionLab/PrecisionLabShell.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS; numeric evidence and provenance remain unchanged.

- [ ] **Step 7: Commit bounded evidence layouts**

~~~bash
git add apps/web/src/components/visualization/LossChart.tsx apps/web/src/components/visualization/LossChart.test.tsx apps/web/src/components/visualization/ConfusionMatrix.tsx apps/web/src/components/visualization/ConfusionMatrix.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "fix: make evidence deck responsive"
~~~

### Task 12: Share saved-run capture and compact the transport

**Files:**
- Create: apps/web/src/components/controls/useSaveCurrentRun.ts
- Create: apps/web/src/components/controls/useSaveCurrentRun.test.tsx
- Modify: apps/web/src/components/controls/TrainingControls.tsx
- Modify: apps/web/src/components/controls/TrainingControls.test.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.tsx
- Modify: apps/web/src/components/controls/RunHistoryPanel.test.tsx
- Modify: apps/web/src/components/layout/MainArea.tsx
- Modify: apps/web/src/App.tsx
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/styles/precisionLab.css

**Interfaces:**
- Consumes: getWorkerApi.captureRunArtifact, createUuid, experimentMemoryStore hydration/pending/error/save/retry/discard actions, and prepared V2 availability.
- Produces:

~~~ts
export type SaveCurrentRunStatus = 'ready' | 'saving' | 'blocked' | 'error';

export interface SaveCurrentRunController {
    readonly status: SaveCurrentRunStatus;
    readonly error: string | null;
    readonly disabledReason: string | null;
    readonly pending: boolean;
    readonly commands: {
        save(): Promise<void>;
        retryPending(): Promise<void>;
        discardPending(): Promise<void>;
        dismissError(): void;
        clearActionError(): void;
    };
}

export function useSaveCurrentRun(): SaveCurrentRunController;
~~~

- TrainingControls receives one SaveCurrentRunController from App.
- RunHistoryPanel receives the same instance from App. The legacy MainArea wrapper creates one controller once and passes it to both legacy children.

- [ ] **Step 1: Write the save-controller state-machine tests**

Assert blocked states for hydration not ready, no prepared V2 document, and pending artifact. Assert concurrent save calls capture once. Assert capture uses one UUID and identical createdAt/updatedAt values, saveRecord receives the exact worker artifact, a capture error is exposed, a persistence failure keeps the exact pending artifact, retry calls retryPersistence without recapture, discard and dismiss delegate once, and successful save returns ready.

~~~tsx
await act(async () => {
    await Promise.all([
        result.current.commands.save(),
        result.current.commands.save(),
    ]);
});
expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
expect(saveRecord).toHaveBeenCalledWith(capturedRecord);
~~~

- [ ] **Step 2: Run the controller test and verify extraction is absent**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/useSaveCurrentRun.test.tsx --pool=forks --reporter=dot
~~~

Expected: FAIL because useSaveCurrentRun.ts does not exist.

- [ ] **Step 3: Extract the exact save and retry behavior**

Move capture state and action error out of RunHistoryPanel. Use a synchronous ref guard in addition to saving state so two calls in one tick cannot capture twice.

~~~ts
const savingRef = useRef(false);
const save = useCallback(async () => {
    if (savingRef.current || pendingSave !== null || disabledReason !== null) return;
    savingRef.current = true;
    setSaving(true);
    setActionError(null);
    try {
        const timestamp = new Date().toISOString();
        const artifact = await (await getWorkerApi()).captureRunArtifact({
            id: createUuid(),
            createdAt: timestamp,
            updatedAt: timestamp,
        });
        await saveRecord(artifact);
    } catch (error) {
        setActionError(error instanceof Error ? error.message : 'Saving the current run failed.');
    } finally {
        savingRef.current = false;
        setSaving(false);
    }
}, [disabledReason, pendingSave, saveRecord]);
~~~

Do not change saved-run bytes, validation, incompatible-envelope retention, rejected records, or apply-recipe behavior.

- [ ] **Step 4: Add compact transport tests**

Assert Start/Pause visible label width class remains the same across states, one-step, reset, Save current run, 1/5/10/25/50, checkpoint timeline, restore, guarantee text, step, and epoch remain present. Assert Save forwards once and disabled reason is exposed. Assert 320px order is primary row then speed/timeline row without duplicated actions.

- [ ] **Step 5: Implement compact controls with direct Tabler icons**

Use direct IconPlayerPlay, IconPlayerPause, IconPlayerTrackNext, IconRefresh, IconDeviceFloppy, and IconRestore imports. Keep icon-only decoration aria-hidden. Use visible Save run with aria-label Save current run. Give the primary button a fixed inline-size large enough for Start/Pause/Resume, and reserve fixed-width numeric metadata cells.

~~~tsx
<button
    type="button"
    className="precision-transport__save"
    aria-label="Save current run"
    disabled={saveController.status !== 'ready'}
    onClick={() => void saveController.commands.save()}
>
    <IconDeviceFloppy aria-hidden="true" />
    <span>Save run</span>
</button>
~~~

Move the persistence error/retry/discard UI in RunHistoryPanel to read the same controller. Keep record listing, comparison, removal, incompatible bytes, legacy bytes, and apply recipe untouched.

- [ ] **Step 6: Pass one controller through App and legacy MainArea**

Create saveController once in CompatiblePlayground. Pass it to TrainingControls and HistoryContent/RunHistoryPanel. In legacy MainArea, create one controller at its root and pass it to both legacy locations. Do not instantiate it inside both children.

- [ ] **Step 7: Add intentional transport wrapping styles**

Wide uses one row. 680–1179px uses a second metadata row with fixed grid areas. Below 680px uses two rows, 44px controls, and no horizontal scroller. Status label changes must not affect button or neighboring positions.

- [ ] **Step 8: Run save, transport, history, and App tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/controls/useSaveCurrentRun.test.tsx src/components/controls/TrainingControls.test.tsx src/components/controls/RunHistoryPanel.test.tsx src/App.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS; exact pending-artifact retry assertions remain green.

- [ ] **Step 9: Commit the compact shared transport**

~~~bash
git add apps/web/src/components/controls/useSaveCurrentRun.ts apps/web/src/components/controls/useSaveCurrentRun.test.tsx apps/web/src/components/controls/TrainingControls.tsx apps/web/src/components/controls/TrainingControls.test.tsx apps/web/src/components/controls/RunHistoryPanel.tsx apps/web/src/components/controls/RunHistoryPanel.test.tsx apps/web/src/components/layout/MainArea.tsx apps/web/src/App.tsx apps/web/src/App.test.tsx apps/web/src/styles/precisionLab.css
git commit -m "refactor: share save capture in compact transport"
~~~

### Task 13: Automate responsive, accessibility, and anti-stutter acceptance

**Files:**
- Modify: apps/web/src/styles/precisionLabResponsive.test.ts
- Modify: apps/web/src/styles/forgeResponsive.test.ts
- Modify: apps/web/src/App.test.tsx
- Modify: apps/web/src/__tests__/appShell.integration.test.tsx
- Create: tests/e2e/precision-lab-layout.spec.ts
- Modify: tests/e2e/playground-smoke.spec.ts

**Interfaces:**
- Consumes: data-precision-workspace and data-precision-region hooks from Task 4.
- Produces reusable Playwright helpers:

~~~ts
type RegionName = 'header' | 'recipe' | 'topology' | 'boundary' | 'evidence' | 'transport';
type Rect = { x: number; y: number; width: number; height: number };

async function readRegionRects(page: Page): Promise<Record<RegionName, Rect>>;
async function readOverflow(page: Page): Promise<{
    documentDelta: number;
    workspaceDelta: number;
}>;
async function sampleStableRegions(
    page: Page,
    baseline: Record<RegionName, Rect>,
    durationMs: 5_000,
): Promise<void>;
~~~

- [ ] **Step 1: Extend static CSS and jsdom accessibility tests**

Assert every new region has min-width: 0, evidence/Confusion/Loss block horizontal overflow, phone targets are 44px, context sheet is full-width on phone, no CSS hides scientific content solely for containment, and reduced motion disables nonessential graph/shell animation. Run axe for Beginner/Explore/Lab in Build and Run and assert zero violations.

- [ ] **Step 2: Write the Playwright layout test before relying on manual QA**

For each viewport 1437x742, 735x860, and 320x844:

1. navigate to the deterministic three-class recipe;
2. wait for data-precision-workspace and a fresh boundary;
3. assert one boundary canvas;
4. select 50 steps per frame;
5. record all six region rectangles and both overflow deltas;
6. start training;
7. reset post-start CLS accounting;
8. sample every 250ms for five seconds;
9. assert every x/y/width/height delta is <= 1;
10. assert document/workspace overflow deltas are <= 1 before, during, and after;
11. on Chromium, assert CLS excluding hadRecentInput is exactly 0;
12. pause training and assert the layout remains stable.

Install the observer before application code:

~~~ts
await page.addInitScript(() => {
    type LayoutShiftEntry = PerformanceEntry & {
        readonly value: number;
        readonly hadRecentInput: boolean;
    };
    const state = { cls: 0 };
    if (PerformanceObserver.supportedEntryTypes.includes('layout-shift')) {
        new PerformanceObserver((list) => {
            for (const entry of list.getEntries() as LayoutShiftEntry[]) {
                if (!entry.hadRecentInput) state.cls += entry.value;
            }
        }).observe({ type: 'layout-shift', buffered: true });
    }
    Object.defineProperty(window, '__precisionLayout', {
        value: {
            reset: () => { state.cls = 0; },
            read: () => state.cls,
        },
    });
});
~~~

Use testInfo.project.name to apply the CLS assertion only in Chromium. Do not add retries or increase the 45-second timeout.

- [ ] **Step 3: Add phone, keyboard, reduced-motion, and 200% zoom scenarios**

At 320x844 assert topology precedes boundary, all visible interactive rectangles are at least 44x44 unless they are inline text links, the utilities menu is keyboard-operable, Build context becomes a sheet, evidence tabs remain reachable, and no horizontal overflow occurs. Emulate reducedMotion: reduce and assert graph flow animation is disabled. Apply documentElement.style.zoom = '2' at a 735px viewport and assert the same stacking and overflow contract.

- [ ] **Step 4: Extend semantic smoke without duplicating layout checks**

Keep existing training/pause/step, preset, checkpoint restore, saved-run reload/reapply, audience/disclosure, concept-help, and mobile focus scenarios. Add:

- selecting a dataset preview changes the recipe and rebuilds data;
- selecting a neuron persists through training and clears after reset generation;
- cycling Boundary/Loss/Confusion/Inspect/Code never creates a second boundary;
- Boundary controls update the pinned canvas;
- Save current run from transport increments saved-run count exactly once.

Record URL/hash, model generation/revision/step, checkpoint labels, and saved-run count before cycling modes/disclosures and assert the specified invariants afterward.

- [ ] **Step 5: Run focused unit/integration tests**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/styles/precisionLabResponsive.test.ts src/styles/forgeResponsive.test.ts src/App.test.tsx src/__tests__/appShell.integration.test.tsx --pool=forks --reporter=dot
~~~

Expected: PASS with zero axe violations.

- [ ] **Step 6: Run Chromium and WebKit acceptance**

Because Product Design work uses the in-app browser for visual QA, obtain the user's confirmation before invoking the Playwright CLI even though this approved plan requires the automated gate.

Run after confirmation:

~~~bash
pnpm test:e2e
~~~

Expected: Chromium and WebKit both PASS with zero retries and no console/page errors; all three five-second samples meet the 1px/CLS/overflow thresholds.

- [ ] **Step 7: Run checkpoint gates**

Run:

~~~bash
pnpm lint
pnpm build
pnpm test:bundle
git diff --check
~~~

Expected: every command exits 0 and all fixed bundle caps pass.

- [ ] **Step 8: Commit automated acceptance coverage**

~~~bash
git add apps/web/src/styles/precisionLabResponsive.test.ts apps/web/src/styles/forgeResponsive.test.ts apps/web/src/App.test.tsx apps/web/src/__tests__/appShell.integration.test.tsx tests/e2e/precision-lab-layout.spec.ts tests/e2e/playground-smoke.spec.ts
git commit -m "test: gate Precision Lab layout stability"
~~~

### Task 14: Prove release readiness and remove only the superseded shell

**Files:**
- Delete only after searches pass: apps/web/src/components/layout/BuildRunShell.tsx
- Delete only after searches pass: apps/web/src/components/layout/BuildRunShell.test.tsx
- Modify only if required by proven imports: apps/web/src/components/layout/MainArea.tsx
- Modify only if required by proven imports: apps/web/src/components/layout/MainArea.test.tsx

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: one production Precision Lab shell, one boundary canvas, preserved legacy scientific/compatibility paths that still have consumers, and a clean verification record in the task handoff.

- [ ] **Step 1: Search production consumers before deletion**

Run:

~~~bash
rg -n "BuildRunShell|<BuildRunShell" apps/web/src tests
rg -n "MainArea|BoundaryContent|DecisionBoundary" apps/web/src tests
rg -n "dataset/regenerateData|activeTabLeft|activeTabRight|history" apps/web/src packages tests
~~~

Expected: BuildRunShell has no production import; remaining hits are its own file/test only. MainArea, aliases, dataset/regenerateData, and legacy render paths are retained wherever a production or compatibility consumer remains.

- [ ] **Step 2: Delete only BuildRunShell and its superseded test**

Use apply_patch deletion for the two proven-unused files. Do not remove MainArea, DecisionBoundary wrapper/helper exports, compatibility aliases, dataset/regenerateData, or another legacy path in this slice unless Step 1 and focused tests prove zero consumers.

- [ ] **Step 3: Run the complete focused web regression set**

Run:

~~~bash
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/components/layout/Header.test.tsx src/components/layout/deriveVisualizationDemand.test.ts src/components/layout/precisionLab/PrecisionLabShell.test.tsx src/components/layout/precisionLab/precisionLabRecipeModel.test.ts src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx src/store/useLayoutStore.test.ts src/components/controls/DataPanel.test.tsx src/components/controls/useSaveCurrentRun.test.tsx src/components/controls/TrainingControls.test.tsx src/components/controls/RunHistoryPanel.test.tsx src/components/visualization/networkSelectionModel.test.ts src/components/visualization/useNetworkSelectionController.test.tsx src/components/visualization/NetworkSelectionDeck.test.tsx src/components/visualization/networkGraphPainter.test.ts src/components/visualization/NetworkGraphCanvas.test.tsx src/components/visualization/NetworkGraphSVG.test.tsx src/components/visualization/NetworkGraph.test.tsx src/components/visualization/decisionBoundaryModel.test.ts src/components/visualization/useDecisionBoundaryModel.test.tsx src/components/visualization/useDecisionBoundaryController.test.tsx src/components/visualization/DecisionBoundaryCanvas.test.tsx src/components/visualization/PinnedBoundaryRail.test.tsx src/components/visualization/BoundaryEvidencePanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx src/components/visualization/LossChart.test.tsx src/components/visualization/ConfusionMatrix.test.tsx src/styles/precisionLabResponsive.test.ts src/App.test.tsx src/__tests__/appShell.integration.test.tsx src/performance/interactionMeasures.test.ts --pool=forks --reporter=dot
~~~

Expected: PASS. The interaction-measure suite confirms browser/tooling measures remain filtered from development slow-interaction warnings.

- [ ] **Step 4: Run all repository and performance gates**

Run on an idle host:

~~~bash
pnpm lint
pnpm test
pnpm build
pnpm test:bundle
pnpm test:perf
pnpm test:perf
pnpm test:perf
git diff --check
~~~

Expected: all commands exit 0; median forced evaluation <= 250ms; median saved-run capture <= 500ms; entry <= 152,245 gzip bytes; InspectionPanel <= 7,373; total JavaScript <= 234,161.

- [ ] **Step 5: Run the final browser gate after the required confirmation**

Run:

~~~bash
pnpm test:e2e
~~~

Expected: zero-retry Chromium and WebKit suites PASS without console/page errors, including all existing smoke workflows and new layout acceptance.

- [ ] **Step 6: Perform in-app-browser visual comparison against the approved prototype**

Use the in-app browser only. At the identical three-class recipe and paused step:

1. capture production and approved prototype at 1437x742, 735x860, and 320x844;
2. compare each pair side by side;
3. verify Precision Lab hierarchy, proportional topology dominance, boundary adjacency, menu treatment, activation tiles, responsive evidence, and compact transport;
4. exercise Build/Run, every rail target, every evidence tab, Presets, Lessons, History, Advanced Tools, graph selection, boundary controls, and transport;
5. correct visible padding, clipping, font, border, radius, alignment, or responsive mismatches and rerun the focused tests for every touched file.

Acceptance: the production UI follows the approved prototype's design language while all content is live production data and every core control works.

- [ ] **Step 7: Re-run consumer and working-tree checks**

Run:

~~~bash
rg -n "BuildRunShell|<BuildRunShell" apps/web/src tests
git status --short
git diff --check
~~~

Expected: no BuildRunShell hit; only task files plus the user's preexisting unrelated changes appear; diff check exits 0.

- [ ] **Step 8: Commit removal of the superseded shell**

~~~bash
git add -u apps/web/src/components/layout/BuildRunShell.tsx apps/web/src/components/layout/BuildRunShell.test.tsx
git commit -m "refactor: remove superseded Build Run shell"
~~~

- [ ] **Step 9: Prepare the completion handoff**

Report:

- the exact commits created;
- focused/full/lint/build/bundle/performance/browser results;
- measured gzip values and performance medians;
- the three viewport stability results;
- the single-boundary canvas count;
- retained compatibility paths and why they remain;
- any hypothesis or separately logged defect not changed by this integration;
- a clickable local URL and the final side-by-side visual comparison.

Do not claim completion if any gate is skipped, blocked, weakened, or failing.
