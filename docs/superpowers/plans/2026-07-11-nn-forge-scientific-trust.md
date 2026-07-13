# NN.FORGE Scientific Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NN.FORGE's unversioned, internally inconsistent experiment runtime with one strict version-2 recipe whose dataset generation, objective math, metrics, persistence, and UI claims are scientifically trustworthy.

**Architecture:** `ExperimentDocumentV2` is the only public experiment configuration. Shared code strictly validates, compiles, and fingerprints it before allocation; the engine owns dataset and objective semantics; the worker owns model generations and evidence capture; the web store publishes atomic recipes and typed evidence. URL, JSON, presets, lessons, saved runs, and session checkpoints cross explicit versioned boundaries.

**Tech Stack:** TypeScript 5.7, pnpm workspaces, Vitest 3, React 19, Zustand 5, Web Workers, Web Crypto SHA-256, RFC 8785 canonical JSON, Playwright/in-app browser verification.

**Approved design:** `docs/superpowers/specs/2026-07-11-nn-forge-scientific-trust-design.md`

## Global Constraints

- Implement this as a clean break. Unversioned, version-1, malformed, and future URL/JSON/runtime payloads are rejected with structured errors; they never become defaults through repair or fallback.
- Keep schema version exactly `2`; keep identity constants and prefixes exactly as specified: `r2.1.`, `d2.1.`, and `o2.1.` using RFC 8785 canonicalization, UTF-8, SHA-256, and unpadded base64url.
- Preserve the approved bounds: samples `2..1_000`, train fraction `0.1..0.9`, noise `0..100`, features `1..9`, hidden layers `0..6`, width `1..16`, parameters `<=2_000`, and the remaining optimizer/schedule/objective bounds in the design.
- Seeds are unsigned 32-bit integers. Reject fractional, negative, non-finite, and greater-than-`2^32 - 1` values before allocation.
- Normal controls cannot create a task/output/target/objective contradiction. Output size, output activation, target encoding, and problem type are derived.
- Version 2 includes only standard experiments. Do not add a failure-demonstration type or a generic validation bypass.
- `prepareExperimentDocument()` is the only public runtime-entry transaction. No worker generation or model allocation begins before validation, compilation, and all identities succeed.
- Batch trend and paired full evaluation remain different types. Generalization and comparison claims use only a same-revision `PairedEvaluation`.
- Keep existing version-1 storage untouched. Version 2 uses exactly `nn-playground-experiment-memory-v2`.
- Saved runs contain recipes and evidence, not weights. The action is named **Apply saved recipe**. Session checkpoints remain in worker memory and state `parameters-and-optimizer-only`.
- Use test-driven development for every task: prove RED for the root cause, implement the smallest coherent slice, prove GREEN, review, then commit.
- Existing uncommitted user changes must be preserved and integrated. Re-read a file immediately before editing it; never replace a whole dirty file from a stale draft.
- Add no runtime dependency for canonical JSON or hashing.

## Locked Cross-Package Interfaces

Engine exports the only pure compiler adapter:

```ts
export interface CompilableExperimentRecipe {
    data: { sampleCount: number; trainFraction: number; noise: number; seed: number };
    inputs: { featureIds: readonly FeatureId[] };
    model: {
        hiddenLayers: readonly number[];
        hiddenActivation: ScalarActivationType;
        initialization: WeightInitType;
        seed: number;
    };
    training: {
        batchSize: number;
        learningRate: number;
        schedule: LearningRateScheduleV2;
        optimizer: OptimizerSpecV2;
        gradientClipping: GradientClipSpecV2;
    };
    task:
        | { kind: 'binary-classification'; dataset: BinaryDatasetId }
        | { kind: 'multiclass-classification'; dataset: 'three-class-clusters' }
        | { kind: 'regression'; dataset: RegressionDatasetId };
    objective: ObjectiveSpecV2;
}

export function compileExperimentRecipe(
    recipe: CompilableExperimentRecipe,
): CompiledExperimentConfig;
```

Shared owns the canonical preparation boundary:

```ts
export function prepareExperimentDocument(
    value: unknown,
): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
```

The worker receives a validated document plus claimed identities, recomputes the preparation transaction, and rejects a mismatch before allocating a dataset or network.

`@nn-playground/engine` must not import `@nn-playground/shared`. The engine declares the structural serializable task/objective input above; the shared V2 recipe is assignable to it after validation. This keeps dependencies one-way: shared may compile through engine, never the reverse.

## Dependency Map

1. Dataset contracts and deterministic generation, then strict schema/compiler.
2. Objective kernels, then network integration and diagnostics.
3. Canonical identities/codecs, then complete catalog and atomic web recipe state.
4. Metric provenance, then worker generation/evaluation integration.
5. Saved-run persistence and checkpoints after worker provenance is stable.
6. UI semantics and legacy removal after all typed contracts land.
7. Browser, performance, and repository verification after every vertical slice is green.

## Pre-execution Baseline and Isolation

- [ ] Create an isolated worktree/branch from a non-destructive snapshot of the current tracked worktree so the existing dirty fixes are present but the user's checkout remains untouched. Use an external worktree path; do not add a repository-local worktree directory.
- [ ] Confirm the isolated tree starts with no uncommitted changes and record the snapshot/base SHA in `.superpowers/sdd/progress.md`.
- [ ] Run `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm test:perf`, and `git diff --check` before Task 1. If a gate fails, stop implementation and use `superpowers:systematic-debugging` to distinguish a baseline failure from the Scientific Trust work.
- [ ] Run `pnpm test:perf` five times before Task 1 and record same-machine medians in `docs/superpowers/verification/2026-07-11-scientific-trust.md`. Task 12 compares post-change results to this baseline; it must not reconstruct a baseline after implementation.
- [ ] Read this plan in full and initialize `.superpowers/sdd/progress.md` before dispatching the first implementer.

---

### Task 1: Engine dataset contracts, seed rules, and bounded generation

**Files:**
- Create: `packages/engine/src/datasetContracts.ts`
- Create: `packages/engine/src/__tests__/datasetContracts.test.ts`
- Modify: `packages/engine/src/prng.ts`
- Modify: `packages/engine/src/__tests__/prng.test.ts`
- Modify: `packages/engine/src/datasets.ts`
- Modify: `packages/engine/src/__tests__/datasets.test.ts`
- Modify: `packages/engine/src/types.ts`
- Modify: `packages/engine/src/index.ts`

**Produces:** `TaskKind`, task-specific dataset ID unions, `DatasetContract`, `getDatasetContract()`, `normalizeUint32Seed()`, deterministic bounded Gaussian sampling, and generation that satisfies each contract.

- [ ] **Step 1: Write the contract and regression tests.** Cover all eleven registered datasets, generator version `2`, declared input/target domains, seed rejection, min/max noise, determinism, and the Checkerboard/Heart label-before-perturb rule.

```ts
it.each([42.9, -1, 4_294_967_296, Number.NaN])('rejects seed %s', (seed) => {
    expect(() => normalizeUint32Seed(seed)).toThrow(/unsigned 32-bit/u);
});

it.each(DATASET_IDS)('%s stays inside its declared contract at noise extremes', (id) => {
    const contract = getDatasetContract(id);
    for (const noise of [contract.noise.minimum, contract.noise.maximum]) {
        const samples = generateDataset({ dataset: id, sampleCount: 1_000, noise, seed: 42 });
        expect(samples).toSatisfyContract(contract);
    }
});
```

- [ ] **Step 2: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/engine exec vitest run src/__tests__/datasetContracts.test.ts src/__tests__/datasets.test.ts src/__tests__/prng.test.ts --pool=forks --reporter=dot
```

Expected: missing registry/seed helper failures plus out-of-domain or wrong-noise-semantics reproductions.

- [ ] **Step 3: Add the engine-owned registry.** Use the exact `DatasetContract` union from the approved design. Reject invalid settings before sample allocation. `reg-plane` bounds are `[-2 - 0.06 * noise, 2 + 0.06 * noise]`; `reg-gauss` bounds are `[-0.06 * noise, 2 + 0.06 * noise]`.

- [ ] **Step 4: Implement deterministic bounded sampling.** Gaussian draws use deterministic rejection within `mean ± 3σ`. Coordinate generators also resample candidates outside the declared domain with at most `10_000` attempts, then return a structured dataset-generation failure. Never clamp a rejected coordinate onto an edge.

- [ ] **Step 5: Fix generator semantics.** Compute clean Checkerboard/Heart labels before coordinate noise. Regression noise perturbs targets only. Every final coordinate/label is finite and contract-valid.

- [ ] **Step 6: Prove GREEN and determinism.** Re-run the focused command twice and confirm byte-for-byte equal samples for identical settings.

- [ ] **Step 7: Commit.**

```bash
git add packages/engine/src/datasetContracts.ts packages/engine/src/prng.ts packages/engine/src/datasets.ts packages/engine/src/types.ts packages/engine/src/index.ts packages/engine/src/__tests__/datasetContracts.test.ts packages/engine/src/__tests__/datasets.test.ts packages/engine/src/__tests__/prng.test.ts
git commit -m "feat(engine): define versioned dataset contracts"
```

---

### Task 2: Stable data-loss, penalty, and gradient-transform primitives

**Files:**
- Create: `packages/engine/src/objective.ts`
- Create: `packages/engine/src/__tests__/objective.test.ts`
- Modify: `packages/engine/src/losses.ts`
- Modify: `packages/engine/src/__tests__/losses.test.ts`
- Modify: `packages/engine/src/types.ts`
- Modify: `packages/engine/src/index.ts`

**Produces:** compiled objective types/functions for stable BCE/categorical CE/true MSE/Huber, L1/L2 penalty, total-objective breakdown, and complete-gradient clip transform.

- [ ] **Step 1: Write numerical invariants and finite-difference tests.** Include logits `-1000` and `1000`, Huber below/at/above delta, standard MSE, penalty values/gradients, excluded biases, and a zero-data-gradient/large-L2 clip case.

```ts
expect(binaryCrossEntropyWithLogits(1_000, 0)).toBeCloseTo(1_000, 12);
expect(binaryCrossEntropyLogitDelta(1_000, 0)).toBeCloseTo(1, 12);
expect(meanSquaredError([2], [0])).toBe(4);
expect(meanSquaredErrorDelta([2], [0])).toEqual([4]);
expect(l2Penalty([2], 0.5)).toBe(1);
```

- [ ] **Step 2: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/engine exec vitest run src/__tests__/objective.test.ts src/__tests__/losses.test.ts src/__tests__/gradient_check.test.ts --pool=forks --reporter=dot
```

Expected: missing objective module and old half-MSE/probability-clamped BCE mismatches.

- [ ] **Step 3: Implement exact formulas.** BCE value is `max(z, 0) - z*y + log1p(exp(-abs(z)))`; derivative is `sigmoid(z) - y`. True MSE is mean squared error with derivative `2 * error / outputCount`. Huber and categorical CE follow the approved formulas and validate public helper inputs.

- [ ] **Step 4: Implement penalty and clipping primitives.** L1 is `c * sum(abs(w))`; L2 is `0.5 * c * sum(w^2)`; neither touches biases. Resolve one global clip scale from the complete objective gradient and expose the pre/post norms without applying optimizer semantics.

- [ ] **Step 5: Prove GREEN.** Re-run the focused suite, including finite-difference checks at ordinary values and finite/correct extreme-logit checks.

- [ ] **Step 6: Commit.**

```bash
git add packages/engine/src/objective.ts packages/engine/src/losses.ts packages/engine/src/types.ts packages/engine/src/index.ts packages/engine/src/__tests__/objective.test.ts packages/engine/src/__tests__/losses.test.ts packages/engine/src/__tests__/gradient_check.test.ts
git commit -m "fix(engine): unify objective value and derivatives"
```

---

### Task 3: Route Network training, diagnostics, and session state through the compiled objective

**Files:**
- Create: `packages/engine/src/trainingContract.ts`
- Create: `packages/engine/src/sessionState.ts`
- Create: `packages/engine/src/__tests__/networkObjective.test.ts`
- Create: `packages/engine/src/__tests__/layerStatistics.test.ts`
- Create: `packages/engine/src/__tests__/sessionState.test.ts`
- Modify: `packages/engine/src/network.ts`
- Modify: `packages/engine/src/types.ts`
- Modify: `packages/engine/src/index.ts`
- Modify: `packages/engine/src/__tests__/network.test.ts`
- Modify: `packages/engine/src/__tests__/gradient_check.test.ts`

**Produces:** `CompiledExperimentConfig`, `CompiledTrainingContractV2`, `Network.trainBatchV2()`, explicit objective-aware evaluation/trace/backprop/landscape methods, deterministic activation statistics, and validated `NetworkSessionStateV2`.

- [ ] **Step 1: Write RED integration tests.** Reproduce complete-gradient clipping with zero data gradient and a weight of `100`, cross-optimizer diagnostic parity, total-objective landscape center, trace separation, bounded activation aggregation, and capture/restore rejection without mutation.

```ts
const result = net.trainBatchV2([[0]], [[0]], training);
expect(result.objective).toEqual({
    dataLoss: 0,
    regularizationPenalty: 5_000,
    totalObjective: 5_000,
});
expect(result.gradients.totalGradientNorm).toBe(100);
expect(result.gradients.clippedGradientNorm).toBeCloseTo(0.01, 12);
expect(net.getWeight(0, 0, 0)).toBeCloseTo(99.99, 12);
```

- [ ] **Step 2: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/engine exec vitest run src/__tests__/networkObjective.test.ts src/__tests__/layerStatistics.test.ts src/__tests__/sessionState.test.ts --pool=forks --reporter=dot
```

Expected: missing V2 network/objective/statistics/session APIs.

- [ ] **Step 3: Implement the authoritative update order.** Average batch data gradients; compute data norm; add weight-only penalty once; compute penalty and total norms; apply one scale to every weight/bias gradient; snapshot the exact clipped gradient; run the optimizer with no internal regularization; zero accumulators; advance step and revision.

```ts
const dataGradientNorm = gradientNorm(this.weightGrads, this.biasGrads);
const penaltyGradientNorm = objective.addPenaltyGradientInto(this.weights, this.weightGrads);
const diagnostics = applyGradientTransformInto(
    this.weightGrads,
    this.biasGrads,
    dataGradientNorm,
    penaltyGradientNorm,
    training.gradientClipping,
);
this.captureRecentGradient();
this.applyOptimizerOnly(training.optimizer, learningRate);
```

- [ ] **Step 4: Unify every evidence path.** `evaluateDataLoss()` reports predictive loss; `evaluateObjective()` adds one model penalty; traces expose `sampleDataLoss` and penalty separately; backprop returns the complete breakdown/diagnostics; landscapes default to `training-objective`; stop-condition projections name their exact basis.

- [ ] **Step 5: Replace last-forward layer statistics.** Aggregate every neuron activation for the first `min(128, trainCount)` deterministic training inputs. Return mean/population-standard-deviation plus model revision and the most recent clipped-gradient revision. An unrelated forward/grid call must not change the result.

- [ ] **Step 6: Add validate-before-mutate session state.** Validate layer shapes, finite arrays, optimizer kind/state shapes, non-negative optimizer step, deep cloning, and a `256 KiB` typed-array budget. Restoring a session state advances the live model revision once. Pre-update momentum/Adam capture materializes correctly shaped zero arrays.

- [ ] **Step 7: Prove GREEN.** Run:

```bash
pnpm --filter @nn-playground/engine exec vitest run src/__tests__/networkObjective.test.ts src/__tests__/layerStatistics.test.ts src/__tests__/sessionState.test.ts src/__tests__/gradient_check.test.ts src/__tests__/network.test.ts --pool=forks --reporter=dot
pnpm --filter @nn-playground/engine test
```

- [ ] **Step 8: Commit.**

```bash
git add packages/engine/src/trainingContract.ts packages/engine/src/sessionState.ts packages/engine/src/network.ts packages/engine/src/types.ts packages/engine/src/index.ts packages/engine/src/__tests__/networkObjective.test.ts packages/engine/src/__tests__/layerStatistics.test.ts packages/engine/src/__tests__/sessionState.test.ts packages/engine/src/__tests__/gradient_check.test.ts packages/engine/src/__tests__/network.test.ts
git commit -m "fix(engine): train on one compiled objective"
```

---

### Task 4: Strict version-2 schema, default document, and pure compiler transaction

**Files:**
- Create: `packages/shared/src/experimentSchema.ts`
- Create: `packages/shared/src/__tests__/fixtures/experimentV2.ts`
- Create: `packages/shared/src/__tests__/experimentSchema.test.ts`
- Modify: `packages/shared/src/constants.ts`
- Modify: `packages/shared/src/types.ts`
- Modify: `packages/shared/src/index.ts`

**Consumes:** Task 1 dataset contracts and Task 3's `compileExperimentRecipe()`.

**Produces:** discriminated `ExperimentDocumentV2`, branded validated types, bounded `SchemaResult<T>`, exact standard task recipes, internal `compileValidatedExperiment()`, and `DEFAULT_EXPERIMENT_DOCUMENT`.

- [ ] **Step 1: Add the exact V2 envelope and task unions from the approved design.** The valid binary fixture is complete—300 samples, `.5` split, zero noise, seeds `42`, `x/y`, `[4,4]` tanh/Xavier, batch 10, LR `.03`, constant/plain SGD, no clip/penalty, BCE-with-logits, mean-per-sample.

- [ ] **Step 2: Write RED validator/compiler tests.** Cover unchanged input branding, unknown-field collection, all task/dataset/loss mismatches, every numeric limit, seed aliases, feature order/duplicates, split populations, batch size, parameter count, and output derivation.

```ts
const invalid = structuredClone(VALID_BINARY_DOCUMENT) as any;
invalid.recipe.model.outputActivation = 'linear';
invalid.recipe.task = { kind: 'regression', dataset: 'circle' };
const result = validateExperimentDocument(invalid);
expect(result).toEqual({
    ok: false,
    issues: expect.arrayContaining([
        expect.objectContaining({ code: 'unknown-field', path: 'recipe.model.outputActivation' }),
        expect.objectContaining({ code: 'incompatible-task', path: 'recipe.task.dataset' }),
    ]),
});
```

- [ ] **Step 3: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/experimentSchema.test.ts
```

Expected: missing `experimentSchema` module.

- [ ] **Step 4: Implement a bounded strict collector.** Validate exact keys at every discriminated layer, cap at 100 issues, never repair or clamp, and return a deeply frozen exact plain-data snapshot branded only when there are no issues. Derive parameter count from `[features, ...hidden, output]`, and require positive train/test populations after `floor(sampleCount * trainFraction)`. **Implementation amendment (2026-07-11):** an independent review reproduced Proxy/getter input that validated one seed and hashed another. The stable snapshot is therefore mandatory; it preserves every submitted data value without normalization while preventing document/compiler/identity divergence.

- [ ] **Step 5: Compile only validated recipes.** Derive output size/activation/target contract/task metrics and the objective. `compileExperimentRecipe()` remains pure and performs no dataset/network allocation. Keep `compileValidatedExperiment()` internal to `prepareExperimentDocument()` and built-in validation so arbitrary callers cannot bypass the public preparation transaction. Validate the built-in default at module initialization and throw only for a programmer invariant.

- [ ] **Step 6: Prove GREEN and commit.**

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/experimentSchema.test.ts
git add packages/shared/src/experimentSchema.ts packages/shared/src/__tests__/fixtures/experimentV2.ts packages/shared/src/__tests__/experimentSchema.test.ts packages/shared/src/constants.ts packages/shared/src/types.ts packages/shared/src/index.ts
git commit -m "feat(shared): add strict v2 experiment schema"
```

---

### Task 5: RFC-8785 identities and clean-break URL/JSON codecs

**Files:**
- Create: `packages/shared/src/canonicalJson.ts`
- Create: `packages/shared/src/__tests__/experimentIdentity.test.ts`
- Rewrite: `packages/shared/src/serialization.ts`
- Rewrite: `packages/shared/src/__tests__/serialization.test.ts`
- Modify: `packages/shared/src/experimentSchema.ts`
- Modify: `packages/shared/src/index.ts`

**Produces:** `canonicalRecipeKey()`, recipe/dataset/objective SHA-256 fingerprints, `prepareExperimentDocument()`, exact V2 URL/JSON codecs, and checked-in digest fixtures.

- [ ] **Step 1: Write RED identity and codec tests.** Verify exact fixture strings, insertion-order invariance, change sensitivity, all fields round-trip through `#v=2&r=`, and distinct legacy/future/malformed issues.

```ts
const prepared = await prepareExperimentDocument(VALID_BINARY_DOCUMENT);
expect(prepared.ok).toBe(true);
if (prepared.ok) {
    expect(prepared.value.identities.recipeFingerprint).toMatch(/^r2\.1\.[A-Za-z0-9_-]{43}$/u);
    expect(prepared.value.identities.datasetKey).toMatch(/^d2\.1\.[A-Za-z0-9_-]{43}$/u);
    expect(prepared.value.identities.objectiveKey).toMatch(/^o2\.1\.[A-Za-z0-9_-]{43}$/u);
}
expect(decodeExperimentUrl('#d=circle&lr=0.03')).toMatchObject({
    ok: false,
    issues: [expect.objectContaining({ code: 'legacy-state' })],
});
```

- [ ] **Step 2: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/experimentIdentity.test.ts src/__tests__/serialization.test.ts
```

- [ ] **Step 3: Implement dependency-free RFC 8785 canonical JSON.** Strict validation makes accepted values I-JSON compatible. Test Unicode key ordering, exponent numbers, `-0`, and rejection of non-finite numbers, `undefined`, functions, and bigint.

- [ ] **Step 4: Implement exact payloads and parallel identities.** Recipe payload includes all five version constants, dataset generator version, and recipe. Dataset payload includes generator/split versions and all generation settings. Objective payload includes task/output/objective/reduction/implementation version. `prepareExperimentDocument()` validates and compiles once, then awaits the three digests in parallel.

- [ ] **Step 5: Replace lenient codecs.** Empty hash loads default V2. Any non-empty incompatible hash is an error. URL is `#v=2&r=<base64url canonical document>`. JSON import/export uses the exact envelope. Do not expose repair, normalization, or short-key helpers.

- [ ] **Step 6: Prove GREEN and commit.**

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/experimentIdentity.test.ts src/__tests__/serialization.test.ts
git add packages/shared/src/canonicalJson.ts packages/shared/src/experimentSchema.ts packages/shared/src/serialization.ts packages/shared/src/index.ts packages/shared/src/__tests__/experimentIdentity.test.ts packages/shared/src/__tests__/serialization.test.ts
git commit -m "feat(shared): add canonical v2 identities and codecs"
```

---

### Task 6: Complete recipe catalog, revision-pinned lessons, and atomic recipe store

**Files:**
- Rewrite: `packages/shared/src/presets.ts`
- Rewrite: `packages/shared/src/__tests__/presets.test.ts`
- Modify: `apps/web/src/lessons/types.ts`
- Modify: `apps/web/src/lessons/lessonRegistry.ts`
- Modify: `apps/web/src/lessons/lessonRegistry.test.ts`
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.tsx`
- Modify: `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`
- Create: `apps/web/src/store/recipeEdits.ts`
- Create: `apps/web/src/store/recipeEdits.test.ts`
- Modify: `apps/web/src/store/recipeIdentity.ts`
- Rewrite: `apps/web/src/store/usePlaygroundStore.ts`
- Rewrite: `apps/web/src/store/usePlaygroundStore.test.ts`
- Modify: `apps/web/src/components/controls/DataPanel.tsx`
- Modify: `apps/web/src/components/controls/FeaturesPanel.tsx`
- Modify: `apps/web/src/components/controls/NetworkConfigPanel.tsx`
- Modify: `apps/web/src/components/controls/HyperparamPanel.tsx`
- Modify: `apps/web/src/components/controls/PresetPanel.tsx`
- Modify: `apps/web/src/components/controls/PresetCard.tsx`

**Produces:** seven complete revision-1 recipes, lesson `{id, revision}` references, pure task-aware edits, and one canonical Zustand document/compiled projection transaction.

- [ ] **Step 1: Re-read dirty overlaps.** Inspect `git diff -- apps/web/src/components/controls/HyperparamPanel.tsx apps/web/src/components/controls/HyperparamPanel.test.tsx apps/web/src/store/usePlaygroundStore.ts` and preserve every unrelated prior fix.

- [ ] **Step 2: Write RED catalog/lesson/all-pairs tests.** Every catalog entry validates, every lesson reference resolves, Regression→XOR equals the exact XOR recipe, and all 49 preset transitions end at the destination canonical recipe key.

```ts
for (const source of PREPARED_PRESETS) for (const target of PREPARED_PRESETS) {
    await store.getState().applyRecipe(source);
    await store.getState().applyRecipe(target);
    expect(store.getState().canonicalRecipeKey, `${source.id} -> ${target.id}`)
        .toBe(target.prepared.identities.canonicalRecipeKey);
}
```

- [ ] **Step 3: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/presets.test.ts
pnpm --filter @nn-playground/web test -- src/lessons/lessonRegistry.test.ts src/components/controls/GuidedLessonPanel.test.tsx src/store/recipeEdits.test.ts src/store/usePlaygroundStore.test.ts
```

Expected: partial presets/lesson `presetId`/fragmented store fail exact-transition checks.

- [ ] **Step 4: Define and prepare the exact seven-entry catalog.** IDs are `single-neuron`, `xor-hidden`, `circle-one-layer`, `spiral-deep`, `regression-plane`, `feature-engineering`, and `three-class-clusters`. Each entry declares all data/model/training/objective fields and `revision: 1`; use the approved catalog matrix and preserve existing educational copy. Prepare/validate the built-ins once and export only entries paired with their branded prepared result; call sites do not cast recipes into brands.

- [ ] **Step 5: Pin lessons and apply complete recipes.** Replace `presetId` with `recipeRef: { id, revision }`. Starting a lesson calls one atomic `applyRecipe()` before reset; it never merges a patch.

- [ ] **Step 6: Replace store fragments with one prepared-document transaction.** Store the last successful `PreparedExperimentDocumentV2` plus preparation/error state. `replaceDocument` awaits `prepareExperimentDocument()` and publishes document, compiled projection, canonical key, and all identities in one `set()` only after success. Guard async races with a monotonically increasing preparation request ID so a slower stale digest cannot overwrite a newer edit. `editRecipe` constructs a candidate then delegates. `applyRecipe` replaces the complete recipe. Task switches install their compatible objective atomically; no action accepts output activation or size.

```ts
editRecipe: async (edit) => {
    const current = get().prepared.document;
    return get().replaceDocument({ ...current, recipe: edit(current.recipe) });
},
applyRecipe: async (entry) => {
    const current = get().prepared.document;
    const result = await get().replaceDocument({ ...current, recipe: entry.recipe });
    if (!result.ok) throw new Error(`Invalid built-in recipe ${entry.id}@${entry.revision}`);
},
```

- [ ] **Step 7: Migrate normal controls.** Controls call typed recipe edits and display derived output/objective values read-only. Regression offers true MSE/Huber; classification shows its derived logits objective. Preserve the approved bounds and disable an edit that would temporarily violate train-population/resource limits rather than publishing invalid state.

- [ ] **Step 8: Prove GREEN and scan fragments.** Run the commands from Step 3, then:

```bash
rg -n 'Partial<AppConfig>|applyPreset|allowMulticlass|setOutputActivation|normalizeLossOutputCompatibility' packages/shared/src apps/web/src
```

Expected: no production recipe/store path uses partial application or repair.

- [ ] **Step 9: Commit.** Stage only Task 6 hunks in previously dirty files.

```bash
git commit -m "feat(web): make v2 recipes atomic across controls"
```

---

### Task 7: Metric provenance types and protocol version 2

**Files:**
- Create: `packages/shared/src/metricProvenance.ts`
- Create: `packages/shared/src/__tests__/metricProvenance.test.ts`
- Modify: `packages/shared/src/workerProtocol.ts`
- Modify: `packages/shared/src/__tests__/workerProtocol.test.ts`
- Modify: `packages/shared/src/index.ts`
- Create: `apps/web/src/test/scientificTrustFixtures.ts`

**Produces:** `ModelRevision`, `DatasetRevision`, `PairedEvaluation`, `LiveTrainingSignal`, `ArtifactProvenance`, `EvaluationPolicy`, runtime assertions, and a V2 worker protocol whose initialization payload is the document plus claimed identities.

- [ ] **Step 1: Write RED type/runtime invariant tests.** A pair owns one model identity; both full-split sample counts equal positive populations; live signal cannot carry accuracy/confusion/gap/total objective; default evaluation cadence is exactly 50 steps with all forced-trigger flags true.

```ts
expect(DEFAULT_EVALUATION_POLICY).toEqual({
    everySteps: 50,
    forceOnPause: true,
    forceOnManualStep: true,
    forceOnCheckpoint: true,
    forceOnSave: true,
});
expectTypeOf<LiveTrainingSignal>().not.toHaveProperty('generalizationGap');
```

- [ ] **Step 2: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/metricProvenance.test.ts src/__tests__/workerProtocol.test.ts
```

- [ ] **Step 3: Implement exact provenance unions.** Copy the approved `ModelRevision`, `DatasetRevision`, split basis, evaluation values, `PairedEvaluation`, `LiveTrainingSignal`, artifact bases, and aliases `TrainingTrendPoint`/`EvaluationPoint`. Add bounded assertions at protocol/persistence boundaries.

- [ ] **Step 4: Version the protocol.** Replace loose config fragments and scalar loss fields with `WorkerExperimentRequestV2`, typed evidence, artifact-specific provenance, structured worker errors, and explicit commands for force evaluation/save/checkpoint. Initialization receives only structured-clone-safe document/identities and recomputes `prepareExperimentDocument()` in the worker before allocation.

- [ ] **Step 5: Prove forged identity rejection.** Add protocol fixtures that mutate recipe, dataset key, and objective key independently and assert rejection before network/dataset constructors are called.

- [ ] **Step 6: Prove GREEN and commit.**

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/metricProvenance.test.ts src/__tests__/workerProtocol.test.ts
git add packages/shared/src/metricProvenance.ts packages/shared/src/workerProtocol.ts packages/shared/src/index.ts packages/shared/src/__tests__/metricProvenance.test.ts packages/shared/src/__tests__/workerProtocol.test.ts apps/web/src/test/scientificTrustFixtures.ts
git commit -m "feat(shared): version metric provenance and worker protocol"
```

---

### Task 8: Worker generations, same-revision paired evaluations, and separated histories

**Files:**
- Create: `apps/web/src/worker/evaluationRuntime.ts`
- Create: `apps/web/src/worker/evaluationRuntime.test.ts`
- Create: `apps/web/src/worker/runtimeMetricHistory.ts`
- Create: `apps/web/src/worker/runtimeMetricHistory.test.ts`
- Create: `apps/web/src/store/metricHistoryBuffer.ts`
- Create: `apps/web/src/store/metricHistoryBuffer.test.ts`
- Modify: `apps/web/src/worker/training.worker.ts`
- Modify: `apps/web/src/worker/training.worker.test.ts`
- Modify: `apps/web/src/worker/stopConditions.ts`
- Modify: `apps/web/src/worker/stopConditions.test.ts`
- Modify: `apps/web/src/worker/frameBuffer.ts`
- Modify: `apps/web/src/worker/workerBridge.ts`
- Modify: `apps/web/src/worker/workerBridge.test.ts`
- Modify: `apps/web/src/store/useTrainingStore.ts`
- Modify: `apps/web/src/store/useTrainingStore.test.ts`
- Modify: `apps/web/src/hooks/useTraining.ts`
- Modify: `apps/web/src/hooks/useTraining.test.tsx`
- Delete after migration: `apps/web/src/store/historyBuffer.ts`

**Produces:** monotonic generations/revisions, one live EMA, forced/cadenced same-revision evaluations, worker-authoritative two-series history, and atomic client publication.

- [ ] **Step 1: Re-read dirty worker/hook changes.** Inspect their diffs and keep the existing lifecycle/frame-buffer fixes.

- [ ] **Step 2: Write RED cadence/trigger/de-duplication tests.** Train and test execute sequentially against one frozen revision; steps 1–49 publish only live trend; step 50 publishes one cadence pair; initial/manual-step/pause/checkpoint/save/stop-condition/restore force a current pair; repeated evaluation IDs append once.

```ts
for (let step = 1; step < 50; step++) {
    runtime.recordBatch(batchAt(step));
    expect(runtime.takeCadenceEvaluation()).toBeUndefined();
}
runtime.recordBatch(batchAt(50));
expect(runtime.takeCadenceEvaluation()?.trigger).toBe('cadence');
```

- [ ] **Step 3: Prove RED.** Run:

```bash
pnpm --filter @nn-playground/web test -- src/worker/evaluationRuntime.test.ts src/worker/runtimeMetricHistory.test.ts src/store/metricHistoryBuffer.test.ts src/worker/training.worker.test.ts src/worker/workerBridge.test.ts src/hooks/useTraining.test.tsx
```

- [ ] **Step 4: Implement the evaluation runtime.** Revision increments on every update/restore; generation increments on rebuild/reset; EMA carries latest batch size and through-step. A forced pair freezes one model identity, evaluates full train then full test without training between, computes one penalty/train objective, increments `evaluationId`, asserts invariants, then publishes atomically.

- [ ] **Step 5: Make stop conditions basis-explicit.** Comparison-sensitive conditions force a current pair and target one of `trainDataLoss`, `testDataLoss`, `trainObjective`, or task accuracy. Non-finite objective/evaluation becomes a terminal divergence error and is excluded from evidence/checkpoints.

- [ ] **Step 6: Separate worker and UI history.** Append every live trend; append a pair only when its `evaluationId` is new. Frame buffer transports artifact arrays and their own provenance only. `useTrainingStore` exposes `latestLiveSignal`, `latestEvaluation`, and separate version counters; never reconstitute a synthetic `HistoryPoint`.

- [ ] **Step 7: Wire bounded activation statistics.** Worker passes the first `min(128, trainCount)` training inputs to the engine aggregator and attaches `bounded-sample` provenance. Unrelated boundary prediction must not alter the result.

- [ ] **Step 8: Prove GREEN and commit.**

```bash
pnpm --filter @nn-playground/web test -- src/worker/evaluationRuntime.test.ts src/worker/runtimeMetricHistory.test.ts src/store/metricHistoryBuffer.test.ts src/worker/training.worker.test.ts src/worker/stopConditions.test.ts src/worker/workerBridge.test.ts src/store/useTrainingStore.test.ts src/hooks/useTraining.test.tsx
git commit -m "feat(worker): publish paired evaluations with provenance"
```

---

### Task 9: Strict version-2 run records and atomic worker-owned save capture

**Files:**
- Rewrite: `packages/shared/src/experimentMemory.ts`
- Rewrite: `packages/shared/src/__tests__/experimentMemory.test.ts`
- Modify: `packages/shared/src/workerProtocol.ts`
- Modify: `packages/shared/src/index.ts`
- Modify: `apps/web/src/worker/training.worker.ts`
- Modify: `apps/web/src/worker/training.worker.test.ts`
- Rewrite: `apps/web/src/store/experimentMemoryStore.ts`
- Rewrite: `apps/web/src/store/experimentMemoryStore.test.ts`
- Modify: `apps/web/src/components/controls/RunHistoryPanel.tsx`
- Modify: `apps/web/src/components/controls/RunHistoryPanel.test.tsx`
- Delete after migration: `apps/web/src/store/experimentRunCapture.ts`
- Delete after migration: `apps/web/src/store/experimentRunCapture.test.ts`

**Produces:** strict `ExperimentRunRecordV2`, deterministic bounded histories, rejected-record isolation, worker-owned `captureRunArtifact()`, legacy-byte preservation, and comparison gating.

- [ ] **Step 1: Write RED contract tests.** Cover a canonical 36-ASCII-character UUID, title length at most 120 Unicode code points, exact `512`/`256` history caps, deterministic endpoint-preserving compaction, duplicate evaluation IDs, `512 KiB` record/`4 MiB` envelope budgets, 20-record limit, identity consistency, valid sibling + rejected raw record reading, and no silent eviction.

```ts
expect(compactEvenly(Array.from({ length: 1025 }, (_, i) => i), 512)).toEqual(
    Array.from({ length: 512 }, (_, i) => Math.round(i * 1024 / 511)),
);
```

- [ ] **Step 2: Prove shared RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/experimentMemory.test.ts
```

- [ ] **Step 3: Implement validate-first persistence.** Use exact key `nn-playground-experiment-memory-v2`; preserve V1 bytes opaquely. Validate UTF-8 bytes, code points, record/evaluation relationships, dataset/objective consistency, generation consistency, and recomputed recipe fingerprint. Return valid records plus structured rejected records/raw JSON. Candidate writes must validate and serialize fully before changing storage/state.

- [ ] **Step 4: Write RED worker/store/UI tests.** Save forces a current `trigger: 'save'` pair; record model equals evaluation model; bounded histories come from one worker generation; record contains no parameters; quota/count errors persist; legacy storage remains untouched; incompatible runs say “Not directly comparable” and have no numeric winner.

- [ ] **Step 5: Prove web RED.** Run:

```bash
pnpm --filter @nn-playground/web test -- src/worker/training.worker.test.ts src/store/experimentMemoryStore.test.ts src/components/controls/RunHistoryPanel.test.tsx
```

- [ ] **Step 6: Implement one queued capture transaction.** Worker mutation RPCs serialize. Capture forces the pair, reads the same worker history, compacts, constructs the record, validates it, and returns it. The UI provides only ID/timestamp/title and persists the returned artifact; it never combines independent store/frame-buffer values.

```ts
async captureRunArtifact(request: CaptureRunArtifactRequestV2) {
    return mutationQueue.run(async () => {
        const evaluation = forcePairedEvaluation('save');
        const history = metricHistory.read();
        return validateOrThrow(buildRunRecord(prepared, evaluation, history, request));
    });
}
```

- [ ] **Step 7: Replace memory UI semantics.** Action is **Apply saved recipe** and starts a fresh model. Legacy notice offers download/dismiss/explicit delete. Rejected records offer raw download/delete. Storage errors remain until successful retry/dismissal. A comparison requires equal dataset and objective keys.

- [ ] **Step 8: Prove GREEN and scan old capture.** Run both focused commands, then:

```bash
rg -n 'captureExperimentRun|createSerializedNetworkFromFrameBuffer|ExperimentRunRecordV1|Restore config|Saved parameters' apps/web/src packages/shared/src
```

Expected: no production matches.

- [ ] **Step 9: Commit.**

```bash
git commit -m "feat(web): capture and persist v2 run evidence atomically"
```

---

### Task 10: Complete version-2 session checkpoints

**Files:**
- Create: `packages/shared/src/sessionCheckpoint.ts`
- Create: `packages/shared/src/__tests__/sessionCheckpoint.test.ts`
- Modify: `packages/shared/src/workerProtocol.ts`
- Modify: `packages/shared/src/index.ts`
- Modify: `apps/web/src/worker/training.worker.ts`
- Modify: `apps/web/src/worker/training.worker.test.ts`
- Modify: `apps/web/src/hooks/useTraining.ts`
- Modify: `apps/web/src/hooks/useTraining.test.tsx`
- Modify: `apps/web/src/components/controls/TrainingControls.tsx`
- Modify: `apps/web/src/components/controls/TrainingControls.test.tsx`

**Consumes:** Task 3 `NetworkSessionStateV2` and Task 8 evaluation runtime.

**Produces:** complete in-memory `SessionCheckpointV2`, strict compatibility validation, clone/capture/restore transactions, and honest UI copy.

- [ ] **Step 1: Write RED shared validation tests.** Reject recipe/objective/dataset mismatch, model/evaluation mismatch, wrong layer shapes, wrong optimizer kind/state shapes, non-finite arrays, invalid cursor/permutation, aliases, and payloads over `256 KiB`. Prove a full valid clone round-trip.

- [ ] **Step 2: Prove shared RED.** Run:

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/sessionCheckpoint.test.ts
```

- [ ] **Step 3: Implement the exact envelope.** Include kind/schema, three identities, model, forced evaluation, network layers, discriminated optimizer state, exact cursor, and `trajectoryGuarantee: 'parameters-and-optimizer-only'`. Count every typed-array byte and validate the shuffled indices as an exact permutation of `0..trainCount-1`.

- [ ] **Step 4: Write RED worker transaction tests.** Capture forces a `checkpoint` pair. A malformed restore leaves live model/cursor byte-identical. Valid restore sets captured parameters/optimizer/cursor, increments live revision, invalidates derived evidence, and publishes a fresh `restore` pair.

- [ ] **Step 5: Prove worker RED.** Run:

```bash
pnpm --filter @nn-playground/web test -- src/worker/training.worker.test.ts src/hooks/useTraining.test.tsx src/components/controls/TrainingControls.test.tsx
```

- [ ] **Step 6: Implement validate-before-mutate restore.** Locate/clone in-memory checkpoint, compare active prepared identities, validate the whole envelope, translate to engine state, restore network/cursor, reset caches, then force/publish restore evaluation. Any failure retains the previous state.

- [ ] **Step 7: Correct UI copy.** Say “Restore in-session parameters and optimizer state” and “Future shuffles may differ; this checkpoint guarantees parameters and optimizer state only.” Never claim exact trajectory resume.

- [ ] **Step 8: Prove GREEN and commit.**

```bash
pnpm --filter @nn-playground/shared test -- src/__tests__/sessionCheckpoint.test.ts
pnpm --filter @nn-playground/web test -- src/worker/training.worker.test.ts src/hooks/useTraining.test.tsx src/components/controls/TrainingControls.test.tsx
git commit -m "feat(worker): validate versioned session checkpoints"
```

---

### Task 11: Provenance-aware UI, explicit compatibility states, and legacy cleanup

**Files:**
- Create: `apps/web/src/store/evidenceSelectors.ts`
- Create: `apps/web/src/components/common/CompatibilityState.tsx`
- Create: `apps/web/src/components/common/CompatibilityState.test.tsx`
- Rewrite: `apps/web/src/components/controls/ConfigPanel.tsx`
- Rewrite: `apps/web/src/components/controls/ConfigPanel.test.tsx`
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/App.test.tsx`
- Modify: `apps/web/src/components/layout/Header.tsx`
- Modify: `apps/web/src/components/layout/ExperimentStateContext.tsx`
- Modify: `apps/web/src/components/controls/CurrentRunCard.tsx`
- Modify: `apps/web/src/components/controls/RecipeSummaryCard.tsx`
- Modify: `apps/web/src/components/controls/InspectionPanel.tsx`
- Modify: `apps/web/src/components/visualization/LossChart.tsx`
- Modify: `apps/web/src/components/visualization/ConfusionMatrix.tsx`
- Modify: `apps/web/src/components/visualization/TrainingExplanationPanel.tsx`
- Modify: `apps/web/src/explanations/trainingExplanations.ts`
- Modify matching tests for every component above.

**Produces:** accurate terminology/basis/sample counts, paired-only generalization, persistent structured incompatibility UI, current-state Copy URL, and no remaining V1/repair runtime paths.

- [ ] **Step 1: Re-read all dirty component diffs.** Preserve existing Tooltip/Header/Inspection/LossChart/ConfusionMatrix behavior and tests; stage only this task's hunks.

- [ ] **Step 2: Write RED copy/provenance/error tests.** A batch trend at step 1240 and evaluation at 1230 display separately; loss chart does not move test/gap from the live point; confusion captions evaluation ID/step/all test samples; activation statistics label N of M; legacy URL preserves hash and blocks worker initialization; import errors persist; Copy URL serializes current state even if `location.hash` is stale.

- [ ] **Step 3: Prove RED.** Run the focused component group:

```bash
pnpm --filter @nn-playground/web test -- src/components/common/CompatibilityState.test.tsx src/components/controls/ConfigPanel.test.tsx src/App.test.tsx src/components/layout/Header.test.tsx src/components/layout/ExperimentStateContext.test.tsx src/components/controls/CurrentRunCard.test.tsx src/components/controls/RecipeSummaryCard.test.tsx src/components/visualization/LossChart.test.tsx src/components/visualization/ConfusionMatrix.test.tsx src/components/controls/InspectionPanel.test.tsx src/components/visualization/TrainingExplanationPanel.test.tsx src/explanations/trainingExplanations.test.ts
```

- [ ] **Step 4: Add one pure evidence selector.** Expose batch trend/step, paired train/test data loss, training objective, penalty, evaluation step/age/counts, and paired-only gap. Components consume typed evidence, not loose snapshot scalar fields.

- [ ] **Step 5: Correct visual claims.** Labels are “Batch trend (EMA),” “Train data loss (full split),” “Test data loss (full split),” and “Training objective.” Confusion is bound to `latestEvaluation`. Trace uses sample data loss plus separate model penalty; backprop includes full objective/gradient breakdown; landscape states its sample/parameter-grid basis; explanations require a pair for generalization.

- [ ] **Step 6: Implement source-preserving compatibility UI.** Bad non-empty hash/document sets `document: null` plus exact original/structured issues. App does not mount training. Only **Start fresh** installs default V2 and changes the hash. Config errors do not auto-dismiss; enforce 4 MiB before parsing; handle `FileReader.onerror`; Copy URL encodes the current validated document synchronously.

- [ ] **Step 7: Remove obsolete public paths.** Delete unversioned codecs, `AppConfig`, partial preset/public config helpers, `allowMulticlass`, generic `loss`, V1 run types, stale-metric Boolean, old history point, output/loss repair, and loose worker protocol fields once `rg` proves no consumer remains.

- [ ] **Step 8: Prove GREEN and run forbidden scans.** Re-run Step 3, then:

```bash
rg -n 'AppConfig|normalizeAppConfig|allowMulticlass|testMetricsStale|snapshot\.(trainLoss|testLoss)|ExperimentRunRecordV1|Restore config|Saved parameters|trainEvalInterval|testEvalInterval' packages/shared/src apps/web/src
```

Expected: no production matches; explicitly named legacy rejection fixtures are allowed only in tests.

- [ ] **Step 9: Commit.**

```bash
git commit -m "fix(web): label evidence and surface v2 incompatibilities"
```

---

### Task 12: Performance budgets, browser scenarios, and final verification

**Files:**
- Create: `apps/web/src/worker/scientificTrust.performance.test.ts`
- Create: `apps/web/vitest.perf.config.ts`
- Modify: `apps/web/package.json`
- Modify: root `package.json`
- Create: `docs/superpowers/verification/2026-07-11-scientific-trust.md`
- Modify only for confirmed failures: focused source/test files from Tasks 1–11

- [ ] **Step 1: Verify the recorded same-machine baseline.** Confirm the pre-execution section captured five complete engine performance runs before Task 1; do not substitute post-change measurements for a missing baseline.

- [ ] **Step 2: Write threshold-enforced worker performance tests.** At the maximum valid 1,000-sample/six-by-16 recipe, median of 20 warmed forced pairs is `<=250 ms`; median of 20 warmed save captures is `<=500 ms`. Diagnostics and provenance remain enabled.

- [ ] **Step 3: Add web/root performance scripts and prove thresholds.** Root `test:perf` runs both existing engine performance coverage and the new web thresholds. Run five complete post-change passes; no existing benchmark median may exceed 120% of baseline.

- [ ] **Step 4: Run focused real-browser scenarios using the browser control skill.** Verify and record screenshots/evidence for:

1. Empty/default V2 URL and identical recipe after reload.
2. Regression→XOR and all destination fingerprints.
3. All public preset-to-preset transitions.
4. Step 51 showing live trend 51 and full evaluation 50.
5. Pause and manual-step forced pairs.
6. Three-class matrix/copy.
7. Regression objective terminology.
8. V1 URL, unversioned JSON, and future JSON incompatibilities.
9. Save pair and **Apply saved recipe**.
10. Untouched legacy localStorage across reload/dismiss.
11. Incompatible saved records without a numeric winner.
12. Session checkpoint restore and limited-guarantee copy.

- [ ] **Step 5: Run the final repository gates from root, one at a time.**

```bash
pnpm test
pnpm lint
pnpm build
pnpm test:perf
git diff --check
```

Expected: all exit `0`; total test count exceeds the pre-wave baseline; any existing build chunk warning is recorded, not hidden.

- [ ] **Step 6: Write the verification record.** Include each slice SHA, exact commands/exit codes/test counts/timestamps, baseline and post-change medians/percentage changes, the 12 browser scenarios and screenshot paths, untouched legacy-key confirmation, and any untested real-hardware limitations.

- [ ] **Step 7: Run broad code review.** Use `superpowers:requesting-code-review` over the complete implementation range. Resolve every critical/high issue and re-run affected focused/full gates.

- [ ] **Step 8: Commit verification artifacts.**

```bash
git add apps/web/src/worker/scientificTrust.performance.test.ts apps/web/vitest.perf.config.ts apps/web/package.json package.json docs/superpowers/verification/2026-07-11-scientific-trust.md
git commit -m "test: enforce scientific trust runtime gates"
```

## Completion Checklist

- [ ] One strict public V2 schema and one preparation/compiler boundary exist.
- [ ] Dataset/output/objective/seed/feature/resource contradictions fail before allocation.
- [ ] Every preset and lesson reaches one complete deterministic fingerprint from any prior state.
- [ ] BCE, categorical CE, true MSE, Huber, L1/L2, total objective, and gradients agree numerically.
- [ ] Global clipping covers the complete objective gradient.
- [ ] Training, evaluation, trace, preview, stop conditions, and landscape share one compiled objective or an explicitly named projection.
- [ ] Live trend and full evaluation cannot be accidentally combined.
- [ ] Generalization uses full train/test data loss at one model/dataset revision.
- [ ] Every streamed/persisted measurement states model, data, objective, sample basis, and step.
- [ ] V2 URL/JSON/run/checkpoint round trips are exact and incompatible versions are explicit.
- [ ] Legacy/bad inputs remain preserved until explicit user action.
- [ ] Focused, full, lint, build, browser, and performance gates all pass.

## Execution Notes

- Execute Tasks 1–12 serially through `superpowers:subagent-driven-development`.
- For every task: dispatch a fresh implementer, run a task-spec review, resolve findings, then update `.superpowers/sdd/progress.md` before moving on.
- Never run two implementation agents concurrently because all agents share the same filesystem.
- Use `superpowers:verification-before-completion` before any completion claim and `superpowers:finishing-a-development-branch` for the final integration choice.
