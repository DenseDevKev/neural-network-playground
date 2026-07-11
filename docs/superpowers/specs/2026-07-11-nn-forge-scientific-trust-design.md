# NN.FORGE Scientific Trust Design

Date: 2026-07-11

Status: Approved design; implementation not started

## Goal

Make NN.FORGE scientifically trustworthy by giving every experiment one strict, versioned, task-valid recipe; making the displayed objective identical to the objective differentiated during training; and ensuring every metric states exactly which model revision, dataset population, and evaluation basis produced it.

This is a clean-break design. It does not preserve the behavior or serialized representation of the current unversioned configuration contract.

## Product Outcome

After this work:

- a binary-classification, multiclass-classification, or regression experiment cannot contain a contradictory dataset, output, target encoding, or loss;
- a preset or lesson always establishes one complete deterministic recipe instead of inheriting hidden state;
- binary cross-entropy, categorical cross-entropy, true mean squared error, Huber, regularization, clipping, evaluation, backprop explanations, traces, stop conditions, and loss landscapes use documented and testable semantics;
- train/test generalization comparisons always use full-split measurements produced at the same model revision;
- batch-loss EMA remains available as a responsive training trend but can never masquerade as a full train evaluation;
- URLs, imported JSON, saved runs, and session checkpoints declare schema version 2 and fail explicitly when incompatible;
- existing version-1 or unversioned saved data is preserved but never silently repaired or loaded as version 2.

## Scope

### Included

- Audit findings C-01 through C-06, C-23, M-01 through M-08, M-24, L-01, and L-02 from `docs/audits/2026-07-10-neural-network-playground/comprehensive-audit.md`.
- A discriminated version-2 experiment recipe and one authoritative validator/compiler.
- A dataset contract registry with task, input-domain, target-domain, class-count, and generator semantics.
- Complete deterministic preset and lesson recipes.
- Stable logit-based cross-entropy, true MSE, conventional Huber, and explicit regularization penalties.
- Global-norm clipping of the complete objective gradient.
- Objective and gradient diagnostic breakdowns.
- Paired train/test evaluations and separate live batch-loss trends.
- Version-2 URL, JSON, saved-run, and session-checkpoint contracts.
- Structured unsupported-version and validation errors.
- Minimum UI changes required to expose corrected terminology, provenance, and clean-break errors.
- Focused numerical, contract, transition, worker, persistence, browser, and performance tests.

### Excluded

- The broader worker generation/lifecycle redesign identified by C-08 and C-16 through C-22.
- The full responsive, accessibility, navigation, and visualization redesign.
- Arbitrary K-class support; version 2 keeps the current three-class public task while removing scalar/multiclass contradictions.
- New datasets, public custom datasets, arbitrary networks, convolutional/recurrent models, or external training frameworks.
- Public creation of intentionally invalid experiments.
- Migration of version-1 or unversioned URLs, JSON, saved runs, or checkpoints.
- Persistent trained-model export and exact resume of saved runs. Version-2 saved runs are evidence records whose action is “Apply saved recipe.”
- AdamW or another decoupled weight-decay optimizer. L1/L2 remain coupled objective penalties.
- Cloud persistence, collaboration, accounts, or remote execution.

These exclusions prevent a correctness repair from becoming an unreviewable product rewrite. Later waves may build on the version-2 contracts without weakening them.

## Design Principles

1. **Persist intent; derive redundancy.** Store task, dataset, inputs, architecture choices, training choices, and objective choices. Derive problem type, input size, output size, output activation, and target encoding.
2. **Reject instead of repair.** Validation returns structured issues and never silently changes a submitted value.
3. **One objective definition.** Loss value, output delta, regularization, gradient norm, clipping, optimizer input, landscape, trace, preview, and stop conditions share one compiled contract.
4. **Provenance is data, not copy.** Evaluation step, model revision, sample basis, and dataset identity travel with every measurement.
5. **Normal experiments are safe by construction.** Intentionally constrained configurations exist only as enumerated built-in lesson demonstrations.
6. **No dual semantic runtime.** Version 1 and version 2 are not executed side by side. Legacy artifacts are rejected and preserved.
7. **Tests describe invariants.** Numerical identity and cross-path consistency matter more than snapshotting incidental strings.

## Architecture

```text
ExperimentDocumentV2
        |
        v
validateExperimentDocument(unknown)
        |
        v
ValidatedExperimentDocumentV2 (branded)
        |
        +--> fingerprintRecipe()
        |
        v
compileExperimentRecipe()
        |
        +--> Compiled dataset contract
        +--> Compiled network config
        +--> Compiled objective
        +--> Compiled optimizer/update policy
        +--> Metric and visualization contract
        |
        v
Worker generation / Network
        |
        +--> LiveTrainingSignal
        +--> PairedEvaluation
        +--> ObjectiveBreakdown
        +--> ArtifactProvenance
        |
        v
UI, history, checkpoints, save artifact, export
```

The canonical recipe is the only persistent configuration. Existing `NetworkConfig`, `TrainingConfig`, `DataConfig`, and `FeatureFlags` may survive as private compiled runtime structures while consumers migrate, but they are not accepted as public version-2 documents.

## Version-2 Experiment Contract

### Envelope

```ts
export const EXPERIMENT_SCHEMA_VERSION = 2 as const;

export interface ExperimentDocumentV2 {
    kind: 'nn-playground-experiment';
    schemaVersion: 2;
    recipe: StandardExperimentRecipeV2;
    view: {
        showTestData: boolean;
        discretizeOutput: boolean;
    };
}
```

View fields remain in the share document because they affect the opened presentation, but they do not participate in the recipe fingerprint or session-checkpoint compatibility.

### Common Recipe

```ts
export interface CommonRecipeV2 {
    data: {
        sampleCount: number;
        trainFraction: number;
        noise: number;
        seed: number;
    };
    inputs: {
        featureIds: readonly FeatureId[];
    };
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
}

export type OptimizerSpecV2 =
    | { kind: 'sgd' }
    | { kind: 'sgd-momentum'; momentum: number }
    | { kind: 'adam'; beta1: number; beta2: number; epsilon: number };

export type LearningRateScheduleV2 =
    | { kind: 'constant' }
    | { kind: 'step'; interval: number; gamma: number }
    | { kind: 'cosine'; totalSteps: number; minimumRate: number };

export type GradientClipSpecV2 =
    | { kind: 'none' }
    | {
        kind: 'global-norm';
        maximumNorm: number;
        scope: 'total-objective-gradient';
    };
```

Optimizer-specific values do not exist when their optimizer is inactive. Adam epsilon therefore cannot be accidentally omitted from a valid Adam recipe.

### Task Variants

```ts
export type StandardExperimentRecipeV2 =
    | (CommonRecipeV2 & {
        task: {
            kind: 'binary-classification';
            dataset: BinaryDatasetId;
        };
        objective: {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' };
            penalty: PenaltySpecV2;
            reduction: 'mean-per-sample';
        };
    })
    | (CommonRecipeV2 & {
        task: {
            kind: 'multiclass-classification';
            dataset: 'three-class-clusters';
        };
        objective: {
            dataLoss: { kind: 'categorical-cross-entropy-with-logits' };
            penalty: PenaltySpecV2;
            reduction: 'mean-per-sample';
        };
    })
    | (CommonRecipeV2 & {
        task: {
            kind: 'regression';
            dataset: RegressionDatasetId;
        };
        objective: {
            dataLoss:
                | { kind: 'mean-squared-error' }
                | { kind: 'huber'; delta: number };
            penalty: PenaltySpecV2;
            reduction: 'mean-per-sample';
        };
    });

export type PenaltySpecV2 =
    | { kind: 'none' }
    | { kind: 'l1' | 'l2'; coefficient: number; applyTo: 'weights' };
```

The compiler derives:

| Task | Output | Activation | Target | Data loss |
|---|---:|---|---|---|
| Binary classification | 1 | sigmoid | scalar 0 or 1 | BCE from logits |
| Three-class classification | 3 | softmax | one-hot length 3 | categorical CE from logits |
| Regression | 1 | linear | finite scalar | true MSE or Huber |

Normal regression does not expose bounded output activations. An intentionally constrained output requires a separately approved failure-demonstration design and is not part of version 2.

### Features and Derived Sizes

`featureIds` is non-empty, unique, and stored in one canonical order. `inputSize` is always derived from its length. The recipe never stores a second input-size value. Multiclass output size is derived from the dataset contract class count. Output activation is derived from the task and never stored.

### Seeds

Data and model seeds are unsigned 32-bit integers from `0` through `4_294_967_295`. Fractional, negative, non-finite, or larger values fail validation. The PRNG consumes the normalized unsigned representation without an additional signed coercion that creates invisible aliases.

## Dataset Contracts

Add one registry owned by the engine:

```ts
export type TaskKind =
    | 'binary-classification'
    | 'multiclass-classification'
    | 'regression';

export interface DatasetContract {
    id: DatasetId;
    taskKind: TaskKind;
    inputDomain: {
        x: readonly [number, number];
        y: readonly [number, number];
    };
    targetDomain:
        | { kind: 'binary'; values: readonly [0, 1] }
        | { kind: 'classes'; classCount: 3 }
        | {
            kind: 'continuous';
            boundsForNoise: (noise: number) => readonly [number, number];
        };
    noise: {
        minimum: number;
        maximum: number;
        meaning: 'coordinate-perturbation' | 'target-perturbation';
    };
    generatorVersion: number;
}
```

### Generator Rules

- Every generated coordinate is finite and inside the declared input domain.
- Every version-2 Gaussian draw uses deterministic rejection sampling bounded to `mean ± 3 * standardDeviation`.
- Coordinate generators additionally reject/resample a candidate whose final coordinate leaves the declared input domain; they do not clamp it onto the edge. A sample has at most `10_000` deterministic attempts, after which generation fails with a structured dataset-generation error instead of emitting a bad point.
- Classification coordinate noise is applied after calculating the clean label for Checkerboard and Heart, matching the intended boundary-ambiguity semantics.
- Regression noise perturbs the target, not the input coordinate, and uses deterministic bounded sampling so `boundsForNoise()` remains true.
- Noise application cannot move a sample outside the declared domain. Deterministic rejection/resampling is preferred to post-hoc clamping that piles points on an edge.
- Every label is finite and valid for the dataset target contract.
- Generator behavior changes increment `generatorVersion` and therefore change the dataset key and experiment fingerprint.
- Dataset generation rejects non-finite or out-of-range settings before allocating samples.

For the current regression datasets, `reg-plane` declares target bounds `[-2 - 0.06 * noise, 2 + 0.06 * noise]` and `reg-gauss` declares `[-0.06 * noise, 2 + 0.06 * noise]`. These conservative bounds follow from the base functions and the bounded `3σ` target perturbation where `σ = 0.02 * noise`.

The version-2 visualization continues to use the declared domain. A later dynamic-domain redesign may expand this contract, but no version-2 training sample may affect metrics while remaining outside its declared visible domain.

## Validation and Compilation

### Structured Result

```ts
export type SchemaResult<T> =
    | { ok: true; value: T }
    | { ok: false; issues: readonly ExperimentSchemaIssue[] };

export interface ExperimentSchemaIssue {
    code:
        | 'unsupported-version'
        | 'legacy-state'
        | 'missing-field'
        | 'unknown-field'
        | 'invalid-field'
        | 'out-of-range'
        | 'duplicate-feature'
        | 'incompatible-task'
        | 'resource-limit';
    path: string;
    message: string;
}
```

Validation collects bounded, actionable issues rather than throwing on the first field. Unknown fields fail strict JSON/URL validation so misspelled behavior-affecting values cannot disappear silently.

### Authoritative Functions

```ts
validateExperimentDocument(value: unknown): SchemaResult<ValidatedExperimentDocumentV2>
canonicalRecipeKey(recipe: ValidatedStandardExperimentRecipeV2): string
fingerprintRecipe(recipe: ValidatedStandardExperimentRecipeV2): Promise<RecipeFingerprint>
fingerprintDataset(recipe: ValidatedStandardExperimentRecipeV2): Promise<DatasetKey>
fingerprintObjective(recipe: ValidatedStandardExperimentRecipeV2): Promise<ObjectiveKey>
prepareExperimentDocument(
    value: unknown,
): Promise<SchemaResult<PreparedExperimentDocumentV2>>
encodeExperimentUrl(document: ValidatedExperimentDocumentV2): string
decodeExperimentUrl(hash: string): SchemaResult<ValidatedExperimentDocumentV2>
```

Validated values are branded. The store, worker bridge, preset catalog, lesson registry, imports, URL decoder, persistence reader, and restore path must pass through this boundary. Type assertions alone do not create the brand.

The validation boundary captures one detached, deeply frozen plain-data snapshot before branding. It rejects accessors, hidden or symbol-keyed state, sparse arrays, cycles, and exotic records. Validation, compilation, canonicalization, and all fingerprints consume only that snapshot. This review-driven amendment prevents Proxy/getter inputs from presenting different values to different phases; it is an exact snapshot, not repair, normalization, or clamping.

`prepareExperimentDocument()` is the only public runtime-entry transaction. It validates synchronously, compiles the validated recipe through an internal pure `compileExperimentRecipe()`, awaits recipe/dataset/objective identities in parallel, and returns:

```ts
export interface PreparedExperimentDocumentV2 {
    document: ValidatedExperimentDocumentV2;
    compiled: CompiledExperimentConfig;
    identities: {
        canonicalRecipeKey: string;
        recipeFingerprint: RecipeFingerprint;
        datasetKey: DatasetKey;
        objectiveKey: ObjectiveKey;
    };
}
```

No worker generation, model allocation, snapshot publication, checkpoint creation, or save capture may begin until this transaction resolves successfully. Worker initialization/configuration receives the validated document plus claimed identities, recomputes the transaction inside the worker, and rejects any mismatch before allocation. `CompiledExperimentConfig` is a one-way internal adapter containing engine-ready network, training, data, feature, objective, task-metric, and visualization contracts. There is no inverse compiler that guesses a canonical recipe from arbitrary runtime fragments.

### Canonical Identity and Fingerprints

Identity uses these exact version constants:

```ts
export type RecipeFingerprint = string & { readonly __brand: 'RecipeFingerprint' };
export type DatasetKey = string & { readonly __brand: 'DatasetKey' };
export type ObjectiveKey = string & { readonly __brand: 'ObjectiveKey' };

export const RECIPE_FINGERPRINT_VERSION = 1 as const;
export const ENGINE_CONTRACT_VERSION = 1 as const;
export const FEATURE_REGISTRY_VERSION = 1 as const;
export const OBJECTIVE_IMPLEMENTATION_VERSION = 1 as const;
export const SPLIT_ALGORITHM_VERSION = 1 as const;
```

The canonical recipe fingerprint payload is:

```ts
{
    fingerprintVersion: 1,
    experimentSchemaVersion: 2,
    engineContractVersion: 1,
    featureRegistryVersion: 1,
    objectiveImplementationVersion: 1,
    datasetGeneratorVersion: getDatasetContract(recipe.task.dataset).generatorVersion,
    recipe
}
```

Canonical serialization follows RFC 8785 JSON Canonicalization Scheme and UTF-8 encoding. `canonicalRecipeKey()` returns the exact canonical JSON string for synchronous in-memory equality such as preset selection. `fingerprintRecipe()` calculates SHA-256 with Web Crypto over those UTF-8 bytes and returns lowercase prefix `r2.1.` followed by unpadded base64url digest bytes.

`datasetKey` uses the same canonicalization/digest algorithm with prefix `d2.1.` over dataset ID, generator version, sample count, train fraction, noise, data seed, and `SPLIT_ALGORITHM_VERSION`. `objectiveKey` uses prefix `o2.1.` over task kind, output contract, complete objective spec, reduction, and `OBJECTIVE_IMPLEMENTATION_VERSION`.

Version-2 artifacts store their digest plus the canonical recipe/document required to recompute it. Consumers recompute and compare rather than trusting the stored string alone. Exact canonical payload and digest fixtures are part of the shared test suite.

### Resource Limits

Limits are checked before network or typed-array allocation:

- sample count is an integer from `2` through `1_000` so both deterministic split populations are non-empty;
- train fraction is from `0.1` through `0.9`, inclusive;
- noise is from `0` through `100`, inclusive;
- one through nine registered features are present;
- hidden-layer count is from zero through six;
- every hidden-layer width is an integer from `1` through `16`;
- total trainable parameter count is at most `2_000`;
- learning rate is finite, greater than `0`, and at most `10`;
- batch size is an integer from `1` through `512` and cannot exceed the deterministic training population;
- SGD-momentum coefficient is finite and in `[0, 1)`;
- Adam beta values are finite and in `[0, 1)`, and epsilon is finite and in `(0, 1]`;
- L1/L2 coefficient is finite and in `(0, 1]`; `{ kind: 'none' }` is used instead of a zero coefficient;
- global clip maximum norm is finite and in `(0, 1_000_000]`;
- Huber delta is finite and in `(0, 1_000_000]`;
- a step schedule uses integer interval `1` through `1_000_000_000` and gamma in `(0, 1]`;
- a cosine schedule uses integer total steps `1` through `1_000_000_000` and minimum rate in `[0, baseLearningRate]`.

Validation fails rather than clamps resource-limit violations. These version-2 limits deliberately match or tightly bound the public UI so every accepted document is covered by the same latency budget. The deterministic split validator also requires both `trainCount >= 1` and `testCount >= 1` after applying `trainFraction`.

## Failure-Demonstration Policy

Version 2 does not ship a failure-demonstration recipe type or validation bypass. All current presets and lessons must use `StandardExperimentRecipeV2` and satisfy every normal invariant.

Intentionally incompatible experiments remain allowed only as a future, separately approved lesson feature. That future feature must define a closed `FailureModeId` union, one exact exception per ID, a private compiler entry point, disabled externalization/persistence actions, and its own tests before any invalid recipe can execute. It may not add `allowUnsafe`, `lenient`, or another generic bypass to the version-2 validator.

## Objective Contract

### Types

```ts
export interface ObjectiveBreakdown {
    dataLoss: number;
    regularizationPenalty: number;
    totalObjective: number;
}

export interface GradientDiagnostics {
    dataGradientNorm: number;
    penaltyGradientNorm: number;
    totalGradientNorm: number;
    clippedGradientNorm: number;
    clipScale: number;
}

export interface BatchTrainingResult {
    revision: number;
    step: number;
    sampleCount: number;
    objective: ObjectiveBreakdown;
    gradients: GradientDiagnostics;
}
```

### Data-Loss Formulas

#### Binary cross-entropy from logits

For logit `z` and target `y` in `{0, 1}`:

```text
max(z, 0) - z*y + log1p(exp(-abs(z)))
```

The output-logit derivative is `sigmoid(z) - y`. Loss and gradient remain finite and consistent at logits `-1000` and `1000`. Probability clamping is not part of the training objective.

#### Categorical cross-entropy from logits

Use the existing stable log-sum-exp value and `softmax(logits) - target` output delta. Probabilities and one-hot targets retain finite, length, and sum validation at public helper boundaries.

#### Mean squared error

For `m` outputs:

```text
sum((prediction_i - target_i)^2) / m
```

The prediction derivative is `2 * (prediction_i - target_i) / m`. With the current scalar regression output, this is ordinary squared error per sample. The old hidden one-half factor is removed.

#### Huber

Use the conventional piecewise Huber value and derivative with finite `delta > 0`, averaged across outputs. The version-2 default delta is exactly `1` and is stored explicitly in a Huber recipe.

### Penalty Formulas

Biases are excluded.

```text
L1 penalty = coefficient * sum(abs(weight))
L1 gradient = coefficient * sign(weight)

L2 penalty = 0.5 * coefficient * sum(weight^2)
L2 gradient = coefficient * weight
```

The penalty is model-wide and is never represented as though it were caused by a single data sample.

### Training Objective

```text
totalObjective = mean(data loss across the mini-batch) + regularizationPenalty
```

The data gradient is averaged over the same reduction used by the data-loss value. The penalty is added once per update, not once per sample.

### Gradient and Update Order

Every optimizer follows this order:

1. Accumulate data gradients for the batch.
2. Average data gradients using the objective reduction.
3. Compute and add penalty gradients to weight gradients.
4. Measure data, penalty, and total gradient norms.
5. Resolve global clip scale from the complete objective gradient.
6. Apply that scale to all weight and bias gradients.
7. Record the exact clipped gradient sent to the optimizer.
8. Execute the optimizer without adding regularization internally.
9. Zero accumulators and advance revision/step.

Clipping is an update transform, not a term in `totalObjective`. Adam's parameter-update norm is not expected to equal or remain below the raw gradient clip threshold; diagnostics describe the clipped gradient entering Adam.

### Compiled Objective API

Add an engine-owned objective module with responsibilities equivalent to:

```ts
compileObjective(spec, networkConfig): CompiledObjective
evaluateDataSample(logits, outputs, target): number
seedOutputDeltaInto(logits, outputs, target, destination): void
regularizationPenalty(weights): number
addPenaltyGradientInto(weights, weightGradients): void
computeGradientTransform(diagnostics, clipSpec): ClipResult
```

Training, full evaluation, prediction trace, backprop explanation, loss-landscape probe, checkpoint evaluation, and stop-condition evaluation consume this compiled objective or an explicitly documented data-loss-only projection.

### Evaluation Semantics

- Full train and test evaluation report predictive `dataLoss` and task metrics.
- `regularizationPenalty` is evaluated once for the model revision.
- `trainObjective = train.dataLoss + regularizationPenalty`.
- Test objective is not used as a generalization measure; the penalty is a model property shared by both splits.
- Generalization gap is `test.dataLoss - train.dataLoss` from one paired evaluation.
- Loss-landscape surfaces default to `training-objective` and state that basis.
- Prediction traces report `sampleDataLoss`; the model penalty appears separately.
- Backprop explanations report the complete objective and gradient breakdown.
- Stop conditions name their target explicitly: `trainDataLoss`, `testDataLoss`, `trainObjective`, or task accuracy.

## Metric Provenance Contract

### Model and Dataset Identity

```ts
export interface ModelRevision {
    generationId: number;
    revision: number;
    step: number;
    epoch: number;
}

export interface DatasetRevision {
    generatorVersion: number;
    datasetKey: string;
    trainCount: number;
    testCount: number;
}
```

`revision` increments whenever weights change or a model is restored. `generationId` changes on a new experiment/reset/rebuild using the existing worker run boundary. Step alone is not an identity.

`datasetKey` is a stable fingerprint of dataset contract version, generation settings, split settings, and feature-independent sample identity. `objectiveKey` is a stable fingerprint of the compiled objective semantics.

### Paired Full Evaluation

```ts
export interface FullSplitBasis {
    kind: 'full-split';
    split: 'train' | 'test';
    sampleCount: number;
    populationCount: number;
}

export interface EvaluationValues {
    dataLoss: number;
    accuracy?: number;
    confusionMatrix?: ConfusionMatrixData | MulticlassConfusionMatrixData;
}

export interface PairedEvaluation {
    evaluationId: number;
    trigger:
        | 'initial'
        | 'cadence'
        | 'manual-step'
        | 'pause'
        | 'checkpoint'
        | 'save'
        | 'stop-condition'
        | 'restore';
    model: ModelRevision;
    dataset: DatasetRevision;
    objectiveKey: string;
    train: {
        basis: FullSplitBasis & { split: 'train' };
        values: EvaluationValues;
    };
    test: {
        basis: FullSplitBasis & { split: 'test' };
        values: EvaluationValues;
    };
    objective: {
        regularizationPenalty: number;
        trainTotalObjective: number;
    };
}
```

Train and test cannot carry different model identities because model identity exists once at bundle level.
For `FullSplitBasis`, `sampleCount` must equal `populationCount` and both must be positive. The train and test counts must equal the corresponding positive counts in `DatasetRevision`.

### Live Training Signal

```ts
export interface LiveTrainingSignal {
    model: ModelRevision;
    dataset: DatasetRevision;
    objectiveKey: string;
    basis: {
        kind: 'mini-batch-ema';
        alpha: number;
        latestBatchSize: number;
        throughStep: number;
    };
    dataLoss: number;
}
```

The live signal never contains cached accuracy, confusion, generalization gap, or total-objective claims. It cannot be passed to APIs that require a `PairedEvaluation`.

### Other Artifacts

```ts
export interface ArtifactProvenance {
    model: ModelRevision;
    dataset: DatasetRevision;
    objectiveKey: string;
    basis:
        | FullSplitBasis
        | {
            kind: 'bounded-sample';
            split: 'train' | 'test';
            sampleCount: number;
            populationCount: number;
        }
        | {
            kind: 'prediction-grid';
            pointCount: number;
            domain: readonly [number, number, number, number];
        }
        | {
            kind: 'parameter-grid';
            sampleCount: number;
            parameterPositions: number;
        };
}
```

Confusion, activation statistics, prediction trace, boundary grid, and loss landscape each carry their own provenance. Remove the global implication that one stale Boolean describes all evidence.

### Activation Statistics

Layer activation `meanActivation` and `activationStd` are aggregates over a deterministic bounded training population, never values left in mutable buffers by the most recent forward pass.

- The sample is the first `min(128, trainCount)` records in the already deterministic shuffled training split.
- For each layer, mean and population standard deviation are calculated across every neuron activation for every selected record.
- `meanAbsWeight` remains a model-wide aggregate over that layer's weights at the same model revision.
- `meanAbsGradient` remains a model-wide aggregate over the most recently applied clipped gradient and states that update revision separately when it differs from the activation revision.
- The result carries `ArtifactProvenance` with `basis.kind: 'bounded-sample'`, `split: 'train'`, the exact selected count, and the full training population count.
- The UI label is “Activation statistics across N of M training examples,” not an unlabeled `μ(a)`/`σ(a)` claim.
- Running an unrelated forward prediction or grid evaluation before requesting statistics cannot change the result.

### Evaluation Policy

```ts
export interface EvaluationPolicy {
    everySteps: number;
    forceOnPause: true;
    forceOnManualStep: true;
    forceOnCheckpoint: true;
    forceOnSave: true;
}
```

The version-2 default policy uses `everySteps: 50`.

The worker:

1. updates `LiveTrainingSignal` after batches;
2. freezes one model revision when full evaluation is due;
3. evaluates complete train and test splits sequentially without training between them;
4. publishes one `PairedEvaluation`;
5. caches and transports the bundle atomically;
6. forces a current pair before pause completion, manual-step completion, checkpoint capture, save capture, and comparison-sensitive stop conditions.

The `everySteps: 50` default is independent of display-frame cadence. The implementation benchmark verifies that this fixed cadence satisfies the performance gates; changing it requires a documented design amendment rather than an implicit tuning change.

Freshness is derived from the difference between the current model revision and the evaluation model revision. The UI can say, for example: “Batch trend through step 1,240; full train/test evaluation at step 1,230 using 210 train and 90 test examples.”

### History

History contains two different series without dropping their identities:

```ts
export type TrainingTrendPoint = LiveTrainingSignal;
export type EvaluationPoint = PairedEvaluation;
```

Cached paired evaluations are not appended again under later steps. Charts label the EMA as “batch trend” and render paired evaluations as train/test points or lines. Best-test, overfitting, and cross-run diagnostics use evaluation points only. Plateau detection may use a specifically configured series and must state which series it uses.

## Presets and Lessons

### Complete Catalog Entries

```ts
export interface RecipeCatalogEntry {
    id: RecipeId;
    revision: number;
    title: string;
    description: string;
    learningGoal?: string;
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    recipe: StandardExperimentRecipeV2;
}
```

- Every current preset declares all data, input, model, objective, optimizer, clipping, schedule, regularization, and seed fields.
- `applyRecipe(id)` replaces the current recipe atomically.
- Preset selection compares the complete `canonicalRecipeKey()`; persisted/transport compatibility uses the SHA-256 recipe fingerprint.
- Lessons reference `{ id, revision }`; changing a catalog entry requires an explicit revision update.
- Pairwise transition tests assert the exact target fingerprint after every preset-to-preset and lesson-to-lesson transition.
- A future “Apply adjustment” feature may provide explicit patches, but no version-2 preset is a patch.

## URL and JSON Contract

### URL

Use:

```text
#v=2&r=<base64url canonical ExperimentDocumentV2 JSON>
```

Canonical JSON uses stable key ordering and the exact validated document. This avoids a hand-maintained short-key codec silently omitting future fields.

Behavior:

- an empty hash opens the default version-2 document;
- a non-empty hash without `v=2`, with a future version, with invalid base64url, or with an invalid document opens an incompatibility state and does not silently apply defaults;
- “Start fresh” is an explicit user action that replaces the bad hash with the default version-2 URL;
- Copy URL serializes the current validated store state synchronously instead of copying a possibly stale location string;

### JSON

- Export emits the exact `ExperimentDocumentV2` envelope.
- Import requires exact kind and version and reports every bounded structured issue.
- Unversioned, version-1, and future documents are rejected.
- Imports never repair output/loss/task combinations and never clamp resource settings.
- Failure-demonstration recipes are not accepted from external JSON.

## Persistence and Session Checkpoints

### Storage Boundary

Version 2 uses the exact storage key `nn-playground-experiment-memory-v2`. The existing version-1 key `nn-playground-experiment-memory` remains untouched.

If legacy storage exists:

- show a persistent notice that earlier runs are incompatible and were not loaded;
- offer dismissal, download of the untouched raw legacy JSON, and explicit deletion;
- never parse failure as an empty successful history;
- never automatically migrate or delete legacy records.

Version-2 persistence limits are exact:

- at most `20` saved records;
- title length at most `120` Unicode code points;
- record ID is a canonical UUID string of `36` ASCII characters;
- at most `512` `TrainingTrendPoint` entries per record;
- at most `256` `EvaluationPoint` entries per record;
- serialized record size at most `512 KiB` in UTF-8;
- serialized version-2 envelope size at most `4 MiB` in UTF-8.

Before capture returns, either history series over its limit is compacted deterministically. For limit `L`, preserve source indices `round(i * (N - 1) / (L - 1))` for `i = 0..L-1`; because compaction only runs when `N > L`, these indices are strictly increasing and preserve the first and last source entries. Evaluation history additionally rejects duplicate `evaluationId` values.

Saving a 21st record or exceeding the record/envelope byte budget returns a persistent structured error and does not evict or rewrite an existing record. Re-saving an existing ID may replace that record if all limits still pass.

The version-2 reader returns valid records and a separate rejected-record list with structured issues and raw JSON. Malformed or oversized records are not treated as valid, not silently dropped, and remain downloadable/deletable by explicit action.

### Saved Run

```ts
export interface ExperimentRunRecordV2 {
    kind: 'nn-playground-run';
    schemaVersion: 2;
    id: string;
    createdAt: string;
    updatedAt: string;
    title?: string;
    recipe: StandardExperimentRecipeV2;
    recipeFingerprint: string;
    snapshot: {
        model: ModelRevision;
        evaluation: PairedEvaluation;
        trendHistory: readonly TrainingTrendPoint[];
        evaluationHistory: readonly EvaluationPoint[];
    };
}
```

The worker authors the record through one atomic `captureRunArtifact()` transaction. It forces a current paired evaluation and returns recipe identity, revision, evaluation, and bounded histories from one generation. The UI does not compose a record from independent stores.

Record validation requires `snapshot.model` to equal `snapshot.evaluation.model`; every trend/evaluation history entry must use the same dataset key and objective key as the saved evaluation; no history model generation may differ from the saved generation; and the stored fingerprint must recompute from `recipe` exactly.

Version-2 saved runs do not persist weights, optimizer state, or PRNG/shuffle state. Their only apply action is named “Apply saved recipe”; it replaces the current recipe and starts a fresh model. Exact trained-model resume belongs to the later lifecycle/artifact wave and must introduce a separately versioned model-state contract before the UI may claim “Restore trained model.”

### Session Checkpoint

Session checkpoints remain worker-memory artifacts and use this complete version-2 envelope:

```ts
export interface SessionCheckpointV2 {
    kind: 'nn-playground-session-checkpoint';
    schemaVersion: 2;
    recipeFingerprint: string;
    objectiveKey: string;
    datasetKey: string;
    model: ModelRevision;
    evaluation: PairedEvaluation;
    network: {
        layers: readonly {
            inputSize: number;
            outputSize: number;
            weights: Float64Array;
            biases: Float64Array;
        }[];
    };
    optimizer: SessionOptimizerStateV2;
    cursor: {
        epoch: number;
        batchStart: number;
        shuffledIndices: Uint32Array;
    };
    trajectoryGuarantee: 'parameters-and-optimizer-only';
}

export type SessionOptimizerStateV2 =
    | { kind: 'sgd'; optimizerStep: number }
    | {
        kind: 'sgd-momentum';
        optimizerStep: number;
        weightVelocity: readonly Float64Array[];
        biasVelocity: readonly Float64Array[];
    }
    | {
        kind: 'adam';
        optimizerStep: number;
        firstWeightMoment: readonly Float64Array[];
        firstBiasMoment: readonly Float64Array[];
        secondWeightMoment: readonly Float64Array[];
        secondBiasMoment: readonly Float64Array[];
    };
```

`validateSessionCheckpointV2()` runs before restore. It requires:

- exact recipe, objective, and dataset-key equality with the active compiled experiment;
- checkpoint model identity equal to the bundled evaluation model identity;
- layer count and every input/output dimension equal to the compiled network;
- weight length `inputSize * outputSize` and bias length `outputSize` for each layer;
- finite parameter and optimizer arrays;
- optimizer kind equal to the active optimizer and every optimizer array shape equal to its parameter array;
- non-negative integer optimizer step, epoch, and batch cursor;
- `shuffledIndices` to be an exact permutation of `0..trainCount-1` and `batchStart <= trainCount`;
- total checkpoint payload no larger than `256 KiB` under the version-2 resource limits.

Checkpoint capture/clone/restore tests provide the version-2 round trip. PRNG state for the *next* shuffle is intentionally absent, hence the explicit `parameters-and-optimizer-only` guarantee.

### Compatibility and Comparison

- Session checkpoints include schema version, recipe fingerprint, objective key, dataset key, tensor shapes, and optimizer kind.
- Session-checkpoint restore rejects any mismatch. It restores the captured in-session parameters/optimizer payload but does not claim future trajectory determinism across a shuffle boundary until PRNG-state restoration is delivered by the lifecycle wave.
- Cross-run numeric test-loss comparisons require matching dataset key and objective key.
- Non-comparable runs may still show configuration differences, but the UI says “not directly comparable” and does not calculate a winner.
- Version-1 records are rejected without executing their parameters.

## Minimal UI Contract

This wave changes semantics, not overall layout.

Required UI changes:

- replace ambiguous “Loss” labels with “Data loss,” “Training objective,” or the exact objective name;
- label EMA as “Batch trend”;
- display full-evaluation step and sample counts where generalization claims appear;
- display regularization penalty and total objective in objective/backprop diagnostics;
- show structured import/URL/version errors with “Start fresh” as an explicit action;
- show the legacy-storage notice without implying records were deleted;
- remove or disable normal controls that can create invalid task/output combinations;
- update “MSE” copy to “Mean squared error” with standard semantics;
- state that gradient clipping covers the complete objective gradient;
- label the version-2 saved-run action “Apply saved recipe”; no exact trained-restore action appears in this wave.

The broader task-aware boundary redesign, compact-layout repair, and accessibility overhaul remain separate work.

## Error Handling

- Schema and protocol boundaries return structured domain errors.
- Programmer invariant failures inside already validated engine code may throw with precise context.
- UI actions never convert schema failure into a default config silently.
- Import, URL, persistence, worker, checkpoint, and restore errors identify source, issue path, and recovery action.
- Unsupported-version states preserve the original URL or stored bytes until the user chooses an action.
- Non-finite objective or evaluation results publish a valid terminal divergence error without entering checkpoint/history metrics.
- Resource validation occurs before large allocation.

## File and Module Boundaries

### Engine

- Add `packages/engine/src/datasetContracts.ts` for dataset metadata and generator invariants.
- Add `packages/engine/src/objective.ts` for compiled objective value, derivative, penalty, and gradient-transform behavior.
- Add `packages/engine/src/trainingContract.ts` for compiled task/network/objective/update policy.
- Modify `packages/engine/src/types.ts` to remove ambiguous public task/loss/metric combinations and export version-2 runtime types.
- Modify `packages/engine/src/losses.ts`, `network.ts`, and dataset generators to consume the compiled contracts.
- Export new public contracts through `packages/engine/src/index.ts`.

### Shared

- Add `packages/shared/src/experimentSchema.ts` for strict validation, branded values, compilation input, canonical URL/JSON codecs, and fingerprints.
- Add `packages/shared/src/metricProvenance.ts` for model/dataset revision, paired evaluation, training signal, and artifact basis types.
- Replace public `AppConfig` and partial `Preset` definitions in `packages/shared/src/types.ts`.
- Define the complete default document in `packages/shared/src/constants.ts`.
- Convert `packages/shared/src/presets.ts` to complete catalog entries.
- Replace version-1 record/envelope definitions in `packages/shared/src/experimentMemory.ts` while retaining explicit legacy detection.
- Version and harden `packages/shared/src/workerProtocol.ts`.
- Export all canonical contracts from `packages/shared/src/index.ts`.

### Web

- Store one canonical recipe in `apps/web/src/store/usePlaygroundStore.ts` and apply atomic recipe transactions.
- Compare full fingerprints in `recipeIdentity.ts`.
- Convert lesson registry/types and `GuidedLessonPanel.tsx` to versioned catalog references and complete standard recipes.
- Update controls to produce complete next recipes through typed actions.
- Update `ConfigPanel.tsx` for version-2 codecs and persistent structured errors.
- Update `useTraining.ts`, worker bridge/protocol, and `training.worker.ts` to accept only validated compiled generations.
- Separate paired evaluation and live signal in `useTrainingStore.ts`, `frameBuffer.ts`, and history buffers.
- Make `captureRunArtifact()` worker-owned; replace UI-composed record capture.
- Update Header, Current Run, Loss, Confusion, Inspection, Backprop, History, and explanation consumers to use explicit provenance and objective fields.

The detailed implementation plan must list exact file ranges and tests after re-reading the current worktree immediately before planning. Existing user edits in these files must be integrated rather than overwritten.

## Test Strategy

### Numerical Invariants

- BCE value and derivative agree under finite differences at ordinary logits and remain finite/correct at logits `-1000` and `1000`.
- Categorical CE value and logit derivative retain finite-difference coverage.
- True MSE reports squared error and derivative `2 * error` for the current scalar output.
- Huber agrees on both sides of and at its transition.
- `totalObjective === dataLoss + regularizationPenalty` within floating-point tolerance.
- L1 and L2 penalty formulas and gradients are exact; biases do not affect penalty.
- A zero-data-gradient, large-L2 case is clipped to the configured complete-gradient norm.
- Diagnostics match the exact gradient delivered to SGD, momentum, and Adam.
- Loss-landscape center equals total-objective evaluation for identical weights and samples.
- Trace, backprop preview, training, evaluation, and stop-condition projections agree at one revision.

### Schema and Dataset Invariants

- Each standard task variant validates and compiles to its declared output/target/objective tuple.
- Cross-task dataset/objective combinations fail.
- Validation never changes submitted values.
- `prepareExperimentDocument()` returns compiled config and all four identities atomically; allocation/publication cannot occur while any digest promise is pending.
- Worker preparation recomputes identities and rejects a forged recipe, dataset, or objective key before network allocation.
- Seeds accept only unsigned 32-bit integers.
- Sample count/train fraction combinations always produce positive train and test populations; counts below `2` fail.
- Features are unique, ordered, and non-empty.
- Resource limits fail before allocation.
- Every dataset produces finite, in-domain points and valid labels at minimum and maximum noise.
- Checkerboard and Heart noise increases boundary ambiguity using label-before-perturb semantics.
- Dataset generation remains deterministic for identical version-2 recipes.

### Preset and Lesson Invariants

- Every catalog recipe validates.
- Every preset-to-preset and lesson-to-lesson pair ends at the exact target fingerprint.
- Lesson catalog references identify an existing recipe revision.

### Codec and Persistence Invariants

- URL and JSON round-trip every behavior-affecting field.
- RFC-8785 canonical recipe, dataset, and objective payload fixtures produce exact checked-in `r2.1.`, `d2.1.`, and `o2.1.` SHA-256 base64url strings in browser and Node test environments.
- Changing a recipe field or referenced dataset/objective/feature contract version changes the corresponding identity; key order alone does not.
- Adam epsilon and all schedule parameters round-trip.
- Missing, version-1, future, malformed, unknown-field, and oversized documents return distinct issues.
- Copy URL uses current store state synchronously.
- Legacy storage is detected, rejected, and preserved.
- Version-2 history compaction preserves exact first/last entries and deterministic evenly spaced indices at `512` trend and `256` evaluation points.
- Record count, title/ID, record-byte, and envelope-byte limits return structured errors without evicting existing records.
- Malformed/oversized version-2 records remain isolated with raw downloadable JSON while valid sibling records still load.
- Version-2 saved records require matching recipe, objective, dataset, and snapshot model-generation identities.
- Session-checkpoint capture/clone/restore preserves exact parameters, optimizer arrays, cursor, model/evaluation identity, and envelope fields.
- Session-checkpoint validation rejects mismatched fingerprints/keys, tensor shapes, optimizer kinds, non-finite arrays, invalid index permutations, and payloads over `256 KiB`.
- Cross-run comparisons reject incompatible dataset/objective keys.

### Metric and Worker Invariants

- Initial, reset, manual step, pause, checkpoint, save, and restore triggers publish current-revision paired evaluations.
- Between evaluation cadence points, batch trend advances while evaluation identity remains unchanged.
- Train and test inside a pair always share model revision, dataset revision, and objective key.
- Cached pairs are not duplicated as later history points.
- Generalization/explanation APIs cannot accept `LiveTrainingSignal`.
- Confusion carries the test evaluation ID and sample basis.
- Activation statistics match a direct aggregate over the deterministic bounded training sample and remain unchanged after unrelated forward/grid evaluations.
- Save forces a fresh pair and returns one atomic artifact.
- Non-finite training results do not enter checkpoints or metric protocol payloads.

### Browser and Copy Invariants

- Applying Regression then XOR produces the exact XOR recipe.
- Applying every public preset from every other preset produces its declared full recipe.
- Legacy URL and JSON states show explicit incompatibility and do not silently train defaults.
- Version-2 share URL reloads to the identical recipe.
- Corrected Data loss, Batch trend, Training objective, evaluation age, and sample-count labels appear in appropriate views.
- Normal controls cannot select incompatible outputs/objectives.

### Performance Gates

- At the maximum public-UI experiment (1,000 samples and six hidden layers of 16), a forced paired evaluation must complete in at most `250 ms` at the median of 20 warmed runs on the existing development benchmark machine.
- A forced save capture at that public-UI maximum, including paired evaluation and bounded-history serialization, must complete in at most `500 ms` at the median of 20 warmed runs.
- The existing grid and optimizer benchmarks must be recorded before changes and may regress by no more than `20%` at the median of five complete benchmark runs on the same machine.
- Objective, penalty, and gradient diagnostics remain enabled in the benchmark so the budget measures shipped semantics rather than a stripped test path.
- Performance tests gain explicit threshold assertions; finite timing alone is not a pass.

### Repository Verification

Every implementation slice runs focused tests for touched behavior. Final verification runs from the repository root:

```text
pnpm test
pnpm lint
pnpm build
pnpm test:perf
git diff --check
```

Browser verification covers presets, lessons, binary classification, three-class classification, regression, URL/JSON errors, history incompatibility, pause/manual-step evaluation, save, and corrected terminology.

## Delivery Sequence

1. Lock version-2 types, dataset contracts, structured issues, and failing contract tests.
2. Implement stable BCE, true MSE, objective decomposition, complete-gradient clipping, and numerical invariants.
3. Implement strict validator, branded result, compiler, canonical fingerprint, and version-2 codecs.
4. Convert defaults, presets, lessons, and all-pairs transition tests.
5. Move the web store and controls to atomic recipes.
6. Introduce model/dataset revisions, paired evaluations, live training signal, and separate histories.
7. Version worker commands and migrate all metric/evidence consumers vertically.
8. Add atomic saved-run version 2, checkpoint/artifact fingerprints, and legacy-storage rejection UX.
9. Remove unversioned codecs, `allowMulticlass`, partial preset application, loose task fields, generic `loss`, and silent repair helpers.
10. Recalibrate lesson thresholds, examples, copy, screenshots, and performance cadence for corrected semantics.
11. Run the complete numerical, contract, transition, browser, repository, and performance gates.

Each delivery slice must begin with a failing test for its root-cause reproduction and end in a focused reviewable commit. Temporary adapters may translate a validated version-2 recipe into current runtime types, but version-1/unversioned data must never cross into the new runtime as though it were version 2.

## Acceptance Criteria

The Scientific Trust wave is complete only when all of these are true:

1. There is exactly one public version-2 experiment schema and one authoritative validation/compiler path.
2. Invalid task, dataset, output, target, objective, seed, feature, and resource combinations fail before training or allocation.
3. Every standard preset and lesson establishes one complete deterministic fingerprint regardless of prior state.
4. Displayed BCE, categorical CE, true MSE, Huber, L1, L2, and total objective agree with their analytic gradients and documented formulas.
5. Global clipping covers the complete objective gradient, including regularization.
6. Training, evaluation, traces, previews, stop conditions, and landscapes use the same compiled objective or a clearly named projection.
7. Batch trend and full paired evaluation are different types and cannot be combined accidentally.
8. Generalization gap always compares full train/test data loss at the same model revision and dataset revision.
9. Every persisted or streamed measurement carries sufficient provenance to identify model, data, objective, sample basis, and evaluation step.
10. Version-2 URLs, JSON, saved runs, and session checkpoints round-trip every field in their declared contracts and reject incompatible versions explicitly.
11. Legacy storage and bad inputs are preserved until explicit user action and never silently become defaults or empty history.
12. All focused and full repository verification passes, browser scenarios reproduce the intended behavior, and performance remains within approved budgets.

## Risks and Mitigations

### Corrected trajectories change

True MSE doubles the current scalar regression gradient; BCE no longer caps extreme loss; complete-gradient clipping changes regularized updates. Preset curves, screenshots, learning-rate recommendations, target thresholds, and lesson success rules will change.

Mitigation: treat version 2 as a new scientific baseline, recalibrate deterministic recipes after numerical invariants pass, and do not compare version-1 histories numerically.

### Broad type blast radius

Configuration, worker protocol, history, persistence, UI, and tests currently consume flat optional structures.

Mitigation: land a one-way recipe compiler first, migrate consumers in vertical slices, and forbid new inverse/repair adapters.

### Evaluation latency

Forced paired evaluation on pause, save, and checkpoint adds work.

Mitigation: benchmark maximum supported experiments, use a step-based cadence independent of rendering, show an explicit evaluating state, and keep the batch trend lightweight.

### Legacy data appears lost

A new storage key can make old records disappear from the normal list.

Mitigation: detect the untouched legacy key and state clearly that earlier data remains stored but is incompatible; provide explicit raw download/delete actions.

### Fingerprint drift

Different canonicalization implementations could identify identical recipes differently.

Mitigation: implement canonicalization once in shared code, include generator/objective schema versions, and cover stable fingerprints with fixtures.

## Rejected Alternatives

### Versioned envelope around the current flat AppConfig

Rejected because it preserves redundant `problemType`, input size, output size, output activation, loss, optimizer-only fields, distributed repair logic, and partial presets. It changes serialization without fixing the root model.

### Minimal objective patch behind existing TrainingConfig

Rejected as the final design because it can correct calculations but leaves invalid experiments representable and keeps generic loss semantics across worker/UI boundaries. It may inform an internal implementation step but cannot be the public result.

### Evaluate train and test on every rendered snapshot

Rejected because it couples scientific evaluation cost to display cadence and may waste bounded worker time. Atomic paired evaluation plus a separate batch trend gives correct semantics at controlled cadence.

### Independently timestamp train and test metrics

Rejected because every consumer must still remember not to compare mismatched revisions. A paired bundle makes the legal comparison structural.

### Generic capability/rule engine

Rejected for version 2 because arbitrary tasks and classes do not justify the additional framework. A small typed dataset/task registry can evolve later.

### Single atomic rewrite

Rejected because schema, engine math, worker protocol, persistence, and UI would change in one unreviewable unit. The staged contract-first sequence reaches the same design with testable checkpoints.

## Approval Record

The user selected:

- **Scientific Trust** as the first repair wave;
- **clean break** rather than version-1 compatibility or migration;
- the recommended policy that normal experiments block incompatible configurations and only explicit built-in failure lessons may demonstrate them;
- the contract-first staged design presented in conversation.

No failure-demonstration lesson is included in the initial version-2 scope.

The user approved the design on 2026-07-11. This approval authorizes writing the implementation plan, not beginning implementation before the written-spec review and plan handoff gates are complete.
