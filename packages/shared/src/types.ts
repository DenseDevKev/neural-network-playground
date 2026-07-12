// ── Shared types for the application layer ──
import type {
    BinaryDatasetId,
    CompiledExperimentConfig,
    FeatureId,
    GradientClipSpecV2,
    LearningRateScheduleV2,
    OptimizerSpecV2,
    PenaltySpecV2,
    RegressionDatasetId,
    ScalarActivationType,
    WeightInitType,
} from '@nn-playground/engine';

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

export type StandardExperimentRecipeV2 =
    | (CommonRecipeV2 & {
        task: { kind: 'binary-classification'; dataset: BinaryDatasetId };
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
        task: { kind: 'regression'; dataset: RegressionDatasetId };
        objective: {
            dataLoss: { kind: 'mean-squared-error' } | { kind: 'huber'; delta: number };
            penalty: PenaltySpecV2;
            reduction: 'mean-per-sample';
        };
    });

export interface ExperimentDocumentV2 {
    kind: 'nn-playground-experiment';
    schemaVersion: 2;
    recipe: StandardExperimentRecipeV2;
    view: {
        showTestData: boolean;
        discretizeOutput: boolean;
    };
}

export type ExperimentSchemaIssueCode =
    | 'unsupported-version'
    | 'legacy-state'
    | 'missing-field'
    | 'unknown-field'
    | 'invalid-field'
    | 'out-of-range'
    | 'duplicate-feature'
    | 'incompatible-task'
    | 'resource-limit';

export interface ExperimentSchemaIssue {
    code: ExperimentSchemaIssueCode;
    path: string;
    message: string;
}

export type SchemaResult<T> =
    | { ok: true; value: T }
    | { ok: false; issues: readonly ExperimentSchemaIssue[] };

declare const validatedRecipeBrand: unique symbol;
declare const validatedDocumentBrand: unique symbol;

export type DeepReadonly<T> = T extends (...args: never[]) => unknown
    ? T
    : T extends readonly (infer Element)[]
        ? readonly DeepReadonly<Element>[]
        : T extends object
            ? { readonly [Key in keyof T]: DeepReadonly<T[Key]> }
            : T;

export type ValidatedStandardExperimentRecipeV2 = DeepReadonly<StandardExperimentRecipeV2> & {
    readonly [validatedRecipeBrand]: true;
};

export type ValidatedExperimentDocumentV2 = Omit<DeepReadonly<ExperimentDocumentV2>, 'recipe'> & {
    readonly recipe: ValidatedStandardExperimentRecipeV2;
    readonly [validatedDocumentBrand]: true;
};

export type RecipeFingerprint = string & { readonly __brand: 'RecipeFingerprint' };
export type DatasetKey = string & { readonly __brand: 'DatasetKey' };
export type ObjectiveKey = string & { readonly __brand: 'ObjectiveKey' };

export interface PreparedExperimentDocumentV2 {
    readonly document: ValidatedExperimentDocumentV2;
    readonly compiled: CompiledExperimentConfig;
    readonly identities: {
        readonly canonicalRecipeKey: string;
        readonly recipeFingerprint: RecipeFingerprint;
        readonly datasetKey: DatasetKey;
        readonly objectiveKey: ObjectiveKey;
    };
}

export type RecipeId =
    | 'single-neuron'
    | 'xor-hidden'
    | 'circle-one-layer'
    | 'spiral-deep'
    | 'regression-plane'
    | 'feature-engineering'
    | 'three-class-clusters';

export interface RecipeRef {
    readonly id: RecipeId;
    readonly revision: number;
}

export interface RecipeCatalogEntry {
    readonly id: RecipeId;
    readonly revision: 1;
    readonly title: string;
    readonly description: string;
    readonly learningGoal?: string;
    readonly difficulty?: 'beginner' | 'intermediate' | 'advanced';
    readonly recipe: ValidatedStandardExperimentRecipeV2;
    readonly prepared: PreparedExperimentDocumentV2;
}

export type TrainingStatus = 'idle' | 'running' | 'paused';

export const PAUSE_REASONS = [
    'target-loss-reached',
    'target-accuracy-reached',
    'plateau',
    'diverged',
    'max-steps',
    'manual',
    'error',
] as const;

export type PauseReason = typeof PAUSE_REASONS[number];

export function isPauseReason(value: unknown): value is PauseReason {
    return typeof value === 'string' && (PAUSE_REASONS as readonly string[]).includes(value);
}
