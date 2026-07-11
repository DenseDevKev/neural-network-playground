import { create, type StoreApi, type UseBoundStore } from 'zustand';
import {
    generateDatasetV2,
    type ActivationType,
    type DataSplit,
    type DatasetType,
    type FeatureFlags,
    type LRSchedule,
    type LossType,
    type NetworkConfig,
    type OptimizerType,
    type RegularizationType,
    type TrainingConfig,
    type WeightInitType,
} from '@nn-playground/engine';
import {
    DEFAULT_DEMAND,
    DEFAULT_EXPERIMENT_DOCUMENT,
    decodeExperimentUrl,
    encodeExperimentUrl,
    prepareExperimentDocument,
    resolveRecipe,
    validateExperimentDocument,
    type AppConfig,
    type ExperimentDocumentV2,
    type ExperimentSchemaIssue,
    type PreparedExperimentDocumentV2,
    type RecipeCatalogEntry,
    type SchemaResult,
    type UIConfig,
    type ValidatedExperimentDocumentV2,
    type ValidatedStandardExperimentRecipeV2,
    type VisualizationDemand,
} from '@nn-playground/shared';
import {
    reshuffleDataSeed,
    setBatchSize as editBatchSize,
    setGradientClipping,
    setHiddenActivation,
    setHiddenLayers as editHiddenLayers,
    setHiddenLayerWidth,
    setInitialization,
    setLearningRate as editLearningRate,
    setNoise as editNoise,
    setOptimizer as editOptimizer,
    setPenalty,
    setRegressionDataLoss,
    setSampleCount,
    setSchedule,
    setTrainFraction,
    switchDataset,
    toggleFeature as editFeature,
    type RecipeEditResult,
    type RecipeEditIssue,
} from './recipeEdits.ts';
import { projectPreparedExperiment } from './legacyProjection.ts';

export interface FeaturesUI {
    canvasNetworkGraph: boolean;
    webgpuGrid: boolean;
}

export interface PreparationState {
    status: 'ready' | 'preparing' | 'error';
    requestId: number;
    issues: readonly ExperimentSchemaIssue[];
}

export interface IncompatibleSource {
    kind: 'url';
    raw: string;
}

export type StoreRecipeEditResult =
    | SchemaResult<PreparedExperimentDocumentV2>
    | { ok: false; issue: RecipeEditIssue };

type PrepareExperiment = (
    value: unknown,
) => Promise<SchemaResult<PreparedExperimentDocumentV2>>;

export interface PlaygroundStoreInitialization {
    prepare?: PrepareExperiment;
    initialPrepared: PreparedExperimentDocumentV2 | null;
    initialIssues: readonly ExperimentSchemaIssue[];
    incompatibleSource: IncompatibleSource | null;
    fallbackPrepared?: PreparedExperimentDocumentV2;
}

export interface PlaygroundStore {
    prepared: PreparedExperimentDocumentV2 | null;
    preparation: PreparationState;
    incompatibleSource: IncompatibleSource | null;

    // Temporary one-way compatibility projection. These fields are published
    // in the same transaction as `prepared` and are never canonical inputs.
    network: NetworkConfig;
    training: TrainingConfig;
    data: AppConfig['data'];
    features: FeatureFlags;
    ui: UIConfig;

    featuresUI: FeaturesUI;
    demand: VisualizationDemand;
    dataset: DataSplit | null;

    replaceDocument(value: unknown): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    editRecipe(
        edit: (recipe: ValidatedStandardExperimentRecipeV2) => RecipeEditResult,
    ): Promise<StoreRecipeEditResult>;
    editView(
        edit: (
            view: ValidatedExperimentDocumentV2['view'],
        ) => ExperimentDocumentV2['view'],
    ): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    applyRecipe(entry: RecipeCatalogEntry): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    syncToUrl(): SchemaResult<string>;
    loadFromUrl(): Promise<SchemaResult<PreparedExperimentDocumentV2>>;

    // Transitional control aliases. They only delegate to typed V2 edits.
    setDataset(type: DatasetType): void;
    setNoise(noise: number): void;
    setTrainTestRatio(ratio: number): void;
    setNumSamples(count: number): void;
    reshuffleDataSeed(): void;
    toggleFeature(feature: keyof FeatureFlags): void;
    setHiddenLayers(layers: number[]): void;
    addLayer(): void;
    removeLayer(): void;
    setNeuronsInLayer(layerIndex: number, count: number): void;
    setActivation(activation: ActivationType): void;
    setLearningRate(learningRate: number): void;
    setBatchSize(batchSize: number): void;
    setLossType(lossType: LossType): void;
    setOptimizer(optimizer: OptimizerType): void;
    setMomentum(momentum: number): void;
    setGradientClip(maximumNorm: number | null): void;
    setAdamBetas(beta1: number, beta2: number): void;
    setHuberDelta(delta: number): void;
    setLRSchedule(schedule: LRSchedule | undefined): void;
    setWeightInit(initialization: WeightInitType): void;
    /** @deprecated Output activation is derived from task and cannot be edited. */
    setOutputActivation(activation: ActivationType): void;
    setRegularization(regularization: RegularizationType): void;
    setRegularizationRate(coefficient: number): void;
    setShowTestData(show: boolean): void;
    setDiscretize(discretize: boolean): void;
    setDemand(demand: VisualizationDemand): void;
    regenerateData(): void;
    /** @deprecated Task 6C callers migrate to applyRecipe. */
    applyPreset(entry: RecipeCatalogEntry): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    getConfig(): AppConfig;
}

export type PlaygroundStoreApi = UseBoundStore<StoreApi<PlaygroundStore>>;

const DEFAULT_FEATURES_UI: FeaturesUI = {
    canvasNetworkGraph: true,
    webgpuGrid: true,
};

const NO_ACTIVE_DOCUMENT: readonly ExperimentSchemaIssue[] = Object.freeze([{
    code: 'invalid-field',
    path: '$',
    message: 'No compatible version-2 experiment is active',
}]);

function unavailableResult<T>(
    issues: readonly ExperimentSchemaIssue[],
): SchemaResult<T> {
    return { ok: false, issues: issues.length > 0 ? issues : NO_ACTIVE_DOCUMENT };
}

function exactCatalogEntry(entry: RecipeCatalogEntry): RecipeCatalogEntry | undefined {
    const resolved = resolveRecipe({ id: entry.id, revision: entry.revision });
    if (resolved !== entry) return undefined;
    if (resolved.recipe !== resolved.prepared.document.recipe) return undefined;
    return resolved;
}

function scheduleFromLegacy(schedule: LRSchedule | undefined) {
    if (!schedule || schedule.type === 'constant') return { kind: 'constant' } as const;
    if (schedule.type === 'step') {
        return { kind: 'step', interval: schedule.stepSize, gamma: schedule.gamma } as const;
    }
    return {
        kind: 'cosine',
        totalSteps: schedule.totalSteps,
        minimumRate: schedule.minLr,
    } as const;
}

export async function initializePlaygroundStateFromHash(
    rawHash: string,
    prepare: PrepareExperiment = prepareExperimentDocument,
): Promise<PlaygroundStoreInitialization> {
    const decoded = decodeExperimentUrl(rawHash);
    if (!decoded.ok) {
        return {
            prepare,
            initialPrepared: null,
            initialIssues: decoded.issues,
            incompatibleSource: { kind: 'url', raw: rawHash },
        };
    }

    const prepared = await prepare(decoded.value);
    if (!prepared.ok) {
        return {
            prepare,
            initialPrepared: null,
            initialIssues: prepared.issues,
            incompatibleSource: rawHash === '' ? null : { kind: 'url', raw: rawHash },
        };
    }
    return {
        prepare,
        initialPrepared: prepared.value,
        initialIssues: [],
        incompatibleSource: null,
    };
}

/**
 * Read the fragment from the full URL instead of `Location.hash` so a bare
 * trailing `#` remains distinguishable from a URL with no fragment.
 */
export function getRawExperimentHash(href: string): string {
    const hashIndex = href.indexOf('#');
    return hashIndex === -1 ? '' : href.slice(hashIndex);
}

export function initializePlaygroundStateFromLocation(
    location: Pick<Location, 'href'>,
    prepare: PrepareExperiment = prepareExperimentDocument,
): Promise<PlaygroundStoreInitialization> {
    return initializePlaygroundStateFromHash(
        getRawExperimentHash(location.href),
        prepare,
    );
}

const defaultPreparation = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
if (!defaultPreparation.ok) {
    const detail = defaultPreparation.issues
        .map((issue) => `${issue.path}: ${issue.message}`)
        .join('; ');
    throw new Error(`Unable to prepare the default experiment: ${detail}`);
}
const DEFAULT_PREPARED = defaultPreparation.value;

export function createPlaygroundStore(
    initialization: PlaygroundStoreInitialization,
): PlaygroundStoreApi {
    const prepare = initialization.prepare ?? prepareExperimentDocument;
    const fallbackPrepared = initialization.fallbackPrepared ?? DEFAULT_PREPARED;
    const initialProjection = projectPreparedExperiment(
        initialization.initialPrepared ?? fallbackPrepared,
    );

    let nextRequestId = 0;
    let latestCandidateDocument: ValidatedExperimentDocumentV2 | null =
        initialization.initialPrepared?.document ?? null;
    let lastSuccessfulPrepared = initialization.initialPrepared;

    return create<PlaygroundStore>((set, get) => {
        const replace = async (
            value: unknown,
            failureSource: IncompatibleSource | null = null,
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const requestId = ++nextRequestId;
            const candidateValidation = validateExperimentDocument(value);
            if (candidateValidation.ok) {
                latestCandidateDocument = candidateValidation.value;
            }

            set({
                preparation: { status: 'preparing', requestId, issues: [] },
            });
            const result = await prepare(
                candidateValidation.ok ? candidateValidation.value : value,
            );
            if (requestId !== nextRequestId) return result;

            if (!result.ok) {
                latestCandidateDocument = lastSuccessfulPrepared?.document ?? null;
                set((state) => ({
                    preparation: { status: 'error', requestId, issues: result.issues },
                    incompatibleSource: failureSource ?? state.incompatibleSource,
                }));
                return result;
            }

            const projection = projectPreparedExperiment(result.value);
            lastSuccessfulPrepared = result.value;
            latestCandidateDocument = result.value.document;
            set({
                prepared: result.value,
                network: projection.network,
                training: projection.training,
                data: projection.data,
                features: projection.features,
                ui: projection.ui,
                preparation: { status: 'ready', requestId, issues: [] },
                incompatibleSource: null,
            });
            return result;
        };

        const editRecipe = async (
            edit: (recipe: ValidatedStandardExperimentRecipeV2) => RecipeEditResult,
        ): Promise<StoreRecipeEditResult> => {
            const base = latestCandidateDocument;
            if (!base) return unavailableResult(get().preparation.issues);
            const edited = edit(base.recipe);
            if (!edited.ok) return edited;
            return replace({ ...base, recipe: edited.recipe });
        };

        const editView = async (
            edit: (
                view: ValidatedExperimentDocumentV2['view'],
            ) => ExperimentDocumentV2['view'],
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const base = latestCandidateDocument;
            if (!base) return unavailableResult(get().preparation.issues);
            return replace({ ...base, view: edit(base.view) });
        };

        const applyRecipe = async (
            entry: RecipeCatalogEntry,
        ): Promise<SchemaResult<PreparedExperimentDocumentV2>> => {
            const base = latestCandidateDocument;
            if (!base) return unavailableResult(get().preparation.issues);
            const resolved = exactCatalogEntry(entry);
            if (!resolved) {
                return {
                    ok: false,
                    issues: [{
                        code: 'invalid-field',
                        path: 'recipe',
                        message: 'Recipe catalog entry is not the resolved built-in identity pair',
                    }],
                };
            }
            return replace({
                kind: 'nn-playground-experiment',
                schemaVersion: 2,
                recipe: resolved.recipe,
                view: base.view,
            });
        };

        const fireEdit = (
            edit: (recipe: ValidatedStandardExperimentRecipeV2) => RecipeEditResult,
        ): void => {
            void editRecipe(edit);
        };

        return {
            prepared: initialization.initialPrepared,
            preparation: initialization.initialPrepared
                ? { status: 'ready', requestId: 0, issues: [] }
                : { status: 'error', requestId: 0, issues: initialization.initialIssues },
            incompatibleSource: initialization.incompatibleSource,
            network: initialProjection.network,
            training: initialProjection.training,
            data: initialProjection.data,
            features: initialProjection.features,
            ui: initialProjection.ui,
            featuresUI: { ...DEFAULT_FEATURES_UI },
            demand: { ...DEFAULT_DEMAND },
            dataset: null,

            replaceDocument: (value) => replace(value),
            editRecipe,
            editView,
            applyRecipe,
            syncToUrl: () => {
                const prepared = get().prepared;
                if (!prepared) return unavailableResult(get().preparation.issues);
                const hash = encodeExperimentUrl(prepared.document);
                window.history.replaceState(null, '', hash);
                return { ok: true, value: hash };
            },
            loadFromUrl: async () => {
                const raw = getRawExperimentHash(window.location.href);
                const decoded = decodeExperimentUrl(raw);
                if (!decoded.ok) {
                    const requestId = ++nextRequestId;
                    latestCandidateDocument = lastSuccessfulPrepared?.document ?? null;
                    set({
                        preparation: { status: 'error', requestId, issues: decoded.issues },
                        incompatibleSource: { kind: 'url', raw },
                    });
                    return decoded;
                }
                return replace(decoded.value, { kind: 'url', raw });
            },

            setDataset: (dataset) => fireEdit((recipe) => switchDataset(recipe, dataset)),
            setNoise: (noise) => fireEdit((recipe) => editNoise(recipe, noise)),
            setTrainTestRatio: (ratio) => fireEdit((recipe) => setTrainFraction(recipe, ratio)),
            setNumSamples: (count) => fireEdit((recipe) => setSampleCount(recipe, count)),
            reshuffleDataSeed: () => fireEdit(reshuffleDataSeed),
            toggleFeature: (feature) => fireEdit((recipe) => editFeature(recipe, feature)),
            setHiddenLayers: (layers) => fireEdit((recipe) => editHiddenLayers(recipe, layers)),
            addLayer: () => fireEdit((recipe) => editHiddenLayers(
                recipe,
                [...recipe.model.hiddenLayers, 4],
            )),
            removeLayer: () => fireEdit((recipe) => editHiddenLayers(
                recipe,
                recipe.model.hiddenLayers.slice(0, -1),
            )),
            setNeuronsInLayer: (layerIndex, count) => fireEdit(
                (recipe) => setHiddenLayerWidth(recipe, layerIndex, count),
            ),
            setActivation: (activation) => {
                if (activation === 'softmax') return;
                fireEdit((recipe) => setHiddenActivation(recipe, activation));
            },
            setLearningRate: (learningRate) => fireEdit(
                (recipe) => editLearningRate(recipe, learningRate),
            ),
            setBatchSize: (batchSize) => fireEdit((recipe) => editBatchSize(recipe, batchSize)),
            setLossType: (lossType) => fireEdit((recipe) => {
                if (recipe.task.kind !== 'regression') return { ok: true, recipe };
                if (lossType === 'mse') {
                    return setRegressionDataLoss(recipe, { kind: 'mean-squared-error' });
                }
                if (lossType === 'huber') {
                    const delta = recipe.objective.dataLoss.kind === 'huber'
                        ? recipe.objective.dataLoss.delta
                        : 1;
                    return setRegressionDataLoss(recipe, { kind: 'huber', delta });
                }
                return { ok: true, recipe };
            }),
            setOptimizer: (optimizer) => fireEdit((recipe) => {
                if (optimizer === 'sgd') return editOptimizer(recipe, { kind: 'sgd' });
                if (optimizer === 'sgdMomentum') {
                    const momentum = recipe.training.optimizer.kind === 'sgd-momentum'
                        ? recipe.training.optimizer.momentum
                        : 0.9;
                    return editOptimizer(recipe, { kind: 'sgd-momentum', momentum });
                }
                const current = recipe.training.optimizer;
                return editOptimizer(recipe, {
                    kind: 'adam',
                    beta1: current.kind === 'adam' ? current.beta1 : 0.9,
                    beta2: current.kind === 'adam' ? current.beta2 : 0.999,
                    epsilon: current.kind === 'adam' ? current.epsilon : 1e-8,
                });
            }),
            setMomentum: (momentum) => fireEdit((recipe) => {
                if (recipe.training.optimizer.kind !== 'sgd-momentum') {
                    return { ok: true, recipe };
                }
                return editOptimizer(recipe, { kind: 'sgd-momentum', momentum });
            }),
            setGradientClip: (maximumNorm) => fireEdit((recipe) => setGradientClipping(
                recipe,
                maximumNorm === null
                    ? { kind: 'none' }
                    : {
                        kind: 'global-norm',
                        maximumNorm,
                        scope: 'total-objective-gradient',
                    },
            )),
            setAdamBetas: (beta1, beta2) => fireEdit((recipe) => {
                if (recipe.training.optimizer.kind !== 'adam') return { ok: true, recipe };
                return editOptimizer(recipe, {
                    ...recipe.training.optimizer,
                    beta1,
                    beta2,
                });
            }),
            setHuberDelta: (delta) => fireEdit((recipe) => {
                if (recipe.task.kind !== 'regression'
                    || recipe.objective.dataLoss.kind !== 'huber') {
                    return { ok: true, recipe };
                }
                return setRegressionDataLoss(recipe, { kind: 'huber', delta });
            }),
            setLRSchedule: (schedule) => fireEdit(
                (recipe) => setSchedule(recipe, scheduleFromLegacy(schedule)),
            ),
            setWeightInit: (initialization) => fireEdit(
                (recipe) => setInitialization(recipe, initialization),
            ),
            setOutputActivation: () => undefined,
            setRegularization: (regularization) => fireEdit((recipe) => {
                if (regularization === 'none') return setPenalty(recipe, { kind: 'none' });
                const current = recipe.objective.penalty;
                const coefficient = current.kind === 'none' ? 0.001 : current.coefficient;
                return setPenalty(recipe, {
                    kind: regularization,
                    coefficient,
                    applyTo: 'weights',
                });
            }),
            setRegularizationRate: (coefficient) => fireEdit((recipe) => {
                const current = recipe.objective.penalty;
                if (current.kind === 'none') return { ok: true, recipe };
                return setPenalty(recipe, { ...current, coefficient });
            }),
            setShowTestData: (showTestData) => {
                void editView((view) => ({ ...view, showTestData }));
            },
            setDiscretize: (discretizeOutput) => {
                void editView((view) => ({ ...view, discretizeOutput }));
            },
            setDemand: (demand) => set({ demand }),
            regenerateData: () => {
                const prepared = get().prepared;
                if (!prepared) return;
                const recipe = prepared.document.recipe;
                const dataset = generateDatasetV2({
                    dataset: recipe.task.dataset,
                    sampleCount: recipe.data.sampleCount,
                    trainFraction: recipe.data.trainFraction,
                    noise: recipe.data.noise,
                    seed: recipe.data.seed,
                });
                set({ dataset });
            },
            applyPreset: (entry) => applyRecipe(entry),
            getConfig: () => {
                const state = get();
                if (!state.prepared) {
                    throw new Error('No compatible version-2 experiment is active.');
                }
                return {
                    network: state.network,
                    training: state.training,
                    data: state.data,
                    features: state.features,
                    ui: state.ui,
                };
            },
        };
    });
}

const productionInitialization = typeof window === 'undefined'
    ? await initializePlaygroundStateFromHash('')
    : await initializePlaygroundStateFromLocation(window.location);

export const usePlaygroundStore = createPlaygroundStore({
    ...productionInitialization,
    fallbackPrepared: DEFAULT_PREPARED,
});
