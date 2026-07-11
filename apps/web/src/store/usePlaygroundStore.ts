import { create, type StoreApi, type UseBoundStore } from 'zustand';
import {
    generateDatasetV2,
    type DataSplit,
    type FeatureFlags,
    type NetworkConfig,
    type TrainingConfig,
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
import type { RecipeEditIssue, RecipeEditResult } from './recipeEdits.ts';
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

    setDemand(demand: VisualizationDemand): void;
    regenerateData(): void;
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
