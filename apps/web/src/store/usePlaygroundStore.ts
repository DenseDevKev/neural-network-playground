import { create, type StoreApi, type UseBoundStore } from 'zustand';
import {
    generateDatasetV2,
    type DataSplit,
} from '@nn-playground/engine';
import {
    DEFAULT_DEMAND,
    DEFAULT_EXPERIMENT_DOCUMENT,
    decodeExperimentUrl,
    encodeExperimentUrl,
    prepareExperimentDocument,
    resolveRecipe,
    validateExperimentDocument,
    type ExperimentDocumentV2,
    type ExperimentSchemaIssue,
    type PreparedExperimentDocumentV2,
    type RecipeCatalogEntry,
    type SchemaResult,
    type ValidatedExperimentDocumentV2,
    type ValidatedStandardExperimentRecipeV2,
    type VisualizationDemand,
} from '@nn-playground/shared';
import type { RecipeEditIssue, RecipeEditResult } from './recipeEdits.ts';

export interface FeaturesUI {
    canvasNetworkGraph: boolean;
    webgpuGrid: boolean;
}

export interface PreparationState {
    status: 'ready' | 'preparing' | 'error';
    requestId: number;
    issues: readonly ExperimentSchemaIssue[];
}

export type ExperimentInputSource =
    | { readonly kind: 'url'; readonly rawHash: string }
    | { readonly kind: 'file'; readonly file: File };

export type ExperimentAccessState =
    | {
        readonly status: 'ready';
        readonly prepared: PreparedExperimentDocumentV2;
    }
    | {
        readonly status: 'incompatible';
        readonly prepared: null;
        readonly source: ExperimentInputSource;
        readonly issues: readonly ExperimentSchemaIssue[];
    };

export type IncompatibleSource = ExperimentInputSource;

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
}

export interface PlaygroundStore {
    access: ExperimentAccessState;
    preparation: PreparationState;
    incompatibleSource: IncompatibleSource | null;

    featuresUI: FeaturesUI;
    demand: VisualizationDemand;
    dataset: DataSplit | null;

    replaceDocument(value: unknown): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    replaceImportedDocument(
        value: unknown,
        file: File,
    ): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    markIncompatible(
        source: ExperimentInputSource,
        issues: readonly ExperimentSchemaIssue[],
    ): void;
    startFresh(): Promise<SchemaResult<PreparedExperimentDocumentV2>>;
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
            incompatibleSource: { kind: 'url', rawHash },
        };
    }

    const prepared = await prepare(decoded.value);
    if (!prepared.ok) {
        return {
            prepare,
            initialPrepared: null,
            initialIssues: prepared.issues,
            incompatibleSource: rawHash === '' ? null : { kind: 'url', rawHash },
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

export function createPlaygroundStore(
    initialization: PlaygroundStoreInitialization,
): PlaygroundStoreApi {
    const prepare = initialization.prepare ?? prepareExperimentDocument;
    const initialAccess: ExperimentAccessState = initialization.initialPrepared
        ? { status: 'ready', prepared: initialization.initialPrepared }
        : {
            status: 'incompatible',
            prepared: null,
            source: initialization.incompatibleSource
                ?? { kind: 'url', rawHash: '' },
            issues: initialization.initialIssues.length > 0
                ? initialization.initialIssues
                : NO_ACTIVE_DOCUMENT,
        };

    let nextRequestId = 0;
    let latestCandidateDocument: ValidatedExperimentDocumentV2 | null =
        initialization.initialPrepared?.document ?? null;
    let lastSuccessfulPrepared = initialization.initialPrepared;

    return create<PlaygroundStore>((set, get) => {
        const markIncompatible = (
            source: ExperimentInputSource,
            issues: readonly ExperimentSchemaIssue[],
        ): void => {
            const requestId = ++nextRequestId;
            latestCandidateDocument = null;
            const boundedIssues = issues.length > 0 ? issues : NO_ACTIVE_DOCUMENT;
            set({
                access: {
                    status: 'incompatible',
                    prepared: null,
                    source,
                    issues: boundedIssues,
                },
                preparation: { status: 'error', requestId, issues: boundedIssues },
                incompatibleSource: source,
            });
        };

        const replace = async (
            value: unknown,
            failureSource: ExperimentInputSource | null = null,
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
                if (failureSource) {
                    latestCandidateDocument = null;
                    set({
                        access: {
                            status: 'incompatible',
                            prepared: null,
                            source: failureSource,
                            issues: result.issues,
                        },
                        preparation: { status: 'error', requestId, issues: result.issues },
                        incompatibleSource: failureSource,
                    });
                } else {
                    latestCandidateDocument = lastSuccessfulPrepared?.document ?? null;
                    set({
                        preparation: { status: 'error', requestId, issues: result.issues },
                    });
                }
                return result;
            }

            lastSuccessfulPrepared = result.value;
            latestCandidateDocument = result.value.document;
            set({
                access: { status: 'ready', prepared: result.value },
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
            access: initialAccess,
            preparation: initialization.initialPrepared
                ? { status: 'ready', requestId: 0, issues: [] }
                : { status: 'error', requestId: 0, issues: initialization.initialIssues },
            incompatibleSource: initialization.incompatibleSource,
            featuresUI: { ...DEFAULT_FEATURES_UI },
            demand: { ...DEFAULT_DEMAND },
            dataset: null,

            replaceDocument: (value) => replace(value),
            replaceImportedDocument: (value, file) => replace(value, { kind: 'file', file }),
            markIncompatible,
            startFresh: async () => {
                const result = await replace(DEFAULT_EXPERIMENT_DOCUMENT);
                if (result.ok) {
                    const access = get().access;
                    if (access.status !== 'ready' || access.prepared !== result.value) {
                        return {
                            ok: false,
                            issues: [{
                                code: 'invalid-field',
                                path: '$',
                                message: 'Start fresh was superseded by a newer experiment request.',
                            }],
                        };
                    }
                    const hash = encodeExperimentUrl(result.value.document);
                    window.history.replaceState(null, '', hash);
                }
                return result;
            },
            editRecipe,
            editView,
            applyRecipe,
            syncToUrl: () => {
                const access = get().access;
                if (access.status !== 'ready') return unavailableResult(access.issues);
                const prepared = access.prepared;
                const hash = encodeExperimentUrl(prepared.document);
                window.history.replaceState(null, '', hash);
                return { ok: true, value: hash };
            },
            loadFromUrl: async () => {
                const raw = getRawExperimentHash(window.location.href);
                const decoded = decodeExperimentUrl(raw);
                if (!decoded.ok) {
                    markIncompatible({ kind: 'url', rawHash: raw }, decoded.issues);
                    return decoded;
                }
                return replace(decoded.value, { kind: 'url', rawHash: raw });
            },

            setDemand: (demand) => set({ demand }),
            regenerateData: () => {
                const access = get().access;
                if (access.status !== 'ready') return;
                const prepared = access.prepared;
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
        };
    });
}

const productionInitialization = typeof window === 'undefined'
    ? await initializePlaygroundStateFromHash('')
    : await initializePlaygroundStateFromLocation(window.location);

export const usePlaygroundStore = createPlaygroundStore({
    ...productionInitialization,
});
