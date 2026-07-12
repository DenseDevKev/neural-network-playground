import { describe, expect, it, vi } from 'vitest';
import { generateDatasetV2 } from '@nn-playground/engine';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    PREPARED_PRESETS,
    encodeExperimentUrl,
    prepareExperimentDocument,
    type ExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
    type SchemaResult,
} from '@nn-playground/shared';
import { setLearningRate, setNoise } from './recipeEdits.ts';
import {
    createPlaygroundStore,
    initializePlaygroundStateFromLocation,
    initializePlaygroundStateFromHash,
    type PlaygroundStoreApi,
} from './usePlaygroundStore.ts';

type Prepare = (
    value: unknown,
) => Promise<SchemaResult<PreparedExperimentDocumentV2>>;

function preset(id: string) {
    const entry = PREPARED_PRESETS.find((candidate) => candidate.id === id);
    if (!entry) throw new Error(`Missing preset ${id}`);
    return entry;
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((next) => {
        resolve = next;
    });
    return { promise, resolve };
}

function controlledPreparation() {
    const calls: Array<{
        value: unknown;
        gate: ReturnType<typeof deferred<void>>;
        result: Promise<SchemaResult<PreparedExperimentDocumentV2>>;
    }> = [];
    const prepare: Prepare = (value) => {
        const gate = deferred<void>();
        const result = prepareExperimentDocument(value);
        calls.push({ value, gate, result });
        return gate.promise.then(() => result);
    };
    return {
        calls,
        prepare,
        resolve(index: number) {
            calls[index].gate.resolve();
        },
    };
}

function createReadyStore(
    initialPrepared = preset('xor-hidden').prepared,
    prepare: Prepare = prepareExperimentDocument,
) {
    return createPlaygroundStore({
        prepare,
        initialPrepared,
        initialIssues: [],
        incompatibleSource: null,
    });
}

function readyPrepared(store: PlaygroundStoreApi): PreparedExperimentDocumentV2 {
    const access = store.getState().access;
    if (access.status !== 'ready') throw new Error('expected ready experiment access');
    return access.prepared;
}

function withNoise(
    prepared: PreparedExperimentDocumentV2,
    noise: number,
): ExperimentDocumentV2 {
    return {
        ...prepared.document,
        recipe: {
            ...prepared.document.recipe,
            data: { ...prepared.document.recipe.data, noise },
        },
    };
}

describe('atomic prepared-document store', () => {
    it('exposes canonical mutation transactions without legacy control aliases', () => {
        const state = createReadyStore().getState();
        const legacyAliases = [
            'setDataset',
            'setNoise',
            'setTrainTestRatio',
            'setNumSamples',
            'reshuffleDataSeed',
            'toggleFeature',
            'setHiddenLayers',
            'addLayer',
            'removeLayer',
            'setNeuronsInLayer',
            'setActivation',
            'setLearningRate',
            'setBatchSize',
            'setLossType',
            'setOptimizer',
            'setMomentum',
            'setGradientClip',
            'setAdamBetas',
            'setHuberDelta',
            'setLRSchedule',
            'setWeightInit',
            'setOutputActivation',
            'setRegularization',
            'setRegularizationRate',
            'setShowTestData',
            'setDiscretize',
            'applyPreset',
        ] as const;

        expect(state).toMatchObject({
            editRecipe: expect.any(Function),
            editView: expect.any(Function),
            applyRecipe: expect.any(Function),
        });
        for (const alias of legacyAliases) {
            expect(state, alias).not.toHaveProperty(alias);
        }
    });

    it('publishes a successful prepared document and every projection in one ready-state set', async () => {
        const initial = preset('xor-hidden').prepared;
        const targetDocument = withNoise(initial, 7);
        const controlled = controlledPreparation();
        const store = createReadyStore(initial, controlled.prepare);
        const snapshots: ReturnType<typeof store.getState>[] = [];
        store.subscribe((state) => snapshots.push(state));

        const pending = store.getState().replaceDocument(targetDocument);
        expect(store.getState().preparation).toMatchObject({ status: 'preparing', requestId: 1 });
        expect(readyPrepared(store)).toBe(initial);

        controlled.resolve(0);
        const result = await pending;
        expect(result.ok).toBe(true);
        if (!result.ok) return;

        const readySnapshots = snapshots.filter((state) => state.preparation.status === 'ready');
        expect(readySnapshots).toHaveLength(1);
        const published = readySnapshots[0];
        expect(published.access).toEqual({ status: 'ready', prepared: result.value });
        expect(published.incompatibleSource).toBeNull();
    });

    it('retains the exact last-successful prepared and projection references after latest failure', async () => {
        const store = createReadyStore();
        const before = readyPrepared(store);

        const result = await store.getState().replaceDocument({
            ...before.document,
            recipe: {
                ...before.document.recipe,
                data: { ...before.document.recipe.data, sampleCount: 1 },
            },
        });

        expect(result.ok).toBe(false);
        const after = store.getState();
        expect(readyPrepared(store)).toBe(before);
        expect(after.preparation.status).toBe('error');
        expect(after.preparation.issues.length).toBeGreaterThan(0);
    });

    it('ignores a stale success after a newer success', async () => {
        const initial = preset('xor-hidden').prepared;
        const controlled = controlledPreparation();
        const store = createReadyStore(initial, controlled.prepare);

        const older = store.getState().replaceDocument(withNoise(initial, 3));
        const newer = store.getState().replaceDocument(withNoise(initial, 8));
        controlled.resolve(1);
        const newerResult = await newer;
        controlled.resolve(0);
        const olderResult = await older;

        expect(newerResult.ok).toBe(true);
        expect(olderResult.ok).toBe(true);
        expect(readyPrepared(store).document.recipe.data.noise).toBe(8);
        expect(store.getState().preparation).toMatchObject({ status: 'ready', requestId: 2 });
    });

    it('does not publish an older success after the latest request fails', async () => {
        const initial = preset('xor-hidden').prepared;
        const controlled = controlledPreparation();
        const store = createReadyStore(initial, controlled.prepare);
        const olderSuccess = store.getState().replaceDocument(withNoise(initial, 6));
        const latestFailure = store.getState().replaceDocument({
            ...initial.document,
            recipe: {
                ...initial.document.recipe,
                data: { ...initial.document.recipe.data, sampleCount: 1 },
            },
        });

        controlled.resolve(1);
        expect((await latestFailure).ok).toBe(false);
        expect(readyPrepared(store)).toBe(initial);
        controlled.resolve(0);
        expect((await olderSuccess).ok).toBe(true);
        expect(readyPrepared(store)).toBe(initial);
        expect(store.getState().preparation).toMatchObject({ status: 'error', requestId: 2 });

        const nextEdit = store.getState().editRecipe((recipe) => setLearningRate(recipe, 0.15));
        controlled.resolve(2);
        expect((await nextEdit).ok).toBe(true);
        expect(readyPrepared(store).document.recipe.data.noise)
            .toBe(initial.document.recipe.data.noise);
    });

    it('does not publish an older failure after the newer request succeeds', async () => {
        const initial = preset('xor-hidden').prepared;
        const controlled = controlledPreparation();
        const store = createReadyStore(initial, controlled.prepare);
        const olderFailure = store.getState().replaceDocument({
            ...initial.document,
            recipe: {
                ...initial.document.recipe,
                data: { ...initial.document.recipe.data, sampleCount: 1 },
            },
        });
        const newerSuccess = store.getState().replaceDocument(withNoise(initial, 13));

        controlled.resolve(1);
        expect((await newerSuccess).ok).toBe(true);
        const published = readyPrepared(store);
        controlled.resolve(0);
        expect((await olderFailure).ok).toBe(false);

        expect(readyPrepared(store)).toBe(published);
        expect(readyPrepared(store).document.recipe.data.noise).toBe(13);
        expect(store.getState().preparation).toMatchObject({ status: 'ready', requestId: 2 });
    });

    it('ignores stale failures before and after a newer success without changing the candidate base', async () => {
        const initial = preset('xor-hidden').prepared;
        const controlled = controlledPreparation();
        const store = createReadyStore(initial, controlled.prepare);
        const invalid = {
            ...initial.document,
            recipe: {
                ...initial.document.recipe,
                data: { ...initial.document.recipe.data, sampleCount: 1 },
            },
        };

        const staleFailure = store.getState().replaceDocument(invalid);
        const newerSuccess = store.getState().replaceDocument(withNoise(initial, 11));
        controlled.resolve(0);
        expect((await staleFailure).ok).toBe(false);
        expect(store.getState().preparation.requestId).toBe(2);
        expect(store.getState().preparation.status).toBe('preparing');
        controlled.resolve(1);
        expect((await newerSuccess).ok).toBe(true);

        const edit = store.getState().editRecipe((recipe) => setLearningRate(recipe, 0.2));
        controlled.resolve(2);
        expect((await edit).ok).toBe(true);
        expect(readyPrepared(store).document.recipe.data.noise).toBe(11);
        expect(readyPrepared(store).document.recipe.training.learningRate).toBe(0.2);
    });

    it('preserves two rapid orthogonal edit intents even when the first preparation resolves last', async () => {
        const controlled = controlledPreparation();
        const store = createReadyStore(preset('xor-hidden').prepared, controlled.prepare);

        const first = store.getState().editRecipe((recipe) => setNoise(recipe, 4));
        const second = store.getState().editRecipe((recipe) => setLearningRate(recipe, 0.12));
        expect(controlled.calls).toHaveLength(2);
        controlled.resolve(1);
        expect((await second).ok).toBe(true);
        controlled.resolve(0);
        expect((await first).ok).toBe(true);

        const recipe = readyPrepared(store).document.recipe;
        expect(recipe.data.noise).toBe(4);
        expect(recipe.training.learningRate).toBe(0.12);
    });

    it('resets the candidate base to the last success after a latest failure', async () => {
        const initial = preset('xor-hidden').prepared;
        const store = createReadyStore(initial);
        await store.getState().replaceDocument({
            ...initial.document,
            recipe: {
                ...initial.document.recipe,
                data: { ...initial.document.recipe.data, sampleCount: 1 },
            },
        });

        const edited = await store.getState().editRecipe((recipe) => setNoise(recipe, 9));
        expect(edited.ok).toBe(true);
        expect(readyPrepared(store).document.recipe.data.sampleCount)
            .toBe(initial.document.recipe.data.sampleCount);
        expect(readyPrepared(store).document.recipe.data.noise).toBe(9);
    });

    it('rejects an edit-level issue without incrementing the request ID or preparing', async () => {
        const prepare = vi.fn<Prepare>(prepareExperimentDocument);
        const store = createReadyStore(preset('xor-hidden').prepared, prepare);
        const before = store.getState().preparation.requestId;

        const result = await store.getState().editRecipe(() => ({
            ok: false,
            issue: { kind: 'empty-feature-set' },
        }));

        expect(result).toEqual({ ok: false, issue: { kind: 'empty-feature-set' } });
        expect(store.getState().preparation.requestId).toBe(before);
        expect(prepare).not.toHaveBeenCalled();
    });

    it('preserves view state across exact whole-recipe catalog replacement', async () => {
        const initial = preset('xor-hidden').prepared;
        const store = createReadyStore(initial);
        await store.getState().editView(() => ({ showTestData: true, discretizeOutput: true }));

        const result = await store.getState().applyRecipe(preset('regression-plane'));
        expect(result.ok).toBe(true);
        const current = readyPrepared(store);
        expect(current.document.recipe).toEqual(preset('regression-plane').recipe);
        expect(current.document.view).toEqual({ showTestData: true, discretizeOutput: true });
        expect(current.identities.canonicalRecipeKey)
            .toBe(preset('regression-plane').prepared.identities.canonicalRecipeKey);
        expect(current.identities.recipeFingerprint)
            .toBe(preset('regression-plane').prepared.identities.recipeFingerprint);
    });

    it('rejects a fabricated catalog entry whose recipe/prepared pair does not match', async () => {
        const store = createReadyStore();
        const target = preset('regression-plane');
        const forged = {
            ...target,
            recipe: preset('xor-hidden').recipe,
        };
        const prepare = vi.spyOn(store.getState(), 'replaceDocument');

        const result = await store.getState().applyRecipe(forged);

        expect(result.ok).toBe(false);
        expect(readyPrepared(store).identities.recipeFingerprint)
            .toBe(preset('xor-hidden').prepared.identities.recipeFingerprint);
        expect(prepare).not.toHaveBeenCalled();
    });

    it('rejects a structural catalog copy even when it reuses the exact recipe/prepared pair', async () => {
        const store = createReadyStore();
        const target = preset('regression-plane');
        const copiedEntry = { ...target };
        const before = readyPrepared(store);

        const result = await store.getState().applyRecipe(copiedEntry);

        expect(result).toMatchObject({
            ok: false,
            issues: [{ path: 'recipe' }],
        });
        expect(readyPrepared(store)).toBe(before);
    });

    it('ends all 49 awaited catalog transitions at the exact destination identity', async () => {
        const store = createReadyStore(PREPARED_PRESETS[0].prepared);

        for (const source of PREPARED_PRESETS) {
            for (const target of PREPARED_PRESETS) {
                expect((await store.getState().applyRecipe(source)).ok).toBe(true);
                expect((await store.getState().applyRecipe(target)).ok).toBe(true);
                expect(
                    readyPrepared(store).identities.canonicalRecipeKey,
                    `${source.id} -> ${target.id}`,
                ).toBe(target.prepared.identities.canonicalRecipeKey);
                expect(readyPrepared(store).identities.recipeFingerprint)
                    .toBe(target.prepared.identities.recipeFingerprint);
            }
        }
    });
});

describe('strict initialization and URL state', () => {
    it('preserves a bare trailing fragment that Location.hash normalizes to empty', async () => {
        window.history.replaceState(null, '', '/#');
        expect(window.location.hash).toBe('');
        expect(window.location.href.endsWith('#')).toBe(true);

        const initial = await initializePlaygroundStateFromLocation(window.location);

        expect(initial.initialPrepared).toBeNull();
        expect(initial.incompatibleSource).toEqual({ kind: 'url', rawHash: '#' });
        expect(initial.initialIssues).toEqual(expect.arrayContaining([
            expect.objectContaining({ code: 'legacy-state' }),
        ]));
    });

    it('prepares the V2 default only for an exactly empty hash', async () => {
        const initial = await initializePlaygroundStateFromHash('');

        expect(initial.initialPrepared?.document).toEqual(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(initial.incompatibleSource).toBeNull();
        expect(initial.initialIssues).toEqual([]);
    });

    it('prepares an exact strict V2 URL', async () => {
        const target = preset('three-class-clusters').prepared;
        const initial = await initializePlaygroundStateFromHash(
            encodeExperimentUrl(target.document),
        );

        expect(initial.initialPrepared?.identities.recipeFingerprint)
            .toBe(target.identities.recipeFingerprint);
        expect(initial.incompatibleSource).toBeNull();
    });

    it.each([
        '#d=xor&n=0.2',
        '#v=3&r=AAAA',
        '#v=2&r=not+base64',
        '#',
    ])('preserves incompatible nonempty input %s without publishing a fallback', async (raw) => {
        const initial = await initializePlaygroundStateFromHash(raw);
        const store = createPlaygroundStore(initial);

        expect(initial.initialPrepared).toBeNull();
        expect(store.getState().access.prepared).toBeNull();
        expect(store.getState().incompatibleSource).toEqual({ kind: 'url', rawHash: raw });
        expect(store.getState().preparation.status).toBe('error');
        expect(store.getState().preparation.issues.length).toBeGreaterThan(0);
        expect((await store.getState().applyRecipe(preset('xor-hidden'))).ok).toBe(false);
    });

    it('syncs the current prepared document even when location.hash is stale', async () => {
        const store = createReadyStore();
        await store.getState().applyRecipe(preset('regression-plane'));
        window.history.replaceState(null, '', '#stale');

        const result = store.getState().syncToUrl();

        expect(result).toEqual({
            ok: true,
            value: encodeExperimentUrl(readyPrepared(store).document),
        });
        expect(window.location.hash).toBe(result.ok ? result.value : '');
    });

    it('turns a strict URL load failure into a source-preserving incompatible state', async () => {
        const store = createReadyStore();
        window.history.replaceState(null, '', '#v=3&r=AAAA');

        const result = await store.getState().loadFromUrl();

        expect(result.ok).toBe(false);
        expect(store.getState().access).toEqual({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'url', rawHash: '#v=3&r=AAAA' },
            issues: expect.arrayContaining([expect.objectContaining({
                code: 'unsupported-version',
            })]),
        });
        expect(store.getState().preparation.status).toBe('error');
    });

    it('loads a browser-normalized bare fragment as incompatible instead of as the default', async () => {
        const store = createReadyStore();
        window.history.replaceState(null, '', '/#');
        expect(window.location.hash).toBe('');

        const result = await store.getState().loadFromUrl();

        expect(result.ok).toBe(false);
        expect(store.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'url', rawHash: '#' },
        });
    });

    it('retains the exact failed import File and clears the active document', async () => {
        const store = createReadyStore();
        const file = new File(['{"schemaVersion":1}'], 'legacy.json', {
            type: 'application/json',
        });
        const issues = [{
            code: 'legacy-state' as const,
            path: 'schemaVersion',
            message: 'schema version 1 is incompatible',
        }];

        store.getState().markIncompatible({ kind: 'file', file }, issues);

        expect(store.getState().access).toEqual({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
            issues,
        });
    });

    it('does not rewrite the hash when a newer incompatible request wins Start fresh', async () => {
        const controlled = controlledPreparation();
        const store = createReadyStore(preset('xor-hidden').prepared, controlled.prepare);
        window.history.replaceState(null, '', '#legacy-preserved');

        const recovery = store.getState().startFresh();
        store.getState().markIncompatible(
            { kind: 'url', rawHash: '#newer-incompatible' },
            [{ code: 'unsupported-version', path: 'schemaVersion', message: 'newer request won' }],
        );
        controlled.resolve(0);

        const result = await recovery;
        expect(result.ok).toBe(false);
        expect(window.location.hash).toBe('#legacy-preserved');
        expect(store.getState().access).toMatchObject({
            status: 'incompatible',
            source: { kind: 'url', rawHash: '#newer-incompatible' },
        });
    });

    it('refuses URL serialization when no compatible experiment is active', async () => {
        const initial = await initializePlaygroundStateFromHash('#legacy');
        const store = createPlaygroundStore(initial);
        window.history.replaceState(null, '', '#legacy');

        const result = store.getState().syncToUrl();

        expect(result.ok).toBe(false);
        expect(window.location.hash).toBe('#legacy');
    });
});

describe('V2 dataset regeneration', () => {
    it('uses the prepared dataset contract and every exact generation setting', async () => {
        const store = createReadyStore(preset('three-class-clusters').prepared);
        const recipe = readyPrepared(store).document.recipe;

        store.getState().regenerateData();

        expect(store.getState().dataset).toEqual(generateDatasetV2({
            dataset: recipe.task.dataset,
            sampleCount: recipe.data.sampleCount,
            trainFraction: recipe.data.trainFraction,
            noise: recipe.data.noise,
            seed: recipe.data.seed,
        }));
    });
});
