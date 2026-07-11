/// <reference lib="dom" />
import { describe, expect, it, vi } from 'vitest';
import { canonicalizeJson } from '../canonicalJson.js';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '../constants.js';
import {
    ENGINE_CONTRACT_VERSION,
    FEATURE_REGISTRY_VERSION,
    OBJECTIVE_IMPLEMENTATION_VERSION,
    RECIPE_FINGERPRINT_VERSION,
    SPLIT_ALGORITHM_VERSION,
    canonicalRecipeKey,
    fingerprintDataset,
    fingerprintObjective,
    fingerprintRecipe,
    prepareExperimentDocument,
    validateExperimentDocument,
} from '../experimentSchema.js';
import type { ExperimentDocumentV2, ValidatedExperimentDocumentV2 } from '../types.js';

const DEFAULT_CANONICAL_RECIPE = '{"data":{"noise":0,"sampleCount":300,"seed":42,"trainFraction":0.5},"inputs":{"featureIds":["x","y"]},"model":{"hiddenActivation":"tanh","hiddenLayers":[4,4],"initialization":"xavier","seed":42},"objective":{"dataLoss":{"kind":"binary-cross-entropy-with-logits"},"penalty":{"kind":"none"},"reduction":"mean-per-sample"},"task":{"dataset":"circle","kind":"binary-classification"},"training":{"batchSize":10,"gradientClipping":{"kind":"none"},"learningRate":0.03,"optimizer":{"kind":"sgd"},"schedule":{"kind":"constant"}}}';
const DEFAULT_RECIPE_PAYLOAD = `{"datasetGeneratorVersion":2,"engineContractVersion":1,"experimentSchemaVersion":2,"featureRegistryVersion":1,"fingerprintVersion":1,"objectiveImplementationVersion":1,"recipe":${DEFAULT_CANONICAL_RECIPE}}`;
const DEFAULT_DATASET_PAYLOAD = '{"dataSeed":42,"datasetId":"circle","generatorVersion":2,"noise":0,"sampleCount":300,"splitAlgorithmVersion":1,"trainFraction":0.5}';
const DEFAULT_OBJECTIVE_PAYLOAD = '{"objective":{"dataLoss":{"kind":"binary-cross-entropy-with-logits"},"penalty":{"kind":"none"},"reduction":"mean-per-sample"},"objectiveImplementationVersion":1,"output":{"activation":"sigmoid","size":1,"targetEncoding":{"kind":"scalar","values":[0,1]}},"taskKind":"binary-classification"}';

function cloneDefaultDocument(): ExperimentDocumentV2 {
    return JSON.parse(JSON.stringify(DEFAULT_EXPERIMENT_DOCUMENT)) as ExperimentDocumentV2;
}

function requireValidatedDocument(value: unknown): ValidatedExperimentDocumentV2 {
    const result = validateExperimentDocument(value);
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(JSON.stringify(result.issues));
    return result.value;
}

describe('canonicalizeJson', () => {
    it('pins RFC 8785 scalar, nested, escaping, and number serialization', () => {
        const value = {
            numbers: [Number('333333333.33333329'), 1e30, 4.5, 2e-3, 1e-27, -0],
            string: '€$\u000f\nA\'B"\\/',
            literals: [null, true, false],
        };

        expect(canonicalizeJson(value)).toBe(
            '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0],"string":"€$\\u000f\\nA\'B\\"\\\\/"}',
        );
    });

    it('sorts object keys by UTF-16 code units at every nesting level', () => {
        const value = {
            '😀': 'Emoji: Grinning Face',
            '€': 'Euro Sign',
            '\r': 'Carriage Return',
            'ö': 'Latin Small Letter O With Diaeresis',
            '1': 'One',
            '': 'Control',
            nested: { z: 1, a: 2 },
        };

        expect(canonicalizeJson(value)).toBe(
            '{"\\r":"Carriage Return","1":"One","nested":{"a":2,"z":1},"":"Control","ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign","😀":"Emoji: Grinning Face"}',
        );
    });

    it('uses UTF-16 rather than Unicode code-point key ordering', () => {
        expect(canonicalizeJson({ '\uE000': 'bmp', '\u{10000}': 'supplementary' })).toBe(
            '{"\u{10000}":"supplementary","\uE000":"bmp"}',
        );
    });

    it('pins ECMAScript number boundary serialization', () => {
        expect(canonicalizeJson([
            Number.MIN_VALUE,
            Number.MAX_VALUE,
            9_007_199_254_740_992,
            1e-6,
            1e-7,
            Number('9.999999999999997e22'),
        ])).toBe(
            '[5e-324,1.7976931348623157e+308,9007199254740992,0.000001,1e-7,9.999999999999997e+22]',
        );
    });

    it('is invariant to insertion order and supports null-prototype records', () => {
        const first = { z: 3, a: { d: 4, b: 2 } };
        const second = { a: { b: 2, d: 4 }, z: 3 };
        const nullPrototype = Object.assign(Object.create(null) as Record<string, unknown>, {
            z: 3,
            a: { d: 4, b: 2 },
        });

        expect(canonicalizeJson(first)).toBe('{"a":{"b":2,"d":4},"z":3}');
        expect(canonicalizeJson(second)).toBe(canonicalizeJson(first));
        expect(canonicalizeJson(nullPrototype)).toBe(canonicalizeJson(first));
    });

    it.each([
        Number.NaN,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        undefined,
        () => undefined,
        Symbol('unsupported'),
        1n,
        new Date(0),
        new Number(1),
        new String('value'),
    ])('rejects non-I-JSON value %#', (value) => {
        expect(() => canonicalizeJson(value)).toThrow();
    });

    it.each(['\ud800', '\udc00', `prefix\ud800suffix`])(
        'rejects lone surrogate string %#',
        (value) => {
            expect(() => canonicalizeJson(value)).toThrow(/surrogate|Unicode/iu);
        },
    );

    it('rejects sparse arrays, inherited indexes, and extra properties', () => {
        const sparse = new Array<unknown>(2);
        sparse[1] = true;

        const inherited = new Array<unknown>(1);
        const inheritedPrototype = Object.create(Array.prototype) as unknown[];
        inheritedPrototype[0] = 'inherited';
        Object.setPrototypeOf(inherited, inheritedPrototype);

        const extra = [1, 2] as unknown[] & { note?: string };
        extra.note = 'hidden from JSON arrays';
        const exotic = [1, 2];
        Object.setPrototypeOf(exotic, { inherited: true });

        expect(() => canonicalizeJson(sparse)).toThrow(/dense|index/iu);
        expect(() => canonicalizeJson(inherited)).toThrow(/array|prototype|index/iu);
        expect(() => canonicalizeJson(extra)).toThrow(/property|array/iu);
        expect(() => canonicalizeJson(exotic)).toThrow(/array|prototype/iu);
    });

    it('rejects hidden, symbol, accessor, and inherited object state', () => {
        const hidden = { visible: true };
        Object.defineProperty(hidden, 'secret', { value: 1, enumerable: false });

        const symbol = { visible: true } as Record<PropertyKey, unknown>;
        symbol[Symbol('secret')] = 1;

        let getterCalls = 0;
        const accessor = {} as Record<string, unknown>;
        Object.defineProperty(accessor, 'value', {
            enumerable: true,
            get: () => {
                getterCalls += 1;
                return 1;
            },
        });

        const inherited = Object.create({ inherited: true }) as Record<string, unknown>;
        inherited.visible = true;

        expect(() => canonicalizeJson(hidden)).toThrow(/enumerable|property/iu);
        expect(() => canonicalizeJson(symbol)).toThrow(/symbol/iu);
        expect(() => canonicalizeJson(accessor)).toThrow(/accessor|data property/iu);
        expect(getterCalls).toBe(0);
        expect(() => canonicalizeJson(inherited)).toThrow(/plain|prototype/iu);
    });

    it('rejects cycles but permits repeated acyclic references', () => {
        const cyclic: { self?: unknown } = {};
        cyclic.self = cyclic;
        const shared = { value: 1 };

        expect(() => canonicalizeJson(cyclic)).toThrow(/cycl/iu);
        expect(canonicalizeJson([shared, shared])).toBe(
            '[{"value":1},{"value":1}]',
        );
    });
});

describe('versioned experiment identities', () => {
    it('pins all identity version constants', () => {
        expect(RECIPE_FINGERPRINT_VERSION).toBe(1);
        expect(ENGINE_CONTRACT_VERSION).toBe(1);
        expect(FEATURE_REGISTRY_VERSION).toBe(1);
        expect(OBJECTIVE_IMPLEMENTATION_VERSION).toBe(1);
        expect(SPLIT_ALGORITHM_VERSION).toBe(1);
    });

    it('pins the exact canonical recipe and literal SHA-256 fixtures', async () => {
        expect(canonicalRecipeKey(DEFAULT_EXPERIMENT_DOCUMENT.recipe)).toBe(
            DEFAULT_CANONICAL_RECIPE,
        );
        await expect(fingerprintRecipe(DEFAULT_EXPERIMENT_DOCUMENT.recipe)).resolves.toBe(
            'r2.1.bRgtD7xOLkRdoP2wsjgY5XUAgSl_7j-apY6W-eWJ5pQ',
        );
        await expect(fingerprintDataset(DEFAULT_EXPERIMENT_DOCUMENT.recipe)).resolves.toBe(
            'd2.1.18nWM8SZXGz04ZWxK-ZXCGkz4YLLJlWGS9ppvqdsn3M',
        );
        await expect(fingerprintObjective(DEFAULT_EXPERIMENT_DOCUMENT.recipe)).resolves.toBe(
            'o2.1.43D3QyDGZQiw7G8i7o9tbgIe6SyYx3GDP54qLRbkL9I',
        );
    });

    it('passes exact canonical payload bytes to all three digests before awaiting any', async () => {
        const payloads: string[] = [];
        const releases: Array<(value: ArrayBuffer) => void> = [];
        const digestSpy = vi.spyOn(globalThis.crypto.subtle, 'digest').mockImplementation(
            (_algorithm, data) => {
                const bytes = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                    : new Uint8Array(data);
                payloads.push(new TextDecoder().decode(bytes));
                return new Promise<ArrayBuffer>((resolve) => releases.push(resolve));
            },
        );

        try {
            const pending = prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
            expect(payloads).toEqual([
                DEFAULT_RECIPE_PAYLOAD,
                DEFAULT_DATASET_PAYLOAD,
                DEFAULT_OBJECTIVE_PAYLOAD,
            ]);
            expect(releases).toHaveLength(3);

            for (const release of releases) release(new Uint8Array(32).buffer);
            const result = await pending;
            expect(result.ok).toBe(true);
            if (result.ok) {
                expect(result.value.identities).toEqual({
                    canonicalRecipeKey: DEFAULT_CANONICAL_RECIPE,
                    recipeFingerprint: `r2.1.${'A'.repeat(43)}`,
                    datasetKey: `d2.1.${'A'.repeat(43)}`,
                    objectiveKey: `o2.1.${'A'.repeat(43)}`,
                });
            }
        } finally {
            digestSpy.mockRestore();
        }
    });

    it('is insertion-order invariant and changes only identities affected by an edit', async () => {
        const baselineResult = await prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(baselineResult.ok).toBe(true);
        if (!baselineResult.ok) return;

        const baseRecipe = cloneDefaultDocument().recipe;
        const reordered = requireValidatedDocument({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: {
                training: baseRecipe.training,
                task: baseRecipe.task,
                objective: baseRecipe.objective,
                model: baseRecipe.model,
                inputs: baseRecipe.inputs,
                data: baseRecipe.data,
            },
            view: { discretizeOutput: false, showTestData: false },
        });
        const reorderedResult = await prepareExperimentDocument(reordered);
        expect(reorderedResult.ok).toBe(true);
        if (!reorderedResult.ok) return;
        expect(reorderedResult.value.identities).toEqual(baselineResult.value.identities);

        const datasetEdit = cloneDefaultDocument();
        datasetEdit.recipe.data.seed += 1;
        const datasetResult = await prepareExperimentDocument(datasetEdit);
        expect(datasetResult.ok).toBe(true);
        if (datasetResult.ok) {
            expect(datasetResult.value.identities.recipeFingerprint).not.toBe(
                baselineResult.value.identities.recipeFingerprint,
            );
            expect(datasetResult.value.identities.datasetKey).not.toBe(
                baselineResult.value.identities.datasetKey,
            );
            expect(datasetResult.value.identities.objectiveKey).toBe(
                baselineResult.value.identities.objectiveKey,
            );
        }

        const objectiveEdit = cloneDefaultDocument();
        objectiveEdit.recipe.objective.penalty = {
            kind: 'l2',
            coefficient: 0.01,
            applyTo: 'weights',
        };
        const objectiveResult = await prepareExperimentDocument(objectiveEdit);
        expect(objectiveResult.ok).toBe(true);
        if (objectiveResult.ok) {
            expect(objectiveResult.value.identities.recipeFingerprint).not.toBe(
                baselineResult.value.identities.recipeFingerprint,
            );
            expect(objectiveResult.value.identities.datasetKey).toBe(
                baselineResult.value.identities.datasetKey,
            );
            expect(objectiveResult.value.identities.objectiveKey).not.toBe(
                baselineResult.value.identities.objectiveKey,
            );
        }
    });

    it('returns structured failures for hidden state and hashing failure', async () => {
        const hidden = cloneDefaultDocument();
        Object.defineProperty(hidden.recipe, 'hidden', { value: true, enumerable: false });
        await expect(prepareExperimentDocument(hidden)).resolves.toMatchObject({
            ok: false,
            issues: [expect.objectContaining({
                code: 'invalid-field',
                path: 'recipe.hidden',
            })],
        });

        const digestSpy = vi.spyOn(globalThis.crypto.subtle, 'digest').mockRejectedValue(
            new Error('digest unavailable'),
        );
        try {
            await expect(
                prepareExperimentDocument(DEFAULT_EXPERIMENT_DOCUMENT),
            ).resolves.toMatchObject({
                ok: false,
                issues: [expect.objectContaining({ code: 'invalid-field', path: '$' })],
            });
        } finally {
            digestSpy.mockRestore();
        }
    });

    it('snapshots Proxy-backed input once so document, compiler, and identities cannot diverge', async () => {
        const candidate = cloneDefaultDocument();
        let seedDescriptorReads = 0;
        candidate.recipe.data = new Proxy(candidate.recipe.data, {
            getOwnPropertyDescriptor(target, key) {
                const descriptor = Reflect.getOwnPropertyDescriptor(target, key);
                if (key === 'seed' && descriptor && 'value' in descriptor) {
                    seedDescriptorReads += 1;
                    return { ...descriptor, value: seedDescriptorReads === 1 ? 42 : 43 };
                }
                return descriptor;
            },
        });

        const result = await prepareExperimentDocument(candidate);
        expect(result.ok).toBe(true);
        if (result.ok) {
            expect(result.value.document.recipe.data.seed).toBe(
                result.value.compiled.data.seed,
            );
            const firstKey = canonicalRecipeKey(result.value.document.recipe);
            const secondKey = canonicalRecipeKey(result.value.document.recipe);
            expect(firstKey).toBe(secondKey);
            expect(Object.isFrozen(result.value.document)).toBe(true);
        }
    });
});
