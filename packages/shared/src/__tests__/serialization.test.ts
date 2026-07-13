/// <reference lib="dom" />

import { describe, expect, it } from 'vitest';
import * as serialization from '../serialization.js';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    MAX_EXPERIMENT_JSON_BYTES,
    canonicalizeJson,
    decodeExperimentJson,
    decodeExperimentUrl,
    encodeExperimentJson,
    encodeExperimentUrl,
    validateExperimentDocument,
} from '../index.js';
import type {
    ExperimentDocumentV2,
    ValidatedExperimentDocumentV2,
} from '../types.js';

function asValidatedForEncoding(
    document: ExperimentDocumentV2,
): ValidatedExperimentDocumentV2 {
    return document as ValidatedExperimentDocumentV2;
}

function requireValidatedForCodec(value: unknown): ValidatedExperimentDocumentV2 {
    const result = validateExperimentDocument(value);
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(JSON.stringify(result.issues));
    return result.value;
}

function encodeTestBase64Url(value: string): string {
    const bytes = new TextEncoder().encode(value);
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return globalThis.btoa(binary)
        .replaceAll('+', '-')
        .replaceAll('/', '_')
        .replace(/=+$/u, '');
}

describe('version-2 experiment URL and JSON codecs', () => {
    it('does not expose the removed unversioned codec or normalizer API', () => {
        for (const legacyExport of [
            'encodeUrlState',
            'decodeUrlState',
            'exportConfigJson',
            'importConfigJson',
            'validateImportedConfig',
            'normalizeAppConfig',
        ]) {
            expect(serialization).not.toHaveProperty(legacyExport);
        }
    });

    it('encodes canonical JSON and round-trips the exact validated document', () => {
        const encoded = encodeExperimentUrl(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(encoded).toMatch(/^#v=2&r=[A-Za-z0-9_-]+$/u);
        const payload = new URLSearchParams(encoded.slice(1)).get('r');
        expect(payload).not.toBeNull();
        expect(payload).not.toContain('=');
        const bytes = Uint8Array.from(
            globalThis.atob(payload ?? ''),
            (character) => character.charCodeAt(0),
        );
        expect(new TextDecoder('utf-8', { fatal: true }).decode(bytes)).toBe(
            canonicalizeJson(DEFAULT_EXPERIMENT_DOCUMENT),
        );
        expect(decodeExperimentUrl(encoded)).toEqual({
            ok: true,
            value: DEFAULT_EXPERIMENT_DOCUMENT,
        });
    });

    it('loads the immutable V2 default only for an actually empty hash', () => {
        const first = decodeExperimentUrl('');
        expect(first).toEqual({ ok: true, value: DEFAULT_EXPERIMENT_DOCUMENT });
        expect(Object.isFrozen(DEFAULT_EXPERIMENT_DOCUMENT)).toBe(true);
        expect(Object.isFrozen(DEFAULT_EXPERIMENT_DOCUMENT.recipe.data)).toBe(true);
        expect(decodeExperimentUrl('#')).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'legacy-state' })],
        });
        expect(decodeExperimentUrl(' ')).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'invalid-field' })],
        });
    });

    it.each([
        ['missing hash marker', 'v=2&r=value', 'invalid-field'],
        ['unversioned legacy', '#d=circle&lr=0.03', 'legacy-state'],
        ['version 1', '#v=1&r=legacy', 'legacy-state'],
        ['future version', '#v=3&r=future', 'unsupported-version'],
        ['malformed version', '#v=two&r=value', 'invalid-field'],
        ['missing payload', '#v=2', 'missing-field'],
        ['duplicate version', '#v=2&v=2&r=value', 'invalid-field'],
        ['duplicate payload', '#v=2&r=value&r=value', 'invalid-field'],
        ['unknown field', '#v=2&r=value&extra=1', 'unknown-field'],
        ['invalid alphabet', '#v=2&r=not+base64', 'invalid-field'],
        ['invalid length', '#v=2&r=A', 'invalid-field'],
        ['invalid UTF-8', '#v=2&r=_w', 'invalid-field'],
        ['invalid JSON', '#v=2&r=bm90LWpzb24', 'invalid-field'],
    ])('rejects %s without falling back', (_label, hash, code) => {
        expect(decodeExperimentUrl(hash)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code })],
        });
    });

    it('rejects incompatible and non-canonical V2 URL payloads', () => {
        const futureDocument = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            schemaVersion: 3,
        } as unknown as ExperimentDocumentV2;
        expect(decodeExperimentUrl(
            encodeExperimentUrl(asValidatedForEncoding(futureDocument)),
        )).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({
                code: 'unsupported-version',
                path: 'schemaVersion',
            })],
        });

        const canonicalUrl = encodeExperimentUrl(DEFAULT_EXPERIMENT_DOCUMENT);
        const payload = new URLSearchParams(canonicalUrl.slice(1)).get('r');
        expect(decodeExperimentUrl(`#r=${payload ?? ''}&v=2`)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'invalid-field', path: '$' })],
        });
        const prettyPayload = encodeTestBase64Url(
            JSON.stringify(DEFAULT_EXPERIMENT_DOCUMENT, null, 2),
        );
        expect(decodeExperimentUrl(`#v=2&r=${prettyPayload}`)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'invalid-field', path: 'r' })],
        });
    });

    it('exports deterministic pretty JSON and strictly imports the exact envelope', () => {
        const json = encodeExperimentJson(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(json.startsWith('{\n  "kind": "nn-playground-experiment",')).toBe(true);
        expect(json.endsWith('\n}')).toBe(true);
        expect(JSON.parse(json)).toEqual(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(decodeExperimentJson(json)).toEqual({
            ok: true,
            value: DEFAULT_EXPERIMENT_DOCUMENT,
        });
    });

    it('round-trips complete multiclass/Adam/step and regression/Huber/cosine branches', () => {
        const multiclass = requireValidatedForCodec({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: {
                data: { sampleCount: 500, trainFraction: 0.7, noise: 20, seed: 123 },
                inputs: { featureIds: ['x', 'y', 'xy'] },
                model: {
                    hiddenLayers: [8, 6],
                    hiddenActivation: 'relu',
                    initialization: 'he',
                    seed: 456,
                },
                training: {
                    batchSize: 32,
                    learningRate: 0.01,
                    schedule: { kind: 'step', interval: 25, gamma: 0.5 },
                    optimizer: { kind: 'adam', beta1: 0.8, beta2: 0.95, epsilon: 1e-8 },
                    gradientClipping: {
                        kind: 'global-norm',
                        maximumNorm: 5,
                        scope: 'total-objective-gradient',
                    },
                },
                task: { kind: 'multiclass-classification', dataset: 'three-class-clusters' },
                objective: {
                    dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
                    penalty: { kind: 'l2', coefficient: 0.01, applyTo: 'weights' },
                    reduction: 'mean-per-sample',
                },
            },
            view: { showTestData: true, discretizeOutput: true },
        });
        const regression = requireValidatedForCodec({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: {
                data: { sampleCount: 200, trainFraction: 0.6, noise: 10, seed: 1 },
                inputs: { featureIds: ['x', 'y', 'xSquared'] },
                model: {
                    hiddenLayers: [5],
                    hiddenActivation: 'tanh',
                    initialization: 'xavier',
                    seed: 2,
                },
                training: {
                    batchSize: 20,
                    learningRate: 0.02,
                    schedule: { kind: 'cosine', totalSteps: 1_000, minimumRate: 0.001 },
                    optimizer: { kind: 'sgd-momentum', momentum: 0.8 },
                    gradientClipping: { kind: 'none' },
                },
                task: { kind: 'regression', dataset: 'reg-gauss' },
                objective: {
                    dataLoss: { kind: 'huber', delta: 1.5 },
                    penalty: { kind: 'l1', coefficient: 0.001, applyTo: 'weights' },
                    reduction: 'mean-per-sample',
                },
            },
            view: { showTestData: false, discretizeOutput: false },
        });

        for (const document of [multiclass, regression]) {
            expect(decodeExperimentUrl(encodeExperimentUrl(document))).toEqual({
                ok: true,
                value: document,
            });
            expect(decodeExperimentJson(encodeExperimentJson(document))).toEqual({
                ok: true,
                value: document,
            });
        }
    });

    it.each([
        ['malformed', '{', 'invalid-field'],
        ['non-document', 'null', 'invalid-field'],
        ['random object', '{"unexpected":true}', 'unknown-field'],
        ['unversioned', JSON.stringify({ kind: 'nn-playground-experiment' }), 'legacy-state'],
        [
            'version 1',
            JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 1 }),
            'legacy-state',
        ],
        [
            'future version',
            JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 3 }),
            'unsupported-version',
        ],
        [
            'unknown field',
            JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, typo: true }),
            'unknown-field',
        ],
    ])('rejects %s JSON explicitly', (_label, json, code) => {
        const result = decodeExperimentJson(json);
        expect(result.ok).toBe(false);
        if (!result.ok) {
            expect(result.issues).toEqual(
                expect.arrayContaining([expect.objectContaining({ code })]),
            );
        }
    });

    it('enforces size limits and rejects duplicate JSON members at every object depth', () => {
        expect(decodeExperimentJson(' '.repeat(MAX_EXPERIMENT_JSON_BYTES + 1))).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'resource-limit', path: '$' })],
        });

        const json = encodeExperimentJson(DEFAULT_EXPERIMENT_DOCUMENT);
        for (const duplicate of [
            json.replace(
                '"schemaVersion": 2',
                '"schemaVersion": 1,\n  "schemaVersion": 2',
            ),
            json.replace('"seed": 42', '"seed": 41,\n      "seed": 42'),
        ]) {
            expect(decodeExperimentJson(duplicate)).toMatchObject({
                ok: false,
                issues: [expect.objectContaining({ code: 'invalid-field', path: '$' })],
            });
        }

        const maximumPayloadCharacters = Math.ceil(MAX_EXPERIMENT_JSON_BYTES * 4 / 3);
        expect(
            decodeExperimentUrl(`#v=2&r=${'A'.repeat(maximumPayloadCharacters + 1)}`),
        ).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'resource-limit', path: 'r' })],
        });
    });
});
