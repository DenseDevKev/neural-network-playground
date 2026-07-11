/// <reference lib="dom" />
import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    MAX_TRAIN_TEST_RATIO,
    MAX_EXPERIMENT_JSON_BYTES,
    DEFAULT_EXPERIMENT_DOCUMENT,
    canonicalizeJson,
    decodeExperimentJson,
    decodeExperimentUrl,
    decodeUrlState,
    encodeExperimentJson,
    encodeExperimentUrl,
    validateImportedConfig,
    normalizeAppConfig,
    encodeUrlState,
    exportConfigJson,
    importConfigJson
} from '../index.js';
import type {
    AppConfig,
    ExperimentDocumentV2,
    ValidatedExperimentDocumentV2,
} from '../types.js';

function asValidatedForEncoding(
    document: ExperimentDocumentV2,
): ValidatedExperimentDocumentV2 {
    return document as ValidatedExperimentDocumentV2;
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
    it('encodes the exact canonical document and round-trips every field', () => {
        const encoded = encodeExperimentUrl(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(encoded).toMatch(/^#v=2&r=[A-Za-z0-9_-]+$/u);

        const encodedPayload = new URLSearchParams(encoded.slice(1)).get('r');
        expect(encodedPayload).not.toBeNull();
        expect(encodedPayload).not.toContain('=');
        const bytes = Uint8Array.from(
            globalThis.atob(encodedPayload ?? ''),
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

    it('loads the validated V2 default only for an actually empty hash', () => {
        expect(decodeExperimentUrl('')).toEqual({
            ok: true,
            value: DEFAULT_EXPERIMENT_DOCUMENT,
        });
        expect(decodeExperimentUrl('#')).toEqual({
            ok: true,
            value: DEFAULT_EXPERIMENT_DOCUMENT,
        });
        expect(decodeExperimentUrl(' ')).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'legacy-state' })],
        });
    });

    it.each([
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

    it('rejects a V2 payload whose document is incompatible', () => {
        const futureDocument = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            schemaVersion: 3,
        } as unknown as ExperimentDocumentV2;
        const futureUrl = encodeExperimentUrl(asValidatedForEncoding(futureDocument));
        expect(decodeExperimentUrl(futureUrl)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({
                code: 'unsupported-version',
                path: 'schemaVersion',
            })],
        });

        const unknownFieldDocument = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            misspelledSetting: true,
        } as unknown as ExperimentDocumentV2;
        const unknownUrl = encodeExperimentUrl(asValidatedForEncoding(unknownFieldDocument));
        expect(decodeExperimentUrl(unknownUrl)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'unknown-field' })],
        });
    });

    it('rejects non-canonical URL syntax and non-canonical document JSON', () => {
        const canonicalUrl = encodeExperimentUrl(DEFAULT_EXPERIMENT_DOCUMENT);
        const payload = new URLSearchParams(canonicalUrl.slice(1)).get('r');
        expect(payload).not.toBeNull();

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

    it('rejects oversized JSON before parsing', () => {
        const oversized = ' '.repeat(MAX_EXPERIMENT_JSON_BYTES + 1);
        expect(decodeExperimentJson(oversized)).toMatchObject({
            ok: false,
            issues: [expect.objectContaining({ code: 'resource-limit', path: '$' })],
        });
    });
});

const validConfig: AppConfig = {
    data: { ...DEFAULT_DATA },
    network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
    training: { ...DEFAULT_TRAINING },
    features: { ...DEFAULT_FEATURES },
    ui: { showTestData: false, discretizeOutput: false },
};

const multiclassConfig: AppConfig = {
    ...validConfig,
    data: {
        ...validConfig.data,
        dataset: 'three-class-clusters' as unknown as AppConfig['data']['dataset'],
        problemType: 'classification',
    },
    network: {
        ...validConfig.network,
        outputSize: 3,
        outputActivation: 'softmax',
    },
    training: {
        ...validConfig.training,
        lossType: 'categoricalCrossEntropy',
    },
};

describe('URL State Serialization', () => {
    it('round-trips a valid configuration', () => {
        const encoded = encodeUrlState(validConfig);
        const decoded = decodeUrlState(encoded);

        // Because encodeUrlState does not serialize the 'noise' default or some other specifics,
        // we might just need to check if the objects match deeply.
        expect(decoded).toEqual(validConfig);
    });

    it('decodes an empty hash to defaults', () => {
        const decoded = decodeUrlState('');

        expect(decoded.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(decoded.network.hiddenLayers).toEqual(DEFAULT_NETWORK.hiddenLayers);
        expect(decoded.training.learningRate).toBe(DEFAULT_TRAINING.learningRate);
    });

    it('round-trips advanced training and network hyperparameters', () => {
        const advanced: AppConfig = {
            ...validConfig,
            network: {
                ...validConfig.network,
                outputActivation: 'linear',
                weightInit: 'he',
            },
            training: {
                ...validConfig.training,
                lossType: 'huber',
                optimizer: 'adam',
                momentum: 0.72,
                gradientClip: 0.25,
                adamBeta1: 0.82,
                adamBeta2: 0.97,
                huberDelta: 0.4,
                lrSchedule: { type: 'step', stepSize: 25, gamma: 0.6 },
            },
        };

        const decoded = decodeUrlState(encodeUrlState(advanced));

        expect(decoded.network.outputActivation).toBe('linear');
        expect(decoded.network.weightInit).toBe('he');
        expect(decoded.training.momentum).toBe(0.72);
        expect(decoded.training.gradientClip).toBe(0.25);
        expect(decoded.training.adamBeta1).toBe(0.82);
        expect(decoded.training.adamBeta2).toBe(0.97);
        expect(decoded.training.huberDelta).toBe(0.4);
        expect(decoded.training.lrSchedule).toEqual({ type: 'step', stepSize: 25, gamma: 0.6 });
    });

    it('round-trips a bounded multiclass config only when the migration contract is enabled', () => {
        const encoded = encodeUrlState(multiclassConfig, { allowMulticlass: true });

        expect(encoded).toContain('os=3');
        expect(encoded).toContain('d=three-class-clusters');
        expect(decodeUrlState(encoded, { allowMulticlass: true })).toEqual(multiclassConfig);

        const publicDefault = decodeUrlState(encoded);
        expect(publicDefault.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(publicDefault.network.outputSize).toBe(1);
        expect(publicDefault.network.outputActivation).not.toBe('softmax');
        expect(publicDefault.training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it('falls back to scalar defaults for dataset-only multiclass URLs', () => {
        const publicDefault = decodeUrlState('d=three-class-clusters');
        const optInMissingPairing = decodeUrlState('d=three-class-clusters', { allowMulticlass: true });

        expect(publicDefault.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(publicDefault.network.outputSize).toBe(1);
        expect(optInMissingPairing.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(optInMissingPairing.network.outputSize).toBe(1);
    });

    it('does not encode unsupported output sizes through the multiclass migration option', () => {
        const encoded = encodeUrlState({
            ...multiclassConfig,
            network: {
                ...multiclassConfig.network,
                outputSize: 4,
            },
        }, { allowMulticlass: true });

        expect(encoded).not.toContain('os=4');
        expect(decodeUrlState(encoded, { allowMulticlass: true }).network.outputSize).toBe(1);
    });
});

describe('JSON Serialization', () => {
    it('round-trips a valid configuration', () => {
        const json = exportConfigJson(validConfig);
        const imported = importConfigJson(json);

        expect(imported).toEqual(validConfig);
    });

    it('returns null for invalid JSON string', () => {
        const imported = importConfigJson('invalid json');
        expect(imported).toBeNull();
    });

    it('returns null for structurally invalid JSON object', () => {
        const imported = importConfigJson(JSON.stringify({ notAConfig: true }));
        expect(imported).toBeNull();
    });
});

describe('validateImportedConfig', () => {
    it('accepts a valid configuration and normalizes input size', () => {
        const result = validateImportedConfig(validConfig);

        expect(result.error).toBeNull();
        expect(result.config).not.toBeNull();
        expect(result.config?.network.inputSize).toBe(2);
    });

    it('rejects unsupported activation functions', () => {
        const result = validateImportedConfig({
            ...validConfig,
            network: {
                ...validConfig.network,
                activation: 'magic',
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Configuration contains an unsupported activation function.');
    });

    it('rejects softmax as a hidden activation even while reserving it for multiclass outputs', () => {
        const result = validateImportedConfig({
            ...validConfig,
            network: {
                ...validConfig.network,
                activation: 'softmax',
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Configuration contains an unsupported activation function.');
    });

    it('rejects the multiclass dataset without the approved multiclass pairing', () => {
        const result = validateImportedConfig({
            ...validConfig,
            data: {
                ...validConfig.data,
                dataset: 'three-class-clusters' as unknown as AppConfig['data']['dataset'],
            },
        }, { allowMulticlass: true });

        expect(result.config).toBeNull();
        expect(result.error).toMatch(/multiclass/i);
    });

    it('rejects invalid learning-rate ranges', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                learningRate: 0,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Learning rate must be greater than 0 and at most 10.');
    });

    it('rejects configurations with no active features', () => {
        const result = validateImportedConfig({
            ...validConfig,
            features: {
                x: false,
                y: false,
                xSquared: false,
                ySquared: false,
                xy: false,
                sinX: false,
                sinY: false,
                cosX: false,
                cosY: false,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('At least one input feature must be enabled.');
    });
});

describe('decodeUrlState', () => {
    it('falls back to defaults for unsupported enum-like URL values', () => {
        const decoded = decodeUrlState('d=invalid&pt=broken&a=magic&oa=wild&wi=bad&l=nope&o=fast&rg=ghost');

        expect(decoded.data.dataset).toBe(DEFAULT_DATA.dataset);
        expect(decoded.data.problemType).toBe(DEFAULT_DATA.problemType);
        expect(decoded.network.activation).toBe(DEFAULT_NETWORK.activation);
        expect(decoded.network.outputActivation).toBe(DEFAULT_NETWORK.outputActivation);
        expect(decoded.network.weightInit).toBe(DEFAULT_NETWORK.weightInit);
        expect(decoded.training.lossType).toBe(DEFAULT_TRAINING.lossType);
        expect(decoded.training.optimizer).toBe(DEFAULT_TRAINING.optimizer);
        expect(decoded.training.regularization).toBe(DEFAULT_TRAINING.regularization);
    });

    it('restores default features when the URL disables every feature', () => {
        const decoded = decodeUrlState('f=000000000');

        expect(decoded.features).toEqual(DEFAULT_FEATURES);
        expect(decoded.network.inputSize).toBe(2);
    });

    it('normalizes incompatible loss/output pairs from URL state', () => {
        const decoded = decodeUrlState('l=mse&oa=sigmoid');

        expect(decoded.training.lossType).toBe('mse');
        expect(decoded.network.outputActivation).toBe('linear');
    });

    it('keeps multiclass URL experiments gated to the current runtime defaults', () => {
        const decoded = decodeUrlState('l=categoricalCrossEntropy&oa=softmax');

        expect(decoded.training.lossType).toBe(DEFAULT_TRAINING.lossType);
        expect(decoded.network.outputActivation).toBe(DEFAULT_NETWORK.outputActivation);
        expect(decoded.network.outputSize).toBe(1);
    });

    it('keeps malicious numeric URL values within safe runtime limits', () => {
        const decoded = decodeUrlState(
            'ns=999999999&bs=0&lr=Infinity&r=2&n=-10&s=NaN&ws=Infinity&hl=9999,0,-3,nope',
        );

        expect(decoded.data.numSamples).toBe(10000);
        expect(decoded.training.batchSize).toBe(DEFAULT_TRAINING.batchSize);
        expect(decoded.training.learningRate).toBe(DEFAULT_TRAINING.learningRate);
        expect(decoded.data.trainTestRatio).toBe(MAX_TRAIN_TEST_RATIO);
        expect(decoded.data.noise).toBe(DEFAULT_DATA.noise);
        expect(decoded.data.seed).toBe(DEFAULT_DATA.seed);
        expect(decoded.network.seed).toBe(DEFAULT_NETWORK.seed);
        expect(decoded.network.hiddenLayers).toEqual([32]);
    });

    it('sanitizes invalid learning-rate schedule URL values instead of dropping old URLs', () => {
        const step = decodeUrlState('lrs=step&lrss=0&lrsg=2');
        expect(step.training.lrSchedule).toEqual({
            type: 'step',
            stepSize: 1,
            gamma: 0.5,
        });

        const cosine = decodeUrlState('lr=0.01&lrs=cosine&lrst=0&lrsm=0.03');
        expect(cosine.training.lrSchedule).toEqual({
            type: 'cosine',
            totalSteps: 1,
            minLr: 0.01,
        });
    });
});

describe('compatibility normalization', () => {
    it('normalizes imported configs with incompatible loss/output pairs', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                lossType: 'mse',
            },
            network: {
                ...validConfig.network,
                outputActivation: 'sigmoid',
            },
        });

        expect(result.error).toBeNull();
        expect(result.config?.training.lossType).toBe('mse');
        expect(result.config?.network.outputActivation).toBe('linear');
    });

    it('recognizes but rejects multiclass imports until worker/runtime support is approved', () => {
        const result = validateImportedConfig({
            ...validConfig,
            data: {
                ...validConfig.data,
                problemType: 'classification',
            },
            network: {
                ...validConfig.network,
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...validConfig.training,
                lossType: 'categoricalCrossEntropy',
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Multiclass configurations are not runtime-enabled yet.');
    });

    it('accepts the bounded three-class config when the migration contract is enabled', () => {
        const result = validateImportedConfig(multiclassConfig, { allowMulticlass: true });

        expect(result.error).toBeNull();
        expect(result.config).toEqual(multiclassConfig);
    });

    it('keeps the multiclass migration contract tied to the approved three-class dataset', () => {
        const result = validateImportedConfig({
            ...multiclassConfig,
            data: {
                ...multiclassConfig.data,
                dataset: 'circle',
            },
        }, { allowMulticlass: true });

        expect(result.config).toBeNull();
        expect(result.error).toMatch(/multiclass|dataset|single-output/i);
    });

    it.each([
        [
            'unsupported class count',
            { network: { outputSize: 4, outputActivation: 'softmax' }, training: { lossType: 'categoricalCrossEntropy' } },
        ],
        [
            'softmax without categorical loss',
            { network: { outputSize: 3, outputActivation: 'softmax' }, training: { lossType: 'crossEntropy' } },
        ],
        [
            'categorical loss without softmax',
            { network: { outputSize: 3, outputActivation: 'sigmoid' }, training: { lossType: 'categoricalCrossEntropy' } },
        ],
        [
            'regression multiclass config',
            {
                data: { problemType: 'regression' },
                network: { outputSize: 3, outputActivation: 'softmax' },
                training: { lossType: 'categoricalCrossEntropy' },
            },
        ],
    ])('rejects multiclass migration configs with %s', (_label, overrides) => {
        const result = validateImportedConfig({
            ...multiclassConfig,
            data: {
                ...multiclassConfig.data,
                ...overrides.data,
            },
            network: {
                ...multiclassConfig.network,
                ...overrides.network,
            },
            training: {
                ...multiclassConfig.training,
                ...overrides.training,
            },
        }, { allowMulticlass: true });

        expect(result.config).toBeNull();
        expect(result.error).toMatch(/multiclass|single-output/i);
    });

    it('falls back to scalar defaults for unsupported multiclass URL contracts', () => {
        const decoded = decodeUrlState('os=4&pt=classification&l=categoricalCrossEntropy&oa=softmax', {
            allowMulticlass: true,
        });

        expect(decoded.network.outputSize).toBe(1);
        expect(decoded.network.outputActivation).not.toBe('softmax');
        expect(decoded.training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it.each([
        ['missing softmax activation', 'os=3&pt=classification&l=categoricalCrossEntropy'],
        ['missing categorical loss', 'os=3&pt=classification&oa=softmax'],
    ])('falls back to scalar defaults for partial multiclass URL contracts: %s', (_label, hash) => {
        const decoded = decodeUrlState(hash, { allowMulticlass: true });

        expect(decoded.network.outputSize).toBe(1);
        expect(decoded.network.outputActivation).not.toBe('softmax');
        expect(decoded.training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it('strictly rejects unsafe imported numeric ranges through the shared normalizer', () => {
        const result = normalizeAppConfig({
            ...validConfig,
            data: {
                ...validConfig.data,
                numSamples: 10001,
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Sample count must be between 1 and 10000.');
    });

    it('preserves optional training hyperparameters during strict import', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                optimizer: 'adam',
                lossType: 'huber',
                momentum: 0.33,
                gradientClip: 1.5,
                adamBeta1: 0.7,
                adamBeta2: 0.95,
                huberDelta: 2.25,
                lrSchedule: { type: 'cosine', totalSteps: 500, minLr: 0.0001 },
            },
        });

        expect(result.error).toBeNull();
        expect(result.config?.training).toMatchObject({
            momentum: 0.33,
            gradientClip: 1.5,
            adamBeta1: 0.7,
            adamBeta2: 0.95,
            huberDelta: 2.25,
            lrSchedule: { type: 'cosine', totalSteps: 500, minLr: 0.0001 },
        });
    });

    it('accepts zero-valued adam betas in strict import mode', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                optimizer: 'adam',
                adamBeta1: 0,
                adamBeta2: 0,
            },
        });

        expect(result.error).toBeNull();
        expect(result.config?.training.adamBeta1).toBe(0);
        expect(result.config?.training.adamBeta2).toBe(0);
    });

    it('strictly rejects invalid learning-rate schedule values', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                lrSchedule: { type: 'step', stepSize: 0, gamma: 2 },
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Step schedule requires a positive integer interval and gamma between 0 and 1.');
    });

    it('strictly rejects cosine schedules whose minimum exceeds the base learning rate', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                learningRate: 0.01,
                lrSchedule: { type: 'cosine', totalSteps: 100, minLr: 0.03 },
            },
        });

        expect(result.config).toBeNull();
        expect(result.error).toBe('Cosine schedule minimum learning rate cannot exceed the base learning rate.');
    });

    it('preserves optional training hyperparameters during strict import', () => {
        const result = validateImportedConfig({
            ...validConfig,
            training: {
                ...validConfig.training,
                optimizer: 'adam',
                lossType: 'huber',
                momentum: 0.33,
                gradientClip: 1.5,
                adamBeta1: 0.7,
                adamBeta2: 0.95,
                huberDelta: 2.25,
                lrSchedule: { type: 'cosine', totalSteps: 500, minLr: 0.0001 },
            },
        });

        expect(result.error).toBeNull();
        expect(result.config?.training).toMatchObject({
            momentum: 0.33,
            gradientClip: 1.5,
            adamBeta1: 0.7,
            adamBeta2: 0.95,
            huberDelta: 2.25,
            lrSchedule: { type: 'cosine', totalSteps: 500, minLr: 0.0001 },
        });
    });
});
