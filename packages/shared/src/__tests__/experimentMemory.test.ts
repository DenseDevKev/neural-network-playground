import { describe, expect, it } from 'vitest';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    EXPERIMENT_MEMORY_MAX_HISTORY,
    createExperimentMemoryEnvelope,
    normalizeExperimentMemoryEnvelope,
    sanitizeExperimentHistory,
    validateExperimentRunRecord,
} from '../index.js';
import type { AppConfig, ExperimentRunRecordV1 } from '../index.js';

const config: AppConfig = {
    data: { ...DEFAULT_DATA },
    network: { ...DEFAULT_NETWORK, inputSize: 2, hiddenLayers: [], outputSize: 1, seed: DEFAULT_DATA.seed },
    training: { ...DEFAULT_TRAINING },
    features: { ...DEFAULT_FEATURES },
    ui: { showTestData: false, discretizeOutput: false },
};

function makeRecord(overrides: Partial<ExperimentRunRecordV1> = {}): ExperimentRunRecordV1 {
    return {
        schemaVersion: 1,
        id: 'run-1',
        createdAt: '2026-04-26T00:00:00.000Z',
        updatedAt: '2026-04-26T00:01:00.000Z',
        title: 'Circle baseline',
        config,
        summary: {
            status: 'paused',
            pauseReason: 'manual',
            step: 12,
            epoch: 1,
            trainLoss: 0.4,
            testLoss: 0.5,
            trainMetrics: { loss: 0.4, accuracy: 0.8 },
            testMetrics: { loss: 0.5, accuracy: 0.75 },
        },
        network: {
            config: config.network,
            weights: [[[0.1, -0.2]]],
            biases: [[0.05]],
        },
        history: [{ step: 12, trainLoss: 0.4, testLoss: 0.5, trainAccuracy: 0.8, testAccuracy: 0.75 }],
        ...overrides,
    };
}

function makeApprovedMulticlassConfig(overrides: Partial<AppConfig> = {}): AppConfig {
    return {
        ...config,
        ...overrides,
        data: {
            ...config.data,
            dataset: 'three-class-clusters' as unknown as AppConfig['data']['dataset'],
            problemType: 'classification',
            ...overrides.data,
        },
        network: {
            ...config.network,
            outputSize: 3,
            outputActivation: 'softmax',
            ...overrides.network,
        },
        training: {
            ...config.training,
            lossType: 'categoricalCrossEntropy',
            ...overrides.training,
        },
    };
}

function makeApprovedMulticlassRecord(overrides: Partial<ExperimentRunRecordV1> = {}): ExperimentRunRecordV1 {
    return makeRecord({
        config: makeApprovedMulticlassConfig(),
        network: null,
        ...overrides,
    });
}

function makeApprovedMulticlassNetwork(): NonNullable<ExperimentRunRecordV1['network']> {
    return {
        config: makeApprovedMulticlassConfig().network,
        weights: [[[0.1, -0.2], [0.2, 0.3], [-0.1, 0.4]]],
        biases: [[0.05, -0.05, 0.1]],
    };
}

function sparseNumberArray(length: number): number[] {
    return new Array(length) as number[];
}

describe('experiment memory schema', () => {
    it('accepts a valid v1 run record', () => {
        const result = validateExperimentRunRecord(makeRecord());

        expect(result.record?.id).toBe('run-1');
        expect(result.error).toBeNull();
        expect(result.record?.network).toEqual(makeRecord().network);
    });

    it('accepts records without a serialized network payload', () => {
        const result = validateExperimentRunRecord(makeRecord({ network: null }));

        expect(result.record?.network).toBeNull();
        expect(result.error).toBeNull();
    });

    it('rejects unversioned or future-version records', () => {
        expect(validateExperimentRunRecord({ ...makeRecord(), schemaVersion: 2 }).record).toBeNull();
        expect(validateExperimentRunRecord({ id: 'run-1' }).record).toBeNull();
    });

    it('rejects malformed configs and non-finite metrics', () => {
        const malformedConfig = validateExperimentRunRecord(makeRecord({
            config: { ...config, features: { ...config.features, x: false, y: false } },
        }));
        const nonFinite = validateExperimentRunRecord(makeRecord({
            summary: { ...makeRecord().summary, trainLoss: Number.NaN },
        }));

        expect(malformedConfig.record).toBeNull();
        expect(nonFinite.record).toBeNull();
    });

    it('rejects multiclass records by default', () => {
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord());

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/not runtime-enabled/i);
    });

    it('accepts approved multiclass records only with explicit persistence opt-in', () => {
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord(), {
            allowMulticlass: true,
        });

        expect(result.error).toBeNull();
        expect(result.record?.config.network.outputSize).toBe(3);
        expect(result.record?.config.network.outputActivation).toBe('softmax');
        expect(result.record?.config.training.lossType).toBe('categoricalCrossEntropy');
        expect(result.record?.network).toBeNull();
    });

    it('accepts matching multiclass serialized network payloads only with explicit persistence opt-in', () => {
        const network = makeApprovedMulticlassNetwork();
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord({ network }), {
            allowMulticlass: true,
        });

        expect(result.error).toBeNull();
        expect(result.record?.network).toEqual(network);
    });

    it('preserves opt-in multiclass records during envelope normalization', () => {
        const valid = makeRecord({ id: 'scalar', updatedAt: '2026-04-26T00:02:00.000Z' });
        const multiclass = makeApprovedMulticlassRecord({ id: 'multiclass' });
        const envelope = createExperimentMemoryEnvelope([multiclass, valid], {
            allowMulticlass: true,
        });
        const normalized = normalizeExperimentMemoryEnvelope(envelope, {
            allowMulticlass: true,
        });

        expect(normalized.records.map((record) => record.id)).toEqual(['scalar', 'multiclass']);
        expect(createExperimentMemoryEnvelope([multiclass, valid]).records.map((record) => record.id)).toEqual(['scalar']);
        expect(normalizeExperimentMemoryEnvelope(envelope).records.map((record) => record.id)).toEqual(['scalar']);
    });

    it.each([
        [
            'unsupported multiclass dataset',
            makeApprovedMulticlassConfig({
                data: { ...makeApprovedMulticlassConfig().data, dataset: 'circle' },
            }),
        ],
        [
            'unsupported output size',
            makeApprovedMulticlassConfig({
                network: { ...makeApprovedMulticlassConfig().network, outputSize: 4 },
            }),
        ],
        [
            'softmax without categorical loss',
            makeApprovedMulticlassConfig({
                training: { ...makeApprovedMulticlassConfig().training, lossType: 'crossEntropy' },
            }),
        ],
        [
            'categorical loss without softmax',
            makeApprovedMulticlassConfig({
                network: { ...makeApprovedMulticlassConfig().network, outputActivation: 'sigmoid' },
            }),
        ],
        [
            'regression problem type',
            makeApprovedMulticlassConfig({
                data: { ...makeApprovedMulticlassConfig().data, problemType: 'regression' },
            }),
        ],
    ])('rejects opt-in multiclass records with %s', (_label, invalidConfig) => {
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord({
            config: invalidConfig,
        }), {
            allowMulticlass: true,
        });

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/multiclass|single-output/i);
    });

    it('rejects opt-in multiclass serialized networks whose payload shape does not match three outputs', () => {
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord({
            network: {
                ...makeApprovedMulticlassNetwork(),
                weights: [[[0.1, -0.2]]],
                biases: [[0.05]],
            },
        }), {
            allowMulticlass: true,
        });

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/network.*parameters/i);
    });

    it('rejects opt-in scalar records with multiclass serialized network payloads', () => {
        const result = validateExperimentRunRecord(makeRecord({
            network: makeApprovedMulticlassNetwork(),
        }), {
            allowMulticlass: true,
        });

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/network config is invalid/i);
    });

    it('rejects opt-in multiclass records with malformed summaries', () => {
        const result = validateExperimentRunRecord(makeApprovedMulticlassRecord({
            summary: {
                ...makeRecord().summary,
                trainLoss: Number.NaN,
            },
        }), {
            allowMulticlass: true,
        });

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/summary/i);
    });

    it('rejects hidden multiclass serialized network payloads in scalar-compatible records', () => {
        const result = validateExperimentRunRecord(makeRecord({
            network: {
                config: {
                    ...config.network,
                    outputSize: 3,
                    outputActivation: 'softmax',
                },
                weights: [[[0.1, -0.2], [0.2, 0.3], [-0.1, 0.4]]],
                biases: [[0.05, -0.05, 0.1]],
            },
        }));

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/network.*not runtime-enabled/i);
    });

    it('drops records with hidden multiclass serialized networks during envelope normalization', () => {
        const valid = makeRecord({ id: 'valid' });
        const hiddenNetwork = makeRecord({
            id: 'hidden-network',
            network: {
                config: {
                    ...config.network,
                    outputSize: 3,
                    outputActivation: 'softmax',
                },
                weights: [[[0.1, -0.2], [0.2, 0.3], [-0.1, 0.4]]],
                biases: [[0.05, -0.05, 0.1]],
            },
        });

        const envelope = createExperimentMemoryEnvelope([hiddenNetwork, valid]);

        expect(envelope.records.map((record) => record.id)).toEqual(['valid']);
    });

    it('accepts valid multi-hidden-layer serialized network payloads unchanged', () => {
        const network = {
            config: {
                ...config.network,
                hiddenLayers: [3, 2],
            },
            weights: [
                [
                    [0.1, -0.2],
                    [0.2, 0.3],
                    [-0.1, 0.4],
                ],
                [
                    [0.5, -0.6, 0.7],
                    [-0.3, 0.2, -0.1],
                ],
                [[0.8, -0.4]],
            ],
            biases: [
                [0.01, 0.02, 0.03],
                [0.04, 0.05],
                [0.06],
            ],
        };

        const result = validateExperimentRunRecord(makeRecord({ network }));

        expect(result.error).toBeNull();
        expect(result.record?.network).toEqual(network);
    });

    it.each([
        [
            'weight layer count mismatch',
            {
                weights: [],
                biases: [[0.05]],
            },
        ],
        [
            'bias layer count mismatch',
            {
                weights: [[[0.1, -0.2]]],
                biases: [],
            },
        ],
        [
            'non-array weight matrix',
            {
                weights: [1],
                biases: [[0.05]],
            },
        ],
        [
            'non-array weight row',
            {
                weights: [[1]],
                biases: [[0.05]],
            },
        ],
        [
            'non-array bias vector',
            {
                weights: [[[0.1, -0.2]]],
                biases: [1],
            },
        ],
        [
            'non-finite weight',
            {
                weights: [[[0.1, Number.NaN]]],
                biases: [[0.05]],
            },
        ],
        [
            'non-finite bias',
            {
                weights: [[[0.1, -0.2]]],
                biases: [[Number.POSITIVE_INFINITY]],
            },
        ],
        [
            'wrong weight matrix row count',
            {
                weights: [[[0.1, -0.2], [0.2, 0.3]]],
                biases: [[0.05]],
            },
        ],
        [
            'wrong weight row width',
            {
                weights: [[[0.1]]],
                biases: [[0.05]],
            },
        ],
        [
            'wrong bias width',
            {
                weights: [[[0.1, -0.2]]],
                biases: [[0.05, 0.06]],
            },
        ],
        [
            'sparse weight row',
            {
                weights: [[sparseNumberArray(2)]],
                biases: [[0.05]],
            },
        ],
        [
            'sparse bias vector',
            {
                weights: [[[0.1, -0.2]]],
                biases: [sparseNumberArray(1)],
            },
        ],
    ])('rejects serialized network payloads with %s', (_label, payload) => {
        const result = validateExperimentRunRecord(makeRecord({
            network: {
                config: config.network,
                weights: payload.weights,
                biases: payload.biases,
            } as any,
        }));

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/network.*parameters/i);
    });

    it('rejects serialized network payloads whose raw input size does not match validated features', () => {
        const result = validateExperimentRunRecord(makeRecord({
            network: {
                config: {
                    ...config.network,
                    inputSize: 3,
                },
                weights: [[[0.1, -0.2, 0.3]]],
                biases: [[0.05]],
            },
        }));

        expect(result.record).toBeNull();
        expect(result.error).toMatch(/network.*input size/i);
    });

    it('bounds history and removes invalid history points', () => {
        const history = Array.from({ length: EXPERIMENT_MEMORY_MAX_HISTORY + 20 }, (_, idx) => ({
            step: idx,
            trainLoss: idx + 0.1,
            testLoss: idx + 0.2,
        }));
        history.splice(5, 0, { step: 5, trainLoss: Number.POSITIVE_INFINITY, testLoss: 1 });

        const sanitized = sanitizeExperimentHistory(history);

        expect(sanitized).toHaveLength(EXPERIMENT_MEMORY_MAX_HISTORY);
        expect(sanitized[0].step).toBe(20);
        expect(sanitized.every((point) => Number.isFinite(point.trainLoss))).toBe(true);
    });

    it('normalizes storage envelopes and drops corrupt records', () => {
        const valid = makeRecord({ id: 'valid' });
        const envelope = createExperimentMemoryEnvelope([valid, { ...valid, id: '', title: 'bad' }]);

        const normalized = normalizeExperimentMemoryEnvelope(envelope);

        expect(normalized.records.map((record) => record.id)).toEqual(['valid']);
        expect(normalized.schemaVersion).toBe(1);
    });
});
