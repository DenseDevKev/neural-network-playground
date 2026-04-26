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
    network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
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

describe('experiment memory schema', () => {
    it('accepts a valid v1 run record', () => {
        const result = validateExperimentRunRecord(makeRecord());

        expect(result.record?.id).toBe('run-1');
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
