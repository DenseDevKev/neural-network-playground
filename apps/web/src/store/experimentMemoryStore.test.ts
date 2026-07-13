import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ExperimentRunRecordV1 } from '@nn-playground/shared';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    createExperimentMemoryStore,
} from './experimentMemoryStore.ts';

function makeRecord(id: string, updatedAt = `2026-04-26T00:00:0${id}.000Z`): ExperimentRunRecordV1 {
    return {
        schemaVersion: 1,
        id,
        createdAt: updatedAt,
        updatedAt,
        config: {
            data: { dataset: 'circle', problemType: 'classification', trainTestRatio: 0.5, noise: 0, numSamples: 200, seed: 42 },
            network: { inputSize: 2, hiddenLayers: [4], outputSize: 1, activation: 'tanh', outputActivation: 'sigmoid', weightInit: 'xavier', seed: 42 },
            training: { learningRate: 0.03, batchSize: 10, lossType: 'crossEntropy', optimizer: 'sgd', momentum: 0.9, regularization: 'none', regularizationRate: 0, gradientClip: null },
            features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
            ui: { showTestData: false, discretizeOutput: false },
        },
        summary: {
            status: 'paused',
            pauseReason: 'manual',
            step: 10,
            epoch: 1,
            trainLoss: 0.4,
            testLoss: 0.5,
            trainMetrics: { loss: 0.4, accuracy: 0.8 },
            testMetrics: { loss: 0.5, accuracy: 0.7 },
        },
        network: null,
        history: [{ step: 10, trainLoss: 0.4, testLoss: 0.5 }],
    };
}

function makeApprovedMulticlassRecord(
    id: string,
    updatedAt = `2026-04-26T00:00:0${id}.000Z`,
): ExperimentRunRecordV1 {
    const record = makeRecord(id, updatedAt);
    return {
        ...record,
        config: {
            ...record.config,
            data: {
                ...record.config.data,
                dataset: 'three-class-clusters',
                problemType: 'classification',
            },
            network: {
                ...record.config.network,
                hiddenLayers: [],
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...record.config.training,
                lossType: 'categoricalCrossEntropy',
            },
        },
        network: null,
    };
}

function makePartialMulticlassRecord(id: string): ExperimentRunRecordV1 {
    const record = makeApprovedMulticlassRecord(id, '2026-04-26T00:00:05.000Z');
    return {
        ...record,
        config: {
            ...record.config,
            data: {
                ...record.config.data,
                dataset: 'circle',
            },
        },
    };
}

const workerAuthoredMulticlassConfusion = {
    classCount: 3,
    classLabels: [0, 1, 2],
    counts: [2, 0, 1, 0, 3, 0, 1, 0, 4],
} as const;

describe('experimentMemoryStore', () => {
    beforeEach(() => {
        window.localStorage.clear();
    });

    it('saves newest records first and persists them locally', () => {
        const store = createExperimentMemoryStore();

        store.getState().saveRecord(makeRecord('1'));
        store.getState().saveRecord(makeRecord('2'));

        expect(store.getState().records.map((record) => record.id)).toEqual(['2', '1']);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toContain('"schemaVersion":1');
    });

    it('replaces existing records by id', () => {
        const store = createExperimentMemoryStore();

        store.getState().saveRecord(makeRecord('1', '2026-04-26T00:00:01.000Z'));
        store.getState().saveRecord({ ...makeRecord('1', '2026-04-26T00:00:02.000Z'), title: 'Updated' });

        expect(store.getState().records).toHaveLength(1);
        expect(store.getState().records[0].title).toBe('Updated');
    });

    it('renames a saved run through the existing title field without changing schema', () => {
        const store = createExperimentMemoryStore();
        store.getState().saveRecord(makeRecord('1', '2026-04-26T00:00:01.000Z'));

        store.getState().renameRecord('1', 'Tuned reference', () => new Date('2026-04-26T00:01:00.000Z'));

        const saved = store.getState().records[0];
        expect(saved.title).toBe('Tuned reference');
        expect(saved.updatedAt).toBe('2026-04-26T00:01:00.000Z');
        expect(JSON.parse(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '{}').schemaVersion).toBe(1);
    });

    it('recovers from corrupt persisted storage', () => {
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, '{not json');

        const store = createExperimentMemoryStore();

        expect(store.getState().records).toEqual([]);
    });

    it('loads approved multiclass records from localStorage without changing schema version', () => {
        const scalar = makeRecord('1', '2026-04-26T00:00:01.000Z');
        const approvedMulticlass = makeApprovedMulticlassRecord('2', '2026-04-26T00:00:02.000Z');
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify({
            schemaVersion: 1,
            records: [approvedMulticlass, scalar],
        }));

        const store = createExperimentMemoryStore();

        expect(store.getState().records.map((record) => record.id)).toEqual(['2', '1']);
        expect(store.getState().records[0].config.training.lossType).toBe('categoricalCrossEntropy');
        expect(JSON.parse(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '{}').schemaVersion).toBe(1);
    });

    it('persists approved multiclass records and drops malformed partial multiclass records', () => {
        const store = createExperimentMemoryStore();

        store.getState().saveRecord(makeApprovedMulticlassRecord('2', '2026-04-26T00:00:02.000Z'));
        store.getState().saveRecord(makePartialMulticlassRecord('3'));

        expect(store.getState().records.map((record) => record.id)).toEqual(['2']);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toContain('categoricalCrossEntropy');
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toContain('"id":"3"');
    });

    it('does not persist worker-authored multiclass confusion metrics to localStorage', () => {
        const store = createExperimentMemoryStore();
        const record = makeApprovedMulticlassRecord('2', '2026-04-26T00:00:02.000Z');

        store.getState().saveRecord({
            ...record,
            summary: {
                ...record.summary,
                testMetrics: {
                    ...record.summary.testMetrics,
                    multiclassConfusionMatrix: workerAuthoredMulticlassConfusion,
                },
            },
        });

        expect(store.getState().records[0].summary.testMetrics.multiclassConfusionMatrix).toBeUndefined();
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toContain('multiclassConfusionMatrix');
        expect(JSON.parse(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '{}').schemaVersion).toBe(1);
    });

    it('keeps the previous state when localStorage writes fail', () => {
        const store = createExperimentMemoryStore();
        const setItem = vi.spyOn(window.localStorage.__proto__, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        expect(() => store.getState().saveRecord(makeRecord('1'))).not.toThrow();
        expect(store.getState().records).toEqual([]);

        setItem.mockRestore();
    });
});
