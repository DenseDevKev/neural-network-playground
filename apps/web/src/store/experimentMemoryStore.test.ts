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

    it('recovers from corrupt persisted storage', () => {
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, '{not json');

        const store = createExperimentMemoryStore();

        expect(store.getState().records).toEqual([]);
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
