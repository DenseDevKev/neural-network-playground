import { beforeEach, describe, expect, it } from 'vitest';
import { useTrainingStore } from './useTrainingStore.ts';
import { readHistory } from './historyBuffer.ts';
import type { NetworkSnapshot } from '@nn-playground/engine';

function makeSnapshot(step: number): NetworkSnapshot {
    return {
        step,
        epoch: 0,
        weights: [],
        biases: [],
        trainLoss: 0.5,
        testLoss: 0.6,
        trainMetrics: { loss: 0.5 },
        testMetrics: { loss: 0.6 },
        outputGrid: [],
        gridSize: 40,
        historyPoint: { step, trainLoss: 0.5, testLoss: 0.6 },
    };
}

describe('useTrainingStore streamed snapshots', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            testMetricsStale: false,
            workerError: 'previous error',
            pauseReason: 'manual',
        });
    });

    it('applies snapshot, frame version, stale flag, and history in one store publication', () => {
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => {
            publications++;
        });

        useTrainingStore.getState().applyStreamedSnapshot({
            snapshot: makeSnapshot(3),
            frameVersions: {
                frameVersion: 7,
                outputGridVersion: 8,
                neuronGridsVersion: 9,
                paramsVersion: 10,
                layerStatsVersion: 11,
                confusionMatrixVersion: 12,
            },
            testMetricsStale: true,
        });

        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(1);
        expect(state.snapshot?.step).toBe(3);
        expect(state.frameVersion).toBe(7);
        expect(state.outputGridVersion).toBe(8);
        expect(state.neuronGridsVersion).toBe(9);
        expect(state.paramsVersion).toBe(10);
        expect(state.layerStatsVersion).toBe(11);
        expect(state.confusionMatrixVersion).toBe(12);
        expect(state.testMetricsStale).toBe(true);
        expect(state.workerError).toBeNull();
        expect(state.pauseReason).toBe('manual');
        expect(readHistory().count).toBe(1);
    });

    it('sets and clears pause reason explicitly', () => {
        useTrainingStore.getState().setPauseReason('diverged');
        expect(useTrainingStore.getState().pauseReason).toBe('diverged');

        useTrainingStore.getState().clearPauseReason();
        expect(useTrainingStore.getState().pauseReason).toBeNull();
    });
});
