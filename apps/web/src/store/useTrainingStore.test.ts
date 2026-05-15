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
            dataConfigLoading: false,
            networkConfigLoading: false,
            featuresConfigLoading: false,
            trainingConfigLoading: false,
            presetConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            multiclassBoundaryVersion: 0,
        });
    });

    it('applies snapshot, frame version, stale flag, and history in one store publication', () => {
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => {
            publications++;
        });

        useTrainingStore.getState().applyStreamedSnapshot({
            snapshot: makeSnapshot(3),
            frameVersion: 7,
            testMetricsStale: true,
        });

        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(1);
        expect(state.snapshot?.step).toBe(3);
        expect(state.frameVersion).toBe(7);
        expect(state.testMetricsStale).toBe(true);
        expect(state.workerError).toBeNull();
        expect(readHistory().count).toBe(1);
    });

    it('publishes the multiclass boundary frame version from streamed frame versions', () => {
        useTrainingStore.getState().applyStreamedSnapshot({
            snapshot: makeSnapshot(4),
            frameVersion: 9,
            frameVersions: {
                frameVersion: 9,
                outputGridVersion: 1,
                neuronGridsVersion: 2,
                paramsVersion: 3,
                layerStatsVersion: 4,
                confusionMatrixVersion: 5,
                activationHistogramsVersion: 6,
                multiclassBoundaryVersion: 7,
                arenaSummariesVersion: 8,
            },
            testMetricsStale: false,
        });

        expect(useTrainingStore.getState().multiclassBoundaryVersion).toBe(7);
    });

    it('tracks preset config transactions and retries with preset loading state', () => {
        useTrainingStore.getState().beginConfigChange('preset');

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);

        useTrainingStore.getState().failConfigChange('Preset failed', 'preset');

        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().presetConfigLoading).toBe(false);
        expect(useTrainingStore.getState().configError).toBe('Preset failed');
        expect(useTrainingStore.getState().configErrorSource).toBe('preset');

        useTrainingStore.getState().retryConfigSync();

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().configError).toBeNull();
    });
});
