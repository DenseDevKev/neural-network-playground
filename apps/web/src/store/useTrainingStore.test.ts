import { beforeEach, describe, expect, it } from 'vitest';
import { useTrainingStore } from './useTrainingStore.ts';
import { readHistory } from './historyBuffer.ts';
import type { NetworkSnapshot } from '@nn-playground/engine';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
} from '@nn-playground/shared';

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

function makeConfig(overrides: Partial<AppConfig> = {}): AppConfig {
    return {
        data: overrides.data ?? { ...DEFAULT_DATA },
        features: overrides.features ?? { ...DEFAULT_FEATURES },
        network: overrides.network ?? {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            hiddenLayers: [...DEFAULT_NETWORK.hiddenLayers],
            seed: DEFAULT_DATA.seed,
        },
        training: overrides.training ?? { ...DEFAULT_TRAINING },
        ui: overrides.ui ?? { showTestData: false, discretizeOutput: false },
    };
}

describe('useTrainingStore streamed snapshots', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            snapshot: null,
            trainedRecipeConfig: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
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

    it('records the config that produced the trained snapshot', () => {
        const config = makeConfig({
            data: { ...DEFAULT_DATA, dataset: 'xor' },
            training: { ...DEFAULT_TRAINING, learningRate: 0.1 },
        });

        useTrainingStore.getState().markTrainedRecipe(config, 'config-sync');

        const state = useTrainingStore.getState();
        expect(state.trainedRecipeConfig?.data.dataset).toBe('xor');
        expect(state.trainedRecipeConfig?.training.learningRate).toBe(0.1);
        expect(state.trainedRecipeRecordedAt).toEqual(expect.any(Number));
        expect(state.trainedRecipeSource).toBe('config-sync');
    });

    it('keeps trained recipe metadata separate from later current recipe mutation', () => {
        const config = makeConfig();

        useTrainingStore.getState().markTrainedRecipe(config, 'initialize');
        config.training.learningRate = 0.3;
        config.network.hiddenLayers.push(12);

        expect(useTrainingStore.getState().trainedRecipeConfig?.training.learningRate).toBe(DEFAULT_TRAINING.learningRate);
        expect(useTrainingStore.getState().trainedRecipeConfig?.network.hiddenLayers).toEqual([4, 4]);
    });
});
