import { beforeEach, describe, expect, it } from 'vitest';
import { useTrainingStore } from './useTrainingStore.ts';

describe('useTrainingStore config lifecycle', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetHistory();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            featuresConfigLoading: false,
            trainingConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            workerError: null,
        });
    });

    it('defaults every config source to idle', () => {
        const state = useTrainingStore.getState();

        expect(state.pendingConfigSource).toBeNull();
        expect(state.dataConfigLoading).toBe(false);
        expect(state.networkConfigLoading).toBe(false);
        expect(state.featuresConfigLoading).toBe(false);
        expect(state.trainingConfigLoading).toBe(false);
        expect(state.configError).toBeNull();
        expect(state.configErrorSource).toBeNull();
    });

    it('marks feature config changes as the active source', () => {
        useTrainingStore.getState().beginConfigChange('features');

        const state = useTrainingStore.getState();
        expect(state.pendingConfigSource).toBe('features');
        expect(state.featuresConfigLoading).toBe(true);
        expect(state.dataConfigLoading).toBe(false);
        expect(state.networkConfigLoading).toBe(false);
        expect(state.trainingConfigLoading).toBe(false);
    });

    it('marks training config changes as the active source', () => {
        useTrainingStore.getState().beginConfigChange('training');

        const state = useTrainingStore.getState();
        expect(state.pendingConfigSource).toBe('training');
        expect(state.trainingConfigLoading).toBe(true);
        expect(state.dataConfigLoading).toBe(false);
        expect(state.networkConfigLoading).toBe(false);
        expect(state.featuresConfigLoading).toBe(false);
    });

    it('clears every loading flag when a config change finishes or fails', () => {
        useTrainingStore.getState().beginConfigChange('features');
        useTrainingStore.getState().finishConfigChange();

        expect(useTrainingStore.getState().featuresConfigLoading).toBe(false);

        useTrainingStore.getState().beginConfigChange('training');
        useTrainingStore.getState().failConfigChange('Failed to update training');

        const state = useTrainingStore.getState();
        expect(state.trainingConfigLoading).toBe(false);
        expect(state.pendingConfigSource).toBeNull();
        expect(state.configErrorSource).toBe('training');
    });

    it('retries the config source that failed most recently', () => {
        useTrainingStore.getState().beginConfigChange('features');
        useTrainingStore.getState().failConfigChange('Failed to update features');

        useTrainingStore.getState().retryConfigSync();

        const state = useTrainingStore.getState();
        expect(state.pendingConfigSource).toBe('features');
        expect(state.featuresConfigLoading).toBe(true);
        expect(state.configError).toBeNull();
        expect(state.configErrorSource).toBeNull();
        expect(state.configSyncNonce).toBe(1);
    });
});
