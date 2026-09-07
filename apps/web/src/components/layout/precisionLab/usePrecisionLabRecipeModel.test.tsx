import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { PREPARED_PRESETS, type PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import { usePrecisionLabRecipeModel } from './usePrecisionLabRecipeModel.ts';

function prepared(id: (typeof PREPARED_PRESETS)[number]['id']): PreparedExperimentDocumentV2 {
    const match = PREPARED_PRESETS.find((entry) => entry.id === id)?.prepared;
    if (!match) throw new Error(`missing preset ${id}`);
    return match;
}

function installCurrent(next: PreparedExperimentDocumentV2) {
    usePlaygroundStore.setState({
        access: { status: 'ready', prepared: next },
        preparation: { status: 'ready', requestId: 0, issues: [] },
    });
}

function installAge(ageSteps: number) {
    const step = 40 + ageSteps;
    useTrainingStore.setState({
        latestLiveSignal: {
            model: { generationId: 1, revision: step, step, epoch: 2 },
            dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 7, testCount: 3 },
            objectiveKey: 'o',
            basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 2, throughStep: step },
            dataLoss: 0.4,
        },
        latestEvaluation: {
            evaluationId: 2,
            trigger: 'cadence',
            model: { generationId: 1, revision: 40, step: 40, epoch: 2 },
            dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 7, testCount: 3 },
            objectiveKey: 'o',
            train: { basis: { kind: 'full-split', split: 'train', sampleCount: 7, populationCount: 7 }, values: { dataLoss: 0.3 } },
            test: { basis: { kind: 'full-split', split: 'test', sampleCount: 3, populationCount: 3 }, values: { dataLoss: 0.4 } },
            objective: { regularizationPenalty: 0, trainTotalObjective: 0.3 },
        },
    });
}

describe('usePrecisionLabRecipeModel', () => {
    const multiclass = prepared('three-class-clusters');

    beforeEach(() => {
        installCurrent(multiclass);
        useTrainingStore.setState({
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
            latestLiveSignal: null,
            latestEvaluation: null,
        });
    });

    it('reads identity values from the prepared V2 document and compiled network', () => {
        const accessBefore = usePlaygroundStore.getState().access;
        const trainingBefore = useTrainingStore.getState();
        const { result } = renderHook(() => usePrecisionLabRecipeModel());
        expect(result.current).toMatchObject({
            dataset: 'three-class-clusters',
            architecture: '2 -> 6 -> 6 -> 3',
            hiddenActivation: 'tanh',
            output: 'softmax · categorical-cross-entropy-with-logits',
            seed: '42',
            tone: 'unavailable',
            evaluationLabel: 'No current evaluation',
        });
        expect(usePlaygroundStore.getState().access).toBe(accessBefore);
        expect(useTrainingStore.getState()).toBe(trainingBefore);
    });

    it('uses fingerprint drift and pending configuration from canonical stores', () => {
        const xor = prepared('xor-hidden');
        act(() => {
            useTrainingStore.getState().markTrainedRecipe(
                xor.document.recipe,
                'initialize',
                xor.identities.recipeFingerprint,
            );
        });
        const { result, rerender } = renderHook(() => usePrecisionLabRecipeModel());
        expect(result.current.tone).toBe('drift');
        act(() => useTrainingStore.setState({ pendingConfigSource: 'network' }));
        rerender();
        expect(result.current).toMatchObject({ tone: 'updating', evaluationLabel: 'Updating' });
    });

    it('updates only freshness copy when paired evaluation age changes', () => {
        act(() => {
            useTrainingStore.getState().markTrainedRecipe(
                multiclass.document.recipe,
                'initialize',
                multiclass.identities.recipeFingerprint,
            );
            installAge(2);
        });
        const { result, rerender } = renderHook(() => usePrecisionLabRecipeModel());
        const identity = {
            dataset: result.current.dataset,
            architecture: result.current.architecture,
            hiddenActivation: result.current.hiddenActivation,
            output: result.current.output,
            seed: result.current.seed,
        };
        expect(result.current).toMatchObject({ tone: 'stale', evaluationLabel: 'Evaluation 2 steps behind' });
        act(() => installAge(9));
        rerender();
        expect(result.current).toMatchObject({
            ...identity,
            tone: 'stale',
            evaluationLabel: 'Evaluation 9 steps behind',
        });
    });
    it('does not call evidence from an older model generation fresh', () => {
        installAge(0);
        const live = useTrainingStore.getState().latestLiveSignal!;
        useTrainingStore.setState({ latestLiveSignal: { ...live, model: { ...live.model, generationId: 2 } } });
        const { result } = renderHook(() => usePrecisionLabRecipeModel());
        expect(result.current).toMatchObject({ tone: 'unavailable', evaluationLabel: 'No current evaluation' });
    });

});
