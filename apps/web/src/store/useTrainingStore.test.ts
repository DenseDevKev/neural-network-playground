import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTrainingStore } from './useTrainingStore.ts';
import { metricHistoryBuffer } from './metricHistoryBuffer.ts';
import {
    PREPARED_PRESETS,
    type WorkerEvidenceMessageV2,
} from '@nn-playground/shared';
import {
    createScientificTrustFixtures,
    type ScientificTrustFixtures,
} from '../test/scientificTrustFixtures.ts';

let fixtures: ScientificTrustFixtures;

beforeAll(async () => {
    fixtures = await createScientificTrustFixtures();
});

const RECIPE = PREPARED_PRESETS.find((entry) => entry.id === 'xor-hidden')!.prepared;

describe('useTrainingStore strict V2 frames', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            frameVersion: 0,
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

    it('publishes strict frame versions without manufacturing scalar evidence', () => {
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => {
            publications++;
        });

        useTrainingStore.getState().applyStreamedFrame({
            frameVersions: {
                frameVersion: 7,
                outputGridVersion: 1,
                neuronGridsVersion: 2,
                paramsVersion: 3,
                layerStatsVersion: 4,
                confusionMatrixVersion: 5,
                activationHistogramsVersion: 6,
                multiclassBoundaryVersion: 7,
            },
        });

        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(1);
        expect(state.frameVersion).toBe(7);
        expect(state.latestLiveSignal).toBeNull();
        expect(state.latestEvaluation).toBeNull();
        // Visual frames must never clear a fatal worker error; only explicit
        // recovery paths (clearWorkerError) may.
        expect(state.workerError).toBe('previous error');
    });

    it('publishes the multiclass boundary frame version from streamed frame versions', () => {
        useTrainingStore.getState().applyStreamedFrame({
            frameVersions: {
                frameVersion: 9,
                outputGridVersion: 1,
                neuronGridsVersion: 2,
                paramsVersion: 3,
                layerStatsVersion: 4,
                confusionMatrixVersion: 5,
                activationHistogramsVersion: 6,
                multiclassBoundaryVersion: 7,
            },
        });

        expect(useTrainingStore.getState().multiclassBoundaryVersion).toBe(7);
    });

    it('tracks preset config transactions and retries with preset loading state', () => {
        useTrainingStore.getState().beginConfigChange('preset');

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().dataConfigLoading).toBe(false);

        useTrainingStore.getState().failConfigChange('Preset failed');

        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().presetConfigLoading).toBe(false);
        expect(useTrainingStore.getState().configError).toBe('Preset failed');
        expect(useTrainingStore.getState().configErrorSource).toBe('preset');

        useTrainingStore.getState().retryConfigSync();

        expect(useTrainingStore.getState().pendingConfigSource).toBe('preset');
        expect(useTrainingStore.getState().presetConfigLoading).toBe(true);
        expect(useTrainingStore.getState().configError).toBeNull();
    });

    it('records the exact recipe and fingerprint that produced evidence', () => {
        useTrainingStore.getState().markTrainedRecipe(
            RECIPE.document.recipe,
            'config-sync',
            RECIPE.identities.recipeFingerprint,
        );

        const state = useTrainingStore.getState();
        expect(state.trainedRecipe).toEqual(RECIPE.document.recipe);
        expect(state.trainedRecipeFingerprint).toBe(RECIPE.identities.recipeFingerprint);
        expect(state.trainedRecipeRecordedAt).toEqual(expect.any(Number));
        expect(state.trainedRecipeSource).toBe('config-sync');
    });

    it('keeps trained recipe metadata separate from later current recipe mutation', () => {
        const recipe = structuredClone(RECIPE.document.recipe);
        useTrainingStore.getState().markTrainedRecipe(recipe, 'initialize');
        (recipe.training as { learningRate: number }).learningRate = 0.3;
        (recipe.model.hiddenLayers as number[]).push(12);

        expect(useTrainingStore.getState().trainedRecipe?.training.learningRate)
            .toBe(RECIPE.document.recipe.training.learningRate);
        expect(useTrainingStore.getState().trainedRecipe?.model.hiddenLayers).toEqual([4, 4]);
    });
});

function evidenceAt(
    evaluationId: number,
    step: number,
    generationId = fixtures.evaluation.model.generationId,
): WorkerEvidenceMessageV2 {
    const model = {
        generationId,
        revision: step,
        step,
        epoch: Math.floor(step / 10),
    };
    const trainDataLoss = 0.6 - evaluationId * 0.01;
    return {
        type: 'evidence',
        protocolVersion: 2,
        liveSignal: {
            ...fixtures.liveSignal,
            model,
            basis: {
                ...fixtures.liveSignal.basis,
                throughStep: step,
            },
            dataLoss: trainDataLoss,
        },
        latestEvaluation: {
            ...fixtures.evaluation,
            evaluationId,
            trigger: evaluationId === 1 ? 'initial' : 'manual-step',
            model,
            train: {
                ...fixtures.evaluation.train,
                values: { dataLoss: trainDataLoss },
            },
            test: {
                ...fixtures.evaluation.test,
                values: { dataLoss: trainDataLoss + 0.1 },
            },
            objective: {
                regularizationPenalty: 0,
                trainTotalObjective: trainDataLoss,
            },
        },
    };
}

describe('useTrainingStore scientific evidence', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetEvidence();
    });

    it('publishes a validated evidence bundle atomically with independent versions', () => {
        const before = useTrainingStore.getState();
        const coldRead = vi.spyOn(metricHistoryBuffer, 'read');
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);

        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(1);
        expect(state.evidenceGenerationId).toBe(fixtures.liveSignal.model.generationId);
        expect(state.latestLiveSignal).toEqual(fixtures.liveSignal);
        expect(state.latestEvaluation).toEqual(fixtures.evaluation);
        expect(state.trainingTrendVersion).toBe(before.trainingTrendVersion + 1);
        expect(state.evaluationHistoryVersion).toBe(before.evaluationHistoryVersion + 1);
        expect(coldRead).not.toHaveBeenCalled();
        coldRead.mockRestore();
        expect(metricHistoryBuffer.read().trendHistory).toHaveLength(1);
        expect(metricHistoryBuffer.read().evaluationHistory).toHaveLength(1);
    });

    it('deduplicates exact stream/direct replay without publishing or bumping versions', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const before = useTrainingStore.getState();
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);

        useTrainingStore.getState().applyEvidence(structuredClone(fixtures.evidence));
        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(0);
        expect(state.trainingTrendVersion).toBe(before.trainingTrendVersion);
        expect(state.evaluationHistoryVersion).toBe(before.evaluationHistoryVersion);
        expect(metricHistoryBuffer.read().trendHistory).toHaveLength(1);
        expect(metricHistoryBuffer.read().evaluationHistory).toHaveLength(1);
    });

    it('rejects a conflicting same-model live signal before appending a new evaluation', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const before = metricHistoryBuffer.read();
        const baseConflict = evidenceAt(2, 0);
        const conflict: WorkerEvidenceMessageV2 = {
            ...baseConflict,
            liveSignal: {
                ...baseConflict.liveSignal!,
                dataLoss: fixtures.liveSignal.dataLoss + 0.25,
            },
        };

        expect(() => useTrainingStore.getState().applyEvidence(conflict))
            .toThrow(/conflicting live signal/i);
        expect(metricHistoryBuffer.read()).toBe(before);
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(1);
    });

    it('rejects non-identical model coordinates at the same live revision', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const before = metricHistoryBuffer.versions;

        expect(() => useTrainingStore.getState().applyEvidence({
            type: 'evidence',
            protocolVersion: 2,
            liveSignal: {
                ...fixtures.liveSignal,
                model: { ...fixtures.liveSignal.model, step: 1 },
                basis: { ...fixtures.liveSignal.basis, throughStep: 1 },
            },
        })).toThrow(/conflicting live signal/i);
        expect(metricHistoryBuffer.versions).toEqual(before);
    });

    it('rejects a conflicting retained evaluation before appending its live signal', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const before = metricHistoryBuffer.read();
        const conflict = evidenceAt(1, 1);

        expect(() => useTrainingStore.getState().applyEvidence(conflict))
            .toThrow(/conflicting evaluationId 1/i);
        expect(metricHistoryBuffer.read()).toBe(before);
        expect(useTrainingStore.getState().latestLiveSignal?.model.step).toBe(0);
    });

    it('rejects a mismatched generation without partially changing either series', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const before = metricHistoryBuffer.read();

        expect(() => useTrainingStore.getState().applyEvidence(evidenceAt(2, 1, 2)))
            .toThrow(/generation/i);
        expect(metricHistoryBuffer.read()).toBe(before);
        expect(useTrainingStore.getState().evidenceGenerationId).toBe(1);
    });

    it('resets generation, latest evidence, and both series in one publication', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);

        useTrainingStore.getState().resetEvidence();
        unsubscribe();

        const state = useTrainingStore.getState();
        expect(publications).toBe(1);
        expect(state.evidenceGenerationId).toBeNull();
        expect(state.latestLiveSignal).toBeNull();
        expect(state.latestEvaluation).toBeNull();
        expect(metricHistoryBuffer.read().trendHistory).toEqual([]);
        expect(metricHistoryBuffer.read().evaluationHistory).toEqual([]);
        expect(state.trainingTrendVersion).toBe(metricHistoryBuffer.versions.trendVersion);
        expect(state.evaluationHistoryVersion).toBe(metricHistoryBuffer.versions.evaluationVersion);
    });

    it('rejects malformed evidence before publishing or mutating buffers', () => {
        const before = metricHistoryBuffer.read();
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);

        expect(() => useTrainingStore.getState().applyEvidence({
            ...fixtures.evidence,
            protocolVersion: 99,
        })).toThrow(/version-2 evidence/i);
        unsubscribe();

        expect(publications).toBe(0);
        expect(metricHistoryBuffer.read()).toBe(before);
    });

    it('publishes prepared replacement and append plans idempotently', () => {
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);
        const replacement = useTrainingStore.getState().prepareEvidenceReplacement(
            fixtures.evidence,
        );

        useTrainingStore.getState().commitEvidenceReplacement(replacement);
        useTrainingStore.getState().commitEvidenceReplacement(replacement);
        expect(publications).toBe(1);

        const append = useTrainingStore.getState().prepareEvidenceAppend(evidenceAt(2, 1));
        useTrainingStore.getState().commitEvidenceAppend(append);
        useTrainingStore.getState().commitEvidenceAppend(append);
        unsubscribe();

        expect(publications).toBe(2);
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(2);
        expect(metricHistoryBuffer.read().evaluationHistory).toHaveLength(2);
    });

    it('keeps store and packed histories unchanged when prepared append validation fails', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const beforeState = useTrainingStore.getState();
        const beforeHistory = metricHistoryBuffer.read();

        expect(() => useTrainingStore.getState().prepareEvidenceAppend({
            ...evidenceAt(2, 1),
            protocolVersion: 99,
        })).toThrow(/version-2 evidence/i);

        expect(useTrainingStore.getState()).toBe(beforeState);
        expect(metricHistoryBuffer.read()).toBe(beforeHistory);
    });

    it('rejects a stale prepared append without publishing its superseded evidence', () => {
        useTrainingStore.getState().applyEvidence(fixtures.evidence);
        const stale = useTrainingStore.getState().prepareEvidenceAppend(evidenceAt(2, 1));
        useTrainingStore.getState().applyEvidence(evidenceAt(3, 2));
        const before = useTrainingStore.getState();
        let publications = 0;
        const unsubscribe = useTrainingStore.subscribe(() => publications++);

        expect(() => useTrainingStore.getState().commitEvidenceAppend(stale)).toThrow(/stale/i);
        unsubscribe();

        expect(publications).toBe(0);
        expect(useTrainingStore.getState()).toBe(before);
        expect(useTrainingStore.getState().latestEvaluation?.evaluationId).toBe(3);
        expect(metricHistoryBuffer.read().evaluationHistory.map(({ evaluationId }) => evaluationId))
            .toEqual([1, 3]);
    });
});
