import { useMemo } from 'react';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import { getRecipeDrift } from '../../../store/recipeIdentity.ts';
import { selectScientificEvidence } from '../../../store/evidenceSelectors.ts';
import { derivePrecisionLabRecipeModel } from './precisionLabRecipeModel.ts';

export function usePrecisionLabRecipeModel() {
    const prepared = usePlaygroundStore((state) =>
        state.access.status === 'ready' ? state.access.prepared : null
    );
    const trainedRecipe = useTrainingStore((state) => state.trainedRecipe);
    const trainedRecipeFingerprint = useTrainingStore((state) => state.trainedRecipeFingerprint);
    const pendingConfigSource = useTrainingStore((state) => state.pendingConfigSource);
    const latestLiveSignal = useTrainingStore((state) => state.latestLiveSignal);
    const latestEvaluation = useTrainingStore((state) => state.latestEvaluation);

    const currentRecipe = prepared?.document.recipe ?? null;
    const currentRecipeFingerprint = prepared?.identities.recipeFingerprint ?? null;

    const evidence = useMemo(
        () => selectScientificEvidence({ latestLiveSignal, latestEvaluation }),
        [latestEvaluation, latestLiveSignal],
    );

    const drift = useMemo(
        () => getRecipeDrift(
            trainedRecipe,
            currentRecipe,
            3,
            trainedRecipeFingerprint === null
                ? undefined
                : { trainedRecipeFingerprint, currentRecipeFingerprint },
        ),
        [currentRecipe, currentRecipeFingerprint, trainedRecipe, trainedRecipeFingerprint],
    );

    return derivePrecisionLabRecipeModel(prepared === null ? null : {
        dataset: prepared.document.recipe.task.dataset,
        architecture: [
            prepared.compiled.network.inputSize,
            ...prepared.compiled.network.hiddenLayers,
            prepared.compiled.network.outputSize,
        ].join(' -> '),
        hiddenActivation: prepared.document.recipe.model.hiddenActivation,
        output: `${prepared.compiled.network.outputActivation} · ${prepared.document.recipe.objective.dataLoss.kind}`,
        seed: prepared.document.recipe.model.seed,
        hasRecipeDrift: drift.hasDrift,
        pendingConfiguration: pendingConfigSource !== null,
        evaluationAgeSteps: evidence.evaluationAgeSteps,
    });
}
