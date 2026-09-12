import { useMemo } from 'react';
import type { DataPoint } from '@nn-playground/engine';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import {
    deriveDecisionBoundaryModel,
    type DecisionBoundaryDisplayModel,
    type DecisionOverlayMode,
} from './decisionBoundaryModel.ts';

export interface UseDecisionBoundaryModelInput {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
    overlayMode?: DecisionOverlayMode;
}

export function useDecisionBoundaryModel({
    trainPoints,
    testPoints,
    showTestData,
    discretize,
    overlayMode = 'none',
}: UseDecisionBoundaryModelInput): DecisionBoundaryDisplayModel {
    const outputGridVersion = useTrainingStore((state) => state.outputGridVersion);
    const multiclassBoundaryVersion = useTrainingStore((state) => state.multiclassBoundaryVersion);
    const task = usePlaygroundStore((state) => (
        state.access.status === 'ready' ? state.access.prepared.compiled.task : null
    ));

    const generation = useTrainingStore((state) => state.evidenceGenerationId);
    const datasetKey = usePlaygroundStore((state) => state.access.status === 'ready' ? state.access.prepared.identities.datasetKey : null);
    const noise = usePlaygroundStore((state) => state.access.status === 'ready' ? state.access.prepared.document.recipe.data.noise : 0);

    return useMemo(() => {
        void outputGridVersion;
        void multiclassBoundaryVersion;
        const frame = getFrameBuffer();
        return deriveDecisionBoundaryModel({
            frame: frame.decisionBoundaryProvenance && (frame.decisionBoundaryProvenance.model.generationId !== generation || frame.decisionBoundaryProvenance.dataset.datasetKey !== datasetKey)
                ? { ...frame, outputGrid:null, multiclassClassGrid:null, multiclassConfidenceGrid:null, decisionBoundaryProvenance:null } : frame,
            task,
            trainPoints,
            testPoints,
            showTestData,
            discretize,
            overlayMode, noise,
        });
    }, [
        outputGridVersion,
        generation, datasetKey,
        noise,
        multiclassBoundaryVersion,
        task,
        trainPoints,
        testPoints,
        showTestData,
        discretize,
        overlayMode,
    ]);
}
