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

    return useMemo(() => {
        void outputGridVersion;
        void multiclassBoundaryVersion;
        const frame = getFrameBuffer();
        return deriveDecisionBoundaryModel({
            frame,
            task,
            trainPoints,
            testPoints,
            showTestData,
            discretize,
            overlayMode,
        });
    }, [
        outputGridVersion,
        multiclassBoundaryVersion,
        task,
        trainPoints,
        testPoints,
        showTestData,
        discretize,
        overlayMode,
    ]);
}
