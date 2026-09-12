import { useCallback, useEffect, useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';
import { useDecisionBoundaryModel } from './useDecisionBoundaryModel.ts';
import { getDecisionOverlayCopy, type DecisionOverlayMode } from './decisionBoundaryModel.ts';

/** One snapshot and local overlay owner, shared by the rail and detailed controls. */
export function useDecisionBoundaryController() {
    const showTestData = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.view.showTestData);
    const discretize = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.view.discretizeOutput);
    const showHelp = usePlaygroundStore((s) => s.access.status === 'ready' && s.access.prepared.document.recipe.task.kind !== 'regression');
    const editView = usePlaygroundStore((s) => s.editView);
    const trainPoints = useTrainingStore((s) => s.trainPoints);
    const testPoints = useTrainingStore((s) => s.testPoints);
    const taskKind = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task.kind : null);
    const [overlayMode, setOverlayMode] = useState<DecisionOverlayMode>('none');
    useEffect(() => { setOverlayMode('none'); }, [taskKind]);
    const guidanceLevel = useAudienceGuidanceLevel();
    const model = useDecisionBoundaryModel({ trainPoints, testPoints, showTestData, discretize, overlayMode });
    const setShowTestData = useCallback((checked: boolean) => editView((view) => ({ ...view, showTestData: checked })), [editView]);
    const setDiscretize = useCallback((checked: boolean) => editView((view) => ({ ...view, discretizeOutput: checked })), [editView]);
    const commands = useMemo(() => ({ setShowTestData, setDiscretize, setOverlayMode }), [setShowTestData, setDiscretize]);
    return { model, taskKind, showTestData, discretize, showHelp, guidanceLevel, overlayMode, overlayCopy: getDecisionOverlayCopy(overlayMode, showTestData, discretize), commands };
}
export type DecisionBoundaryController = ReturnType<typeof useDecisionBoundaryController>;
