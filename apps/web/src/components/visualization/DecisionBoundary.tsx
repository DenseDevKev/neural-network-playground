// Compatibility wrapper for standalone consumers. Production Precision Lab shares
// one controller between its pinned canvas and the no-canvas evidence panel.
import { memo } from 'react';
import type { DataPoint } from '@nn-playground/engine';
import { useDecisionBoundaryModel } from './useDecisionBoundaryModel.ts';
import { DecisionBoundaryCanvas } from './DecisionBoundaryCanvas.tsx';
import type { DecisionOverlayMode } from './decisionBoundaryModel.ts';
export { classifyPointFromGrid } from './DecisionBoundaryCanvas.tsx';
export { getDecisionOverlayCopy, type DecisionOverlayCopy, type DecisionOverlayMode } from './decisionBoundaryModel.ts';
export interface DecisionBoundaryProps {
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
    overlayMode?: DecisionOverlayMode;
}
export const DecisionBoundary = memo(function DecisionBoundary(props: DecisionBoundaryProps) {
    const model = useDecisionBoundaryModel(props);
    return <DecisionBoundaryCanvas model={model} />;
});
