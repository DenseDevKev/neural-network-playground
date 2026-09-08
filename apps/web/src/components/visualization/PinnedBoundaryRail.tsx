import { memo } from 'react';
import { DecisionBoundaryCanvas } from './DecisionBoundaryCanvas.tsx';
import type { DecisionBoundaryController } from './useDecisionBoundaryController.ts';

/** Pinned means always visible; it is the live boundary, not a saved model. */
export const PinnedBoundaryRail = memo(function PinnedBoundaryRail({ controller, onExpand }: {
    readonly controller: DecisionBoundaryController;
    readonly onExpand: () => void;
}) {
    return <div className="precision-boundary-rail">
        <div className="precision-boundary-rail__heading"><h2>Decision boundary</h2><button type="button" onClick={onExpand}>Boundary details</button></div>
        <DecisionBoundaryCanvas model={controller.model} />
        <p className="precision-boundary-rail__basis">Live prediction grid. Full-split measurements and freshness are in the evidence panel.</p>
    </div>;
});
