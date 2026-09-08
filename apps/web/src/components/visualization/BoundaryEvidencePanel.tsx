import { memo } from 'react';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { EvidenceFrame } from '../layout/ExperimentStateContext.tsx';
import type { DecisionBoundaryController } from './useDecisionBoundaryController.ts';
import type { DecisionOverlayMode } from './decisionBoundaryModel.ts';

const MODES: readonly [DecisionOverlayMode, string][] = [
    ['none', 'Output'], ['uncertainty', 'Uncertain'], ['misclassification', 'Errors'], ['split', 'Split'],
];

/** Detailed controls intentionally contain no canvas or second model subscription. */
export const BoundaryEvidencePanel = memo(function BoundaryEvidencePanel({ controller }: { readonly controller: DecisionBoundaryController }) {
    const { model, commands, showHelp, guidanceLevel, showTestData, discretize, overlayMode, overlayCopy } = controller;
    return <EvidenceFrame view="Boundary">
        <div className="precision-boundary-details">
            {showHelp && <div className="decision-boundary-concept"><span>Decision boundary</span><ConceptHelp conceptId="decision-boundary" guidanceLevel={guidanceLevel} className="concept-help--block" /></div>}
            <label className="checkbox-row"><input type="checkbox" checked={showTestData} onChange={(e) => { void commands.setShowTestData(e.currentTarget.checked); }} />Show test data</label>
            <label className="checkbox-row"><input type="checkbox" checked={discretize} onChange={(e) => { void commands.setDiscretize(e.currentTarget.checked); }} />Discretize output</label>
            <div className="decision-overlay-controls" aria-label="Decision overlay controls">
                {MODES.map(([mode, label]) => <button key={mode} type="button" aria-pressed={overlayMode === mode} onClick={() => commands.setOverlayMode(mode)}>{label}</button>)}
            </div>
            <p className="decision-overlay-note" aria-live="polite">{overlayCopy.description}</p>
            <p>{model.kind === 'empty' || model.kind === 'unavailable' ? model.description : model.accessibleDescription}</p>
        </div>
    </EvidenceFrame>;
});
