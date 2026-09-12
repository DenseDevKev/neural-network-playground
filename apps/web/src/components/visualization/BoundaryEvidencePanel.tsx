import { memo } from 'react';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import type { DecisionBoundaryController } from './useDecisionBoundaryController.ts';
import type { DecisionOverlayMode } from './decisionBoundaryModel.ts';

const MODES: readonly [DecisionOverlayMode, string][] = [
    ['none', 'Output'], ['uncertainty', 'Uncertain'], ['misclassification', 'Errors'], ['split', 'Split'],
];

/** Detailed controls intentionally contain no canvas or second model subscription. */
export const BoundaryEvidencePanel = memo(function BoundaryEvidencePanel({ controller }: { readonly controller: DecisionBoundaryController }) {
    const { model, taskKind, commands, showHelp, guidanceLevel, showTestData, discretize, overlayMode, overlayCopy } = controller;
    return <section aria-label="Prediction controls">
        <div className="precision-boundary-details">
            {showHelp && <div className="decision-boundary-concept"><span>Decision boundary</span><ConceptHelp conceptId="decision-boundary" guidanceLevel={guidanceLevel} className="concept-help--block" /></div>}
            {model.kind !== 'empty' && model.kind !== 'unavailable' && <p>{model.provenance ? `Prediction grid at model step ${model.provenance.model.step}. A sampled field, not a full-split evaluation.` : 'Prediction grid provenance is not yet available.'}</p>}
            <label className="checkbox-row"><input type="checkbox" checked={showTestData} onChange={(e) => { void commands.setShowTestData(e.currentTarget.checked); }} />Show test data</label>
            {taskKind !== 'regression' && <label className="checkbox-row"><input type="checkbox" checked={discretize} onChange={(e) => { void commands.setDiscretize(e.currentTarget.checked); }} />Discretize output</label>}
            <div className="decision-overlay-controls" aria-label="Decision overlay controls">
                {MODES.filter(([mode]) => taskKind !== 'regression' || mode === 'none' || mode === 'split').map(([mode, label]) => <button key={mode} type="button" aria-pressed={overlayMode === mode} onClick={() => commands.setOverlayMode(mode)}>{label}</button>)}
            </div>
            <p className="decision-overlay-note" aria-live="polite">{taskKind === 'regression' ? 'Continuous predictions and numerical target values. Test points use outlined markers.' : taskKind === 'multiclass-classification' && overlayMode === 'uncertainty' ? 'Winning-class confidence across all three classes; lower values indicate ambiguous regions.' : taskKind === 'multiclass-classification' && overlayMode === 'none' ? discretize ? 'Output mode shows flat winning-class regions for Class 0, Class 1, and Class 2.' : 'Output mode shows the winning class across all three classes, with color strength indicating winning-class confidence.' : overlayCopy.description}</p>
            <p>{model.kind === 'empty' || model.kind === 'unavailable' ? model.description : model.accessibleDescription}</p>
        </div>
    </section>;
});
