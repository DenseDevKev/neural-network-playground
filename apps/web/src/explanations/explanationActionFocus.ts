import { useLayoutStore } from '../store/useLayoutStore.ts';
import type { RelatedPanelId } from './trainingExplanations.ts';

interface FocusOptions {
    root?: ParentNode;
    scheduleFocus?: (focus: () => void) => void;
}

export function focusExplanationActionTarget(panelId: RelatedPanelId, { root = document, scheduleFocus = requestAnimationFrame }: FocusOptions = {}) {
    const state = useLayoutStore.getState();
    let selector: string;
    if (panelId === 'config' || panelId === 'code') {
        state.requestExport(panelId === 'code' ? 'code' : 'setup');
        selector = '.atelier-dialog';
    } else if (panelId === 'history') {
        state.navigate('saved-runs'); selector = '#main-content';
    } else if (['presets','data','features','network','hyperparams'].includes(panelId)) {
        state.openSetup(panelId === 'hyperparams' ? 'training' : panelId === 'network' || panelId === 'features' ? 'network' : 'dataset');
        selector = '.atelier-setup-tabs [aria-current="page"]';
    } else {
        state.navigate('playground', panelId === 'inspection' ? 'inspect' : 'results');
        if (panelId !== 'inspection') state.setResultsTab(panelId === 'loss' ? 'learning' : panelId === 'confusion' ? 'errors' : 'boundary');
        selector = panelId === 'inspection' ? '#workspace-tab-inspect' : '[aria-label="Results views"] [aria-selected="true"]';
    }
    scheduleFocus(() => root.querySelector<HTMLElement>(selector)?.focus());
}
