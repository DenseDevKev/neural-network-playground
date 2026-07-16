import { useLayoutStore, type EvidenceViewId, type RecipeSectionId } from '../store/useLayoutStore.ts';
import type { RelatedPanelId } from './trainingExplanations.ts';

const LEFT_TARGETS = ['presets', 'data', 'features', 'network', 'hyperparams', 'config'] as const satisfies readonly RecipeSectionId[];
const RIGHT_TARGETS = ['boundary', 'loss', 'confusion', 'inspection', 'code', 'history'] as const satisfies readonly EvidenceViewId[];

type ScheduleFocus = (focus: () => void) => void;

interface FocusExplanationActionOptions {
    root?: ParentNode;
    scheduleFocus?: ScheduleFocus;
}

function isLeftTarget(panelId: RelatedPanelId): panelId is RecipeSectionId {
    return (LEFT_TARGETS as readonly string[]).includes(panelId);
}

function isRightTarget(panelId: RelatedPanelId): panelId is EvidenceViewId {
    return (RIGHT_TARGETS as readonly string[]).includes(panelId);
}

function getScheduler(scheduleFocus?: ScheduleFocus): ScheduleFocus {
    if (scheduleFocus) return scheduleFocus;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
        return (focus) => window.requestAnimationFrame(focus);
    }
    return (focus) => window.setTimeout(focus, 0);
}

function panelContainerSelector(panelId: RelatedPanelId): string {
    return `[data-forge-panel-targets~="${panelId}"]`;
}

function focusPanelContainer(root: ParentNode, panelId: RelatedPanelId) {
    root.querySelector<HTMLElement>(panelContainerSelector(panelId))?.focus();
}

function focusPanelTarget(root: ParentNode, side: 'left' | 'right', panelId: RelatedPanelId) {
    const candidates = [
        `#forge-${side}-tab-${panelId}`,
        `#forge-${side}-panel-${panelId}`,
        panelContainerSelector(panelId),
    ];

    for (const selector of candidates) {
        const element = root.querySelector<HTMLElement>(selector);
        if (element) {
            element.focus();
            return;
        }
    }
}

export function scheduleExplanationPanelFocus(
    panelId: RelatedPanelId,
    options: FocusExplanationActionOptions = {},
) {
    const root = options.root ?? document;
    const scheduleFocus = getScheduler(options.scheduleFocus);
    scheduleFocus(() => focusPanelContainer(root, panelId));
}

export function focusExplanationActionTarget(
    panelId: RelatedPanelId,
    options: FocusExplanationActionOptions = {},
) {
    const state = useLayoutStore.getState();
    const root = options.root ?? document;
    const scheduleFocus = getScheduler(options.scheduleFocus);

    if (isLeftTarget(panelId)) {
        state.setView('build');
        state.setActiveRecipeSection(panelId);
        scheduleFocus(() => focusPanelTarget(root, 'left', panelId));
        return;
    }

    if (isRightTarget(panelId)) {
        state.setView('run');
        state.setActiveEvidenceView(panelId);
        scheduleFocus(() => focusPanelTarget(root, 'right', panelId));
    }
}
