import { useLayoutStore, type LeftTabId, type RightTabId } from '../store/useLayoutStore.ts';
import type { RelatedPanelId } from './trainingExplanations.ts';

const LEFT_TARGETS = ['presets', 'data', 'features', 'network', 'hyperparams', 'config'] as const satisfies readonly LeftTabId[];
const RIGHT_TARGETS = ['boundary', 'loss', 'confusion', 'inspection', 'code', 'history'] as const satisfies readonly RightTabId[];

type ScheduleFocus = (focus: () => void) => void;

interface FocusExplanationActionOptions {
    root?: ParentNode;
    scheduleFocus?: ScheduleFocus;
}

function isLeftTarget(panelId: RelatedPanelId): panelId is LeftTabId {
    return (LEFT_TARGETS as readonly string[]).includes(panelId);
}

function isRightTarget(panelId: RelatedPanelId): panelId is RightTabId {
    return (RIGHT_TARGETS as readonly string[]).includes(panelId);
}

function getScheduler(scheduleFocus?: ScheduleFocus): ScheduleFocus {
    if (scheduleFocus) return scheduleFocus;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
        return (focus) => window.requestAnimationFrame(focus);
    }
    return (focus) => window.setTimeout(focus, 0);
}

function focusPanelTarget(root: ParentNode, side: 'left' | 'right', panelId: RelatedPanelId) {
    const candidates = [
        `#forge-${side}-tab-${panelId}`,
        `#forge-${side}-panel-${panelId}`,
        `[data-forge-panel-targets~="${panelId}"]`,
    ];

    for (const selector of candidates) {
        const element = root.querySelector<HTMLElement>(selector);
        if (element) {
            element.focus();
            return;
        }
    }
}

export function focusExplanationActionTarget(
    panelId: RelatedPanelId,
    options: FocusExplanationActionOptions = {},
) {
    const state = useLayoutStore.getState();
    const root = options.root ?? document;
    const scheduleFocus = getScheduler(options.scheduleFocus);

    if (isLeftTarget(panelId)) {
        state.setActiveTabLeft(panelId);
        if (state.layout === 'split') state.setPhase('build');
        scheduleFocus(() => focusPanelTarget(root, 'left', panelId));
        return;
    }

    if (isRightTarget(panelId)) {
        state.setActiveTabRight(panelId);
        if (state.layout === 'split') state.setPhase('run');
        scheduleFocus(() => focusPanelTarget(root, 'right', panelId));
    }
}
