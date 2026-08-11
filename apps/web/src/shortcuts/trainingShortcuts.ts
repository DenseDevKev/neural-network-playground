export type TrainingShortcutAction = 'play-pause' | 'step' | 'reset';

export const TRAINING_SHORTCUTS = [
    { code: 'Space', label: 'Space', description: 'Play or pause training' },
    { code: 'ArrowRight', label: '→', description: 'Step training once' },
    { code: 'KeyR', label: 'R', description: 'Reset training' },
] as const;

const SHORTCUT_ACTION_BY_CODE: Record<string, TrainingShortcutAction> = {
    Space: 'play-pause',
    ArrowRight: 'step',
    KeyR: 'reset',
};

const SHORTCUT_BLOCKED_ROLES = new Set(['button', 'tab', 'switch', 'slider']);

function isEditableShortcutTarget(target: EventTarget | null) {
    if (!(target instanceof Element)) return false;
    if (target === document.body || target === document.documentElement) return false;

    let element: Element | null = target;
    while (element) {
        if (
            element instanceof HTMLButtonElement ||
            element instanceof HTMLInputElement ||
            element instanceof HTMLSelectElement ||
            element instanceof HTMLTextAreaElement ||
            element instanceof HTMLAnchorElement
        ) {
            return true;
        }

        const role = element.getAttribute('role');
        if (role && SHORTCUT_BLOCKED_ROLES.has(role)) return true;

        const tabIndex = element.getAttribute('tabindex');
        if (tabIndex !== null && tabIndex !== '-1') return true;

        const contentEditable = element.getAttribute('contenteditable');
        if (contentEditable !== null && contentEditable.toLowerCase() !== 'false') return true;

        element = element.parentElement;
    }

    return false;
}

export function resolveTrainingShortcut(event: KeyboardEvent): TrainingShortcutAction | null {
    if (event.repeat || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return null;
    if (isEditableShortcutTarget(event.target)) return null;
    return SHORTCUT_ACTION_BY_CODE[event.code] ?? null;
}
