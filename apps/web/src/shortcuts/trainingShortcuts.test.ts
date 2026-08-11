import { describe, expect, it } from 'vitest';
import { TRAINING_SHORTCUTS, resolveTrainingShortcut } from './trainingShortcuts.ts';

function keyboardEvent(code: string, options: KeyboardEventInit = {}) {
    return new KeyboardEvent('keydown', { code, bubbles: true, ...options });
}

function resolveFromTarget(target: HTMLElement, code = 'KeyR') {
    let action: ReturnType<typeof resolveTrainingShortcut> = null;
    target.addEventListener('keydown', (event) => {
        action = resolveTrainingShortcut(event);
    }, { once: true });
    target.dispatchEvent(keyboardEvent(code));
    return action;
}

describe('resolveTrainingShortcut', () => {
    it('maps every registered training key to its action on the page background', () => {
        for (const { action, code } of TRAINING_SHORTCUTS) {
            expect(resolveFromTarget(document.body, code)).toBe(action);
        }
        expect(resolveTrainingShortcut(keyboardEvent('KeyS'))).toBeNull();
    });

    it('ignores modified and repeated keys so browser and assistive shortcuts remain available', () => {
        expect(resolveTrainingShortcut(keyboardEvent('KeyR', { metaKey: true }))).toBeNull();
        expect(resolveTrainingShortcut(keyboardEvent('KeyR', { ctrlKey: true }))).toBeNull();
        expect(resolveTrainingShortcut(keyboardEvent('KeyR', { altKey: true }))).toBeNull();
        expect(resolveTrainingShortcut(keyboardEvent('KeyR', { shiftKey: true }))).toBeNull();
        expect(resolveTrainingShortcut(keyboardEvent('KeyR', { repeat: true }))).toBeNull();
    });

    it('ignores keys sent from editable targets', () => {
        expect(resolveFromTarget(document.createElement('input'))).toBeNull();
        expect(resolveFromTarget(document.createElement('textarea'))).toBeNull();

        const editable = document.createElement('div');
        editable.setAttribute('contenteditable', 'true');
        expect(resolveFromTarget(editable)).toBeNull();
    });

    it('ignores every registered shortcut from an implicitly focusable native summary', () => {
        const nativeDetails = document.createElement('details');
        const nativeSummary = document.createElement('summary');
        nativeDetails.append(nativeSummary);

        for (const { code } of TRAINING_SHORTCUTS) {
            expect(resolveFromTarget(nativeSummary, code)).toBeNull();
        }
    });
});

describe('TRAINING_SHORTCUTS', () => {
    it('lists each supported shortcut with a display label and description', () => {
        expect(TRAINING_SHORTCUTS).toEqual([
            { code: 'Space', label: 'Space', description: 'Play or pause training', action: 'play-pause' },
            { code: 'ArrowRight', label: '→', description: 'Step training once', action: 'step' },
            { code: 'KeyR', label: 'R', description: 'Reset training', action: 'reset' },
        ]);
    });
});
