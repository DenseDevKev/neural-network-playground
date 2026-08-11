import { describe, expect, it } from 'vitest';
import { TRAINING_SHORTCUTS, resolveTrainingShortcut } from './trainingShortcuts.ts';

function keyboardEvent(code: string, options: KeyboardEventInit = {}) {
    return new KeyboardEvent('keydown', { code, bubbles: true, ...options });
}

function resolveFromTarget(target: HTMLElement) {
    let action: ReturnType<typeof resolveTrainingShortcut> = null;
    target.addEventListener('keydown', (event) => {
        action = resolveTrainingShortcut(event);
    }, { once: true });
    target.dispatchEvent(keyboardEvent('KeyR'));
    return action;
}

describe('resolveTrainingShortcut', () => {
    it('maps the supported training keys to their actions', () => {
        expect(resolveTrainingShortcut(keyboardEvent('Space'))).toBe('play-pause');
        expect(resolveTrainingShortcut(keyboardEvent('ArrowRight'))).toBe('step');
        expect(resolveTrainingShortcut(keyboardEvent('KeyR'))).toBe('reset');
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
});

describe('TRAINING_SHORTCUTS', () => {
    it('lists each supported shortcut with a display label and description', () => {
        expect(TRAINING_SHORTCUTS).toEqual([
            { code: 'Space', label: 'Space', description: 'Play or pause training' },
            { code: 'ArrowRight', label: '→', description: 'Step training once' },
            { code: 'KeyR', label: 'R', description: 'Reset training' },
        ]);
    });
});
