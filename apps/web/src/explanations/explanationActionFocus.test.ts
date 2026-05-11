import { beforeEach, describe, expect, it } from 'vitest';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { focusExplanationActionTarget } from './explanationActionFocus.ts';

function resetLayout() {
    useLayoutStore.setState({
        layout: 'dock',
        phase: 'build',
        activeTabLeft: 'data',
        activeTabRight: 'boundary',
        activeLessonId: null,
        activeLessonStepIndex: null,
    });
}

describe('focusExplanationActionTarget', () => {
    beforeEach(() => {
        window.localStorage.clear();
        document.body.innerHTML = '';
        resetLayout();
    });

    it('selects and focuses an existing left dock tab target', () => {
        document.body.innerHTML = '<button id="forge-left-tab-hyperparams">Hyperparams</button>';

        focusExplanationActionTarget('hyperparams', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        expect(document.activeElement).toBe(document.getElementById('forge-left-tab-hyperparams'));
    });

    it('selects and focuses an existing right dock tab target', () => {
        document.body.innerHTML = '<button id="forge-right-tab-loss">Loss</button>';

        focusExplanationActionTarget('loss', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        expect(document.activeElement).toBe(document.getElementById('forge-right-tab-loss'));
    });

    it('switches split layout to the target phase before focusing', () => {
        useLayoutStore.setState({ layout: 'split', phase: 'build' });
        document.body.innerHTML = '<section id="forge-right-panel-loss" tabindex="-1">Loss</section>';

        focusExplanationActionTarget('loss', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().phase).toBe('run');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        expect(document.activeElement).toBe(document.getElementById('forge-right-panel-loss'));
    });

    it('falls back to visible panel containers when dock tabs are not present', () => {
        useLayoutStore.setState({ layout: 'focus' });
        document.body.innerHTML = '<section data-forge-panel-targets="hyperparams config" tabindex="-1">Controls</section>';

        focusExplanationActionTarget('hyperparams', { scheduleFocus: (focus) => focus() });

        expect(document.activeElement).toBe(document.querySelector('[data-forge-panel-targets~="hyperparams"]'));
    });
});
