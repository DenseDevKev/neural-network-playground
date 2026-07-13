import { beforeEach, describe, expect, it } from 'vitest';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { focusExplanationActionTarget } from './explanationActionFocus.ts';

function resetLayout() {
    useLayoutStore.setState({
        view: 'build',
        activeRecipeSection: 'data',
        activeEvidenceView: 'boundary',
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

    it('selects and focuses an existing Build recipe target', () => {
        document.body.innerHTML = '<button id="forge-left-tab-hyperparams">Hyperparams</button>';

        focusExplanationActionTarget('hyperparams', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().view).toBe('build');
        expect(useLayoutStore.getState().activeRecipeSection).toBe('hyperparams');
        expect(useLayoutStore.getState().activeTabLeft).toBe('hyperparams');
        expect(document.activeElement).toBe(document.getElementById('forge-left-tab-hyperparams'));
    });

    it('selects and focuses an existing Run evidence target', () => {
        document.body.innerHTML = '<button id="forge-right-tab-loss">Loss</button>';

        focusExplanationActionTarget('loss', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        expect(document.activeElement).toBe(document.getElementById('forge-right-tab-loss'));
    });

    it('switches to Run before focusing an evidence panel', () => {
        useLayoutStore.setState({ view: 'build', phase: 'build' });
        document.body.innerHTML = '<section id="forge-right-panel-loss" tabindex="-1">Loss</section>';

        focusExplanationActionTarget('loss', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().view).toBe('run');
        expect(useLayoutStore.getState().phase).toBe('run');
        expect(useLayoutStore.getState().activeEvidenceView).toBe('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');
        expect(document.activeElement).toBe(document.getElementById('forge-right-panel-loss'));
    });

    it('falls back to visible panel containers when direct tab targets are not present', () => {
        document.body.innerHTML = '<section data-forge-panel-targets="hyperparams config" tabindex="-1">Controls</section>';

        focusExplanationActionTarget('hyperparams', { scheduleFocus: (focus) => focus() });

        expect(useLayoutStore.getState().view).toBe('build');
        expect(document.activeElement).toBe(document.querySelector('[data-forge-panel-targets~="hyperparams"]'));
    });
});
