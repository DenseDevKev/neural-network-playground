import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY } from './useLayoutStore.ts';

describe('Precision Lab Build context disclosure', () => {
    beforeEach(() => {
        window.localStorage.clear();
    });

    it('defaults closed and selects Build context atomically with compatibility aliases', () => {
        const store = createLayoutStore();
        expect(store.getState().buildContextOpen).toBe(false);
        store.setState({ view: 'run', phase: 'run' });
        const transitions: Array<ReturnType<typeof store.getState>> = [];
        const unsubscribe = store.subscribe((state) => transitions.push(state));

        store.getState().selectBuildContext('network');
        unsubscribe();

        expect(transitions).toHaveLength(1);
        expect(store.getState()).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'network',
            activeTabLeft: 'network',
            buildContextOpen: true,
        });
    });

    it('opens Advanced Tools when the selected context is hidden for the current profile', () => {
        const store = createLayoutStore();
        store.getState().setAudienceMode('beginner');
        store.getState().selectBuildContext('hyperparams');
        expect(store.getState()).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'hyperparams',
            activeTabLeft: 'hyperparams',
            buildContextOpen: true,
            advancedToolsOpen: true,
            audienceMode: 'beginner',
        });
    });

    it('closes the disclosure without changing the selected recipe section', () => {
        const store = createLayoutStore();
        store.getState().selectBuildContext('network');
        store.getState().setBuildContextOpen(false);
        expect(store.getState()).toMatchObject({
            activeRecipeSection: 'network',
            activeTabLeft: 'network',
            buildContextOpen: false,
        });
    });

    it('closes Build context on Run navigation and does not reopen it on Build navigation', () => {
        const store = createLayoutStore();
        store.getState().selectBuildContext('network');
        store.getState().setView('run');
        expect(store.getState()).toMatchObject({
            view: 'run',
            phase: 'run',
            activeRecipeSection: 'network',
            buildContextOpen: false,
        });
        store.getState().setView('build');
        expect(store.getState().buildContextOpen).toBe(false);
    });

    it('routes canonical and legacy recipe selection through the same Build-context transition', () => {
        const store = createLayoutStore();
        store.setState({ view: 'run', phase: 'run' });

        store.getState().setActiveRecipeSection('network');
        expect(store.getState()).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'network',
            activeTabLeft: 'network',
            buildContextOpen: true,
        });

        store.getState().setView('run');
        store.getState().setActiveTabLeft('features');
        expect(store.getState()).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'features',
            activeTabLeft: 'features',
            buildContextOpen: true,
        });
    });

    it('opens explicit advanced recipe navigation in Build context atomically', () => {
        const store = createLayoutStore();
        store.getState().setAudienceMode('beginner');
        store.setState({ view: 'run', phase: 'run', advancedToolsOpen: false });
        const transitions: Array<ReturnType<typeof store.getState>> = [];
        const unsubscribe = store.subscribe((state) => transitions.push(state));

        store.getState().openAdvancedRecipeSection('hyperparams');
        unsubscribe();

        expect(transitions).toHaveLength(1);
        expect(store.getState()).toMatchObject({
            view: 'build',
            phase: 'build',
            activeRecipeSection: 'hyperparams',
            activeTabLeft: 'hyperparams',
            buildContextOpen: true,
            advancedToolsOpen: true,
        });
    });

    it('keeps Build context ephemeral and resets it closed on hydration', async () => {
        const store = createLayoutStore();
        store.getState().selectBuildContext('network');
        const persisted = JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? '{}');
        expect(persisted.state).not.toHaveProperty('buildContextOpen');

        window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify({
            state: {
                view: 'build',
                audienceMode: 'beginner',
                advancedToolsOpen: false,
                activeRecipeSection: 'hyperparams',
                activeEvidenceView: 'boundary',
                buildContextOpen: true,
            },
            version: 0,
        }));
        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());
        expect(freshStore.getState()).toMatchObject({
            view: 'build',
            activeRecipeSection: 'hyperparams',
            activeTabLeft: 'hyperparams',
            advancedToolsOpen: true,
            buildContextOpen: false,
        });
    });

    it('keeps deprecated setPhase synchronized and closes context when navigating to Run', () => {
        const store = createLayoutStore();
        store.getState().selectBuildContext('network');
        store.getState().setPhase('run');
        expect(store.getState()).toMatchObject({
            view: 'run',
            phase: 'run',
            activeRecipeSection: 'network',
            activeTabLeft: 'network',
            buildContextOpen: false,
        });
    });
});
