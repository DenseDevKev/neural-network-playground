import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, useLayoutStore } from './useLayoutStore.ts';

describe('useLayoutStore', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('defaults to dock layout and build phase', () => {
        const state = useLayoutStore.getState();
        expect(state.layout).toBe('dock');
        expect(state.phase).toBe('build');
    });

    it('defaults activeTabLeft to data and activeTabRight to boundary', () => {
        const state = useLayoutStore.getState();
        expect(state.activeTabLeft).toBe('data');
        expect(state.activeTabRight).toBe('boundary');
    });

    it('setLayout switches between dock, focus, grid, and split', () => {
        const { setLayout } = useLayoutStore.getState();

        setLayout('focus');
        expect(useLayoutStore.getState().layout).toBe('focus');

        setLayout('grid');
        expect(useLayoutStore.getState().layout).toBe('grid');

        setLayout('split');
        expect(useLayoutStore.getState().layout).toBe('split');

        setLayout('dock');
        expect(useLayoutStore.getState().layout).toBe('dock');
    });

    it('setPhase toggles between build and run', () => {
        const { setPhase } = useLayoutStore.getState();

        setPhase('run');
        expect(useLayoutStore.getState().phase).toBe('run');

        setPhase('build');
        expect(useLayoutStore.getState().phase).toBe('build');
    });

    it('setActiveTabLeft updates the left panel tab', () => {
        useLayoutStore.getState().setActiveTabLeft('network');
        expect(useLayoutStore.getState().activeTabLeft).toBe('network');

        useLayoutStore.getState().setActiveTabLeft('presets');
        expect(useLayoutStore.getState().activeTabLeft).toBe('presets');
    });

    it('setActiveTabRight updates the right panel tab', () => {
        useLayoutStore.getState().setActiveTabRight('loss');
        expect(useLayoutStore.getState().activeTabRight).toBe('loss');

        useLayoutStore.getState().setActiveTabRight('history');
        expect(useLayoutStore.getState().activeTabRight).toBe('history');
    });

    it('persists the code export tab across panel remounts', () => {
        useLayoutStore.getState().setCodeExportTab('numpy');
        expect(useLayoutStore.getState().codeExportTab).toBe('numpy');

        useLayoutStore.getState().setCodeExportTab('tfjs');
        expect(useLayoutStore.getState().codeExportTab).toBe('tfjs');
    });

    it('persists state to localStorage under nn-playground-layout', () => {
        useLayoutStore.getState().setLayout('grid');
        useLayoutStore.getState().setPhase('run');

        const stored = JSON.parse(window.localStorage.getItem(LAYOUT_STORAGE_KEY) ?? '{}');
        expect(stored.state?.layout).toBe('grid');
        expect(stored.state?.phase).toBe('run');
    });

    it('rehydrates state from localStorage with a fresh store instance', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    layout: 'focus',
                    phase: 'run',
                    activeTabLeft: 'features',
                    activeTabRight: 'history',
                    codeExportTab: 'numpy',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().layout).toBe('focus');
        expect(freshStore.getState().phase).toBe('run');
        expect(freshStore.getState().activeTabLeft).toBe('features');
        expect(freshStore.getState().activeTabRight).toBe('history');
        expect(freshStore.getState().codeExportTab).toBe('numpy');
    });

    it('sanitizes invalid persisted layout state on rehydrate', async () => {
        window.localStorage.setItem(
            LAYOUT_STORAGE_KEY,
            JSON.stringify({
                state: {
                    layout: 'wide-open',
                    phase: 'debug',
                    activeTabLeft: 'missing',
                    activeTabRight: 'also-missing',
                    codeExportTab: 'also-missing',
                },
                version: 0,
            }),
        );

        const freshStore = createLayoutStore();
        await Promise.resolve(freshStore.persist.rehydrate());

        expect(freshStore.getState().layout).toBe('dock');
        expect(freshStore.getState().phase).toBe('build');
        expect(freshStore.getState().activeTabLeft).toBe('data');
        expect(freshStore.getState().activeTabRight).toBe('boundary');
        expect(freshStore.getState().codeExportTab).toBe('pseudocode');
    });
});
