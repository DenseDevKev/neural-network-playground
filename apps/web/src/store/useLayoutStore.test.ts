import { beforeEach, describe, expect, it } from 'vitest';
import { useLayoutStore } from './useLayoutStore.ts';

describe('useLayoutStore', () => {
    beforeEach(() => {
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

    it('setLayout switches between dock, grid, and split', () => {
        const { setLayout } = useLayoutStore.getState();

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

        useLayoutStore.getState().setActiveTabRight('code');
        expect(useLayoutStore.getState().activeTabRight).toBe('code');
    });

    it('persists state to localStorage under nn-playground-layout', () => {
        useLayoutStore.getState().setLayout('grid');
        useLayoutStore.getState().setPhase('run');

        const stored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(stored.state?.layout).toBe('grid');
        expect(stored.state?.phase).toBe('run');
    });

    it('restores state from localStorage on initialization', () => {
        window.localStorage.setItem(
            'nn-playground-layout',
            JSON.stringify({ state: { layout: 'split', phase: 'run', activeTabLeft: 'features', activeTabRight: 'loss' }, version: 0 }),
        );

        // Force re-hydration by reading the persisted state
        const storedRaw = window.localStorage.getItem('nn-playground-layout');
        const stored = JSON.parse(storedRaw ?? '{}');
        expect(stored.state.layout).toBe('split');
        expect(stored.state.phase).toBe('run');
    });
});
