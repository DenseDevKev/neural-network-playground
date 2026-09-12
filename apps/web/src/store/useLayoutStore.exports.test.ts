import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, LEGACY_LAYOUT_STORAGE_KEY } from './useLayoutStore.ts';
beforeEach(() => localStorage.clear());
describe('Session export navigation', () => {
    it.each(['setActiveTabLeft', 'setActiveRecipeSection', 'selectBuildContext', 'openAdvancedRecipeSection'] as const)('maps %s config navigation to setup export', (method) => {
        const store = createLayoutStore();
        store.getState()[method]('config');
        expect(store.getState().exportRequest?.mode).toBe('setup');
    });
    it.each(['setActiveTabRight', 'setActiveEvidenceView'] as const)('maps %s code navigation to code export', (method) => {
        const store = createLayoutStore(); store.getState()[method]('code');
        expect(store.getState().exportRequest?.mode).toBe('code');
    });
    it('migrates legacy code preference without editing its old key', () => {
        const legacy = JSON.stringify({ state: { activeTabRight: 'code', codeExportTab: 'numpy', phase: 'run' } });
        localStorage.setItem(LEGACY_LAYOUT_STORAGE_KEY, legacy);
        const store = createLayoutStore();
        expect(store.getState().exportRequest?.mode).toBe('code');
        expect(store.getState().codeExportTab).toBe('numpy');
        store.getState().clearExportRequest();
        expect(localStorage.getItem(LEGACY_LAYOUT_STORAGE_KEY)).toBe(legacy);
        expect(JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY)!).state.exportRequest).toBeUndefined();
        expect(createLayoutStore().getState().exportRequest).toBeNull();
    });
    it('does not persist or reopen an explicit utility request', () => {
        const store = createLayoutStore(); store.getState().requestExport('code');
        expect(JSON.parse(localStorage.getItem(LAYOUT_STORAGE_KEY)!).state.exportRequest).toBeUndefined();
        expect(createLayoutStore().getState().exportRequest).toBeNull();
    });
});
