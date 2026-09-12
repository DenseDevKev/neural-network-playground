import { beforeEach, describe, expect, it } from 'vitest';
import { createLayoutStore, LAYOUT_STORAGE_KEY, LEGACY_LAYOUT_STORAGE_KEY } from './useLayoutStore.ts';
beforeEach(() => localStorage.clear());
describe('Session export navigation', () => {
    it.each(['setup','code'] as const)('opens an explicit %s utility request', (mode) => {
        const store = createLayoutStore(); store.getState().requestExport(mode);
        expect(store.getState().exportRequest?.mode).toBe(mode);
        expect(store.getState().workspaceTab).toBe('network');
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
