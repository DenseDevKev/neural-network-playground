import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { readThemePreference, THEME_STORAGE_KEY, useThemeEffect, useThemeStore } from './theme.ts';

function ThemeOwner() { useThemeEffect(); return null; }
describe('theme preference', () => {
    let dark = false;
    let changed: (() => void) | undefined;
    beforeEach(() => {
        window.localStorage.clear(); dark = false; changed = undefined;
        vi.stubGlobal('matchMedia', vi.fn(() => ({ get matches() { return dark; }, addEventListener: (_: string, callback: () => void) => { changed = callback; }, removeEventListener: vi.fn() })));
        useThemeStore.setState({ preference:'system',resolved:'light' });
    });
    afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
    it('follows device changes only while System is selected', () => {
        render(<ThemeOwner />);
        expect(document.documentElement.dataset.theme).toBe('light');
        act(() => { dark = true; changed?.(); });
        expect(document.documentElement.dataset.theme).toBe('dark');
        act(() => useThemeStore.getState().setPreference('light'));
        act(() => { dark = false; changed?.(); dark = true; changed?.(); });
        expect(document.documentElement.dataset.theme).toBe('light');
        expect(readThemePreference()).toBe('light');
        act(() => useThemeStore.getState().setPreference('system'));
        expect(document.documentElement.dataset.theme).toBe('dark');
    });
    it('keeps a session override when preference storage is unavailable', () => {
        render(<ThemeOwner />);
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
        act(() => useThemeStore.getState().setPreference('dark'));
        expect(useThemeStore.getState().preference).toBe('dark');
        expect(document.documentElement.dataset.theme).toBe('dark');
    });
    it('restores an explicit preference and accepts cross-tab preference changes', () => {
        window.localStorage.setItem(THEME_STORAGE_KEY,'dark');
        expect(readThemePreference()).toBe('dark');
        render(<ThemeOwner />);
        act(() => window.dispatchEvent(new StorageEvent('storage',{key:THEME_STORAGE_KEY,newValue:'dark'})));
        expect(document.documentElement.dataset.theme).toBe('dark');
        window.localStorage.setItem(THEME_STORAGE_KEY,'invalid');
        act(() => window.dispatchEvent(new StorageEvent('storage',{key:THEME_STORAGE_KEY,newValue:'invalid'})));
        expect(useThemeStore.getState().preference).toBe('system');
    });
});
