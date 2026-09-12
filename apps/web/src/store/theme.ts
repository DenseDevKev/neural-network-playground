import { useEffect } from 'react';
import { create } from 'zustand';
import type { ThemePreference } from '../productShell/atelierTypes.ts';

export const THEME_STORAGE_KEY = 'nn-playground-theme';
export function isThemePreference(value: unknown): value is ThemePreference {
    return value === 'system' || value === 'light' || value === 'dark';
}
export function readThemePreference(): ThemePreference {
    try { const value = window.localStorage.getItem(THEME_STORAGE_KEY); return isThemePreference(value) ? value : 'system'; }
    catch { return 'system'; }
}
export function resolveTheme(preference: ThemePreference, systemDark: boolean): 'light' | 'dark' {
    return preference === 'system' ? systemDark ? 'dark' : 'light' : preference;
}
function systemDark(): boolean { return typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches; }
function paintTheme(theme: 'light' | 'dark') {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
}
const preference = readThemePreference();
export const useThemeStore = create<{ preference: ThemePreference; resolved: 'light' | 'dark'; setPreference(value: ThemePreference): void }>((set) => ({
    preference, resolved: resolveTheme(preference, systemDark()),
    setPreference: (value) => {
        try { window.localStorage.setItem(THEME_STORAGE_KEY, value); } catch { /* Session choice still applies. */ }
        const resolved = resolveTheme(value, systemDark());
        paintTheme(resolved);
        set({ preference: value, resolved });
    },
}));
export function useThemeEffect() {
    const theme = useThemeStore((state) => state.resolved);
    useEffect(() => {
        paintTheme(theme);
    }, [theme]);
    useEffect(() => {
        const media = window.matchMedia?.('(prefers-color-scheme: dark)');
        const sync = () => {
            const resolved = resolveTheme(useThemeStore.getState().preference, Boolean(media?.matches));
            paintTheme(resolved); useThemeStore.setState({ resolved });
        };
        const storage = (event: StorageEvent) => {
            if (event.key !== THEME_STORAGE_KEY && event.key !== null) return;
            const next = readThemePreference();
            const resolved = resolveTheme(next, Boolean(media?.matches));
            paintTheme(resolved); useThemeStore.setState({ preference: next, resolved });
        };
        sync(); media?.addEventListener('change', sync); window.addEventListener('storage', storage);
        return () => { media?.removeEventListener('change', sync); window.removeEventListener('storage', storage); };
    }, []);
}

/** Canvas/SVG adapters read the same semantic CSS tokens as HTML. */
export function readPlotPalette() {
    const style = getComputedStyle(document.documentElement);
    const read = (key: string, fallback: string) => style.getPropertyValue(key).trim() || fallback;
    return { background: read('--bg-primary', '#17191B'), surface: read('--bg-secondary', '#202326'), text: read('--text-primary', '#EEEDE8'), muted: read('--text-secondary', '#BCC0C2'), rule: read('--border-color', '#43484D'), accent: read('--color-primary', '#EF653F') };
}
