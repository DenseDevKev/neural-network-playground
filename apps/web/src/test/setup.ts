import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

function createMemoryStorage(): Storage {
    const store = new Map<string, string>();

    return {
        get length() {
            return store.size;
        },
        clear() {
            store.clear();
        },
        getItem(key: string) {
            return store.get(key) ?? null;
        },
        key(index: number) {
            return Array.from(store.keys())[index] ?? null;
        },
        removeItem(key: string) {
            store.delete(key);
        },
        setItem(key: string, value: string) {
            store.set(key, String(value));
        },
    };
}

if (
    typeof window.localStorage?.getItem !== 'function' ||
    typeof window.localStorage?.setItem !== 'function' ||
    typeof window.localStorage?.removeItem !== 'function' ||
    typeof window.localStorage?.clear !== 'function'
) {
    Object.defineProperty(window, 'localStorage', {
        configurable: true,
        value: createMemoryStorage(),
    });
}

// JSDOM doesn't implement Path2D, but the canvas-based NetworkGraph (AS-5)
// instantiates them during paint. Provide a minimal no-op shim so render
// tests can run; production paths still use the real browser Path2D.
if (typeof globalThis.Path2D === 'undefined') {
    class Path2DShim {
        addPath(): void {}
        moveTo(): void {}
        lineTo(): void {}
        bezierCurveTo(): void {}
        arc(): void {}
        closePath(): void {}
    }
    (globalThis as unknown as { Path2D: typeof Path2DShim }).Path2D = Path2DShim;
}

// Cleanup after each test
afterEach(() => {
  cleanup();
  if (vi.isFakeTimers()) {
    try {
      vi.runOnlyPendingTimers();
    } finally {
      vi.useRealTimers();
    }
  }
});
