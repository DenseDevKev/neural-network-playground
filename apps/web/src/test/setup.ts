import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

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
});
