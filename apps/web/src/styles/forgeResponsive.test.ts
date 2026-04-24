import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('forge compact dock CSS', () => {
    it('keeps the compact dock to rail plus one content column under narrow media rules', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-dock.forge-dock--compact');
        expect(css).toContain('grid-template-columns: 40px minmax(0, 1fr)');
    });

    it('keeps compact dock rows fluid enough for short mobile viewports', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('minmax(140px, 0.8fr)');
        expect(css).toContain('minmax(170px, 1fr)');
    });

    it('makes stacked grid panels scrollable instead of forcing every nested panel to full height', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-panel-stack');
        expect(css).toContain('overflow-y: auto');
        expect(css).not.toContain('.forge-grid .forge-panel { height: 100%; }');
    });
});
