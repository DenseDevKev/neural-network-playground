import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('forge compact dock CSS', () => {
    it('keeps the compact dock to rail plus one content column under narrow media rules', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-dock.forge-dock--compact');
        expect(css).toContain('grid-template-columns: 40px minmax(0, 1fr)');
    });
});
