import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const css = readFileSync(resolve(__dirname, 'precisionLab.css'), 'utf8');
describe('Precision Lab layout contracts', () => {
    it('owns its desktop, intermediate, phone and reduced-motion breakpoints', () => {
        for (const header of ['@media (min-width: 1180px)', '@media (min-width: 680px) and (max-width: 1179px)', '@media (max-width: 679px)', '@media (prefers-reduced-motion: reduce)']) {
            expect(css).toContain(header);
        }
    });
    it('contains paint and layout, constrains horizontal overflow, and reserves real touch targets', () => {
        expect(css).toContain('contain: layout paint');
        expect(css).toMatch(/overflow-x:\s*(clip|hidden)/);
        expect(css).toMatch(/min-height:\s*44px/);
        expect(css).toContain('grid-template-areas:');
        expect(css).toContain('.precision-transport');
        expect(css).toContain('.precision-context');
        expect(css).toContain('.precision-drawer');
    });
    it('is imported after legacy chrome without changing the engine or base theme', () => {
        const entry = readFileSync(resolve(__dirname, '../main.tsx'), 'utf8');
        expect(entry.indexOf('precisionLab.css')).toBeGreaterThan(entry.indexOf('forge.css'));
        expect(css).not.toMatch(/(?:^|\n)\s*:root\s*\{/);
    });
});
