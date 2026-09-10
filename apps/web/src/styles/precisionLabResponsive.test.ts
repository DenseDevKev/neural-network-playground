import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const css = readFileSync(resolve(__dirname, 'precisionLab.css'), 'utf8');
describe('Precision Lab layout contracts', () => {
    it('owns its desktop, intermediate, phone and reduced-motion breakpoints', () => {
        for (const header of ['@container forge-viewport (min-width: 1180px)', '@container forge-viewport (min-width: 680px) and (max-width: 1179px)', '@container forge-viewport (max-width: 679px)', '@media (prefers-reduced-motion: reduce)']) {
            expect(css).toContain(header);
        }
    });
    it('uses the actual root dimensions for document-zoom reflow', () => {
        expect(css).toMatch(/#root:has\(\.precision-layout\)\s*\{[^}]*container:\s*forge-viewport\s*\/\s*size/);
        expect(css).toMatch(/\.forge-shell:has\(\.precision-layout\)\s*\{[^}]*height:\s*100%/);
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
    it('keeps topology overlays in their own stacking context below Build controls', () => {
        expect(css).toMatch(/\.precision-topology\s*\{[^}]*isolation:\s*isolate/);
    });
    it('does not reserve the removed legacy transport column', () => {
        expect(css).toMatch(/\.precision-transport \.forge-transport-cluster\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/);
    });
    it('allows short viewports to scroll the workspace instead of overlapping the status bar', () => {
        expect(css).toMatch(/\.forge-workspace:has\(\.precision-shell\)\s*\{[^}]*overflow-y:\s*auto/);
        expect(css).toMatch(/\.precision-layout\s*\{[^}]*min-height:\s*max\(100%,\s*640px\)/);
    });
    it('is imported after legacy chrome without changing the engine or base theme', () => {
        const entry = readFileSync(resolve(__dirname, '../main.tsx'), 'utf8');
        expect(entry.indexOf('precisionLab.css')).toBeGreaterThan(entry.indexOf('forge.css'));
        expect(css).not.toMatch(/(?:^|\n)\s*:root\s*\{/);
    });
});
