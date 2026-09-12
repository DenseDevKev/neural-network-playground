import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

function extractMediaBlocks(css: string, header: string): string[] {
    const blocks: string[] = [];
    let cursor = 0;
    while (true) {
        const headerIndex = css.indexOf(header, cursor);
        if (headerIndex < 0) return blocks;
        const open = css.indexOf('{', headerIndex);
        if (open < 0) throw new Error(`Missing block for ${header}`);
        let depth = 1;
        let index = open + 1;
        for (; index < css.length && depth > 0; index += 1) {
            if (css[index] === '{') depth += 1;
            if (css[index] === '}') depth -= 1;
        }
        if (depth !== 0) throw new Error(`Unclosed block for ${header}`);
        blocks.push(css.slice(open + 1, index - 1));
        cursor = index;
    }
}

describe('Precision Lab supporting CSS', () => {
    it('defines local hierarchy polish tokens without overriding base tokens', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('--forge-surface-muted');
        expect(css).toContain('--forge-surface-active');
        expect(css).toContain('--forge-border-muted');
        expect(css).toContain('--forge-shadow-muted');
        expect(css).toContain('--forge-control-gap');
    });

    it('keeps the live first-visit lesson module chrome', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-instrument-module');
        expect(css).toContain('.forge-instrument-module__head');
        expect(css).toContain('.forge-instrument-module__grip');
        expect(css).toContain('.forge-instrument-module__title');
        expect(css).toContain('.forge-instrument-module__body');
    });

    it('keeps production topology and evidence controls touch-sized', () => {
        const css = readFileSync(resolve(__dirname, 'precisionLab.css'), 'utf8');

        expect(css).toMatch(
            /\.forge-shell \.network-graph-frame :is\([^}]+\)\s*\{[^}]*min-width:\s*44px;[^}]*min-height:\s*44px/,
        );
        expect(css).toMatch(
            /\.precision-evidence \.decision-overlay-controls button\s*\{[^}]*min-width:\s*44px;[^}]*min-height:\s*44px/,
        );
    });

    it('keeps the compact evaluation disclosure hidden by default and wholly owned by one compact block', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');
        const baseDisclosure = css.match(/\.forge-compact-outcome\s*\{([^}]*)\}/);
        expect(baseDisclosure?.[1]).toMatch(/display:\s*none/);

        const compactBlocks = extractMediaBlocks(css, '@media (max-width: 900px)');
        const owningBlocks = compactBlocks.filter((block) => (
            /\.forge-compact-outcome\s*\{/.test(block)
            && /\.forge-compact-outcome\s*>\s*summary\s*\{/.test(block)
            && /\.forge-compact-outcome\[open\]\s*>\s*\.forge-compact-outcome__body\s*\{/.test(block)
        ));
        expect(owningBlocks).toHaveLength(1);

        const owningBlock = owningBlocks[0] ?? '';
        const disclosure = owningBlock.match(/\.forge-compact-outcome\s*\{([^}]*)\}/)?.[1] ?? '';
        const summary = owningBlock.match(/\.forge-compact-outcome\s*>\s*summary\s*\{([^}]*)\}/)?.[1] ?? '';
        const openBody = owningBlock.match(
            /\.forge-compact-outcome\[open\]\s*>\s*\.forge-compact-outcome__body\s*\{([^}]*)\}/,
        )?.[1] ?? '';

        expect(disclosure).toMatch(/display:\s*block/);
        expect(disclosure).toMatch(/flex:\s*1\s+0\s+100%/);
        expect(disclosure).toMatch(/min-width:\s*0/);
        expect(disclosure).toMatch(/max-width:\s*100%/);
        expect(disclosure).toMatch(/box-sizing:\s*border-box/);
        expect(disclosure).not.toMatch(/(?:^|;)\s*width:\s*\d+(?:px|r?em)\b/);

        expect(summary).toMatch(/display:\s*flex/);
        expect(summary).toMatch(/min-width:\s*0/);
        expect(summary).toMatch(/min-height:\s*44px/);
        expect(summary).toMatch(/box-sizing:\s*border-box/);
        expect(summary).toMatch(/overflow-wrap:\s*anywhere/);
        expect(summary).not.toMatch(/(?:^|;)\s*width:\s*\d+(?:px|r?em)\b/);

        expect(openBody).toMatch(/display:\s*grid/);
        expect(openBody).toMatch(/gap:\s*4px/);
        expect(openBody).toMatch(/min-width:\s*0/);
        expect(openBody).toMatch(/overflow-wrap:\s*anywhere/);
        expect(openBody).not.toMatch(/(?:^|;)\s*width:\s*\d+(?:px|r?em)\b/);
    });

    it('styles app scrollbars with dark chrome', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('scrollbar-color');
        expect(css).toContain('::-webkit-scrollbar-thumb');
        expect(css).toContain('rgba(137, 132, 170, 0.42)');
    });

    it('keeps concept disclosures focus-visible, contained, and touch-sized', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-shell .concept-help__trigger:focus-visible');
        expect(css).toContain('outline: 2px solid var(--color-primary)');
        expect(css).toContain('.forge-shell .concept-help--end .concept-help__content');
        expect(css).toContain('overflow-wrap: anywhere');
        expect(css).toMatch(
            /@media \(max-width: 900px\)[\s\S]*?\.forge-shell \.concept-help__trigger,[\s\S]*?\.forge-shell \.concept-help__target \{[\s\S]*?min-width: 44px/,
        );
        expect(css).not.toContain('var(--accent-cyan)');
    });

    it('pins viewport-overlay concept help after label-relative modifiers', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');
        const aboveIndex = css.indexOf('.forge-shell .concept-help--above .concept-help__content');
        const overlayIndex = css.indexOf(
            '.forge-shell .concept-help.concept-help--viewport-overlay .concept-help__content',
        );

        expect(overlayIndex).toBeGreaterThan(aboveIndex);
        expect(css).toMatch(
            /\.forge-shell \.concept-help\.concept-help--viewport-overlay \.concept-help__content\s*\{[^}]*position:\s*fixed;[^}]*inset:\s*auto 12px 36px auto;/,
        );
    });

    it('keeps the keyboard shortcut disclosure visible, full-width, and touch-sized on compact screens', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-transport-cluster .training-shortcuts > summary:focus-visible');
        expect(css).toMatch(
            /@media \(max-width: 900px\)[\s\S]*?\.forge-transport-cluster \.training-shortcuts \{[\s\S]*?display: block;[\s\S]*?width: 100%;/,
        );
        expect(css).toMatch(
            /@media \(max-width: 900px\)[\s\S]*?\.forge-transport-cluster \.training-shortcuts > summary \{[\s\S]*?min-height: 44px;/,
        );
        expect(css).toMatch(
            /\.forge-transport-cluster \.training-shortcuts\[open\] dl \{[^}]*display: grid;/,
        );
        expect(css).not.toMatch(
            /\.forge-transport-cluster \.training-shortcuts dl \{[^}]*display\s*:/,
        );
    });

    it('keeps topology controls attached to a single graph toolbar surface', () => {
        const css = readFileSync(resolve(__dirname, 'index.css'), 'utf8');

        expect(css).toContain('.network-graph-toolbar');
        expect(css).toContain('.network-graph-toolbar .network-graph-controls');
        expect(css).toContain('.network-graph-toolbar .network-graph-mode-toggle');
    });

    it('defines lab notebook context cards and evidence ownership chrome', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-experiment-context');
        expect(css).toContain('.forge-context-card');
        expect(css).toContain('.forge-state-badge');
        expect(css).toContain('.forge-evidence-context');
        expect(css).toContain('.forge-evidence-frame');
        expect(css).toContain('.forge-cockpit-strip');
        expect(css).toContain('.run-comparison-loop');
    });

});
