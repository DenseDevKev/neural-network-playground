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

describe('forge Build/Run instrument CSS', () => {
    it('defines local hierarchy polish tokens without overriding base tokens', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('--forge-surface-muted');
        expect(css).toContain('--forge-surface-active');
        expect(css).toContain('--forge-border-muted');
        expect(css).toContain('--forge-shadow-muted');
        expect(css).toContain('--forge-control-gap');
    });

    it('defines the Build/Run instrument grids and modules', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-buildrun__grid--build');
        expect(css).toContain('.forge-buildrun__grid--run');
        expect(css).toContain('.forge-instrument-module');
        expect(css).toContain('.forge-instrument-module__tag--build');
        expect(css).toContain('.forge-instrument-module__tag--run');
    });

    it('keeps menus and history in drawer chrome instead of permanent panels', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-menu-button');
        expect(css).toContain('.forge-instrument-drawer');
        expect(css).toContain('.forge-instrument-drawer--history');
    });

    it('keeps the compact active-run transport reachable on mobile', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-buildrun__transport');
        expect(css).toContain('position: sticky');
        expect(css).toContain('bottom: 0');
    });

    it('defines compact mobile graph and neuron stepper controls', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.neuron-stepper');
        expect(css).toContain('.neuron-stepper__input');
        expect(css).toContain('.forge-buildrun__transport[data-status="running"]');
        expect(css).toContain('.forge-buildrun__topology-stage .network-graph-toolbar');
        expect(css).toContain('height: 300px');
    });

    it('touch-sizes compact graph and evidence controls in one owning media block', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');
        const compactBlocks = extractMediaBlocks(css, '@media (max-width: 900px)');
        const owningBlock = compactBlocks.find((block) => {
            const targets = block.match(
                /\.forge-buildrun__topology-stage \.network-graph-toolbar button\s*,\s*\.forge-buildrun__topology-stage \.network-graph-legend__filter\s*,\s*\.forge-buildrun__evidence-body \.decision-overlay-controls button\s*\{([^}]*)\}/,
            );
            const summary = block.match(
                /\.forge-buildrun__topology-stage \.network-graph-summary\s*\{([^}]*)\}/,
            );
            return Boolean(
                targets
                && /min-width:\s*44px/.test(targets[1])
                && /min-height:\s*44px/.test(targets[1])
                && summary
                && /top:\s*62px/.test(summary[1]),
            );
        });
        expect(owningBlock).toBeDefined();
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

    it('keeps compact active-run transport reachable on mobile', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-dock--compact .forge-dock__bottom');
        expect(css).toContain('position: sticky');
        expect(css).toContain('bottom: 0');
    });
});
