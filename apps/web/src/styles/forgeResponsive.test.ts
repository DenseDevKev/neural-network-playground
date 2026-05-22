import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

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

    it('styles app scrollbars with dark chrome', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('scrollbar-color');
        expect(css).toContain('::-webkit-scrollbar-thumb');
        expect(css).toContain('rgba(137, 132, 170, 0.42)');
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
