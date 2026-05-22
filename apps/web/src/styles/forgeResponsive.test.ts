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

    it('docks the lesson drawer inside the transport cluster with a compact breakpoint', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('.forge-transport-cluster');
        expect(css).toContain('grid-template-columns: minmax(0, 1fr) minmax(290px, 360px)');
        expect(css).toContain('grid-template-columns: minmax(0, 1fr)');
    });

    it('defines local hierarchy polish tokens without overriding base tokens', () => {
        const css = readFileSync(resolve(__dirname, 'forge.css'), 'utf8');

        expect(css).toContain('--forge-surface-muted');
        expect(css).toContain('--forge-surface-active');
        expect(css).toContain('--forge-border-muted');
        expect(css).toContain('--forge-shadow-muted');
        expect(css).toContain('--forge-control-gap');
    });

    it('keeps topology controls attached to a single graph toolbar surface', () => {
        const css = readFileSync(resolve(__dirname, 'index.css'), 'utf8');

        expect(css).toContain('.network-graph-toolbar');
        expect(css).toContain('.network-graph-toolbar .network-graph-controls');
        expect(css).toContain('.network-graph-toolbar .network-graph-mode-toggle');
    });

    it('keeps guided lesson visually secondary until active lesson state', () => {
        const css = readFileSync(resolve(__dirname, 'index.css'), 'utf8');

        expect(css).toContain('.guided-lesson--active');
        expect(css).toContain('.guided-lesson:not(.guided-lesson--active)');
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
