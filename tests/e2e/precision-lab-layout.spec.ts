import { Buffer } from 'node:buffer';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { identity, ready, transport, workspace } from './atelier-helpers';

async function horizontalOverflow(page: Page) {
    return page.evaluate(() => {
        const main = document.querySelector<HTMLElement>('.atelier-main')!;
        return { document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            main: main.scrollWidth - main.clientWidth };
    });
}
async function expectControlContained(control: Locator, container: Locator, label: string) {
    const [box, parent] = await Promise.all([control.boundingBox(), container.boundingBox()]);
    expect(box, `${label} has measurable bounds`).not.toBeNull();
    expect(parent, `${label} container has measurable bounds`).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(parent!.x - 1);
    expect(box!.x + box!.width).toBeLessThanOrEqual(parent!.x + parent!.width + 1);
    expect(box!.height).toBeGreaterThanOrEqual(44);
}

for (const viewport of [{ width: 1280, height: 720 }, { width: 1437, height: 742 }, { width: 390, height: 844 }]) {
    test(`first-visit lesson invitation works at ${viewport.width}x${viewport.height}`, async ({ page }) => {
        await page.setViewportSize(viewport);
        await page.goto('./');
        await ready(page);
        await expect(page.getByRole('tablist', { name: 'Experiment workspace' }).getByRole('tab', { name: 'Network', exact: true })).toHaveAttribute('aria-selected', 'true');
        const cue = page.getByRole('complementary', { name: 'Getting started' });
        const start = cue.getByRole('button', { name: 'Explore lessons', exact: true });
        const dismiss = cue.getByRole('button', { name: 'Dismiss lesson suggestion', exact: true });
        await expectControlContained(start, cue, 'Explore lessons');
        await expectControlContained(dismiss, cue, 'Dismiss suggestion');
        await start.click();
        await expect(page.getByRole('heading', { name: 'Learn by experimenting' })).toBeVisible();
        await page.getByRole('button', { name: 'Playground', exact: true }).click();
        await dismiss.click();
        await expect(cue).toHaveCount(0);
        await page.reload();
        await ready(page);
        await expect(cue).toHaveCount(0);
    });
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1024, height: 900 }, { width: 768, height: 900 }, { width: 390, height: 844 }, { width: 360, height: 800 }]) {
    test(`visible graph artifacts and training layout stay stable at ${viewport.width}x${viewport.height}`, async ({ page, browserName }, info) => {
        await page.setViewportSize(viewport);
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await ready(page);
        await page.getByRole('button', { name: 'Dismiss lesson suggestion' }).click();
        const initial = await identity(page);
        const boundary = page.locator('[data-decision-boundary-canvas]');
        const graph = page.getByRole('img', { name: 'Neural network graph', exact: true });
        for (const view of ['Setup', 'Results', 'Inspect', 'Network'] as const) {
            await workspace(page, view);
            expect(await identity(page)).toEqual(initial);
            if (view === 'Setup' || view === 'Inspect') { await expect(boundary).toHaveCount(0); await expect(graph).toHaveCount(0); }
        }
        if (viewport.width < 760) {
            const regions = page.getByRole('tablist', { name: 'Network regions' });
            await regions.getByRole('tab', { name: 'Prediction', exact: true }).click();
            await expect(boundary).toHaveCount(1);
            await expect(graph).toHaveCount(0);
            await regions.getByRole('tab', { name: 'Network', exact: true }).click();
            await expect(boundary).toHaveCount(0);
        } else await expect(boundary).toHaveCount(1);
        await expect(graph).toBeVisible();
        const graphGeometry = await page.locator('.network-graph-frame').evaluate((frame) => {
            const plot = frame.querySelector('.network-graph-container')!.getBoundingClientRect();
            const legend = frame.querySelector('.network-graph-legend')!.getBoundingClientRect();
            return { plot: plot.toJSON(), legend: legend.toJSON() };
        });
        expect(graphGeometry.legend.top, 'Legend stays outside the interactive graph').toBeGreaterThanOrEqual(graphGeometry.plot.bottom - 1);
        await page.evaluate(async () => { await document.fonts.ready; });
        await page.getByRole('combobox', { name: 'Steps per frame' }).selectOption('50');
        const regions = page.locator('.atelier-network, .atelier-metrics, .atelier-transport');
        const bounds = () => regions.evaluateAll((nodes) => nodes.map((node) => ({ name: node.className, rect: { ...node.getBoundingClientRect().toJSON(), y: node.getBoundingClientRect().y + window.scrollY } })));
        await page.getByRole('button', { name: 'Start training', exact: true }).scrollIntoViewIfNeeded();
        const before = await bounds();
        const overflowBefore = await horizontalOverflow(page);
        await page.evaluate(() => {
            const target = window as typeof window & { __atelierShifts?: {value:number;sources:string[]}[] };
            target.__atelierShifts = [];
            if (PerformanceObserver.supportedEntryTypes.includes('layout-shift')) {
                new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        const shift = entry as PerformanceEntry & { value: number; hadRecentInput: boolean; sources:{node?:Node}[] };
                        if (!shift.hadRecentInput) target.__atelierShifts!.push({value:shift.value,sources:shift.sources.map(source=>source.node instanceof Element ? source.node.outerHTML : source.node?.textContent ?? '')});
                    }
                }).observe({ type: 'layout-shift' });
            }
        });
        await page.getByRole('button', { name: 'Start training', exact: true }).click();
        await expect(transport(page)).toHaveAttribute('data-status', 'running');
        await expect(page.getByRole('button', { name: 'Run one training step' })).toBeDisabled();
        await page.waitForTimeout(2500);
        const after = await bounds();
        const overflowDuring = await horizontalOverflow(page);
        const shifts = await page.evaluate(() => (window as typeof window & { __atelierShifts?: {value:number;sources:string[]}[] }).__atelierShifts ?? []);
        await info.attach('region-stability', { body: Buffer.from(JSON.stringify({ viewport, before, after, shifts, overflowBefore, overflowDuring }, null, 2)), contentType: 'application/json' });
        for (let i = 0; i < before.length; i++) for (const dimension of ['x', 'y', 'width', 'height'] as const) {
            expect(Math.abs(after[i].rect[dimension] - before[i].rect[dimension]), `${before[i].name}.${dimension}`).toBeLessThanOrEqual(1);
        }
        if (browserName === 'chromium') expect(shifts.reduce((sum, shift) => sum + shift.value, 0)).toBe(0);
        await page.getByRole('button', { name: 'Pause training', exact: true }).click();
        await expect(transport(page)).toHaveAttribute('data-status', 'paused');
        expect(Number(await transport(page).getAttribute('data-model-step'))).toBeGreaterThan(0);
        for (const sample of [overflowBefore, overflowDuring, await horizontalOverflow(page)]) for (const [region, overrun] of Object.entries(sample)) expect(overrun, `${region} horizontal overflow`).toBeLessThanOrEqual(1);
        expect(page.url()).toBe(initial.url);
        await info.attach('workspace', { body: await page.screenshot(), contentType: 'image/png' });
        expect(errors).toEqual([]);
    });
}
