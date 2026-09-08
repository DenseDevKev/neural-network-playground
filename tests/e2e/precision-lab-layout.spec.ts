import { Buffer } from 'node:buffer';
import { expect, test, type Page } from '@playwright/test';

const REGIONS = ['recipe', 'rail', 'topology', 'boundary', 'selection', 'evidence', 'transport'] as const;
async function bounds(page: Page) {
    return page.locator('[data-precision-region]').evaluateAll((nodes) => Object.fromEntries(nodes.map((node) => {
        const { x, y, width, height } = node.getBoundingClientRect();
        return [node.getAttribute('data-precision-region'), { x, y, width, height }];
    })));
}

async function horizontalOverflow(page: Page) {
    return page.evaluate(() => {
        const workspace = document.querySelector<HTMLElement>('[data-precision-workspace]');
        const main = document.querySelector<HTMLElement>('.forge-workspace');
        if (!workspace || !main) throw new Error('Precision Lab workspace is missing');
        return {
            document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            workspace: workspace.scrollWidth - workspace.clientWidth,
            main: main.scrollWidth - main.clientWidth,
        };
    });
}

for (const viewport of [{ width: 1437, height: 742 }, { width: 735, height: 860 }, { width: 320, height: 844 }]) {
    for (const view of ['build', 'run'] as const) {
    test(`Precision Lab retains one live boundary and stable regions at ${viewport.width}x${viewport.height} in ${view}`, async ({ page, browserName }, info) => {
        await page.setViewportSize(viewport);
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await expect(page.locator('[data-precision-workspace]')).toBeVisible();
        await page.getByRole('button', { name: 'Presets', exact: true }).click();
        await page.getByRole('dialog', { name: 'Presets' }).getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers', exact: true }).click();
        await page.getByRole('combobox', { name: 'Workspace profile' }).selectOption('lab');
        await page.getByRole('button', { name: view, exact: true }).click();
        const timeline = page.getByRole('region', { name: 'Timeline strip' });
        await expect(timeline.getByRole('slider', { name: 'Checkpoint timeline', exact: true })).toHaveAttribute('aria-valuetext', 'Step 0');
        if (view === 'run') await expect(page.getByRole('region', { name: 'Current run', exact: true })).toContainText(/Full evaluation \d+ at step 0(?!\d)/);
        const canvas = page.locator('[data-decision-boundary-canvas]');
        await expect(canvas).toHaveCount(1);
        const originalCanvas = await canvas.elementHandle();
        expect(originalCanvas).not.toBeNull();
        const url = page.url();
        const advanced = page.getByRole('button', { name: 'Advanced Tools', exact: true });
        await advanced.click();
        await advanced.click();
        expect(page.url()).toBe(url);
        const context = page.locator('.precision-context');
        if (await context.count()) await context.getByRole('button', { name: /^Close / }).click();
        for (const label of ['Loss', 'Confusion', 'Inspect', 'Code', 'Boundary']) {
            const tab = page.getByRole('tab', { name: label, exact: true });
            await tab.click();
            await expect(tab).toHaveAttribute('aria-selected', 'true');
            await expect(canvas).toHaveCount(1);
            expect(await originalCanvas!.evaluate((element) => element.isConnected)).toBe(true);
            expect(page.url()).toBe(url);
        }
        await page.evaluate(async () => { await document.fonts.ready; });
        await timeline.getByRole('button', { name: '50 steps per frame', exact: true }).click();
        await timeline.getByRole('button', { name: 'Start training', exact: true }).scrollIntoViewIfNeeded();
        const before = await bounds(page);
        const overflowBefore = await horizontalOverflow(page);
        await page.evaluate(() => {
            const target = window as typeof window & { __precisionLayoutShifts?: number[] };
            target.__precisionLayoutShifts = [];
            if (PerformanceObserver.supportedEntryTypes.includes('layout-shift')) {
                const observer = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        const shift = entry as PerformanceEntry & { value: number; hadRecentInput: boolean };
                        if (!shift.hadRecentInput) target.__precisionLayoutShifts!.push(shift.value);
                    }
                });
                observer.observe({ type: 'layout-shift' });
            }
        });
        await timeline.getByRole('button', { name: 'Start training', exact: true }).click();
        await expect(page.getByRole('group', { name: 'Status bar' })).toHaveAttribute('data-status', 'running');
        await page.waitForTimeout(5500);
        const after = await bounds(page);
        const overflowDuring = await horizontalOverflow(page);
        const shifts = await page.evaluate(() => (window as typeof window & { __precisionLayoutShifts?: number[] }).__precisionLayoutShifts ?? []);
        await info.attach('region-stability', { body: Buffer.from(JSON.stringify({ viewport, view, before, after, shifts, overflowBefore, overflowDuring }, null, 2)), contentType: 'application/json' });
        for (const name of REGIONS) {
            expect(before[name], `${name} exists before training`).toBeDefined();
            expect(after[name], `${name} exists after training`).toBeDefined();
            expect(after[name].width).toBeGreaterThan(0);
            expect(after[name].height).toBeGreaterThan(0);
            for (const dimension of ['x', 'y', 'width', 'height'] as const) {
                expect(Math.abs(after[name][dimension] - before[name][dimension]), `${name}.${dimension}`).toBeLessThanOrEqual(1);
            }
        }
        if (browserName === 'chromium') expect(shifts.reduce((sum, value) => sum + value, 0)).toBe(0);
        await timeline.getByRole('button', { name: 'Pause training', exact: true }).click();
        await expect(page.getByRole('group', { name: 'Status bar' })).toHaveAttribute('data-status', 'paused');
        const overflowAfter = await horizontalOverflow(page);
        for (const sample of [overflowBefore, overflowDuring, overflowAfter]) {
            for (const [region, overrun] of Object.entries(sample)) expect(overrun, `${region} horizontal overflow`).toBeLessThanOrEqual(1);
        }
        expect(await originalCanvas!.evaluate((element) => element.isConnected)).toBe(true);
        expect(page.url()).toBe(url);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
        await page.getByRole('button', { name: 'History', exact: true }).click();
        const drawer = page.getByRole('dialog', { name: 'History', exact: true });
        const close = drawer.getByRole('button', { name: 'Close History', exact: true });
        await expect(close).toBeInViewport();
        await close.click();
        await expect(page.getByRole('button', { name: 'History', exact: true })).toBeFocused();
        await info.attach('workspace', { body: await page.screenshot(), contentType: 'image/png' });
        expect(errors).toEqual([]);
    });
}
}
