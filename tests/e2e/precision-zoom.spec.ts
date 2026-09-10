import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';

for (const view of ['build', 'run'] as const) {
    test(`Precision Lab retains phone stacking and usable controls at 200% document zoom in ${view}`, async ({ page }, info) => {
        // July 17 Task 13 explicitly specifies documentElement.style.zoom at 735px.
        // This is a document-zoom/reflow test, not a claim about browser UI shortcuts.
        await page.setViewportSize({ width: 735, height: 860 });
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await page.getByRole('combobox', { name: 'Workspace profile' }).selectOption('lab');
        await page.getByRole('group', { name: 'Workspace view' }).getByRole('button', { name: view, exact: true }).click();
        const canvas = page.locator('[data-decision-boundary-canvas]');
        await expect(canvas).toHaveCount(1);
        const original = await canvas.elementHandle();
        const url = page.url();
        await page.evaluate(async () => { document.documentElement.style.zoom = '2'; await document.fonts.ready; });
        await expect(page.locator('html')).toHaveCSS('zoom', '2');
        // Do not let 100vh become a double-height, clipped application at CSS zoom.
        const shell = await page.locator('.forge-shell').boundingBox();
        expect(shell).not.toBeNull();
        expect(shell!.height).toBeLessThanOrEqual(861);
        await expect(page.locator('.precision-transport')).toHaveCSS('position', 'static');
        const topology = page.locator('[data-precision-region="topology"]');
        const boundary = page.locator('[data-precision-region="boundary"]');
        await topology.scrollIntoViewIfNeeded();
        const geometry = await page.evaluate(() => {
            const rect = (name: string) => {
                const element = document.querySelector(`[data-precision-region="${name}"]`)!;
                const { x, y, width, height } = element.getBoundingClientRect();
                return { x, y, width, height };
            };
            return { topology: rect('topology'), boundary: rect('boundary'),
                overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth };
        });
        await info.attach('document-zoom-geometry', { body: Buffer.from(JSON.stringify(geometry, null, 2)), contentType: 'application/json' });
        expect(geometry.boundary.y).toBeGreaterThanOrEqual(geometry.topology.y + geometry.topology.height - 1);
        expect(geometry.overflow).toBeLessThanOrEqual(1);
        await expect(boundary).toBeVisible();
        for (const label of ['Loss', 'Confusion', 'Boundary']) {
            const tab = page.getByRole('tab', { name: label, exact: true });
            await tab.click();
            await expect(tab).toHaveAttribute('aria-selected', 'true');
        }
        await page.getByRole('button', { name: 'History', exact: true }).click();
        const history = page.getByRole('dialog', { name: 'History', exact: true });
        await history.getByRole('button', { name: 'Close History', exact: true }).click();
        await expect(history).toBeHidden();
        await expect(page.getByRole('button', { name: 'History', exact: true })).toBeFocused();
        const timeline = page.getByRole('region', { name: 'Timeline strip', exact: true });
        const step = timeline.getByRole('button', { name: 'Run one training step', exact: true });
        await step.click();
        await expect(page.getByRole('group', { name: 'Status bar' })).toContainText('STEP 1');
        await expect(canvas).toHaveCount(1);
        expect(await original!.evaluate((element) => element.isConnected)).toBe(true);
        expect(page.url()).toBe(url);
        await info.attach('document-zoom', { body: await page.screenshot(), contentType: 'image/png' });
        expect(errors).toEqual([]);
    });
}
