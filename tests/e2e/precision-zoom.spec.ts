import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';
import { identity, ready, transport, utility, workspace } from './atelier-helpers';

for (const view of ['Setup', 'Network'] as const) {
    test(`Signal Atelier retains readable reflow and usable controls at 200% document zoom in ${view}`, async ({ page }, info) => {
        await page.setViewportSize({ width: 735, height: 860 });
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await ready(page);
        await workspace(page, view);
        const before = await identity(page);
        await page.evaluate(async () => { document.documentElement.style.zoom = '2'; await document.fonts.ready; });
        await expect(page.locator('html')).toHaveCSS('zoom', '2');
        const geometry = await page.evaluate(() => ({
            viewport: { width: innerWidth, height: innerHeight },
            main: document.querySelector('main')!.getBoundingClientRect().toJSON(),
            overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        }));
        await info.attach('document-zoom-geometry', { body: Buffer.from(JSON.stringify(geometry, null, 2)), contentType: 'application/json' });
        expect(geometry.overflow).toBeLessThanOrEqual(1);
        await expect(transport(page)).toHaveCSS('position', 'static');
        for (const tab of ['Results', 'Inspect', view] as const) {
            await workspace(page, tab);
            await expect(page.getByRole('tablist', { name: 'Experiment workspace' }).getByRole('tab', { name: tab, exact: true })).toHaveAttribute('aria-selected', 'true');
        }
        await utility(page, 'Guidance');
        await page.getByRole('button', { name: 'Close Guidance', exact: true }).click();
        await expect(page.getByRole('button', { name: 'Utilities', exact: true })).toBeFocused();
        expect(await identity(page)).toEqual(before);
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expect(transport(page)).toHaveAttribute('data-model-step', '1');
        expect(page.url()).toBe(before.url);
        await info.attach('document-zoom', { body: await page.screenshot(), contentType: 'image/png' });
        expect(errors).toEqual([]);
    });
}

test('phone transport and setup actions remain reachable without covering the form across viewport changes', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('./');
    await ready(page);
    await workspace(page, 'Setup');
    await page.getByRole('navigation', { name: 'Setup sections' }).getByRole('button', { name: 'Inputs & layers', exact: true }).click();
    const before = await identity(page);
    for (const [height, zoom] of [[844, 1], [599, 1], [601, 1], [844, 2], [844, 1]]) {
        await page.setViewportSize({ width: 390, height });
        await page.evaluate((value) => { document.documentElement.style.zoom = String(value); }, zoom);
        await expect(transport(page)).toHaveCSS('position', 'static');
        const seed = page.getByLabel('Model seed', { exact: true });
        await seed.focus();
        await seed.scrollIntoViewIfNeeded();
        const [field, footer] = await Promise.all([seed.boundingBox(), page.locator('.atelier-setup-footer').boundingBox()]);
        expect(field).not.toBeNull();
        expect(footer).not.toBeNull();
        expect(field!.y + field!.height).toBeLessThanOrEqual(footer!.y + 1);
        await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeVisible();
        expect(await identity(page)).toEqual(before);
    }
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
});
