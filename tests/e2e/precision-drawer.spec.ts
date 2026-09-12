import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';

test.use({ hasTouch: true, isMobile: true });

test('Precision drawers remain below a wrapping header and close by touch without changing the experiment', async ({ page }, info) => {
    test.setTimeout(90_000);
    for (const width of [320, 390]) {
        for (const expanded of [false, true]) {
            await page.setViewportSize({ width, height: 844 });
            await page.goto('./');
            const timeline = page.getByRole('region', { name: 'Timeline strip' });
            const checkpoint = timeline.getByRole('slider', { name: 'Checkpoint timeline', exact: true });
            await expect(checkpoint).toHaveAttribute('aria-valuetext', 'Step 0');
            await page.evaluate(async () => { await document.fonts.ready; });
            const outcome = page.getByRole('group', { name: 'Evaluation outcome' });
            if (expanded) await outcome.locator('summary').tap();
            const url = page.url();
            for (const surface of ['Presets', 'Lessons', 'History']) {
                const trigger = page.getByRole('button', { name: surface, exact: true });
                await trigger.tap();
                const drawer = page.getByRole('dialog', { name: surface, exact: true });
                const close = drawer.getByRole('button', { name: `Close ${surface}`, exact: true });
                await expect(close).toBeFocused();
                const hit = await close.evaluate((element) => {
                    const box = element.getBoundingClientRect();
                    const header = document.querySelector('header')!.getBoundingClientRect();
                    const recipient = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
                    return {
                        box: box.toJSON(),
                        header: header.toJSON(),
                        recipient: recipient?.outerHTML,
                        reachesTarget: recipient !== null && element.contains(recipient),
                    };
                });
                await info.attach(`drawer-hit-${width}-${expanded}-${surface}`, {
                    body: Buffer.from(JSON.stringify(hit, null, 2)), contentType: 'application/json',
                });
                expect(hit.reachesTarget, `close control is covered by ${hit.recipient}`).toBe(true);
                expect(hit.box.top).toBeGreaterThanOrEqual(hit.header.bottom);
                expect(hit.box.height).toBeGreaterThanOrEqual(44);
                expect(hit.box.width).toBeGreaterThanOrEqual(44);
                await page.touchscreen.tap(hit.box.x + hit.box.width / 2, hit.box.y + hit.box.height / 2);
                await expect(drawer).toBeHidden();
                await expect(trigger).toBeFocused();
                expect(await outcome.getAttribute('open')).toBe(expanded ? '' : null);
                expect(page.url()).toBe(url);
                await expect(checkpoint).toHaveAttribute('aria-valuetext', 'Step 0');
                await expect(page.getByRole('group', { name: 'Status bar' })).toHaveAttribute('data-status', 'idle');
            }
        }
    }
});
