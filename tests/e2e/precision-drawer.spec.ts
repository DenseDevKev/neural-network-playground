import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';
import { identity, ready } from './atelier-helpers';

test.use({ hasTouch: true, isMobile: true });

test('utility dialogs remain reachable beneath the modal layer with a wrapping header', async ({ page }, info) => {
    for (const width of [320, 390]) {
        await page.setViewportSize({ width, height: 844 });
        await page.goto('./');
        await ready(page);
        await page.evaluate(async () => { await document.fonts.ready; });
        const before = await identity(page);
        for (const [surface, title] of [['Export / import', 'Export / import'], ['Session checkpoints', 'Session checkpoints'], ['Guidance', 'Guidance'], ['Shortcuts & help', 'Shortcuts & help']]) {
            const trigger = page.getByRole('button', { name: 'Utilities', exact: true });
            await trigger.tap();
            await page.getByRole('menuitem', { name: surface, exact: true }).tap();
            const dialog = page.getByRole('dialog', { name: title, exact: true });
            const close = dialog.getByRole('button', { name: `Close ${title}`, exact: true });
            await expect(close).toBeInViewport();
            const hit = await close.evaluate((element) => {
                const box = element.getBoundingClientRect();
                const recipient = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
                return { box: box.toJSON(), recipient: recipient?.outerHTML,
                    reachesTarget: recipient !== null && element.contains(recipient) };
            });
            await info.attach(`dialog-hit-${width}-${surface.replaceAll('/', '-')}`, {
                body: Buffer.from(JSON.stringify(hit, null, 2)), contentType: 'application/json',
            });
            expect(hit.reachesTarget, `close control is covered by ${hit.recipient}`).toBe(true);
            expect(hit.box.height).toBeGreaterThanOrEqual(44);
            expect(hit.box.width).toBeGreaterThanOrEqual(44);
            await page.touchscreen.tap(hit.box.x + hit.box.width / 2, hit.box.y + hit.box.height / 2);
            await expect(dialog).toBeHidden();
            await expect(trigger).toBeFocused();
            expect(await identity(page)).toEqual(before);
        }
    }
});
