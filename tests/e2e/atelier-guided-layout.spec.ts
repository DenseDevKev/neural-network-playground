import { expect, test } from '@playwright/test';
import { ready, workspace } from './atelier-helpers';

for (const width of [1440, 1280, 390]) {
    test(`active lesson aligns with the workspace at ${width}px in both themes`, async ({ page }) => {
        await page.setViewportSize({ width, height: 1024 });
        await page.goto('./');
        await ready(page);
        await page.getByRole('button', { name: 'Lessons', exact: true }).click();
        if (width < 900) {
            await page.getByRole('button', { name: 'Expand guided lesson drawer' }).click();
            await page.getByRole('combobox', { name: 'Guided lesson', exact: true }).selectOption({ label: 'Circle With One Hidden Layer' });
        } else await page.locator('.lesson-list').getByRole('button', { name: /Circle With One Hidden Layer/ }).click();
        await page.getByRole('button', { name: 'Start lesson and reset', exact: true }).click();
        await ready(page);
        for (const theme of ['light', 'dark']) {
            await page.getByLabel('Color theme').selectOption(theme);
            for (const tab of ['Setup', 'Network'] as const) {
                await workspace(page, tab);
                const guide = page.getByRole('complementary', { name: 'Guided lesson mode' });
                await expect(guide).toBeVisible();
                const [lessonBox, workspaceBox] = await Promise.all([
                    guide.boundingBox(), page.locator('.atelier-workspace').boundingBox(),
                ]);
                expect(lessonBox).not.toBeNull();
                expect(workspaceBox).not.toBeNull();
                if (width >= 1200) {
                    expect(Math.abs(lessonBox!.y - workspaceBox!.y)).toBeLessThanOrEqual(1);
                    expect(lessonBox!.x).toBeGreaterThan(workspaceBox!.x + workspaceBox!.width);
                } else expect(lessonBox!.y).toBeGreaterThanOrEqual(workspaceBox!.y + workspaceBox!.height);
                expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
            }
        }
    });
}
