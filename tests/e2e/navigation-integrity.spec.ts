import { expect, test } from '@playwright/test';

for (const activation of ['keyboard', 'pointer'] as const) {
    test(`skip-to-main ${activation} preserves the experiment and trained snapshot`, async ({ page }) => {
        await page.goto('./');
        const main = page.getByRole('main', { name: 'Neural network playground workspace' });
        await expect(main).toBeVisible();
        await page.getByRole('button', { name: 'Presets', exact: true }).click();
        const presets = page.getByRole('dialog', { name: 'Presets' });
        await presets.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }).click();
        await expect(presets).toBeHidden();
        await page.getByRole('button', { name: 'run', exact: true }).click();
        const run = page.locator('section[role="region"][aria-label="Current run"]');
        await expect(run).toContainText(/Full evaluation \d+ at step 0(?![0-9,])/);
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expect(run).toContainText(/Full evaluation \d+ at step 1(?![0-9,])/);
        const url = page.url();
        const generation = await run.getAttribute('data-model-generation');
        const revision = await run.getAttribute('data-model-revision');
        expect(generation).not.toBeNull();
        expect(revision).not.toBeNull();
        expect(new URL(url).hash).toMatch(/^#v=2&r=/);

        const skip = page.getByRole('link', { name: 'Skip to main content' });
        await skip.focus();
        if (activation === 'keyboard') await skip.press('Enter');
        else await skip.click();

        await expect(page).toHaveURL(url);
        await expect(main).toBeFocused();
        await expect(run).toHaveAttribute('data-model-generation', generation!);
        await expect(run).toHaveAttribute('data-model-revision', revision!);
        await expect(run).toContainText(/Full evaluation \d+ at step 1(?![0-9,])/);
        await expect(page.getByRole('group', { name: 'Status bar' }))
            .toContainText(/STEP\s+1(?![0-9,])/);
    });
}
