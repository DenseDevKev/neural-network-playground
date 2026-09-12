import { expect, test } from '@playwright/test';
import { applyDataset, currentRun, expectEvidence, identity, learning, ready, transport } from './atelier-helpers';

for (const activation of ['keyboard', 'pointer'] as const) {
    test(`skip-to-main ${activation} preserves the experiment and trained snapshot`, async ({ page }) => {
        await page.goto('./');
        const main = page.getByRole('main', { name: 'Neural network playground workspace' });
        await expect(main).toBeVisible();
        await ready(page);
        await applyDataset(page, 'XOR');
        await learning(page);
        const run = currentRun(page);
        await expectEvidence(page, 0);
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expectEvidence(page, 1);
        const before = await identity(page);
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
        await expect(transport(page)).toHaveAttribute('data-model-step', '1');
        expect(await identity(page)).toEqual(before);
    });
}
