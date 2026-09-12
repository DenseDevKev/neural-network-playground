import { expect, test } from '@playwright/test';
import { applyPreset, identity, learning, ready, transport, workspace } from './atelier-helpers.ts';

for (const recipe of ['Circle with One Hidden Layer', 'Three-Class Softmax Lab']) {
    test(`paused navigation refreshes all ${recipe} activation maps without a model update`, async ({ page }) => {
        await page.setViewportSize({ width: 1440, height: 1000 });
        await page.goto('./');
        await ready(page);
        await applyPreset(page, recipe);
        await learning(page);
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expect(transport(page)).toHaveAttribute('data-model-step', '1');
        const before = await identity(page);
        await workspace(page, 'Network');
        const hidden = page.getByRole('group', { name: 'Select a neuron' }).getByRole('button', { name: /^Hidden/ });
        await expect(hidden.first()).toHaveAttribute('data-grid-available', 'true');
        const outputs = page.getByRole('group', { name: 'Select a neuron' }).getByRole('button', { name: /^Output/ });
        await expect(outputs).toHaveCount(recipe.startsWith('Three') ? 3 : 1);
        for (const node of [...await hidden.all(), ...await outputs.all()]) {
            await expect(node).toHaveAttribute('data-grid-available', 'true');
        }
        expect(await identity(page)).toEqual(before);
        await expect(page.getByRole('button', { name: 'Start training' })).toBeVisible();
    });
}
