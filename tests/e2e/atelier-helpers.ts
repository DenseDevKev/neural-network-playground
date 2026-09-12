import { expect, type Page } from '@playwright/test';

export const transport = (page: Page) => page.getByRole('region', { name: 'Training controls', exact: true });
export const currentRun = (page: Page) => page.locator('section[role="region"][aria-label="Current run"]');

export async function ready(page: Page): Promise<void> {
    await expect(page.getByRole('main', { name: 'Neural network playground workspace' })).toBeVisible();
    await expect(transport(page)).toHaveAttribute('data-model-generation', /.+/);
    await expect(page.getByRole('button', { name: 'Run one training step' })).toBeEnabled();
}

export async function workspace(page: Page, name: 'Setup' | 'Network' | 'Results' | 'Inspect'): Promise<void> {
    await page.getByRole('button', { name: 'Playground', exact: true }).click();
    await page.getByRole('tablist', { name: 'Experiment workspace' }).getByRole('tab', { name, exact: true }).click();
}

export async function learning(page: Page): Promise<void> {
    await workspace(page, 'Results');
    await page.getByRole('tab', { name: 'Learning progress', exact: true }).click();
    await expect(currentRun(page)).toBeVisible();
}

export async function expectEvidence(page: Page, step: number): Promise<void> {
    await expect(transport(page)).toHaveAttribute('data-model-step', String(step));
    await expect(currentRun(page)).toContainText(new RegExp(`Full evaluation \\d+ at step ${step.toLocaleString('en-US')}(?![0-9,])`));
}

export async function utility(page: Page, name: 'Export / import' | 'Session checkpoints' | 'Guidance' | 'Shortcuts & help'): Promise<void> {
    await page.getByRole('button', { name: 'Utilities', exact: true }).click();
    await page.getByRole('menuitem', { name, exact: true }).click();
}

export async function identity(page: Page) {
    return {
        url: page.url(),
        generation: await transport(page).getAttribute('data-model-generation'),
        revision: await transport(page).getAttribute('data-model-revision'),
        step: await transport(page).getAttribute('data-model-step'),
    };
}

export async function applyDataset(page: Page, name: string): Promise<void> {
    await workspace(page, 'Setup');
    await page.getByRole('navigation', { name: 'Setup sections' }).getByRole('button', { name: 'Data', exact: true }).click();
    await page.getByRole('button', { name, exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Experiment setup' })).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled();
    await ready(page);
}

export async function applyPreset(page: Page, title: string): Promise<void> {
    await workspace(page, 'Setup');
    const presets = page.locator('.atelier-recipe-presets');
    if (await presets.getAttribute('open') === null) await page.getByText('Start from a preset', { exact: true }).click();
    await page.getByRole('group', { name: 'Recipe presets' }).getByRole('button', { name: new RegExp(`^${title}`) }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Experiment setup' })).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled();
    await ready(page);
}
