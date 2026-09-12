import { expect, test, type Page } from '@playwright/test';
import { expectEvidence, learning, utility } from './atelier-helpers';

function collectBrowserErrors(page: Page): string[] {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
    return errors;
}

async function expectEvidenceConvergence(page: Page, step: number): Promise<void> {
    await expectEvidence(page, step);
    await utility(page, 'Session checkpoints');
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' }))
        .toHaveAttribute('aria-valuetext', `Step ${step.toLocaleString('en-US')}`);
    await page.keyboard.press('Escape');
}

async function runOneStep(page: Page): Promise<void> {
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expectEvidence(page, 1);
}

test('@fault-enabled recovers from an injected startup failure', async ({ page }) => {
    const errors = collectBrowserErrors(page);
    await page.goto('./?e2eWorkerFault=startup-once');

    // Fault injection only exists in bundles built with VITE_E2E_FAULTS=1
    // (test:e2e:recovery); the plain smoke build must skip this scenario.
    const faultsCompiledIn = await page.evaluate(() =>
        (window as { __nnpE2EFaultsEnabled?: boolean }).__nnpE2EFaultsEnabled === true);
    test.skip(!faultsCompiledIn, 'Bundle was built without VITE_E2E_FAULTS=1');

    const dialog = page.getByRole('alertdialog', { name: 'Worker connection lost' });
    await expect(dialog).toBeVisible();
    await expect(dialog).toHaveAccessibleDescription(
        'Injected E2E worker startup failure. Refresh the page to restart the playground.',
    );
    await expect(dialog).toBeFocused();
    const shell = page.locator('.atelier');
    await expect(shell).toHaveAttribute('inert', '');
    await expect(shell).toHaveAttribute('aria-hidden', 'true');
    const refresh = dialog.getByRole('button', { name: 'Refresh page' });
    await refresh.focus();
    await page.keyboard.press('Tab');
    await expect(refresh).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(refresh).toBeFocused();

    const reloaded = page.waitForNavigation({ waitUntil: 'domcontentloaded' });
    await refresh.click();
    await reloaded;
    await expect(dialog).toBeHidden();
    await expect(shell).not.toHaveAttribute('inert', '');
    await expect(shell).not.toHaveAttribute('aria-hidden', 'true');
    await learning(page);
    await expectEvidenceConvergence(page, 0);
    await runOneStep(page);
    expect(errors).toEqual([]);
});

test('@fault-disabled normal builds ignore the fault query', async ({ page }) => {
    const errors = collectBrowserErrors(page);
    await page.goto('./?e2eWorkerFault=startup-once');
    await expect(page.getByRole('alertdialog', { name: 'Worker connection lost' })).toHaveCount(0);
    await learning(page);
    await expectEvidenceConvergence(page, 0);
    await runOneStep(page);
    expect(errors).toEqual([]);
});
